"""A whole run coordinated on a lab server (`perception remote run | status | attach | fetch`), for runs that go on with
this machine asleep or gone: run-anywhere phase 7. The other half, which runs there, is remote_coordinator.py.

    perception remote run HOST GRAPH --input video=PATH [--only ...] [--set ...] [--place ...] [--max-cost-usd N]
                          [--allow-uploads] [--allow-downloads] [--run-id ID] [--dry-run] [--detach]
    perception remote status HOST [RUN]
    perception remote attach HOST RUN
    perception remote fetch HOST RUN [--output DIR]

What leaves this machine, and when. Nothing before the checks below pass, and then, in this order:

1. A listed copy of the pipeline, by content (<root>/pipelines/<hash>): the files git tracks for the pipeline (the
   folders the package carries: cloud, envs, graphs, nodes, schema, sdk; with the orchestrator's own source and its
   pinned requirements), or, from an installed package, the files its RELEASE.json lists. The copy gets a RELEASE.json
   of its own that lists them, as a package's does, so the coordinator there lists a node's code as this machine does
   (remote.code_files) and its cache keys are the ones this machine computes for the same node on that host
   (Backend.identity: the code, runner, accelerator and platform, never the machine). Results made there are then hits
   here once imported, and the same as a run coordinated here with its steps placed on that host. Sent once per
   content, from a snapshot checked against its hash (a file saved meanwhile is not filed under the old hash).
2. The orchestrator's environment there, once per set of requirements (<root>/envs/orchestrator-<key>).
3. Each graph input (the video), filed in the host's root by its sha256 (objects/<sha256>/data), as a step placed there
   is sent its inputs (remote_driver.send_inputs): once per host, checked there against its hash.
4. The run's spec on the SSH connection's stdin: the graph (with --only, --set, --place and --max-cost-usd applied, and
   each input named only by its sha256 and file name, not this machine's path), the run's consent, and the keys its steps
   declare (below). The coordinator there is started detached (remote_coordinator: setsid), so closing this machine
   ends nothing there.

The checks, before anything is sent:
- **The graph's privacy policy, with HOST as the destination.** With the coordinator on HOST every step is local to it,
  so a step's own `sends()` would see nothing leave; running there is consent for HOST itself (RUN_ANYWHERE_PLAN.md,
  "Privacy across routes"). The graph's `policy.allow_upload` must name every kind in the run's inputs (the video) and
  every kind its steps make (an output type that could hold anything counts as every kind), and the run needs
  `--allow-uploads`. A graph that keeps its nodes on this machine (`privacy: local-only`, `placement: local`) is
  refused.
- **Places.** Only HOST's own (native, self): `runpod:*` and `cloud:*` are refused (the RunPod key and AWS credentials
  never leave this machine), and so are other hosts (remote_coordinator.refused_places, checked there too).
- **Keys.** The keys the run's steps declare (and only those) go with it, if HOST has the user's consent for that:
  `perception hosts keys HOST on` (off by default). Without it a run whose steps need keys is refused, naming them.
  The keys go in the spec on stdin (never a command line, a file or a log, here or there), into the memory of the
  coordinator there, and from it only into the environment of the steps that declare them (child_env), as on this
  machine. `--dry-run` (the `remote_plan` object) says which keys go to which steps. HF_TOKEN goes to the steps that
  declare it, as every key does (narrowed to downloads in run-anywhere phase 5).

The run's id is chosen here and said first (`run.created`), so a start whose answer is lost can be asked again with
`--run-id`: a run already being coordinated there is attached to, not started a second time, and one whose coordinator
is gone (the host restarted) is resumed.

`fetch` exports the run's bundle there and brings it back, and imports the run into this machine's store: its run,
task and event rows, its results' cache rows and every object they name (copied, then checked against its hash here,
as the store publishes). Imported tasks get attempt numbers of this store's own (one past the last it has for that
node), their attempt logs beside them, and where they ran says HOST (`ran_on`); a cache row this store already has is
never replaced. After a fetch, planning the same steps placed on HOST here finds them cached.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import shutil
import subprocess
import time
import uuid
from functools import cached_property
from pathlib import Path

from . import artifacts
from .artifacts import sha256_file, sha256_tree, special_files
from .hosts import Host
from .ir import UPLOAD_KINDS, Graph
from .release import REPO, home, release
from .remote_coordinator import refused_places
from .remote_driver import PUBLISH, Unreachable
from .store import Artifact, Store, _read_only, dumps

# what the package carries (scripts/build-dist.sh), and what a copy for a coordinator elsewhere carries too
PIPELINE = ["cloud", "envs", "graphs", "nodes", "schema", "sdk"]
ORCHESTRATOR = ["orchestrator/src"]
REQUIREMENTS = "orchestrator/requirements.txt"
COPY_LAYOUT = "copy-1"   # how a copy is made (its RELEASE.json and requirements): a change gives every copy a new hash
POLL_S = 2.0


class RemoteError(RuntimeError):
    """The host refused, or could not answer as expected: said as an `error` (problems)."""
    def __init__(self, *problems: str):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


# --- the checks, before anything leaves -----------------------------------------------------------------------------

def made_kinds(graph: Graph) -> dict[str, list[str]]:
    """Each data kind the run would hold on the host -> what brings it there: a graph input, or the steps that make it
    (an output type that could hold anything, json.v1 or text.v1, counts as every kind)."""
    kinds: dict[str, list[str]] = {}
    for name, spec in graph.spec.inputs.items():
        for k in artifacts.get(spec.type).kinds or UPLOAD_KINDS:
            kinds.setdefault(k, []).append(f"input {name}")
    for node in graph.nodes.values():
        for t in node.manifest.outputs.values():
            for k in artifacts.get(t).kinds if artifacts.get(t).kinds is not None else UPLOAD_KINDS:
                if node.id not in kinds.setdefault(k, []):
                    kinds[k].append(node.id)
    return {k: kinds[k] for k in sorted(kinds)}


def remote_plan(host: Host, graph: Graph, doc: dict, secrets: dict[str, str], allow_uploads: bool) -> dict:
    """What `remote run` would send HOST and why it would not start: the `remote_plan` object (docs/cli-json.md)."""
    policy = graph.spec.policy
    problems = []
    if policy.privacy == "local-only" or policy.placement == "local":
        problems.append("the graph keeps its nodes on this machine (policy.placement / policy.privacy): it cannot run on "
                        f"{host.name}")
    kinds = made_kinds(graph)
    if refused := {k: by for k, by in kinds.items() if k not in policy.allow_upload}:
        problems.append(f"{host.name} would hold " + "; ".join(f"{k} (from {', '.join(by)})" for k, by in refused.items())
                        + ", which policy.allow_upload does not let leave this machine")
    if not allow_uploads:
        problems.append(f"running on {host.name} sends it the video and what every step makes: run with --allow-uploads")
    problems += refused_places(doc)
    keys: dict[str, list[str]] = {}
    for node in graph.nodes.values():
        for s in node.manifest.secrets:
            keys.setdefault(s, []).append(node.id)
    if keys and not host.keys:
        problems.append(f"{', '.join(sorted({n for ns in keys.values() for n in ns}))} need {', '.join(sorted(keys))}, and "
                        f"{host.name} may not hold your keys for its runs: `perception hosts keys {host.name} on` lets it "
                        "(or leave those steps out with --only)")
    if missing := sorted(k for k in keys if k not in secrets):
        problems.append(f"{', '.join(missing)} not set here (perception keys set NAME): {', '.join(sorted({n for k in missing for n in keys[k]}))} need it")
    nodes = [{"id": n.id, "uses": n.manifest.ref, "secrets": list(n.manifest.secrets), "service": n.manifest.service,
              "uploads": list(n.manifest.privacy.external_upload), "place": n.spec.place} for n in graph.nodes.values()]
    return {"host": host.name, "ok": not problems, "problems": problems, "nodes": nodes, "uploads": list(kinds),
            "upload_to": f"{host.name} (SSH)", "keys": {k: keys[k] for k in sorted(keys)}, "keys_allowed": host.keys,
            "max_cost_usd": policy.max_cost_usd}


# --- the copy, and commands there -----------------------------------------------------------------------------------

def git_files(paths: list[str]) -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--", *paths], capture_output=True, text=True)
    if out.returncode:
        raise RemoteError(f"{REPO} is not a git checkout: only files git tracks are sent ({out.stderr.strip()})")
    return sorted(f for f in out.stdout.split("\0") if f)


class Remote:
    """One host, as a coordinator for whole runs. `python`: an interpreter there that has the orchestrator's
    dependencies, used instead of the environment made from the copy's requirements (the tests' stand-in host)."""

    def __init__(self, host: Host, nodes: list[str] | None = None, python: str | None = None):
        self.host, self.python = host, python
        self.root = host.root
        self.home = f"{self.root}/coordinator"
        self.ssh = host.ssh()   # every command to a host goes through it (its overrides: hosts.Host.ssh)
        # node folders given with --nodes, as the copy names them: only ones inside the pipeline's files are sent
        self.nodes = []
        for n in nodes or []:
            path = Path(n).resolve()
            if not path.is_relative_to(REPO):
                raise RemoteError(f"--nodes {n}: not in the pipeline's repository, so not sent to {host.name}")
            self.nodes.append(path.relative_to(REPO).as_posix())

    def sh(self, script: str, data: bytes | None = None, timeout: float = 600, check: bool = True) -> subprocess.CompletedProcess:
        out = subprocess.run([*self.ssh, self.host.dest, f"sh -c {shlex.quote(script)}"], input=data, capture_output=True,
                             timeout=timeout)
        if out.returncode == 255:
            raise Unreachable(out.stderr.decode(errors="replace").strip()[-300:])
        if check and out.returncode:
            raise RemoteError(f"on {self.host.name}: exit {out.returncode}: {out.stderr.decode(errors='replace').strip()[-500:]}")
        return out

    def rsync(self, *args: str):
        shell = " ".join(shlex.quote(a) for a in self.ssh)
        out = subprocess.run(["rsync", "-rlt", "-e", shell, *args], capture_output=True, text=True, timeout=24 * 3600)
        if out.returncode:
            raise RemoteError(f"rsync {' '.join(args[-2:])} failed ({out.returncode}): {out.stderr.strip()[-500:]}")

    # the copy ---------------------------------------------------------------------------------------------------

    @cached_property
    def files(self) -> dict[str, Path]:
        """What the copy holds (its path there -> this machine's file): module docstring, step 1."""
        if r := release():   # an installed package: its listed files, and its own source as the orchestrator's
            pkg = Path(__file__).resolve().parent
            out = {f: REPO / f for f in r["files"]}
            out |= {f"orchestrator/src/perception/{p.relative_to(pkg).as_posix()}": p for p in sorted(pkg.rglob("*.py"))
                    if "_pipeline" not in p.relative_to(pkg).parts and "__pycache__" not in p.parts}
            return out
        from .backends.remote import refuse_links
        listed = git_files([*PIPELINE, *ORCHESTRATOR, *self.nodes])
        refuse_links(listed)
        return {f: REPO / f for f in listed}

    @cached_property
    def requirements(self) -> str:
        """The orchestrator's dependencies, pinned: a checkout's lock (uv export), or the versions a package requires
        (scripts/build-dist.sh pins them exactly)."""
        if release():
            from importlib.metadata import requires
            return "".join(f"{r}\n" for r in requires("hyperstudy-perception") or [])
        uv = shutil.which("uv") or "uv"
        out = subprocess.run([uv, "export", "--project", str(REPO / "orchestrator"), "--frozen", "--no-dev", "--no-hashes",
                              "--no-emit-project", "--no-header", "--no-annotate"], capture_output=True, text=True)
        if out.returncode:
            raise RemoteError(f"could not list the orchestrator's requirements (uv export): {out.stderr.strip()[-300:]}")
        return out.stdout

    def commit(self) -> str:
        if r := release():
            return r["commit"]
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
        return out.stdout.strip() or "unknown"

    def copy_hash(self, files: dict[str, Path]) -> str:
        """The copy's content: every file by path and sha256, its requirements and its commit (what its RELEASE.json says)."""
        import hashlib
        h = hashlib.sha256(f"{COPY_LAYOUT} {self.commit()}\n".encode())
        for rel in sorted(files):
            h.update(rel.encode() + b"\0" + hashlib.sha256(files[rel].read_bytes()).digest())
        h.update(b"requirements\0" + self.requirements.encode())
        return h.hexdigest()[:20]

    @cached_property
    def private_root(self) -> str:
        """The host's root, checked to be still the user's own private folder (hosts.PRIVATE_ROOT) before anything is
        sent there or read from it: what is kept there (copies, inputs, the coordinator's store) is trusted for that."""
        from .hosts import private_root_script
        self.sh(private_root_script(self.root))
        return self.root

    @cached_property
    def copy(self) -> str:
        """The copy's folder there, sent first if it is not there yet (module docstring, step 1)."""
        q = shlex.quote
        h = self.copy_hash(self.files)
        target = f"{self.private_root}/pipelines/{h}"
        if self.sh(f"test -e {q(target)}/.complete", check=False).returncode == 0:
            return target
        stage = home() / "tmp" / f"{os.getpid()}-copy-{uuid.uuid4().hex}"   # named for this process (Store.__init__)
        try:
            for rel, src in self.files.items():
                (stage / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, stage / rel)
            snapshot = {rel: stage / rel for rel in self.files}
            if self.copy_hash(snapshot) != h:
                raise RemoteError("the pipeline's files changed while they were being copied: try again")
            (stage / REQUIREMENTS).parent.mkdir(parents=True, exist_ok=True)
            (stage / REQUIREMENTS).write_text(self.requirements)
            (stage / "RELEASE.json").write_text(json.dumps({"commit": self.commit(), "copy": h, "files": sorted(self.files)}, indent=0))
            incoming = f"{self.root}/pipelines/.incoming-{uuid.uuid4().hex[:12]}"
            self.sh(f"mkdir -p {q(incoming)}")
            try:
                self.rsync(f"{stage}/", f"{self.host.dest}:{incoming}/")
                # read-only once published: its name is its content
                self.sh(f"{PUBLISH}\ntouch {q(incoming)}/.complete && publish {q(incoming)} {q(target)} && chmod -R a-w {q(target)}")
            finally:
                self.sh(f"[ -e {q(incoming)} ] && {{ chmod -R u+w {q(incoming)}; rm -rf {q(incoming)}; }}; true", check=False)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        return target

    @cached_property
    def interpreter(self) -> str:
        """The orchestrator's Python there: an environment made once per set of requirements (module docstring, step 2)."""
        if self.python:
            return self.python
        import hashlib
        q = shlex.quote
        uv = self.host.caps.tools.get("uv")
        if not uv:
            raise RemoteError(f"{self.host.name} has no uv (perception hosts check {self.host.name} after installing it)")
        key = hashlib.sha256(f"{COPY_LAYOUT}\n{self.requirements}".encode()).hexdigest()[:20]
        env = f"{self.root}/envs/orchestrator-{key}"
        if self.sh(f"test -e {q(env)}/.complete", check=False).returncode:
            done = self.sh(f"{q(uv)} venv --quiet --allow-existing --python '>=3.12' {q(env)} && "
                           f"{q(uv)} pip install --quiet --python {q(env)}/bin/python -r {q(self.copy + '/' + REQUIREMENTS)} && "
                           f"touch {q(env)}/.complete", check=False, timeout=1800)
            if done.returncode:
                raise RemoteError(f"could not make the orchestrator's environment on {self.host.name}: "
                                  f"{done.stderr.decode(errors='replace').strip()[-500:]}")
        return f"{env}/bin/python"

    def call(self, action: str, *args: str, data: bytes | None = None, timeout: float = 600) -> dict:
        """remote_coordinator ACTION there, from the copy, with the coordinator's store as PERCEPTION_HOME: its last
        JSON answer. Its refusal is a RemoteError with its problems."""
        q = shlex.quote
        path = ":".join([*sorted({str(Path(p).parent) for p in self.host.caps.tools.values() if p}), "$HOME/.local/bin"])
        extra = " ".join(f"--nodes {q(n)}" for n in self.nodes)
        script = (f"umask 077; mkdir -p {q(self.home)} && cd {q(self.home)} && "
                  f"PERCEPTION_HOME={q(self.home)} PERCEPTION_NO_KEYCHAIN=1 PYTHONPATH={q(self.copy + '/orchestrator/src')} "
                  f"PATH=\"{path}:$PATH\" exec {q(self.interpreter)} -m perception.remote_coordinator {action} "
                  + " ".join(q(a) for a in args) + (f" {extra}" if extra else ""))
        out = self.sh(script, data=data, timeout=timeout, check=False)
        lines = [line for line in out.stdout.decode(errors="replace").splitlines() if line.startswith("{")]
        try:
            last = json.loads(lines[-1])
        except (IndexError, ValueError):
            raise RemoteError(f"{self.host.name} did not answer {action} (exit {out.returncode}): "
                              f"{out.stderr.decode(errors='replace').strip()[-800:]}")
        if not last.get("ok"):
            e = RemoteError(*last.get("problems", [f"{action} failed"]))
            e.answer = last
            raise e
        return last

    def send_input(self, path: Path, sha: str, run: str, name: str):
        """A graph input as objects/<sha256>/data there, once per host, checked there against its hash and published in
        one step, read-only (remote_driver.send_inputs files a step's inputs so: the same folder serves both). Held for
        the run first, as a step's attempt holds its inputs (remote_driver.HOLD, under the objects' lock), so the host's
        retention cleanup never removes it while the run may read it: the hold names coordinator/remote/<run>/held, a
        folder the run's coordinator removes when the run ends (remote_coordinator.release), and the hold with it."""
        from .remote_driver import HOLD, locked
        q = shlex.quote
        root = self.private_root
        folder = f"{root}/objects/{sha}"
        rel = f"coordinator/remote/{run}/held"
        held = self.sh(locked(root, f"{HOLD}\nhold {q(root)} {q(root + '/' + rel)} {q(rel)} {q(run + '-remote-' + name)} {q(sha)}"),
                       check=False)
        if held.returncode:
            raise RemoteError(f"could not hold {name} on {self.host.name} (is its disk full?): "
                              f"{held.stderr.decode(errors='replace').strip()[-300:]}")
        if f"there {sha}" in held.stdout.decode(errors="replace"):
            return
        tag = f"{run}-input-{name}"
        local = path.resolve()
        incoming = f"{self.root}/objects/.incoming-{tag}"
        self.sh(f"rm -rf {q(incoming)} && mkdir -p {q(incoming)}")
        try:
            self.rsync(str(local), f"{self.host.dest}:{incoming}/data")
            self.sh(f"{PUBLISH}\n"
                    f"got=$( (sha256sum {q(incoming)}/data 2>/dev/null || shasum -a 256 {q(incoming)}/data) | cut -d' ' -f1)\n"
                    f"[ \"$got\" = {q(sha)} ] || {{ echo \"{path.name} arrived as $got, not {sha}: it changed since it was read\" >&2; exit 5; }}\n"
                    f"touch {q(incoming)}/.complete && publish {q(incoming)} {q(folder)} && chmod -R a-w {q(folder)}")
        finally:
            self.sh(f"[ -e {q(incoming)} ] && {{ chmod -R u+w {q(incoming)}; rm -rf {q(incoming)}; }}; true", check=False)


# --- run, attach -----------------------------------------------------------------------------------------------------

def record(run: str) -> Path:
    """What this machine remembers of a run it started elsewhere: its host and its inputs' paths here (fetch puts them
    back in the imported run), and how far a fetch has imported its events."""
    return home() / "remote" / f"{run}.json"


def start(remote: Remote, run: str, graph: Graph, doc: dict, secrets: dict[str, str], allow_uploads: bool,
          allow_downloads: bool) -> dict:
    """Send what the run needs (module docstring, steps 1-4) and start it there: the host's answer."""
    store = Store(home())
    inputs, here = {}, {}
    for name in graph.spec.inputs:
        path = graph.input_path(name)
        if not path.exists():
            raise RemoteError(f"graph input {name}: {path} does not exist")
        sha = store.source(path, graph.spec.inputs[name].type).hash
        inputs[name] = {"sha256": sha, "filename": path.name}
        here[name] = str(path)
    record(run).parent.mkdir(parents=True, exist_ok=True)
    record(run).write_text(json.dumps({"host": remote.host.name, "inputs": here, "graph": str(graph.source or "")}))
    remote.copy, remote.interpreter   # the copy and its environment before the video: a host that cannot run it gets no clip
    for name in graph.spec.inputs:
        remote.send_input(Path(here[name]), inputs[name]["sha256"], run, name)
    # nothing of this machine's paths: each input is named by its sha256 and file name (remote_coordinator.with_inputs)
    sent = {**doc, "inputs": {k: {**v, "uri": inputs[k]["filename"]} for k, v in doc["inputs"].items()}}
    declared = {s for n in graph.nodes.values() for s in n.manifest.secrets}
    keys = {k: v for k, v in secrets.items() if k in declared} if remote.host.keys else {}
    spec = {"run": run, "graph": sent, "inputs": inputs, "allow_uploads": allow_uploads, "allow_downloads": allow_downloads,
            "nodes": remote.nodes, "keys": keys}
    data = json.dumps(spec).encode()
    del keys, spec
    return remote.call("start", data=data)


def follow(remote: Remote, run: str, on_event, after: int = 0) -> str:
    """Stream the run's events (on_event, as they come) until it ends there; its last status. A host that cannot be
    reached is waited for (this machine changed networks, or slept): the run goes on there regardless. A run whose
    coordinator is gone is said so (RemoteError): `remote run --run-id` resumes it."""
    waiting = False
    while True:
        try:
            got = remote.call("events", run, "--after", str(after))
        except Unreachable as e:
            if not waiting:
                print(f"perception: {remote.host.name} cannot be reached ({e}); waiting (the run goes on there)", file=sys.stderr)
                waiting = True
            time.sleep(POLL_S * 5)
            continue
        waiting = False
        for e in got["events"]:
            on_event(e)
            after = e["id"]
        if got["status"] != "running":
            return got["status"]
        if got["coordinator"] == "gone":
            raise RemoteError(f"run {run}'s coordinator on {remote.host.name} is gone (the host restarted?): "
                              f"`perception remote run {remote.host.name} GRAPH ... --run-id {run}` resumes it")
        time.sleep(POLL_S)


# --- fetch and import --------------------------------------------------------------------------------------------------

def fetch(remote: Remote, run: str, output: Path) -> dict:
    """The run's bundle, exported there, into `output`, and the run imported into this machine's store (module
    docstring). What was imported."""
    dump = remote.call("dump", run, timeout=3600)
    store = Store(home())
    stage = store.root / "tmp" / f"{os.getpid()}-fetch-{uuid.uuid4().hex}"   # named for this process (Store.__init__)
    stage.mkdir(parents=True)
    try:
        wanted = [o for o in dump["objects"] if not (store.root / o).exists()] + [l["path"] for l in dump["logs"]]
        if dump["bundle"]:
            wanted.append(dump["bundle"])
        if wanted:
            listing = stage / "files.txt"
            listing.write_text("".join(f"{w}\n" for w in wanted))
            remote.rsync("--files-from", str(listing), f"{remote.host.dest}:{remote.home}/", f"{stage}/got/")
            for d in [stage / "got", *(p for p in (stage / "got").rglob("*") if p.is_dir() and not p.is_symlink())]:
                d.chmod(0o700)   # rsync keeps the host's read-only folders so: they are checked, moved and removed here
        bundle = None
        if dump["bundle"]:
            if (output / "manifest.json").exists():
                raise RemoteError(f"{output} already holds a bundle")
            output.parent.mkdir(parents=True, exist_ok=True)
            got = stage / "got" / dump["bundle"]
            if special_files(got):
                raise RemoteError("the bundle there holds a link or a special file: not brought back")
            shutil.copytree(got, output, dirs_exist_ok=True)
            for f in output.rglob("*"):
                if f.is_file():
                    f.chmod(0o644)   # as export writes a bundle
            bundle = str(output)
        imported = import_run(store, remote.host, dump, stage / "got")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {"run": run, "host": remote.host.name, "path": bundle, "bundle_problem": dump.get("bundle_problem"), **imported}


def _objects(store: Store, dump: dict, got: Path) -> int:
    """Each object the run names, checked against its hash here (as Store.publish would compute it) and moved into the
    store unless it is there already. A folder that does not hash to its name is refused, and nothing is imported."""
    types = {a["hash"]: a for a in dump["artifacts"]}
    brought = 0
    for rel in dump["objects"]:
        parts = Path(rel).parts
        if len(parts) != 3 or parts[0] not in ("objects", "trees") or parts[1] != parts[2][:2] or not artifacts_hash(parts[2]):
            raise RemoteError(f"{rel}: not a store object's folder")
        final = store.root / rel
        if final.exists():
            continue
        src = got / rel
        if special_files(src) or not src.is_dir():
            raise RemoteError(f"{rel}: missing, or holds a link or a special file")
        if parts[0] == "objects":
            files = list(src.iterdir())
            if len(files) != 1 or not files[0].is_file() or sha256_file(files[0]) != parts[2]:
                raise RemoteError(f"{rel} is not the file its hash names")
        elif sha256_tree(src) != parts[2]:
            raise RemoteError(f"{rel} is not the folder its hash names")
        final.parent.mkdir(parents=True, exist_ok=True)
        _read_only(src)
        os.replace(src, final)
        brought += 1
        a = types.get(parts[2])
        if a:
            artifacts.get(a["type"])   # a type this store knows
            store.db.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?,?,?,?)", (a["hash"], a["shape"], a["type"], a["bytes"], a["created_at"]))
    return brought


def artifacts_hash(name: str) -> bool:
    return len(name) == 64 and all(c in "0123456789abcdef" for c in name)


def _local(store: Store, a: dict) -> dict:
    """An artifact as the host's store named it, as this store names it: objects and trees by hash; a source (the video)
    at its path here, when this machine started the run."""
    if a["shape"] == "source":
        return a
    folder = store.object_dir(a["shape"], a["hash"])
    path = folder if a["shape"] == "tree" else next(folder.iterdir())
    return {**a, "path": str(path)}


def _here(usage: str | None, host: Host) -> str | None:
    """Where a result was made, as this machine names it: a step the coordinator there ran on its own machine (native,
    self) ran on HOST, which is `ssh-HOST:...` here; the run says it was coordinated there too."""
    if not usage:
        return usage
    u = json.loads(usage)
    ran = u.get("ran_on")
    if isinstance(ran, dict) and str(ran.get("place", "")).split(":")[0] in ("native", "self"):
        u["ran_on"] = {**ran, "place": f"ssh-{host.name}:{ran['place'].split(':', 1)[1]}", "title": f"{host.name} (SSH)",
                       "coordinator": host.name}
    return dumps(u)


def _outputs(store: Store, outputs: str | None) -> str | None:
    if not outputs:
        return outputs
    return dumps({k: _local(store, a) for k, a in json.loads(outputs).items()})


def import_run(store: Store, host: Host, dump: dict, got: Path) -> dict:
    """The run, its tasks, events and cache rows, into this store (module docstring). A run of this id that this store
    has and did not import from HOST is another run: refused."""
    run = dump["run"]
    run_id = run["id"]
    mine = json.loads(record(run_id).read_text()) if record(run_id).exists() else {}
    local = store.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    imported_from = store.db.execute("SELECT 1 FROM events WHERE run_id=? AND kind='run.imported' AND message LIKE ?",
                                     (run_id, f"from {host.name}:%")).fetchone()
    if local and not imported_from:
        raise RemoteError(f"this machine's store has another run {run_id}: not imported over it")
    brought = _objects(store, dump, got)
    # the graph's inputs where they are here (this machine started the run): its doc and its input rows
    graph = json.loads(run["graph"])
    inputs = json.loads(run["inputs"])
    for name, path in (mine.get("inputs") or {}).items():
        if name in graph.get("inputs", {}):
            graph["inputs"][name]["uri"] = Path(path).as_uri()
        if name in inputs:
            inputs[name]["path"] = path
    tasks_in, kept, added = 0, 0, 0
    with store.tx() as db:
        if not local:
            db.execute("INSERT INTO runs (id, pipeline, graph, graph_path, inputs, status, created_at, finished_at, max_cost_usd, spent_usd, allow_uploads) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (run_id, run["pipeline"], dumps(graph), mine.get("graph") or None, dumps(inputs), run["status"],
                        run["created_at"], run["finished_at"], run["max_cost_usd"], run["spent_usd"], run["allow_uploads"]))
        else:
            db.execute("UPDATE runs SET status=?, finished_at=?, spent_usd=? WHERE id=?",
                       (run["status"], run["finished_at"], run["spent_usd"], run_id))
        logs = {(l["node"], l["attempt"]): l["path"] for l in dump["logs"]}
        for t in dump["tasks"]:
            now = db.execute("SELECT * FROM tasks WHERE run_id=? AND node_id=?", (run_id, t["node_id"])).fetchone()
            outputs = _outputs(store, t["outputs"])
            if now and (now["state"], now["cache_key"], now["outputs"]) == (t["state"], t["cache_key"], outputs):
                continue   # imported before, unchanged
            # an attempt number of this store's own: one past the last it has for this node (none: the first)
            attempt = (now["attempt"] if now else 0) + 1 if t["attempt"] else (now["attempt"] if now else 0)
            runner = t["runner"]
            if runner and runner.split(":")[0] in ("native", "self"):
                runner = f"ssh-{host.name}:{runner.split(':', 1)[1]}"
            row = (t["state"], attempt, attempt or 1, t["cache_key"], runner, t["started_at"], t["finished_at"], t["error_code"],
                   t["error"], t["cost_usd"], _here(t["usage"], host), outputs, t["gpu"], t["gpu_vram_gb"])
            if now:
                db.execute("UPDATE tasks SET state=?, attempt=?, first_attempt=?, cache_key=?, runner=?, started_at=?, finished_at=?, "
                           "error_code=?, error=?, cost_usd=?, usage=?, outputs=?, gpu=?, gpu_vram_gb=?, lease_owner=NULL, "
                           "lease_expires=NULL, handle=NULL, not_before=NULL WHERE run_id=? AND node_id=?", (*row, run_id, t["node_id"]))
            else:
                db.execute("INSERT INTO tasks (state, attempt, first_attempt, cache_key, runner, started_at, finished_at, error_code, "
                           "error, cost_usd, usage, outputs, gpu, gpu_vram_gb, run_id, node_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (*row, run_id, t["node_id"]))
            tasks_in += 1
            log = logs.get((t["node_id"], t["attempt"]))
            if attempt and log and (got / log).is_file() and not (got / log).is_symlink():
                folder = store.root / "work" / run_id / t["node_id"] / str(attempt)
                folder.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(got / log, folder / "log.txt")
        for c in dump["cache"]:   # never over this store's own: a key it has keeps its row
            if db.execute("SELECT 1 FROM cache WHERE key=?", (c["key"],)).fetchone():
                kept += 1
                continue
            db.execute("INSERT INTO cache VALUES (?,?,?,?,?)", (c["key"], c["node"], _outputs(store, c["outputs"]),
                                                                 _here(c["usage"], host), c["created_at"]))
            added += 1
        after = mine.get("imported_events", 0)
        for e in dump["events"]:
            if e["id"] > after:
                db.execute("INSERT INTO events (run_id, node_id, at, kind, message) VALUES (?,?,?,?,?)",
                           (run_id, e["node_id"], e["at"], e["kind"], e["message"]))
                after = e["id"]
        db.execute("INSERT INTO events (run_id, node_id, at, kind, message) VALUES (?,?,?,?,?)",
                   (run_id, None, time.time(), "run.imported",
                    f"from {host.name}: {tasks_in} tasks, {added} cache rows ({kept} this machine had kept), {brought} objects"))
    record(run_id).parent.mkdir(parents=True, exist_ok=True)
    record(run_id).write_text(json.dumps({**mine, "host": host.name, "imported_events": after}))
    return {"tasks": tasks_in, "cache_added": added, "cache_kept": kept, "objects": brought}
