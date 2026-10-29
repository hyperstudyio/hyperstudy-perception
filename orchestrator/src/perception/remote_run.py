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

`fetch` imports the run into this machine's store: its run,
task and event rows and every object its results name (copied, then checked against its hash here, as the store
publishes), and a cache row for each result whose key this machine computes itself for that step (its node's identity
on HOST, its config, its inputs' hashes). Nothing the host answers is trusted beyond what is checked (the host is
shared: root or another of the user's processes there could rewrite its store): the graph, nodes and inputs are this
machine's own record of the run (only runs it started are imported), every name must be a node of that graph and every
path one of the store's own layouts, and nothing comes back as a link. The bundle is exported here, from the run just
imported, never taken from the host. Imported tasks get attempt numbers of this
store's own (one past the last it has for that node), their attempt logs beside them, and where they ran says HOST
(`ran_on`); a cache row this store already has is never replaced. After a fetch, planning the same steps placed on HOST
here finds them cached.
"""

from __future__ import annotations

import json
import os
import re
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
        held = artifacts.get(spec.type).kinds   # () holds none; None (json.v1, text.v1) could hold any (Backend.sends)
        for k in held if held is not None else UPLOAD_KINDS:
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

    def rsync(self, *args: str, links: bool = True):
        """`links`: copy links as links (what this machine sends: the copy's files, never followed). Bringing anything
        back, none: a link there is skipped, never made here, where it could point at a file of this machine's."""
        shell = " ".join(shlex.quote(a) for a in self.ssh)
        out = subprocess.run(["rsync", "-rlt" if links else "-rt", "-e", shell, *args], capture_output=True, text=True,
                             timeout=24 * 3600)
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

    def use_copy_of(self, run: str | None = None):
        """Call the host from the copy a run started with (its record here; without a run, the latest one this machine
        started there), if it is still there: reading a run sends nothing, whatever was edited here since (code review
        of #46). Otherwise this machine's own copy is sent first, as for a start."""
        records = [record(run)] if run else sorted((home() / "remote").glob("*.json"), reverse=True)
        q = shlex.quote
        for path in records:
            try:
                mine = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if mine.get("host") != self.host.name or not isinstance(mine.get("copy"), str):
                continue
            copy = mine["copy"]
            if not copy.startswith(f"{self.root}/pipelines/") or \
                    self.sh(f"test -e {q(copy)}/.complete && test -x {q(mine.get('interpreter') or '')}", check=False).returncode:
                return   # gone there (hosts clean): this machine's copy then
            self.__dict__["copy"] = copy
            self.__dict__["interpreter"] = mine["interpreter"]
            return

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
    # the graph as this machine ran it (its inputs at their paths here): what an import trusts, never the host's copy.
    # Written once: a start under the same id with another graph (another --set or --only, other inputs, another host)
    # is another run, refused before anything is sent, and never changes the record fetch checks the first one against
    # each input by its hash too: one rewritten in place (the same path, other bytes) is another run's input, which a
    # resumed run must never read under the first one's cache keys (re-review of #46)
    mine = {"host": remote.host.name, "inputs": here, "input_sha256": {k: v["sha256"] for k, v in inputs.items()},
            "graph": str(graph.source or ""), "graph_doc": doc}
    if record(run).exists():
        had = json.loads(record(run).read_text())
        if any(had.get(k) != mine[k] for k in ("host", "inputs", "input_sha256", "graph_doc")):
            raise RemoteError(f"run {run} was started with another graph (or other inputs, or on another host): "
                              "a run's id names one run")
    else:
        record(run).parent.mkdir(parents=True, exist_ok=True)
        record(run).write_text(json.dumps(mine))
    remote.copy, remote.interpreter   # the copy and its environment before the video: a host that cannot run it gets no clip
    had = json.loads(record(run).read_text())
    if "copy" not in had:   # the copy it started with: status, attach and fetch call it (Remote.use_copy_of)
        record(run).write_text(json.dumps({**had, "copy": remote.copy, "interpreter": remote.interpreter}))
    for name in graph.spec.inputs:
        remote.send_input(Path(here[name]), inputs[name]["sha256"], run, name)
    # nothing of this machine's paths: each input is named by its sha256 and file name (remote_coordinator.with_inputs)
    sent = {**doc, "inputs": {k: {**v, "uri": inputs[k]["filename"]} for k, v in doc["inputs"].items()}}
    declared = {s for n in graph.nodes.values() for s in n.manifest.secrets}
    keys = {k: v for k, v in secrets.items() if k in declared} if remote.host.keys else {}
    spec = {"run": run, "graph": sent, "inputs": inputs, "allow_uploads": allow_uploads, "allow_downloads": allow_downloads,
            "nodes": remote.nodes, "keys": keys,
            # the host's retention, for the runs coordinated there (remote_coordinator.prune)
            "keep_s": remote.host.keep_s, "min_free_gb": remote.host.min_free_gb}
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
        if got.get("status") not in RUN_STATES | {"running"} or not isinstance(got.get("events"), list):
            raise RemoteError(f"{remote.host.name} did not answer as a run's events")
        for e in got["events"]:   # data from the host, said as it is (JSON-escaped under --json), never a path here
            if not isinstance(e, dict) or type(e.get("id")) is not int or e["id"] <= after:
                raise RemoteError(f"{remote.host.name} answered with an event that is not one")
            on_event({"node": _text(e.get("node"), "event node", 64), "kind": _text(e.get("kind"), "event kind", 64) or "",
                      "message": _text(e.get("message"), "event") or "", "at": _text(e.get("at"), "event time", 32)})
            after = e["id"]
        if got["status"] != "running":
            return got["status"]
        if got["coordinator"] == "gone":
            raise RemoteError(f"run {run}'s coordinator on {remote.host.name} is gone (the host restarted?): "
                              f"`perception remote run {remote.host.name} GRAPH ... --run-id {run}` resumes it")
        time.sleep(POLL_S)


# --- fetch and import --------------------------------------------------------------------------------------------------
#
# Everything a host sends back is data that root or another process of the user's there could have rewritten (the host is
# shared). So an import trusts nothing of it but bytes it checks: the run's graph, its nodes and its inputs come from this
# machine's own record of the run it started (record); every name the dump gives is checked against them before it
# names a path here; each object is checked against its hash here; and a cache row is added only for a task of this run
# whose key this machine computes itself (cache_key: its node's identity on the host's platform and accelerator, its
# config, its inputs' hashes), so no host can plant a result under a key another step or other inputs would look up.

NODE_ID = re.compile(r"^[A-Za-z0-9_-]+$")           # as ir.NodeSpec.id
PLACE = re.compile(r"^[a-z][a-z0-9-]*:[a-z]+$")      # as ir.NodeSpec.place
HEX64 = re.compile(r"^[0-9a-f]{64}$")
KIND = re.compile(r"^[a-z][a-z._-]{0,40}$")
TASK_STATES = {"succeeded", "cached", "failed", "canceled"}
RUN_STATES = {"succeeded", "failed", "canceled"}
ERROR_CODES = {"transient", "input", "schema", "privacy", "licence", "resources", "internal", "budget", "upstream", "policy"}


def _number(x, what: str, none: bool = True) -> float | None:
    if x is None and none:
        return None
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise RemoteError(f"the host's {what} is not a number")
    return x


def _text(x, what: str, limit: int = 4000) -> str | None:
    if x is None:
        return None
    if not isinstance(x, str):
        raise RemoteError(f"the host's {what} is not text")
    return x[:limit]


def _usage(x) -> dict:
    try:
        u = json.loads(x) if isinstance(x, str) else {}
    except ValueError:
        u = {}
    return u if isinstance(u, dict) else {}


def _within(got: Path, rel: str) -> Path:
    """A path the dump named, as it lies under what was brought back: never outside it (after links are resolved)."""
    path = (got / rel).resolve()
    if not path.is_relative_to(got.resolve()):
        raise RemoteError(f"{rel} is not where the host's store keeps it")
    return path


def checked(dump: dict, run: str, graph: Graph) -> dict:
    """The dump, checked against this machine's own record of the run (module comment above): the run it names, one task
    per node of the graph, each name and number of the kind it must be, each output a store object of the type its node
    makes, and every path one of the layouts the host's store has. A RemoteError names the first thing that is not."""
    if not isinstance(dump, dict) or not isinstance(dump.get("run"), dict) or dump["run"].get("id") != run:
        raise RemoteError(f"the host answered for another run than {run}: nothing imported")
    r = dump["run"]
    if r.get("status") not in RUN_STATES:
        raise RemoteError(f"run {run} has not ended there")
    nodes = graph.nodes
    tasks = {}
    for t in dump.get("tasks") or []:
        node = t.get("node_id") if isinstance(t, dict) else None
        if not isinstance(node, str) or not NODE_ID.match(node) or node not in nodes or node in tasks:
            raise RemoteError(f"the host's dump names {node!r}, not a node of the run (or twice)")
        attempt = t.get("attempt")
        if type(attempt) is not int or not 0 <= attempt < 10_000:
            raise RemoteError(f"{node}: the host's attempt {attempt!r} is not a number of one")
        if t.get("state") not in TASK_STATES:
            raise RemoteError(f"{node}: the host's state {t.get('state')!r} is not an ended task's")
        runner = t.get("runner")
        if runner is not None and not (isinstance(runner, str) and PLACE.match(runner)):
            raise RemoteError(f"{node}: the host's place {runner!r} is not one")
        key = t.get("cache_key")
        if key is not None and not (isinstance(key, str) and HEX64.match(key)):
            raise RemoteError(f"{node}: the host's cache key is not one")
        code = t.get("error_code")
        if code is not None and code not in ERROR_CODES:
            raise RemoteError(f"{node}: the host's error code {code!r} is not one")
        outputs = {}
        if t["state"] in ("succeeded", "cached"):
            m = nodes[node].manifest
            try:
                claimed = json.loads(t.get("outputs") or "")
            except ValueError:
                claimed = None
            if not isinstance(claimed, dict) or set(claimed) != set(m.outputs):
                raise RemoteError(f"{node}: the host's outputs are not what {m.ref} makes")
            for name, a in claimed.items():
                want = m.outputs[name]
                if not isinstance(a, dict) or a.get("type") != want or a.get("shape") != artifacts.get(want).shape:
                    if isinstance(a, dict) and a.get("shape") not in ("file", "tree"):
                        raise RemoteError(f"{node}.{name}: not a store object (a {a.get('shape')!r})")
                    raise RemoteError(f"{node}.{name}: not what {m.ref} makes ({want})")
                if a["shape"] not in ("file", "tree"):
                    raise RemoteError(f"{node}.{name}: not a store object (a {a['shape']!r})")
                if not isinstance(a.get("hash"), str) or not HEX64.match(a["hash"]):
                    raise RemoteError(f"{node}.{name}: its hash is not one")
                outputs[name] = {"hash": a["hash"], "shape": a["shape"], "type": want}
        gpu = t.get("gpu")
        if gpu is not None and type(gpu) is not int:
            raise RemoteError(f"{node}: the host's card is not a number")
        tasks[node] = {"node_id": node, "attempt": attempt, "state": t["state"], "runner": runner, "cache_key": key,
                       "outputs": outputs, "usage": _usage(t.get("usage")), "error_code": code,
                       "error": _text(t.get("error"), "error"), "gpu": gpu,
                       **{f: _number(t.get(f), f, none=f in ("started_at", "finished_at"))
                          for f in ("started_at", "finished_at", "cost_usd", "gpu_vram_gb")}}
    if set(tasks) != set(nodes):
        raise RemoteError(f"the host's dump does not have the run's nodes ({', '.join(sorted(nodes))})")
    logs = {}
    for entry in dump.get("logs") or []:
        node, attempt = entry.get("node") if isinstance(entry, dict) else None, entry.get("attempt") if isinstance(entry, dict) else None
        if node not in tasks or type(attempt) is not int or attempt != tasks[node]["attempt"] or attempt == 0 \
                or entry.get("path") != f"work/{run}/{node}/{attempt}/log.txt":
            raise RemoteError(f"a log the host named ({entry.get('path') if isinstance(entry, dict) else entry!r}) is not where its "
                              "store keeps an attempt's")
        logs[node] = entry["path"]
    events = []
    for e in dump.get("events") or []:
        node = e.get("node_id") if isinstance(e, dict) else "?"
        if node is not None and node not in nodes:
            raise RemoteError(f"an event names {node!r}, not a node of the run")
        if type(e.get("id")) is not int or not (isinstance(e.get("kind"), str) and KIND.match(e["kind"])):
            raise RemoteError("an event of the host's is not one")
        events.append({"id": e["id"], "node_id": node, "kind": e["kind"], "message": _text(e.get("message"), "event") or "",
                       "at": _number(e.get("at"), "event time", none=False)})
    run_row = {f: _number(r.get(f), f, none=f == "finished_at") for f in ("created_at", "finished_at", "max_cost_usd", "spent_usd")}
    return {"run": {**run_row, "status": r["status"]}, "tasks": tasks, "logs": logs, "events": events}


def fetch(remote: Remote, run: str, output: Path) -> dict:
    """The run's bundle, exported there, into `output`, and the run imported into this machine's store (module
    docstring). What was imported. Only a run this machine started there: its record holds the graph and inputs that the
    host's answer is checked against."""
    from .manifests import Registry
    from .ir import GraphError, compile_graph
    mine = json.loads(record(run).read_text()) if record(run).exists() else {}
    if mine.get("host") != remote.host.name or not isinstance(mine.get("graph_doc"), dict):
        raise RemoteError(f"this machine did not start run {run} on {remote.host.name}: only runs it started there are "
                          "imported (their graph and inputs are its own record)")
    registry = Registry.load(*[r for r in [REPO / "nodes", *(REPO / n for n in remote.nodes)] if r.exists()])
    # each input at the absolute path the run was started with (a URI relative to the graph file would otherwise be
    # read relative to wherever fetch runs: code review of #46)
    doc = json.loads(json.dumps(mine["graph_doc"]))
    for name, path in (mine.get("inputs") or {}).items():
        if name in doc.get("inputs", {}):
            doc["inputs"][name]["uri"] = Path(path).as_uri()
    mine = {**mine, "graph_doc": doc}
    try:
        graph = compile_graph(mine["graph_doc"], registry)
    except GraphError as e:
        raise RemoteError(f"run {run}'s graph no longer compiles here: {'; '.join(e.problems)}")
    dump = checked(remote.call("dump", run, timeout=3600), run, graph)
    store = Store(home())
    stage = store.root / "tmp" / f"{os.getpid()}-fetch-{uuid.uuid4().hex}"   # named for this process (Store.__init__)
    stage.mkdir(parents=True)
    got = stage / "got"
    try:
        # every type claimed for each object (each the graph's for its output: checked), so an object two outputs name
        # is checked against both, never only the one claimed last (security review of #46)
        objects: dict[str, set[str]] = {}
        for t in dump["tasks"].values():
            for a in t["outputs"].values():
                objects.setdefault(store.object_dir(a["shape"], a["hash"]).relative_to(store.root).as_posix(), set()).add(a["type"])
        wanted = [o for o in objects if not (store.root / o).exists()] + list(dump["logs"].values())
        if dump["run"]["status"] == "succeeded" and (output / "manifest.json").exists() and not own_bundle(output, run):
            raise RemoteError(f"{output} already holds a bundle (of another run)")
        got.mkdir()
        if wanted:
            listing = stage / "files.txt"
            listing.write_text("".join(f"{w}\n" for w in wanted))
            remote.rsync("--files-from", str(listing), f"{remote.host.dest}:{remote.home}/", f"{got}/", links=False)
            for d in [got, *(p for p in got.rglob("*") if p.is_dir() and not p.is_symlink())]:
                d.chmod(0o700)   # rsync keeps the host's read-only folders so: they are checked, moved and removed here
        imported = import_run(store, remote.host, run, graph, mine, dump, got, objects)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    # the bundle, exported here from the run just imported (its objects checked against their hashes), never one the host
    # kept: that could have been rewritten there after the run (a layer, a licence mode), and is what the annotator reads
    bundle, problem = None, None
    if dump["run"]["status"] == "succeeded":
        from . import export as export_mod
        from .scheduler import Coordinator
        # exported beside it, then put in its place: a fetch again to the same folder (bundles/RUN by default) replaces
        # the bundle an earlier fetch of this run wrote there (code review of #46)
        fresh = output.parent / f".{output.name}.{os.getpid()}"
        shutil.rmtree(fresh, ignore_errors=True)
        try:
            export_mod.bundle(Coordinator(store, registry, []), run, fresh)
            if own_bundle(output, run):
                shutil.rmtree(output)
            os.replace(fresh, output)
            bundle = str(output)
        except SystemExit as e:   # export's own refusals (unrelated writers of a layer, no decode node)
            problem = str(e)
        finally:
            shutil.rmtree(fresh, ignore_errors=True)
    return {"run": run, "host": remote.host.name, "path": bundle, "bundle_problem": problem, **imported}


def own_bundle(folder: Path, run: str) -> bool:
    """Whether `folder` holds a bundle exported from this run (by an earlier fetch): one a fetch may replace."""
    try:
        manifest = json.loads((folder / "manifest.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(manifest, dict) and isinstance(manifest.get("pipeline"), dict) \
        and manifest["pipeline"].get("source") == f"perception run {run}"


def _objects(store: Store, objects: dict[str, set[str]], got: Path) -> int:
    """Each object the run's outputs name (its folder in the store -> every artifact type claimed for it), checked
    against its hash here (as Store.publish computes it) and against each type claimed for it (as every commit checks an
    output: scheduler._publish), one this store has already too, and moved into the store unless it is there already. One that is missing, does not hash to its name, or is not what its
    type says refuses the import."""
    brought = 0
    for rel in objects:
        final = store.root / rel
        shape, h = ("tree" if rel.startswith("trees/") else "file"), Path(rel).name
        if final.exists():   # this store's own (checked against its hash when published here): its type still checked
            src = final
        else:
            src = _within(got, rel)
        if not src.is_dir() or special_files(src):
            raise RemoteError(f"{rel}: missing there, or holds a link or a special file")
        if shape == "file":
            files = list(src.iterdir())
            if len(files) != 1 or not files[0].is_file() or sha256_file(files[0]) != h:
                raise RemoteError(f"{rel} is not the file its hash names")
        elif sha256_tree(src) != h:
            raise RemoteError(f"{rel} is not the folder its hash names")
        readable = src if shape == "tree" else next(src.iterdir())
        for type_id in sorted(objects[rel]):
            if artifacts.get(type_id).shape != shape:
                raise RemoteError(f"{rel} is not a {type_id} (a {shape})")
            if problems := artifacts.get(type_id).validate(readable):
                raise RemoteError(f"{rel} is not a {type_id}: {'; '.join(problems[:3])}")
    for rel in objects:   # only once every one has been checked: a refused import leaves this store as it was
        final = store.root / rel
        if final.exists():
            continue
        src = _within(got, rel)
        final.parent.mkdir(parents=True, exist_ok=True)
        _read_only(src)
        os.replace(src, final)
        brought += 1
    return brought


def _artifact(store: Store, a: dict) -> Artifact:
    folder = store.object_dir(a["shape"], a["hash"])
    return Artifact(a["hash"], a["shape"], a["type"], folder if a["shape"] == "tree" else next(folder.iterdir()))


def _here(usage: dict, host: Host) -> str:
    """Where a result was made, as this machine names it: a step the coordinator there ran on its own machine (native,
    self) ran on HOST, which is `ssh-HOST:...` here; the run says it was coordinated there too."""
    ran = usage.get("ran_on")
    if isinstance(ran, dict) and isinstance(ran.get("place"), str) and PLACE.match(ran["place"]) \
            and ran["place"].split(":")[0] in ("native", "self"):
        usage = {**usage, "ran_on": {**ran, "place": f"ssh-{host.name}:{ran['place'].split(':', 1)[1]}",
                                     "title": f"{host.name} (SSH)", "coordinator": host.name}}
    return dumps(usage)


def expected_keys(coord, host: Host, node, inputs: dict[str, Artifact]) -> set[str]:
    """The cache keys this machine computes for the node run on HOST with these inputs: one per runner of its that the
    coordinator there could have used (its own machine's: native or self), each with the node's identity on HOST's
    platform (Backend.identity, as SshBackend gives it: the code this machine sent, never the machine)."""
    from .backends.remote import SshBackend
    from .scheduler import Placement
    keys = set()
    for r in node.manifest.runners:
        if r.kind in ("native", "self"):
            keys.add(coord.cache_key(node, Placement(SshBackend(host, runner_kind=r.kind), r), inputs))
    return keys


def import_run(store: Store, host: Host, run_id: str, graph: Graph, mine: dict, dump: dict, got: Path,
               objects: dict[str, set[str]]) -> dict:
    """The run, its tasks and events, and the cache rows of the results whose keys this machine computes, into this
    store (the comment above). A run of this id that this store has and did not import from HOST is another run:
    refused."""
    from .scheduler import Coordinator
    local = store.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    imported_from = store.db.execute("SELECT 1 FROM events WHERE run_id=? AND kind='run.imported' AND message LIKE ?",
                                     (run_id, f"from {host.name}:%")).fetchone()
    if local and not imported_from:
        raise RemoteError(f"this machine's store has another run {run_id}: not imported over it")
    coord = Coordinator(store, graph_registry(graph), [])
    # the graph's inputs as this machine has them: the run's input rows, and what each node's key is computed from;
    # checked before anything moves into this store, so a refused import changes nothing (code review of #46). Hashed,
    # not published: a json.v1 input is published below only once the import goes ahead
    for name in graph.spec.inputs:
        path = graph.input_path(name)
        if not path.exists() or artifacts.sha256_file(path) != (mine.get("input_sha256") or {}).get(name):
            raise RemoteError(f"run {run_id}'s inputs here are not the ones it was started with (changed since?): not imported")
    brought = _objects(store, objects, got)
    sources = {name: coord._graph_input(graph, name) for name in graph.spec.inputs}
    made: dict[str, dict[str, Artifact]] = {}
    verified: dict[str, str] = {}
    refused = 0
    for node_id, node in graph.nodes.items():   # in order: a node's inputs are its upstream's outputs, checked first
        t = dump["tasks"][node_id]
        if t["state"] not in ("succeeded", "cached"):
            continue
        made[node_id] = {k: _artifact(store, a) for k, a in t["outputs"].items()}
        inputs = {}
        for name, ref in node.inputs.items():
            if ref.node is None:
                inputs[name] = sources[ref.name]
            elif ref.node in made:
                inputs[name] = made[ref.node][ref.name]
        if len(inputs) == len(node.inputs) and t["cache_key"] in expected_keys(coord, host, node, inputs):
            verified[node_id] = t["cache_key"]
        else:
            refused += 1
    r = dump["run"]
    tasks_in, kept, added = 0, 0, 0
    # the attempt each task had when an import last wrote it: one this store has moved on from since (a rerun here) is
    # this store's own result, and a later fetch leaves it, and the run's status and spend, as they are (review of #46)
    last = mine.get("imported_attempts") or {}
    rerun_here = {n for n, t in ((n, store.db.execute("SELECT attempt FROM tasks WHERE run_id=? AND node_id=?", (run_id, n)).fetchone())
                                 for n in dump["tasks"]) if t is not None and local and last.get(n) != t["attempt"]}
    imported_attempts = dict(last)
    with store.tx() as db:
        if not local:
            db.execute("INSERT INTO runs (id, pipeline, graph, graph_path, inputs, status, created_at, finished_at, max_cost_usd, spent_usd, allow_uploads) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (run_id, graph.spec.pipeline, dumps(mine["graph_doc"]), mine.get("graph") or None,
                        dumps({k: a.to_json() for k, a in sources.items()}), r["status"], r["created_at"], r["finished_at"],
                        r["max_cost_usd"], r["spent_usd"], 1))
        elif not rerun_here:
            db.execute("UPDATE runs SET status=?, finished_at=?, spent_usd=? WHERE id=?", (r["status"], r["finished_at"], r["spent_usd"], run_id))
        for node_id, t in dump["tasks"].items():
            if node_id in rerun_here:
                continue
            now = db.execute("SELECT * FROM tasks WHERE run_id=? AND node_id=?", (run_id, node_id)).fetchone()
            outputs = dumps({k: a.to_json() for k, a in made[node_id].items()}) if node_id in made else None
            key = verified.get(node_id)
            if now and (now["state"], now["cache_key"], now["outputs"]) == (t["state"], key, outputs):
                imported_attempts[node_id] = now["attempt"]
                continue   # imported before, unchanged
            # an attempt number of this store's own: one past the last it has for this node (none: the first)
            attempt = (now["attempt"] if now else 0) + 1 if t["attempt"] else (now["attempt"] if now else 0)
            imported_attempts[node_id] = attempt
            runner = t["runner"]
            if runner and runner.split(":")[0] in ("native", "self"):
                runner = f"ssh-{host.name}:{runner.split(':', 1)[1]}"
            row = (t["state"], attempt, attempt or 1, key, runner, t["started_at"], t["finished_at"], t["error_code"],
                   t["error"], t["cost_usd"], _here(t["usage"], host), outputs, t["gpu"], t["gpu_vram_gb"])
            if now:
                db.execute("UPDATE tasks SET state=?, attempt=?, first_attempt=?, cache_key=?, runner=?, started_at=?, finished_at=?, "
                           "error_code=?, error=?, cost_usd=?, usage=?, outputs=?, gpu=?, gpu_vram_gb=?, lease_owner=NULL, "
                           "lease_expires=NULL, handle=NULL, not_before=NULL WHERE run_id=? AND node_id=?", (*row, run_id, node_id))
            else:
                db.execute("INSERT INTO tasks (state, attempt, first_attempt, cache_key, runner, started_at, finished_at, error_code, "
                           "error, cost_usd, usage, outputs, gpu, gpu_vram_gb, run_id, node_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (*row, run_id, node_id))
            tasks_in += 1
            log = dump["logs"].get(node_id)
            if attempt and log:
                src = _within(got, log)
                if src.is_file() and not src.is_symlink():
                    folder = store.root / "work" / run_id / node_id / str(attempt)
                    folder.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, folder / "log.txt")
        # a cache row for each result whose key this machine computed, never over this store's own
        for node_id, key in verified.items():
            if db.execute("SELECT 1 FROM cache WHERE key=?", (key,)).fetchone():
                kept += 1
                continue
            t = dump["tasks"][node_id]
            db.execute("INSERT INTO cache VALUES (?,?,?,?,?)", (key, graph.nodes[node_id].manifest.ref,
                                                                 dumps({k: a.to_json() for k, a in made[node_id].items()}),
                                                                 _here(t["usage"], host), time.time()))
            added += 1
        for a in (a for outs in made.values() for a in outs.values()):
            size = sum(p.stat().st_size for p in store.object_dir(a.shape, a.hash).rglob("*") if p.is_file())
            db.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?,?,?,?)", (a.hash, a.shape, a.type, size, time.time()))
        after = mine.get("imported_events", 0)
        for e in dump["events"]:
            if e["id"] > after:
                db.execute("INSERT INTO events (run_id, node_id, at, kind, message) VALUES (?,?,?,?,?)",
                           (run_id, e["node_id"], e["at"], e["kind"], e["message"]))
                after = e["id"]
        db.execute("INSERT INTO events (run_id, node_id, at, kind, message) VALUES (?,?,?,?,?)",
                   (run_id, None, time.time(), "run.imported",
                    f"from {host.name}: {tasks_in} tasks, {added} cache rows ({kept} this machine had kept, {refused} refused: "
                    f"keys this machine does not compute), {brought} objects"))
    # the record as it was written at the start (its graph as started: what a later start is compared with), and how far
    # imports have gone
    kept_record = json.loads(record(run_id).read_text())
    record(run_id).write_text(json.dumps({**kept_record, "imported_events": after, "imported_attempts": imported_attempts}))
    return {"tasks": tasks_in, "tasks_kept": len(rerun_here), "cache_added": added, "cache_kept": kept,
            "cache_refused": refused, "objects": brought}


def graph_registry(graph: Graph):
    """A registry of the graph's own manifests (what cache_key and _graph_input need of one)."""
    class Of:
        def get(self, ref):
            return next(n.manifest for n in graph.nodes.values() if n.manifest.ref == ref)
    return Of()
