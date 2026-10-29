"""A coordinator on a lab server, for a run another machine started there (`perception remote run`: remote_run.py is the
other half, on that machine). It runs from the listed copy of the pipeline that machine sent (<root>/pipelines/<hash>,
with its RELEASE.json), with PERCEPTION_HOME set to the host's own store for such runs (<root>/coordinator):

    python -m perception.remote_coordinator start                 the run's spec on stdin (its keys in it)
    python -m perception.remote_coordinator serve RUN             started by `start`, in a session of its own; keys on stdin
    python -m perception.remote_coordinator status [RUN]
    python -m perception.remote_coordinator events RUN [--after N]
    python -m perception.remote_coordinator dump RUN              what `remote fetch` imports
    python -m perception.remote_coordinator sweep --keep-s S      started by `serve` as it ends: prune, hourly

Each prints JSON objects, one per line (an internal protocol between two copies of this code: the machine that started
the run sends its copy before every call).

`start` creates the run under the id the other machine chose (it attaches instead of starting a second coordinator when
that run is already being coordinated here, so a start whose answer was lost can be asked again), checks it can run here
(the host's own plan: places, missing weights, keys), and starts `serve` for it: a process in a session of its own
(setsid), whose output goes to a log beside the run, which outlives the SSH connection that started it, so the run goes
on with the laptop closed. A run whose coordinator is gone (the host restarted) is resumed by the same `start`.

Keys. The keys the run's steps declare come in the spec on stdin, and only those are kept (any other is dropped here as
well). `start` hands them to `serve` through a pipe, its stdin: they are in no file, no command line, no log and no
environment variable on the host, only in the coordinator's memory, and from it, as on any machine (child_env), in the
environment of the steps that declare them and no other; a download secret (a gated model's token) only in their
prepare's, while it runs (prepare_first), and the steps then read that Hugging Face account's cache here offline (the
spec names the account: hosts.hf_home, GATED_HF_HOME). While the run lasts, root and the same user's processes can
read a step's environment there (/proc/<pid>/environ): what `perception hosts keys HOST on` agrees to. The host's own
keys (its environment, keychain, ~/.config/hyperstudy) are never read: a run here has the keys its starter sent.

Places. Every step runs here, natively (its uv environment from the copy, made under this store by its lock and the
copy: release.env_dir), on the cards the coordinator gives it, leased against every other coordinator here (GPU_LEASES).
A place on RunPod or AWS (`runpod:*`, `cloud:*`) is refused: the account-wide RunPod key and the AWS credentials never
leave the machine that started the run (GRAPH_MVP_PLAN.md). A place on another host (`ssh-*`) is refused too: a run
coordinated here runs every step here.

Inputs. The other machine files each graph input in the host's root as any input sent there is (objects/<sha256>/data,
remote_driver.send_inputs), and the run reads it through a link under its own name (remote/<run>/inputs/<name>/<file>; a
copy for an input the store publishes, which takes no link), the name it has on that machine: decode records it, and the layers then match a run coordinated there.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import artifacts
from .backends.base import GATED_HF_HOME, SENT_TOKEN_ONLY
from .backends.native import NativeBackend, SelfBackend
from .hosts import hf_home
from .ir import GraphError, compile_graph
from .keycheck import ACCOUNT_NAME
from .manifests import Registry
from .release import REPO, home
from .scheduler import RUN_ID, Coordinator, LeaseHeld
from .store import Store

SHA256 = re.compile(r"^[0-9a-f]{64}$")
# a graph input's file name, as the other machine has it: one plain name (a link of that name points at the object)
FILE_NAME = re.compile(r"^[^/\0\n]{1,255}$")


class Refusal(Exception):
    """The run cannot start here, or the call cannot be answered: said to the other machine as {"ok": false, problems}."""
    def __init__(self, *problems: str):
        super().__init__("; ".join(problems))
        self.problems = list(problems)


def answer(**fields):
    print(json.dumps({"ok": True, **fields}), flush=True)


def refused_places(doc: dict) -> list[str]:
    """Why a graph's chosen places cannot be used by a coordinator on a host: only this machine's own (native, self).
    Checked on the machine that starts the run too, before anything leaves it."""
    problems = []
    for node in doc.get("nodes", []):
        place = node.get("place") or ""
        kind = place.split(":")[0]
        if kind in ("runpod", "cloud"):
            problems.append(f"{node.get('id')}: {place} is refused for a run coordinated on a host: the RunPod key and AWS "
                            "credentials never leave the machine that starts the run")
        elif kind.startswith("ssh-"):
            problems.append(f"{node.get('id')}: {place} is another machine: a run coordinated on a host runs every step there")
    return problems


def root() -> Path:
    """The host's root (the folder `perception hosts add` made private): the coordinator's store is its coordinator/."""
    return home().resolve().parent


def run_folder(run: str) -> Path:
    if not RUN_ID.match(run):
        raise Refusal(f"not a run id: {run!r}")
    return home() / "remote" / run


class HostBackend(NativeBackend):
    """This machine's native runs, with each node's environment kept by its lock, as a host keeps them for steps placed
    there (remote_driver.make_env, #41): envs/<env_key> in this store, made of the node's uv project and the SDK's
    project file, synced without the SDK (uv sync --no-install-package), which the node imports from the copy it runs
    from (PYTHONPATH). Every copy of the pipeline (each named by its whole content: remote_run.Remote.copy) with that
    lock shares it, so an edit elsewhere in the pipeline makes no new environment here (review of #46)."""

    def env_project(self, manifest, runner) -> Path:
        from .backends.remote import code_files, env_files, env_key
        files = code_files(manifest, runner)
        folder = home() / "envs" / env_key(files, runner.env)
        if not (folder / ".complete").exists():
            incoming = home() / "envs" / f".incoming-{os.getpid()}-{folder.name}"
            shutil.rmtree(incoming, ignore_errors=True)
            for f in env_files(files, runner.env):
                (incoming / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / f, incoming / f)
            (incoming / ".complete").touch()
            try:
                os.rename(incoming, folder)
            except OSError:   # made meanwhile by another node with the same lock: the same content
                shutil.rmtree(incoming, ignore_errors=True)
        return folder / runner.env

    def command(self, manifest, runner, request: Path) -> list[str]:
        if not runner.env:
            raise ValueError(f"{manifest.ref}: a native runner needs env (a uv project)")
        from .backends.remote import SDK_PACKAGE
        uv = shutil.which("uv") or "uv"
        # synced (a moment once it is), then run without syncing: uv itself makes two syncs of one environment wait
        script = (f'"$0" sync --quiet --project "$1" --frozen --no-dev --no-install-package {SDK_PACKAGE} && '
                  'exec "$0" run --project "$1" --frozen --no-dev --no-sync python "$2" --request "$3"')
        return ["sh", "-c", script, uv, str(self.env_project(manifest, runner)), str(manifest.directory / runner.entry),
                str(request)]

    def project_env(self, runner) -> dict[str, str]:
        return {"PYTHONPATH": str(REPO / "sdk" / "src")} if runner.env else {}

    def sync_command(self, manifest, runner) -> list[str] | None:
        """Its environment by its lock, without the SDK, as command() syncs it (prepare_first: before a prepare)."""
        from .backends.remote import SDK_PACKAGE
        return [shutil.which("uv") or "uv", "sync", "--quiet", "--project", str(self.env_project(manifest, runner)), "--frozen",
                "--no-dev", "--no-install-package", SDK_PACKAGE]

    def prepare_command(self, manifest, runner, config: dict) -> list[str]:
        return [shutil.which("uv") or "uv", "run", "--project", str(self.env_project(manifest, runner)), "--frozen", "--no-dev",
                "--no-sync", "python", str(manifest.directory / manifest.prepare), json.dumps(config)]


def backends() -> list:
    """Every step here, on this machine (module docstring, Places)."""
    return [HostBackend(), SelfBackend()]


def registry(nodes: list[str]) -> Registry:
    """The copy's nodes, and the other node folders the run was started with (within the copy: `--nodes`)."""
    extra = []
    for rel in nodes:
        path = (REPO / rel).resolve()
        if not path.is_relative_to(REPO.resolve()) or not path.is_dir():
            raise Refusal(f"--nodes {rel}: not a folder of the pipeline's copy")
        extra.append(path)
    return Registry.load(*[r for r in [REPO / "nodes", *extra] if r.exists()])


@contextlib.contextmanager
def locked(path: Path, shared: bool = False):
    """A run's own lock (`.lock` in its folder: one `start` of a run at a time, two at once would both find it missing,
    or its coordinator gone), or the store's prune lock (PRUNE_LOCK): exclusive for a prune, from its look for running
    coordinators to its last removal; shared for a start, from its look at the run to its coordinator's pid recorded.
    So no coordinator appears between a prune's check and its removals (review of #46)."""
    import fcntl
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


PRUNE_LOCK = ".prune"   # in the coordinator's store (home())
# how long after its end (or its last activity) a run is kept whatever the retention: an attach or a fetch of it right
# after it ends must find it (lab check (d): with nothing kept, a run's own sweeper removed it before its attach read it)
PRUNE_GRACE_S = 600
# the host's retention, as `perception hosts set` and every start there write it (remote_driver's sweeper reads the same
# file, for what the host keeps for steps): "<keep seconds> <free KB, or -1 for the default>"
RETENTION = "holds/.retention"   # in the host's root (root())


def retention(keep_s: float, min_free_gb: float | None) -> tuple[float, float | None]:
    """The host's retention now (RETENTION), or `keep_s` and `min_free_gb` where it has none that can be read."""
    try:
        keep, free = (root() / RETENTION).read_text().split()[:2]
        keep_s, free_kb = float(keep), float(free)
        return max(0.0, keep_s), None if free_kb < 0 else free_kb / 2**20
    except (OSError, ValueError):
        return keep_s, min_free_gb


def write_retention(keep_s: float, min_free_gb: float | None):
    """The host's retention, for the sweepers here (both this one and remote_driver's), as a start was told it."""
    folder = root() / "holds"
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".retention.{os.getpid()}"
    tmp.write_text(f"{int(keep_s)} {-1 if min_free_gb is None else round(min_free_gb * 2**20)}\n")
    os.replace(tmp, folder / ".retention")


def alive(run: str) -> bool:
    """Whether the run's coordinator (`serve RUN`) is running here: the pid it recorded, whose command line still names
    it (pids are reused), or has none for the moment (a process mid-exec, not a zombie or a kernel thread;
    backends/native._shown)."""
    try:
        pid = int((run_folder(run) / "coordinator.pid").read_text().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    from .backends.native import _shown
    command = _shown(pid)
    return command is not None and (f"perception.remote_coordinator serve {run}" in command or command == "")


def coordinator_state(store: Store, run: str) -> tuple[str, str]:
    """The run's coordinator, and the run's status read after it: running (a coordinator here is working on it),
    finished (it ended), or gone (it says running, and nothing here is: the host restarted, or the process was killed;
    `remote run` with its id resumes it). The coordinator is looked at first and the status read after: one that ends
    between the two is seen ended, never gone with its run still running (code review of #46)."""
    up = alive(run)
    row = store.db.execute("SELECT status FROM runs WHERE id=?", (run,)).fetchone()
    status = row["status"] if row else "running"
    return ("running" if up else "gone" if status == "running" else "finished"), status


def with_inputs(doc: dict, inputs: dict, run: str, make: bool = True) -> dict:
    """The graph with each input read through a link under its own name to the object the other machine filed it as
    (module docstring, Inputs). Nothing of that machine's paths came here: only each input's sha256 and file name.
    `make`: make the links (a run made or resumed); without, only what the graph names them (a run going on, whose links
    are left as they are: a node may be opening one)."""
    if not isinstance(inputs, dict) or set(inputs) != set(doc.get("inputs", {})):
        raise Refusal("the run's inputs and its graph's do not match")
    out = {**doc, "inputs": {}}
    for name, entry in inputs.items():
        sha, file = entry.get("sha256"), entry.get("filename")
        if not (isinstance(sha, str) and SHA256.match(sha)) or not (isinstance(file, str) and FILE_NAME.match(file)) \
                or file in (".", "..") or not re.match(r"^[A-Za-z0-9_-]+$", name):
            raise Refusal(f"input {name!r}: not a sha256 and a plain file name")
        obj = root() / "objects" / sha
        if not (obj / ".complete").exists():
            raise Refusal(f"input {name} ({sha[:12]}) was not sent here")
        link = run_folder(run) / "inputs" / name / file
        try:
            shape = artifacts.get(doc["inputs"][name]["type"]).shape
        except KeyError as e:
            raise Refusal(f"input {name}: {e.args[0]}")
        if make:
            # made beside it and renamed into place: a node of a run being resumed that opens it meanwhile finds it
            link.parent.mkdir(parents=True, exist_ok=True)
            tmp = link.parent / f".{file}.{os.getpid()}"
            shutil.rmtree(tmp, ignore_errors=True) if tmp.is_dir() and not tmp.is_symlink() else tmp.unlink(missing_ok=True)
            if shape == "source":   # read where it is (the video): a link
                tmp.symlink_to(obj / "data")
            elif (obj / "data").is_dir():   # published into the store as it is read (Store.publish refuses a link): a copy
                shutil.copytree(obj / "data", tmp)
            else:
                shutil.copyfile(obj / "data", tmp)
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            os.replace(tmp, link)
        out["inputs"][name] = {**doc["inputs"][name], "uri": link.as_uri()}
    return out


def declared(graph) -> set[str]:
    """The keys the run's steps declare: at run time, or for their prepare alone (download_secrets)."""
    return {s for n in graph.nodes.values() for s in (*n.manifest.secrets, *n.manifest.download_secrets)}


def spawn(run: str, nodes: list[str], keys: dict[str, str], retention: list[str] = ()) -> subprocess.Popen:
    """`serve RUN` in a session of its own, its keys written to its stdin and the pipe closed: never on its command line,
    in its environment or in a file. Its output (no keys: the coordinator never prints one) goes to the run's log."""
    folder = run_folder(run)
    log = open(folder / "coordinator.log", "ab")
    proc = subprocess.Popen([sys.executable, "-m", "perception.remote_coordinator", "serve", run,
                             *[x for rel in nodes for x in ("--nodes", rel)], *retention],
                            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, cwd=home(),
                            start_new_session=True, close_fds=True)
    log.close()
    (folder / "coordinator.pid").write_text(str(proc.pid))   # at once: a second start finds it (alive)
    proc.stdin.write(json.dumps(keys).encode())
    proc.stdin.close()
    return proc


def prune(store: Store, keep_s: float, min_free_gb: float | None, starting: str | None = None) -> list[str]:
    """The runs coordinated here that ended more than `keep_s` ago, and, while the disk has less than `min_free_gb`
    free (None: hosts.MIN_FREE_GB, or a tenth of the disk if that is less), the ones that ended longest ago too: their
    rows, their folders (work/<run>, remote/<run>), and every object of the store no run left here reads, with the
    cache rows that name one. What a run coordinated here made (face crops, transcripts, embeddings) is kept on the
    host no longer than what the host keeps for steps run there (RUN_ANYWHERE_PLAN.md, Privacy across routes; review of
    #46). Not while any run is being coordinated here: one may be publishing an object this would remove. Fetch the
    runs to keep before then. The runs removed."""
    with locked(home() / PRUNE_LOCK):
        return _prune(store, keep_s, min_free_gb, starting)


def last_activity(store: Store, run: str) -> float:
    """When anything last happened in a run: its last event, or a task's start or end, or its creation."""
    row = store.db.execute("SELECT MAX(x) FROM (SELECT MAX(at) AS x FROM events WHERE run_id=? UNION ALL "
                           "SELECT MAX(COALESCE(finished_at, started_at)) FROM tasks WHERE run_id=? UNION ALL "
                           "SELECT created_at FROM runs WHERE id=?)", (run, run, run)).fetchone()
    return row[0] or 0.0


def _prune(store: Store, keep_s: float, min_free_gb: float | None, starting: str | None = None) -> list[str]:
    import shutil as _shutil
    from .backends.native import _alive
    from .hosts import MIN_FREE_GB
    if any(p.is_file() and alive(p.parent.name) for p in (home() / "remote").glob("*/coordinator.pid")):
        return []
    # ended runs, oldest first; and a run left `running` by a coordinator that died (the host restarted) and was never
    # resumed, once nothing has happened in it for the keep period and no node it started still runs (review of #46)
    runs = [(r["id"], r["finished_at"]) for r in store.db.execute(
        "SELECT id, finished_at FROM runs WHERE status IN ('succeeded','failed','canceled') AND finished_at IS NOT NULL")]
    for r in store.db.execute("SELECT id FROM runs WHERE status='running'").fetchall():
        nodes_alive = any(_alive(h["pid"], h["request"]) for t in store.db.execute(
            "SELECT handle FROM tasks WHERE run_id=? AND state IN ('running','leased') AND handle IS NOT NULL", (r["id"],))
            for h in [json.loads(t["handle"])] if isinstance(h, dict) and type(h.get("pid")) is int)
        if not nodes_alive:
            runs.append((r["id"], last_activity(store, r["id"])))
    # never the run a start is for: the holds its inputs were just given name its folder (review of #46)
    runs = sorted((r for r in runs if r[0] != starting), key=lambda r: r[1])
    if not runs:
        return []
    disk = _shutil.disk_usage(store.root)
    need = (min(MIN_FREE_GB, disk.total / 2**30 / 10) if min_free_gb is None else min_free_gb) * 2**30
    # never within the grace after its end, whatever the retention: an attach or a fetch right after it finds it
    gone = [run for run, ended in runs if ended < time.time() - max(keep_s, PRUNE_GRACE_S)]
    short = disk.free < need
    for run, ended in runs:   # oldest first, while the disk is short (one more each time: the next prune looks again)
        if not short:
            break
        if run not in gone and ended < time.time() - max(3600, PRUNE_GRACE_S):   # never one that ended (or went quiet) within the hour
            gone.append(run)
            short = False
    if not gone:
        return []
    with store.tx() as db:
        for run in gone:
            for table, col in (("events", "run_id"), ("tasks", "run_id"), ("runs", "id")):
                db.execute(f"DELETE FROM {table} WHERE {col}=?", (run,))
        kept = {a["hash"] for t in db.execute("SELECT outputs FROM tasks") for a in json.loads(t["outputs"] or "{}").values()}
        # and the graph inputs a remaining run was given, which the store published (json.v1...): named by its row alone
        # (code review of #46)
        kept |= {a["hash"] for r in db.execute("SELECT inputs FROM runs") for a in json.loads(r["inputs"] or "{}").values()
                 if isinstance(a, dict) and a.get("shape") in ("file", "tree")}
        for c in db.execute("SELECT key, outputs FROM cache").fetchall():
            if any(a["hash"] not in kept for a in json.loads(c["outputs"] or "{}").values()):
                db.execute("DELETE FROM cache WHERE key=?", (c["key"],))
        for a in db.execute("SELECT hash, shape FROM artifacts").fetchall():
            if a["hash"] not in kept and a["shape"] in ("file", "tree"):
                db.execute("DELETE FROM artifacts WHERE hash=?", (a["hash"],))
    for sub in ("objects", "trees"):
        for folder in (store.root / sub).glob("*/*"):
            if folder.is_dir() and folder.name not in kept:
                for p in [folder, *folder.rglob("*")]:
                    with contextlib.suppress(OSError):
                        p.chmod(0o700 if p.is_dir() else 0o600)
                _shutil.rmtree(folder, ignore_errors=True)
    for run in gone:
        _shutil.rmtree(store.root / "work" / run, ignore_errors=True)
        _shutil.rmtree(home() / "remote" / run, ignore_errors=True)   # its holds on its inputs go with it
    return gone


def cmd_sweep(args) -> int:
    """The host's sweeper for runs coordinated here (prune), as a host has one for what it keeps for steps (remote_driver
    SWEEP): started detached by each coordinator as it ends, one at a time (a lock it holds while it lives), it prunes
    every hour (or half the keep period, if shorter), with the host's retention as it is at each pass (RETENTION), and
    ends once no ended run is left, so a host used once is cleaned too."""
    import fcntl
    fd = os.open(home() / ".sweeper", os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 0   # one is running
    while True:
        if not (home() / "perception.db").is_file():
            return 0   # the store is gone (hosts clean): nothing to sweep, and not to be made again
        # the host's retention as it is now, never the one it was started with: one changed since (hosts set, or a start
        # told another) applies at once (lab check (d))
        keep_s, min_free_gb = retention(args.keep_s, args.min_free_gb)
        store = Store(home())
        prune(store, keep_s, min_free_gb)
        left = store.db.execute("SELECT 1 FROM runs").fetchone()
        store.close()
        if not left:
            return 0
        # the next pass in an hour (or half the keep period, if shorter), or as soon as the retention changes
        said = _said()
        deadline = time.time() + max(1.0, min(3600.0, keep_s / 2))
        while time.time() < deadline and _said() == said:
            time.sleep(1)


def _said() -> str:
    try:
        return (root() / RETENTION).read_text()
    except OSError:
        return ""


def sweeper(keep_s: float, min_free_gb: float | None, nodes: list[str]):
    """Start the sweeper (cmd_sweep) in a session of its own, unless one is running (it then exits at once)."""
    args = ["--keep-s", str(keep_s)] + (["--min-free-gb", str(min_free_gb)] if min_free_gb is not None else [])
    subprocess.Popen([sys.executable, "-m", "perception.remote_coordinator", "sweep", *args], cwd=home(),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True, close_fds=True)


def release(run: str):
    """The holds on the run's inputs in the host's root (remote_run.Remote.send_input): they name the run's held/
    folder, and a hold whose folder is gone is stale (remote_driver.CLEANUP). Kept while the run may still need its
    inputs: a run whose coordinator dies keeps them, for the start that resumes it."""
    shutil.rmtree(run_folder(run) / "held", ignore_errors=True)


def cmd_start(args) -> int:
    spec = json.loads(sys.stdin.read())
    keys = spec.pop("keys", None) or {}
    if not isinstance(keys, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in keys.items()):
        raise Refusal("keys: not names and values")
    run, nodes = spec.get("run", ""), spec.get("nodes", [])
    folder = run_folder(run)
    folder.mkdir(parents=True, exist_ok=True)
    # the gated weights its steps read are its Hugging Face account's (hosts.hf_home): for its plan here, and for serve,
    # which inherits it, and gives it to the steps with download secrets alone as their HF_HOME (backends/base.node_env)
    if (account := spec.get("hf_account")) is not None:
        if not isinstance(account, str) or not ACCOUNT_NAME.fullmatch(account):
            raise Refusal(f"hf_account: not an account's name: {account!r}")
        os.environ[GATED_HF_HOME] = hf_home(str(root()), account)
    # nothing a run here does uses the host's own Hugging Face login: only the token its starter sent (SENT_TOKEN_ONLY)
    os.environ.update(SENT_TOKEN_ONLY)
    # under the run's lock, which its coordinator takes too for its last step (releasing the holds, then its pid): so a
    # start either sees it running (and it releases them after) or sees it ended
    from .hosts import KEEP_DAYS
    keep_s, min_free = spec.get("keep_s", KEEP_DAYS * 86400), spec.get("min_free_gb")
    if _number_ok(keep_s) and (min_free is None or _number_ok(min_free)):
        write_retention(float(keep_s), min_free)   # for the sweepers here: the host's retention as this start was told it
        for gone in prune(Store(home()), float(keep_s), min_free, starting=run):
            print(f"removed run {gone}: it ended more than the host's keep days ago (or its disk is short)", file=sys.stderr)
    with locked(home() / PRUNE_LOCK, shared=True), locked(folder / ".lock"):
        running = False
        try:
            code, running = _start(spec, keys, run, nodes, folder)
            return code
        finally:
            # the run's inputs were held for it before they were sent (remote_run.Remote.send_input): a start that ends
            # with no coordinator for it (refused, its plan refused, the run ended already) releases them, or the host's
            # retention would never remove them (review of #46)
            if not running and not alive(run):
                release(run)


def _number_ok(x) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and x >= 0


def _start(spec: dict, keys: dict, run: str, nodes: list[str], folder: Path) -> tuple[int, bool]:
    """cmd_start under the run's lock: its exit code, and whether a coordinator is running for the run when it ends."""
    store = Store(home())
    reg = registry(nodes)
    doc = with_inputs(spec["graph"], spec.get("inputs", {}), run, make=False)
    if problems := refused_places(doc):
        raise Refusal(*problems)
    row = store.db.execute("SELECT status, graph, inputs FROM runs WHERE id=?", (run,)).fetchone()
    sent = {k: v.get("sha256") for k, v in spec.get("inputs", {}).items()}
    if row:
        if json.loads(row["graph"]) != doc:
            raise Refusal(f"run {run} is here already, with another graph")
        # by hash too: the graph names its inputs by link, and a resume remakes the links (re-review of #46)
        if {k: v["hash"] for k, v in json.loads(row["inputs"]).items()} != sent:
            raise Refusal(f"run {run} is here already, with other inputs")
        if alive(run):   # a start whose answer was lost, asked again: the run goes on as it was
            answer(run=run, started=False, status="running")
            return 0, True
        if row["status"] != "running":
            answer(run=run, started=False, status=row["status"])
            return 0, False
        with_inputs(spec["graph"], spec.get("inputs", {}), run)   # its coordinator is gone: resumed below
        graph = compile_graph(doc, reg)
    else:
        with_inputs(spec["graph"], spec.get("inputs", {}), run)
        for name, sha in sent.items():   # what the run will read is what was sent under that hash
            link = Path(unquote(urlparse(doc["inputs"][name]["uri"]).path))
            if artifacts.get(doc["inputs"][name]["type"]).shape == "source" and store.source(link, "video.v1").hash != sha:
                raise Refusal(f"input {name} here is not {sha[:12]}")
        try:
            graph = compile_graph(doc, reg)
        except GraphError as e:
            raise Refusal(*e.problems)
    keys = {k: v for k, v in keys.items() if k in declared(graph)}   # only what its steps declare (module docstring)
    if not row:
        from .cli import plan_json
        from .plan import plan as make_plan
        coord = Coordinator(store, reg, backends(), keys)
        p = make_plan(coord, graph, allow_downloads=bool(spec.get("allow_downloads")),
                      allow_uploads=bool(spec.get("allow_uploads")))
        if not p.ok:   # it would not run here: the host's own plan says why, and nothing is created
            print(json.dumps({"ok": False, "plan": plan_json(p),
                              "problems": [f"{n.id}: {n.why_not or 'cannot run here (see the plan)'}" for n in p.nodes
                                           if not n.where or n.missing_secrets or n.downloads]}), flush=True)
            return 1, False
        coord.create_run(graph, doc, allow_uploads=bool(spec.get("allow_uploads")), run_id=run)
    from .hosts import KEEP_DAYS
    keep_s, min_free = spec.get("keep_s", KEEP_DAYS * 86400), spec.get("min_free_gb")
    retention = ["--keep-s", str(float(keep_s))] + (["--min-free-gb", str(float(min_free))] if min_free is not None else []) \
        if _number_ok(keep_s) and (min_free is None or _number_ok(min_free)) else []
    proc = spawn(run, nodes, keys, retention)
    del keys
    try:   # one that cannot even start (a broken copy) says so now; one that runs is left to run
        code = proc.wait(timeout=1.5)
    except subprocess.TimeoutExpired:
        code = None
    if code:
        tail = (folder / "coordinator.log").read_text(errors="replace")[-1500:]
        raise Refusal(f"the coordinator for {run} exited at once ({code}): {tail}")
    status = store.db.execute("SELECT status FROM runs WHERE id=?", (run,)).fetchone()["status"]
    answer(run=run, started=True, resumed=bool(row), pid=proc.pid, status=status)
    return 0, True


def cmd_serve(args) -> int:
    """The run's coordinator, for as long as the run lasts. Its keys from stdin, then stdin closed."""
    keys = json.loads(sys.stdin.read() or "{}")
    sys.stdin.close()
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    run = args.run
    folder = run_folder(run)
    pidfile = folder / "coordinator.pid"
    pidfile.write_text(str(os.getpid()))
    coord = Coordinator(Store(home()), registry(args.nodes or []), backends(), keys)
    del keys
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} coordinating run {run} (pid {os.getpid()})", flush=True)
    status = None
    try:
        for _ in range(10):
            try:
                status = asyncio.run(coord.run(run))
                break
            except LeaseHeld as e:   # resumed while the last coordinator's leases had not run out yet
                print(f"waiting: {e}", flush=True)
                time.sleep(coord.lease_s / 2)
    finally:
        with locked(folder / ".lock"):   # a start then sees it either running, or ended with its holds released
            if status:   # the run has ended: its inputs need no hold any more (remote_run.Remote.send_input)
                release(run)
            with contextlib.suppress(OSError, ValueError):
                if int(pidfile.read_text()) == os.getpid():
                    pidfile.unlink()
        if args.keep_s is not None:
            sweeper(args.keep_s, args.min_free_gb, args.nodes or [])
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} run {run}: {status}", flush=True)
    return 0 if status else 1


def cmd_status(args) -> int:
    from .cli import iso, run_json
    store = Store(home())
    if args.run:
        row = store.db.execute("SELECT * FROM runs WHERE id=?", (args.run,)).fetchone()
        if row is None:
            raise Refusal(f"no run {args.run} on this host")
        state, status = coordinator_state(store, args.run)
        row = store.db.execute("SELECT * FROM runs WHERE id=?", (args.run,)).fetchone()
        answer(status=run_json(store, row), coordinator=state)
        return 0
    runs = [{"run": r["id"], "pipeline": r["pipeline"], "status": r["status"], "created_at": iso(r["created_at"]),
             "coordinator": coordinator_state(store, r["id"])[0]}
            for r in store.db.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 20")]
    answer(runs=runs)
    return 0


def cmd_events(args) -> int:
    from .cli import iso
    store = Store(home())
    row = store.db.execute("SELECT status FROM runs WHERE id=?", (args.run,)).fetchone()
    if row is None:
        raise Refusal(f"no run {args.run} on this host")
    events = [{"id": e["id"], "node": e["node_id"], "kind": e["kind"], "message": e["message"] or "", "at": iso(e["at"])}
              for e in store.events(args.run, args.after)]
    # the run's status after its events: a run that ends between the two reads is seen ending at the next call
    state, status = coordinator_state(store, args.run)
    answer(events=events, status=status, coordinator=state)
    return 0


def cmd_dump(args) -> int:
    """What `remote fetch` reads, once the run has ended: its run, task and event rows and each final attempt's log
    (relative to the store; the bundle is exported on that machine, from what it checked). The other machine trusts none of it but what it checks: it
    copies the objects the tasks' outputs name from their places in the store (Store.object_dir) and checks each against
    its hash, and makes cache rows only from tasks whose keys it computes itself (remote_run, fetch and import)."""
    store = Store(home())
    run = args.run
    row = store.db.execute("SELECT * FROM runs WHERE id=?", (run,)).fetchone()
    if row is None:
        raise Refusal(f"no run {run} on this host")
    if row["status"] not in ("succeeded", "failed", "canceled"):
        state = coordinator_state(store, run)[0]
        raise Refusal(f"run {run} is still running here" if state == "running" else
                      f"run {run}'s coordinator is gone: `perception remote run` with --run-id {run} resumes it")
    tasks = [dict(t) for t in store.db.execute("SELECT * FROM tasks WHERE run_id=?", (run,))]
    logs = []
    for t in tasks:
        log = store.root / "work" / run / t["node_id"] / str(t["attempt"]) / "log.txt"
        if t["attempt"] and log.is_file() and not log.is_symlink():
            logs.append({"node": t["node_id"], "attempt": t["attempt"], "path": log.relative_to(store.root).as_posix()})
    events = [dict(e) for e in store.events(run)]
    answer(run=dict(row), tasks=tasks, logs=logs, events=events)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="perception.remote_coordinator")
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("start").add_argument("--nodes", action="append")   # start reads them from its spec
    sub.choices["start"].set_defaults(fn=cmd_start)
    sw = sub.add_parser("sweep")
    sw.add_argument("--keep-s", type=float, required=True)
    sw.add_argument("--min-free-gb", type=float)
    sw.set_defaults(fn=cmd_sweep)
    for name, fn in (("serve", cmd_serve), ("status", cmd_status), ("events", cmd_events), ("dump", cmd_dump)):
        p = sub.add_parser(name)
        p.add_argument("run", nargs="?" if name == "status" else None)
        p.add_argument("--nodes", action="append")
        if name == "serve":
            p.add_argument("--keep-s", type=float)
            p.add_argument("--min-free-gb", type=float)
        if name == "events":
            p.add_argument("--after", type=int, default=0)
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    try:
        return args.fn(args) or 0
    except Refusal as e:
        print(json.dumps({"ok": False, "problems": e.problems}), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
