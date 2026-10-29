"""A coordinator on a lab server, for a run another machine started there (`perception remote run`: remote_run.py is the
other half, on that machine). It runs from the listed copy of the pipeline that machine sent (<root>/pipelines/<hash>,
with its RELEASE.json), with PERCEPTION_HOME set to the host's own store for such runs (<root>/coordinator):

    python -m perception.remote_coordinator start                 the run's spec on stdin (its keys in it)
    python -m perception.remote_coordinator serve RUN             started by `start`, in a session of its own; keys on stdin
    python -m perception.remote_coordinator status [RUN]
    python -m perception.remote_coordinator events RUN [--after N]
    python -m perception.remote_coordinator dump RUN              what `remote fetch` imports; the bundle exported here

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
environment of the steps that declare them and no other. While the run lasts, root and the same user's processes can
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

from . import artifacts
from .backends.native import NativeBackend, SelfBackend
from .ir import GraphError, compile_graph
from .manifests import Registry
from .release import REPO, home
from .scheduler import DONE, RUN_ID, Coordinator, LeaseHeld
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


def backends() -> list:
    """Every step here, on this machine (module docstring, Places)."""
    return [NativeBackend(), SelfBackend()]


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
def locked(path: Path):
    """One `start` of a run at a time (two at once would both find it missing, or its coordinator gone)."""
    import fcntl
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def alive(run: str) -> bool:
    """Whether the run's coordinator (`serve RUN`) is running here: the pid it recorded, whose command line still names
    it (pids are reused), or has none for the moment (a process mid-exec; backends/native._alive)."""
    try:
        pid = int((run_folder(run) / "coordinator.pid").read_text().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    out = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    if out.returncode:
        return False
    command = out.stdout.strip()
    return f"perception.remote_coordinator serve {run}" in command or command == "" or command.startswith("[")


def coordinator_state(store: Store, run: str, status: str) -> str:
    """running: a coordinator here is working on it; finished: it ended; gone: it says running, and nothing here is
    (the host restarted, or the process was killed): `remote run` with its id resumes it."""
    if alive(run):
        return "running"
    return "gone" if status == "running" else "finished"


def with_inputs(doc: dict, inputs: dict, run: str) -> dict:
    """The graph with each input read through a link under its own name to the object the other machine filed it as
    (module docstring, Inputs). Nothing of that machine's paths came here: only each input's sha256 and file name."""
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
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.exists():
            shutil.rmtree(link)
        try:
            shape = artifacts.get(doc["inputs"][name]["type"]).shape
        except KeyError as e:
            raise Refusal(f"input {name}: {e.args[0]}")
        if shape == "source":   # read where it is (the video): a link
            link.symlink_to(obj / "data")
        elif (obj / "data").is_dir():   # published into the store as it is read (Store.publish refuses a link): a copy
            shutil.copytree(obj / "data", link)
        else:
            shutil.copyfile(obj / "data", link)
        out["inputs"][name] = {**doc["inputs"][name], "uri": link.as_uri()}
    return out


def declared(graph) -> set[str]:
    return {s for n in graph.nodes.values() for s in n.manifest.secrets}


def spawn(run: str, nodes: list[str], keys: dict[str, str]) -> subprocess.Popen:
    """`serve RUN` in a session of its own, its keys written to its stdin and the pipe closed: never on its command line,
    in its environment or in a file. Its output (no keys: the coordinator never prints one) goes to the run's log."""
    folder = run_folder(run)
    log = open(folder / "coordinator.log", "ab")
    proc = subprocess.Popen([sys.executable, "-m", "perception.remote_coordinator", "serve", run,
                             *[x for rel in nodes for x in ("--nodes", rel)]],
                            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, cwd=home(),
                            start_new_session=True, close_fds=True)
    log.close()
    (folder / "coordinator.pid").write_text(str(proc.pid))   # at once: a second start finds it (alive)
    proc.stdin.write(json.dumps(keys).encode())
    proc.stdin.close()
    return proc


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
    with locked(folder / ".lock"):
        store = Store(home())
        reg = registry(nodes)
        doc = with_inputs(spec["graph"], spec.get("inputs", {}), run)
        if problems := refused_places(doc):
            raise Refusal(*problems)
        row = store.db.execute("SELECT status, graph FROM runs WHERE id=?", (run,)).fetchone()
        if row:
            if json.loads(row["graph"]) != doc:
                raise Refusal(f"run {run} is here already, with another graph")
            if alive(run):   # a start whose answer was lost, asked again: the run goes on as it was
                answer(run=run, started=False, status="running")
                return 0
            if row["status"] != "running":
                answer(run=run, started=False, status=row["status"])
                return 0
            graph = compile_graph(doc, reg)   # its coordinator is gone: resumed below, with the keys sent now
        else:
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
                return 1
            coord.create_run(graph, doc, allow_uploads=bool(spec.get("allow_uploads")), run_id=run)
        proc = spawn(run, nodes, keys)
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
        return 0


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
        if status:   # the run has ended: its inputs need no hold any more (remote_run.Remote.send_input)
            release(run)
    finally:
        with contextlib.suppress(OSError, ValueError):
            if int(pidfile.read_text()) == os.getpid():
                pidfile.unlink()
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} run {run}: {status}", flush=True)
    return 0 if status else 1


def cmd_status(args) -> int:
    from .cli import iso, run_json
    store = Store(home())
    if args.run:
        row = store.db.execute("SELECT * FROM runs WHERE id=?", (args.run,)).fetchone()
        if row is None:
            raise Refusal(f"no run {args.run} on this host")
        answer(status=run_json(store, row), coordinator=coordinator_state(store, args.run, row["status"]))
        return 0
    runs = [{"run": r["id"], "pipeline": r["pipeline"], "status": r["status"], "created_at": iso(r["created_at"]),
             "coordinator": coordinator_state(store, r["id"], r["status"])}
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
    row = store.db.execute("SELECT status FROM runs WHERE id=?", (args.run,)).fetchone()
    answer(events=events, status=row["status"], coordinator=coordinator_state(store, args.run, row["status"]))
    return 0


def cmd_dump(args) -> int:
    """Everything `remote fetch` imports, once the run has ended: its run, task and event rows, the cache rows of its
    results, the artifact rows and store folders of every object they name (relative to the store: the other machine
    copies them, then checks each against its hash), each final attempt's log, and the bundle exported here."""
    from . import export as export_mod
    store = Store(home())
    run = args.run
    row = store.db.execute("SELECT * FROM runs WHERE id=?", (run,)).fetchone()
    if row is None:
        raise Refusal(f"no run {run} on this host")
    if row["status"] not in ("succeeded", "failed", "canceled"):
        state = coordinator_state(store, run, row["status"])
        raise Refusal(f"run {run} is still running here" if state == "running" else
                      f"run {run}'s coordinator is gone: `perception remote run` with --run-id {run} resumes it")
    tasks = [dict(t) for t in store.db.execute("SELECT * FROM tasks WHERE run_id=?", (run,))]
    keys = [t["cache_key"] for t in tasks if t["state"] in DONE and t["cache_key"]]
    cache = [dict(c) for k in keys for c in store.db.execute("SELECT * FROM cache WHERE key=?", (k,))]
    named = {}
    for doc in [t["outputs"] for t in tasks] + [c["outputs"] for c in cache]:
        for a in json.loads(doc or "{}").values():
            if a["shape"] in ("file", "tree"):
                named[a["hash"]] = a["shape"]
    artifacts = [dict(a) for h in named for a in store.db.execute("SELECT * FROM artifacts WHERE hash=?", (h,))]
    objects = [store.object_dir(shape, h).relative_to(store.root).as_posix() for h, shape in sorted(named.items())]
    logs = []
    for t in tasks:
        log = store.root / "work" / run / t["node_id"] / str(t["attempt"]) / "log.txt"
        if t["attempt"] and log.is_file() and not log.is_symlink():
            logs.append({"node": t["node_id"], "attempt": t["attempt"], "path": log.relative_to(store.root).as_posix()})
    bundle, problem = None, None
    if row["status"] == "succeeded":
        out = run_folder(run) / "bundle"
        if not (out / "manifest.json").exists():
            try:
                coord = Coordinator(store, registry(args.nodes or []), backends())
                export_mod.bundle(coord, run, out)
            except SystemExit as e:   # export's own refusals (an unfinished node, unrelated writers)
                problem = str(e)
        if (out / "manifest.json").exists():
            bundle = out.relative_to(store.root).as_posix()
    events = [dict(e) for e in store.events(run)]
    answer(run=dict(row), tasks=tasks, cache=cache, artifacts=artifacts, objects=objects, logs=logs, bundle=bundle,
           bundle_problem=problem, events=events)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="perception.remote_coordinator")
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("start").add_argument("--nodes", action="append")   # start reads them from its spec
    sub.choices["start"].set_defaults(fn=cmd_start)
    for name, fn in (("serve", cmd_serve), ("status", cmd_status), ("events", cmd_events), ("dump", cmd_dump)):
        p = sub.add_parser(name)
        p.add_argument("run", nargs="?" if name == "status" else None)
        p.add_argument("--nodes", action="append")
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
