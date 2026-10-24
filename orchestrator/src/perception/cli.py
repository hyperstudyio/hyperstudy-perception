"""perception: run video perception graphs and publish track bundles.

    perception doctor
    perception models list
    perception graph validate GRAPH.yaml
    perception plan GRAPH.yaml [--explain]
    perception run GRAPH.yaml [--input video=PATH] [--max-cost-usd N] [--allow-downloads]
    perception status [RUN_ID]
    perception logs RUN_ID [--follow] [--node ID]
    perception resume RUN_ID
    perception rerun RUN_ID --from NODE [--no-cache]
    perception cancel RUN_ID
    perception artifacts RUN_ID
    perception export RUN_ID --format bundle --output DIR
    perception hosts add NAME USER@HOST [--root DIR] | list | check NAME | remove NAME | clean NAME

The store (artifacts, run state) lives in $PERCEPTION_HOME (default ~/.hyperstudy-perception). Secrets come from the
environment and from ~/.config/hyperstudy/*.env; their values are never printed or written to run files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import yaml

from . import export as export_mod
from .backends.base import Backend
from .backends.native import NativeBackend, SelfBackend
from .backends.probe import probe
from .backends.remote import SshBackend
from .hosts import HostError, Hosts, ssh_command
from .ir import GraphError, compile_graph
from .manifests import Registry
from .plan import plan as make_plan
from .scheduler import Coordinator
from .store import Store

REPO = Path(__file__).resolve().parents[3]


def home() -> Path:
    return Path(os.environ.get("PERCEPTION_HOME", Path.home() / ".hyperstudy-perception"))


def load_secrets() -> dict[str, str]:
    secrets = {}
    conf = Path.home() / ".config" / "hyperstudy"
    for f in sorted(conf.glob("*.env")) if conf.is_dir() else []:
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                secrets[k.strip()] = v.strip()
    secrets.update({k: v for k, v in os.environ.items() if k in secrets or k.endswith(("_KEY", "_TOKEN"))})
    return secrets


def backends() -> list[Backend]:
    """This machine's, then the hosts `perception hosts add` recorded (their places are only ever chosen, never a
    default: they receive a node's inputs)."""
    return [NativeBackend(), SelfBackend(), *(SshBackend(h) for h in Hosts(home()).load().values())]


def registry(args) -> Registry:
    roots = [REPO / "nodes", *map(Path, args.nodes or [])]
    return Registry.load(*[r for r in roots if r.exists()])


def coordinator(args) -> Coordinator:
    return Coordinator(Store(home()), registry(args), backends(), load_secrets())


def only(doc: dict, chosen: list[str]) -> dict:
    """The graph with just the chosen nodes and the nodes they need (their inputs' $nodes.X, transitively); graph outputs
    of nodes left out go too. The app's stage picker: run part of a graph."""
    nodes = {n.get("id"): n for n in doc.get("nodes", [])}
    keep, todo = set(), [c for item in chosen for c in item.split(",") if c]
    while todo:
        name = todo.pop()
        if name not in nodes:
            raise GraphError([f"--only {name}: the graph has no node {name!r}"])
        if name in keep:
            continue
        keep.add(name)
        for ref in nodes[name].get("inputs", {}).values():
            if isinstance(ref, str) and ref.startswith("$nodes."):
                todo.append(ref.split(".")[1])
    outputs = {k: v for k, v in doc.get("outputs", {}).items() if not (isinstance(v, str) and v.startswith("$nodes.") and v.split(".")[1] not in keep)}
    return {**doc, "nodes": [n for n in doc["nodes"] if n.get("id") in keep], **({"outputs": outputs} if "outputs" in doc else {})}


def read_graph(path: Path, reg: Registry, max_cost: float | None = None, inputs: list[str] | None = None,
               places: list[str] | None = None, chosen: list[str] | None = None, sets: list[str] | None = None):
    doc = yaml.safe_load(path.read_text())
    for item in sets or []:   # --set node.key=value: a node's setting for this run (compile checks it against the manifest)
        target, eq, value = item.partition("=")
        name, dot, key = target.partition(".")
        if not (eq and dot and key):
            raise GraphError([f"--set {item}: expected NODE.KEY=VALUE"])
        node = next((n for n in doc.get("nodes", []) if n.get("id") == name), None)
        if node is None:
            raise GraphError([f"--set {name}: the graph has no node {name!r}"])
        node.setdefault("config", {})[key] = yaml.safe_load(value)   # 8 -> int, true -> bool, claude-opus-5-5 -> str
    for item in places or []:   # --place node=backend:accelerator chooses where a node runs (plan lists the options)
        name, _, place = item.partition("=")
        node = next((n for n in doc.get("nodes", []) if n.get("id") == name), None)
        if node is None:
            raise GraphError([f"--place {name}: the graph has no node {name!r}"])
        node["place"] = place
    if chosen:   # after --place and --set: they may name a node --only leaves out (it goes, with them)
        doc = only(doc, chosen)
    for item in inputs or []:   # --input name=path overrides a graph input's location
        name, _, value = item.partition("=")
        if name not in doc.get("inputs", {}):
            raise GraphError([f"--input {name}: the graph has no input {name!r}"])
        doc["inputs"][name]["uri"] = Path(value).expanduser().resolve().as_uri()
    if max_cost is not None:
        doc.setdefault("policy", {})["max_cost_usd"] = max_cost
    graph = compile_graph(doc, reg)
    graph.source = path.resolve()
    return graph, doc


# --- machine-readable output (--json): what the annotator app reads when it drives this CLI -------------------------

JSON_PROTOCOL = 1   # bump when a field changes meaning or goes away; adding fields is compatible


def emit(type_: str, /, **fields):
    """One JSON object per line, flushed at once: the app reads events as they happen. Positional-only, so a field may be
    called `kind` (events have one)."""
    print(json.dumps({"v": JSON_PROTOCOL, "type": type_, **fields}), flush=True)


def iso(t: float | None) -> str | None:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if t else None


def graph_error(args, e: GraphError) -> int:
    if args.json:
        emit("error", problems=e.problems)
    else:
        print("invalid graph:\n" + "\n".join(f"  - {p}" for p in e.problems))
    return 1


def plan_json(p) -> dict:
    nodes = [{"id": n.id, "uses": n.uses, "where": n.where, "why_not": n.why_not, "cached": n.cached,
              "downloads": [m.model_dump(include={"id", "source", "size_gb", "gated", "licence_mode"}) for m in n.downloads],
              "missing_secrets": n.missing_secrets, "cost_usd": n.cost_usd, "cost_note": n.cost_note, "uploads": n.uploads, "upload_to": n.upload_to,
              "place": n.place, "options": n.options, "estimate_s": n.estimate_s, "estimate_basis": n.estimate_basis,
              "needs": n.needs, "estimates": n.estimates, "title": n.title, "service": n.service, "description": n.description, "outputs": n.outputs,
              "models": [m.model_dump(include={"id", "source", "licence_mode", "size_gb"}) for m in n.models], "settings": n.settings}
             for n in p.nodes]
    return {"ok": p.ok, "nodes": nodes, "places": p.places, "estimated_usd": round(sum(n.cost_usd for n in p.nodes if not n.cached), 4)}


# --- commands -----------------------------------------------------------------------------------------------------

def cmd_doctor(args):
    caps = probe()
    if args.json:
        reg = registry(args)
        coord = Coordinator(Store(home()), reg, backends())
        nodes = []
        for m in reg:
            placement, why = coord.place(m)
            nodes.append({"ref": m.ref, "stage": m.stage or m.id, "description": m.description,
                          "where": f"{placement.backend.kind} ({placement.runner.accelerator})" if placement else None,
                          "why_not": "" if placement else why})
        emit("doctor", platform=caps.platform, cpu_cores=caps.cpu_cores, memory_gb=caps.memory_gb,
             accelerators=sorted(caps.accelerators), vram_gb=caps.vram_gb, tools=caps.tools, store=str(home()), nodes=nodes)
        return 0
    print(f"platform     {caps.platform}")
    print(f"cpu          {caps.cpu_cores} cores")
    print(f"memory       {caps.memory_gb} GB")
    print(f"accelerators {', '.join(sorted(caps.accelerators))}" + (f" ({caps.vram_gb} GB)" if caps.vram_gb else ""))
    for tool, path in caps.tools.items():
        print(f"{tool:12s} {path or 'not found'}")
    print(f"store        {home()}")
    reg = registry(args)
    coord = Coordinator(Store(home()), reg, backends())
    print("\nnodes")
    for m in reg:
        placement, why = coord.place(m)
        where = f"{placement.backend.kind} ({placement.runner.accelerator})" if placement else f"cannot run here: {why}"
        print(f"  {m.ref:32s} {where}")


def cmd_nodes(args):
    """Every node the app can offer: what it takes and makes, where it can run here, what it costs and sends, its models."""
    reg = registry(args)
    coord = Coordinator(Store(home()), reg, backends())
    nodes = []
    for m in reg:
        options, why = coord.placements(m)
        nodes.append({
            "ref": m.ref, "id": m.id, "version": m.version, "stage": m.stage or m.id, "description": m.description,
            "inputs": m.inputs, "outputs": m.outputs, "config": m.config, "places": [p.label for p in options], "why_not": why,
            "cost": m.cost.model_dump() if m.cost else None, "uploads": list(m.privacy.external_upload), "secrets": m.secrets,
            "models": [x.model_dump(include={"id", "source", "size_gb", "gated", "licence_mode"}) for x in m.models],
            "gpu": m.resources.gpu})
    if args.json:
        emit("nodes", nodes=nodes)
    else:
        for n in nodes:
            print(f"{n['ref']:32s} {', '.join(n['places']) or 'cannot run here: ' + n['why_not']}")
    return 0


def cmd_models(args):
    reg = registry(args)
    if args.json:
        emit("models", models=[{"node": m.ref, **x.model_dump(include={"id", "source", "size_gb", "gated", "licence_mode"})}
                               for m in reg for x in m.models])
        return 0
    for m in reg:
        for x in m.models:
            size = f"{x.size_gb:.1f} GB" if x.size_gb else "?"
            print(f"{m.ref:30s} {x.id:45s} {x.licence_mode:14s} {size:>8s} {'gated ' if x.gated else ''}{x.source or ''}")


def cmd_validate(args):
    try:
        graph, _ = read_graph(Path(args.graph), registry(args))
    except GraphError as e:
        if args.json:
            emit("graph", valid=False, nodes=[], problems=e.problems)
            return 1
        print(f"{args.graph}: invalid")
        for p in e.problems:
            print(f"  - {p}")
        return 1
    if args.json:
        emit("graph", valid=True, nodes=list(graph.nodes), problems=[])
        return 0
    print(f"{args.graph}: valid ({len(graph.nodes)} nodes: {' -> '.join(graph.nodes)})")
    return 0


def cmd_plan(args):
    reg = registry(args)
    try:
        graph, _ = read_graph(Path(args.graph), reg, args.max_cost_usd, args.input, args.place, args.only, args.set)
    except GraphError as e:
        return graph_error(args, e)
    p = make_plan(coordinator(args), graph)
    if args.json:
        emit("plan", **plan_json(p))
    else:
        print(p.render(explain=args.explain))
    return 0 if p.ok else 1


def last_event(store: Store, run_id: str) -> int:
    return store.db.execute("SELECT COALESCE(MAX(id), 0) FROM events WHERE run_id=?", (run_id,)).fetchone()[0]


def _run(coord: Coordinator, run_id: str, as_json: bool = False, after: int = 0) -> int:
    """Run and stream events; `after` skips the run's earlier history (resume and rerun show only what they do)."""
    if as_json:
        emit("run.created", run=run_id, store=str(coord.store.root))
    else:
        print(f"run {run_id}  (store {coord.store.root})")
    seen = after

    def flush():
        nonlocal seen
        for e in coord.store.events(run_id, seen):
            seen = e["id"]
            if as_json:
                emit("event", run=run_id, node=e["node_id"], kind=e["kind"], message=e["message"] or "", at=iso(e["at"]))
            else:
                print(f"  {time.strftime('%H:%M:%S', time.localtime(e['at']))} {e['node_id'] or '':14s} {e['kind']:16s} {e['message'] or ''}")

    async def watch():
        while True:
            flush()
            await asyncio.sleep(0.3)

    async def main():
        w = asyncio.create_task(watch())
        status = await coord.run(run_id)
        w.cancel()   # flush() below prints what it had not seen yet
        return status

    status = asyncio.run(main())
    flush()   # events written after the watcher's last look
    if as_json:
        emit("run.finished", run=run_id, status=status)
    else:
        print(f"run {run_id}: {status}")
    return 0 if status == "succeeded" else 1


def cmd_run(args):
    coord = coordinator(args)
    try:
        graph, doc = read_graph(Path(args.graph), coord.registry, args.max_cost_usd, args.input, args.place, args.only, args.set)
    except GraphError as e:
        return graph_error(args, e)
    p = make_plan(coord, graph, allow_downloads=args.allow_downloads, allow_uploads=args.allow_uploads)
    if not p.ok:
        if args.json:
            emit("plan", **plan_json(p))
        else:
            print(p.render(explain=True))
        return 1
    return _run(coord, coord.create_run(graph, doc, allow_uploads=args.allow_uploads), args.json)


def cmd_resume(args):
    coord = coordinator(args)
    if args.allow_uploads:
        coord.allow_uploads(args.run_id)
    return _run(coord, args.run_id, args.json, after=last_event(coord.store, args.run_id))


def cmd_rerun(args):
    coord = coordinator(args)
    if args.allow_uploads:
        coord.allow_uploads(args.run_id)
    after = last_event(coord.store, args.run_id)   # before the rerun writes its own run.rerun event
    coord.rerun(args.run_id, args.from_node, use_cache=not args.no_cache)
    return _run(coord, args.run_id, args.json, after=after)


def cmd_cancel(args):
    coord = coordinator(args)
    if not known_run(args, coord.store):
        return 1
    coord.cancel(args.run_id)
    if args.json:
        emit("canceled", run=args.run_id)
    else:
        print(f"run {args.run_id}: canceled")


def known_run(args, store: Store) -> bool:
    """Whether the run exists; if not, says so (as an error object under --json) for the caller to return 1."""
    if store.db.execute("SELECT 1 FROM runs WHERE id=?", (args.run_id,)).fetchone():
        return True
    if args.json:
        emit("error", problems=[f"no run {args.run_id}"])
    else:
        print(f"no run {args.run_id}")
    return False


def run_json(store: Store, run) -> dict:
    tasks = [{"node": t["node_id"], "state": t["state"], "attempt": t["attempt"], "runner": t["runner"],
              "started_at": iso(t["started_at"]), "finished_at": iso(t["finished_at"]),
              "seconds": round(t["finished_at"] - t["started_at"], 1) if t["finished_at"] and t["started_at"] else None,
              "cost_usd": t["cost_usd"], "error_code": t["error_code"], "error": t["error"]}
             for t in store.db.execute("SELECT * FROM tasks WHERE run_id=?", (run["id"],))]
    return {"run": run["id"], "pipeline": run["pipeline"], "status": run["status"], "created_at": iso(run["created_at"]),
            "spent_usd": run["spent_usd"], "max_cost_usd": run["max_cost_usd"], "tasks": tasks}


def cmd_status(args):
    store = Store(home())
    if args.json:
        if not args.run_id:
            emit("runs", runs=[{"run": r["id"], "pipeline": r["pipeline"], "status": r["status"], "created_at": iso(r["created_at"])}
                               for r in store.db.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 20")])
            return 0
        run = store.db.execute("SELECT * FROM runs WHERE id=?", (args.run_id,)).fetchone()
        if not run:
            emit("error", problems=[f"no run {args.run_id}"])
            return 1
        emit("status", **run_json(store, run))
        return 0
    if not args.run_id:
        for r in store.db.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 20"):
            print(f"{r['id']}  {r['pipeline']:20s} {r['status']:10s} {time.strftime('%Y-%m-%d %H:%M', time.localtime(r['created_at']))}")
        return 0
    run = store.db.execute("SELECT * FROM runs WHERE id=?", (args.run_id,)).fetchone()
    if not run:
        print(f"no run {args.run_id}")
        return 1
    print(f"{run['id']}  {run['pipeline']}  {run['status']}  spent ${run['spent_usd']:.2f} of ${run['max_cost_usd']:.2f}")
    for t in store.db.execute("SELECT * FROM tasks WHERE run_id=?", (args.run_id,)):
        took = f"{t['finished_at'] - t['started_at']:.1f} s" if t["finished_at"] and t["started_at"] else ""
        err = f"  {t['error_code']}: {t['error']}" if t["state"] == "failed" else ""
        print(f"  {t['node_id']:16s} {t['state']:10s} attempt {t['attempt']}  {t['runner'] or '':12s} {took}{err}")
    return 0


def cmd_logs(args):
    store = Store(home())
    if not known_run(args, store):   # --follow would otherwise wait forever for a run that never ends
        return 1
    seen = 0
    while True:
        for e in store.events(args.run_id, seen):
            seen = e["id"]
            if args.node and e["node_id"] != args.node:
                continue
            if args.json:
                emit("event", run=args.run_id, node=e["node_id"], kind=e["kind"], message=e["message"] or "", at=iso(e["at"]))
            else:
                print(f"{time.strftime('%H:%M:%S', time.localtime(e['at']))} {e['node_id'] or '':14s} {e['kind']:16s} {e['message'] or ''}")
        if args.node and not args.json:   # the node's own log is text for people; as JSON, logs streams events only
            t = store.db.execute("SELECT attempt FROM tasks WHERE run_id=? AND node_id=?", (args.run_id, args.node)).fetchone()
            if t and t["attempt"]:
                log = store.root / "work" / args.run_id / args.node / str(t["attempt"]) / "log.txt"
                if log.exists() and not args.follow:
                    print(f"--- {log}\n{log.read_text()[-20000:]}")
        run = store.db.execute("SELECT status FROM runs WHERE id=?", (args.run_id,)).fetchone()
        if not args.follow or (run and run["status"] != "running"):
            return 0
        time.sleep(0.5)


def cmd_artifacts(args):
    store = Store(home())
    if not known_run(args, store):
        return 1
    if args.json:
        emit("artifacts", run=args.run_id, artifacts=[
            {"node": t["node_id"], "name": name, "type": a["type"], "hash": a["hash"], "path": a["path"]}
            for t in store.db.execute("SELECT node_id, outputs FROM tasks WHERE run_id=?", (args.run_id,))
            for name, a in json.loads(t["outputs"] or "{}").items()])
        return 0
    for t in store.db.execute("SELECT node_id, state, outputs FROM tasks WHERE run_id=?", (args.run_id,)):
        for name, a in json.loads(t["outputs"] or "{}").items():
            print(f"{t['node_id']}.{name:12s} {a['type']:22s} {a['hash'][:12]}  {a['path']}")


def cmd_hosts(args):
    """Lab servers nodes can run on, reached over SSH with your own keys (batch mode: nothing asks for a password)."""
    hosts = Hosts(home())
    try:
        if args.action == "add":
            if not args.dest:
                raise HostError("hosts add NAME USER@HOST")
            hosts.add(args.name, args.dest, args.root)
        elif args.action in ("check", "remove", "clean") and not args.name:
            raise HostError(f"hosts {args.action} NAME")
        elif args.action == "check":
            hosts.check(args.name)
        elif args.action == "remove":
            # a node running there could no longer be stopped (cancel finds its place by the host's name)
            busy = Store(home()).db.execute("SELECT run_id, node_id FROM tasks WHERE state IN ('running', 'leased') AND runner LIKE ?",
                                            (f"ssh-{args.name}:%",)).fetchone()
            if busy:
                raise HostError(f"run {busy['run_id']} has {busy['node_id']} on {args.name}: cancel or finish it first "
                                f"(perception cancel {busy['run_id']})")
            hosts.remove(args.name)
        elif args.action == "clean":   # the inputs and code kept on the host between runs (never a running attempt's folder)
            host = hosts.load().get(args.name)
            if host is None:
                raise HostError(f"no host {args.name!r}")
            import shlex
            import subprocess
            q = shlex.quote
            root = q(host.root)
            # not while an attempt's folder is there: a node running (or its watchdog not done) still uses them
            script = (f'if [ -n "$(ls -A {root}/work 2>/dev/null)" ]; then echo "a node is running there" >&2; exit 4; fi; '
                      f"rm -rf {root}/objects {root}/code")
            out = subprocess.run([*ssh_command(), host.dest, f"sh -c {q(script)}"], capture_output=True, text=True, timeout=600)
            if out.returncode:
                raise HostError(f"could not clean {host.name}: {out.stderr.strip()[-300:]}")
    except HostError as e:
        if args.json:
            emit("error", problems=[str(e)])
        else:
            print(f"perception hosts: {e}", file=sys.stderr)
        return 1
    listed = [h.to_json() for h in hosts.load().values()]
    if args.json:
        emit("hosts", hosts=listed)
        return 0
    for h in listed:
        c = h["caps"]
        tools = ", ".join(f"{t} {'yes' if p else 'missing'}" for t, p in c["tools"].items())
        print(f"{h['name']:12s} {h['dest']:28s} {c['platform']}, {c['cpu_cores']} cores, {c['memory_gb']} GB, "
              f"{'/'.join(c['accelerators'])}" + (f" ({c['vram_gb']} GB)" if "cuda" in c["accelerators"] else "") + f"; {tools}")
    return 0


def cmd_export(args):
    coord = coordinator(args)
    out = export_mod.bundle(coord, args.run_id, Path(args.output))
    if args.json:
        emit("exported", run=args.run_id, path=str(out))
    else:
        print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="perception", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", action="append", help="another directory of node manifests")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output: one JSON object per line, each with a protocol version (for the annotator app)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("nodes", help="every node, with where it can run here").set_defaults(fn=cmd_nodes)
    models = sub.add_parser("models")
    models.add_argument("action", choices=["list"])
    models.set_defaults(fn=cmd_models)
    g = sub.add_parser("graph")
    g.add_argument("action", choices=["validate"])
    g.add_argument("graph")
    g.set_defaults(fn=cmd_validate)
    p = sub.add_parser("plan")
    p.add_argument("graph")
    p.add_argument("--explain", action="store_true")
    p.add_argument("--estimate-cost", action="store_true", help="(always shown)")
    p.add_argument("--max-cost-usd", type=float)
    p.add_argument("--input", action="append", metavar="NAME=PATH", help="set a graph input's file")
    p.add_argument("--place", action="append", metavar="NODE=BACKEND:ACCELERATOR", help="where a node runs (plan lists its options)")
    p.add_argument("--only", action="append", metavar="NODE[,NODE]", help="just these nodes and what they need")
    p.add_argument("--set", action="append", metavar="NODE.KEY=VALUE", help="a node's setting (plan lists the choices)")
    p.set_defaults(fn=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("graph")
    r.add_argument("--max-cost-usd", type=float)
    r.add_argument("--input", action="append", metavar="NAME=PATH", help="set a graph input's file")
    r.add_argument("--allow-downloads", action="store_true", help="let nodes fetch model weights that are not on this machine yet")
    r.add_argument("--place", action="append", metavar="NODE=BACKEND:ACCELERATOR", help="where a node runs (plan lists its options)")
    r.add_argument("--only", action="append", metavar="NODE[,NODE]", help="just these nodes and what they need")
    r.add_argument("--set", action="append", metavar="NODE.KEY=VALUE", help="a node's setting (plan lists the choices)")
    r.add_argument("--allow-uploads", action="store_true", help="let nodes send what the plan says off this machine (within policy.allow_upload)")
    r.set_defaults(fn=cmd_run)
    for name, fn in (("resume", cmd_resume), ("cancel", cmd_cancel), ("artifacts", cmd_artifacts)):
        x = sub.add_parser(name)
        x.add_argument("run_id")
        if name == "resume":
            x.add_argument("--allow-uploads", action="store_true", help="let the run's nodes send data off this machine from now on")
        x.set_defaults(fn=fn)
    s = sub.add_parser("status")
    s.add_argument("run_id", nargs="?")
    s.set_defaults(fn=cmd_status)
    lg = sub.add_parser("logs")
    lg.add_argument("run_id")
    lg.add_argument("--follow", action="store_true")
    lg.add_argument("--node")
    lg.set_defaults(fn=cmd_logs)
    rr = sub.add_parser("rerun")
    rr.add_argument("run_id")
    rr.add_argument("--from", dest="from_node", required=True)
    rr.add_argument("--no-cache", action="store_true", help="run the nodes even if the cache has their results")
    rr.add_argument("--allow-uploads", action="store_true", help="let the run's nodes send data off this machine from now on")
    rr.set_defaults(fn=cmd_rerun)
    h = sub.add_parser("hosts", help="lab servers nodes can run on, over SSH")
    h.add_argument("action", choices=["add", "list", "check", "remove", "clean"])
    h.add_argument("name", nargs="?")
    h.add_argument("dest", nargs="?", help="add: an ssh destination (user@host, or a ~/.ssh/config alias)")
    h.add_argument("--root", default=".cache/hyperstudy-perception", help="add: where on the host its files go (under its home if relative)")
    h.set_defaults(fn=cmd_hosts)
    ex = sub.add_parser("export")
    ex.add_argument("run_id")
    ex.add_argument("--format", choices=["bundle"], default="bundle")
    ex.add_argument("--output", required=True)
    ex.set_defaults(fn=cmd_export)
    args = ap.parse_args(argv)
    if not args.json:
        return args.fn(args) or 0
    try:
        return args.fn(args) or 0
    except Exception as e:   # the app reads one error object, not a traceback meant for people
        emit("error", problems=[f"{type(e).__name__}: {e}"])
        return 1


if __name__ == "__main__":
    sys.exit(main())
