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
    perception cloud on | off | status [aws | runpod]
    perception keys set NAME | list | remove NAME | check NAME
    perception setup [GRAPH.yaml] [--only NODE[,NODE]] [--place NODE=PLACE]

The store (artifacts, run state) lives in $PERCEPTION_HOME (default ~/.hyperstudy-perception). Keys come from the
environment, the system keychain (`perception keys set`) and ~/.config/hyperstudy/*.env files only you can read (keys.py);
their values are never printed or written to run files.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
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
from .backends.cloud import Cloud, CloudBackend, RunPodBackend
from .backends.remote import SshBackend
from .hosts import HostError, Hosts, probe_dest
from .keys import load_secrets
from .ir import GraphError, compile_graph
from .manifests import Registry
from .plan import plan as make_plan
from .scheduler import Coordinator
from .store import Store

from .release import REPO, home, release


def backends() -> list[Backend]:
    """This machine's, then the hosts `perception hosts add` recorded (their places are only ever chosen, never a
    default: they receive a node's inputs)."""
    offered = Cloud(home())
    cloud = [CloudBackend()] if offered.enabled else []
    if offered.runpod:   # its key for the driver alone (RunPodBackend.submit)
        cloud.append(RunPodBackend(api_key=load_secrets().get("RUNPOD_API_KEY")))
    return [NativeBackend(), SelfBackend(), *(SshBackend(h) for h in Hosts(home()).load().values()), *cloud]


def registry(args) -> Registry:
    roots = [REPO / "nodes", *map(Path, args.nodes or [])]
    return Registry.load(*[r for r in roots if r.exists()])


def coordinator(args) -> Coordinator:
    return Coordinator(Store(home()), registry(args), backends(), load_secrets())


def only(doc: dict, chosen: list[str], reg: Registry) -> dict:
    """The graph with just the chosen nodes and the nodes they need (their inputs' $nodes.X, transitively); graph outputs
    of nodes left out go too. An optional input (the manifest's optional_inputs) does not make a node needed: it stays
    connected when its node is chosen (or needed otherwise), and is dropped when it is not. The app's stage picker: run
    part of a graph."""
    nodes = {n.get("id"): n for n in doc.get("nodes", [])}

    def optional(node: dict) -> list[str]:
        try:
            return reg.get(node.get("uses", "")).optional_inputs
        except KeyError:   # an unknown node: compile_graph reports it
            return []

    def node_of(ref) -> str | None:
        return ref.split(".")[1] if isinstance(ref, str) and ref.startswith("$nodes.") else None

    keep, todo = set(), [c for item in chosen for c in item.split(",") if c]
    while todo:
        name = todo.pop()
        if name not in nodes:
            raise GraphError([f"--only {name}: the graph has no node {name!r}"])
        if name in keep:
            continue
        keep.add(name)
        todo += [node_of(ref) for key, ref in nodes[name].get("inputs", {}).items()
                 if node_of(ref) and key not in optional(nodes[name])]
    kept = []
    for n in doc["nodes"]:
        if n.get("id") in keep:
            # a producer the graph lacks stays named, for compile_graph to report
            dropped = [k for k, ref in n.get("inputs", {}).items() if k in optional(n) and node_of(ref) in nodes.keys() - keep]
            kept.append({**n, "inputs": {k: v for k, v in n["inputs"].items() if k not in dropped}} if dropped else n)
    outputs = {k: v for k, v in doc.get("outputs", {}).items() if not (node_of(v) and node_of(v) not in keep)}
    return {**doc, "nodes": kept, **({"outputs": outputs} if "outputs" in doc else {})}


def graph_path(arg: str) -> Path:
    """A graph argument: a file, or else one of the pipeline's own graphs by name (`perception graphs` lists them; an
    installed package has no graphs/ folder of the user's to point at)."""
    path = Path(arg)
    named = REPO / "graphs" / arg
    if not path.exists() and path.name == arg and named.is_file():
        return named
    return path


def graphs() -> list[dict]:
    """The pipeline's graphs: file name, pipeline name, the comment each starts with, and where it is."""
    out = []
    for path in sorted((REPO / "graphs").glob("*.yaml")):
        lines = path.read_text().splitlines()
        comment = itertools.takewhile(lambda l: l.startswith("#"), lines)
        description = " ".join(c for c in (l.lstrip("#").strip() for l in comment) if c)
        pipeline = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("pipeline:")), "")
        out.append({"file": path.name, "pipeline": pipeline, "description": description, "path": str(path)})
    return out


def cmd_graphs(args):
    if args.json:
        emit("graphs", graphs=graphs())
    else:
        for g in graphs():
            print(f"{g['file']:20s} {g['description']}")
    return 0


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
        doc = only(doc, chosen, reg)
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
              "needs": n.needs, "optional_needs": n.optional_needs, "estimates": n.estimates, "title": n.title, "service": n.service, "description": n.description, "outputs": n.outputs,
              "models": [m.model_dump(include={"id", "source", "licence_mode", "size_gb"}) for m in n.models], "settings": n.settings}
             for n in p.nodes]
    return {"ok": p.ok, "nodes": nodes, "places": p.places, "max_cost_usd": p.max_cost_usd, "estimated_usd": round(sum(n.cost_usd for n in p.nodes if not n.cached), 4)}


# --- commands -----------------------------------------------------------------------------------------------------

def pipeline_version() -> dict:
    """This pipeline: its package version (None in a checkout not installed), its commit (the package's, from
    RELEASE.json; None in a checkout) and where its files are."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        v = version("hyperstudy-perception")
    except PackageNotFoundError:
        v = None
    r = release()
    return {"version": v, "commit": r["commit"] if r else None, "root": str(REPO)}


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
             accelerators=sorted(caps.accelerators), vram_gb=caps.vram_gb, tools=caps.tools, store=str(home()), nodes=nodes,
             pipeline=pipeline_version())
        return 0
    v = pipeline_version()
    print(f"pipeline     {v['version'] or 'not installed'}" + (f" ({v['commit'][:7]})" if v["commit"] else " (checkout)") + f", {v['root']}")
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
            "inputs": m.inputs, "optional_inputs": m.optional_inputs, "outputs": m.outputs, "config": m.config, "places": [p.label for p in options], "why_not": why,
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
        # each node too: what it is called, the API it calls (service) and the keys it needs, and its fixed choices
        # (a model, an effort) at their defaults, with prices: what the app's settings show
        from .plan import settings
        emit("models", models=[{"node": m.ref, **x.model_dump(include={"id", "source", "size_gb", "gated", "licence_mode"})}
                               for m in reg for x in m.models],
             nodes=[{"node": m.ref, "title": m.title, "service": m.service, "secrets": list(m.secrets),
                     "settings": settings(m, {k: s.get("default") for k, s in m.config.items()})} for m in reg])
        return 0
    for m in reg:
        for x in m.models:
            size = f"{x.size_gb:.1f} GB" if x.size_gb else "?"
            print(f"{m.ref:30s} {x.id:45s} {x.licence_mode:14s} {size:>8s} {'gated ' if x.gated else ''}{x.source or ''}")


def cmd_validate(args):
    try:
        graph, _ = read_graph(graph_path(args.graph), registry(args))
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
        graph, _ = read_graph(graph_path(args.graph), reg, args.max_cost_usd, args.input, args.place, args.only, args.set)
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
        graph, doc = read_graph(graph_path(args.graph), coord.registry, args.max_cost_usd, args.input, args.place, args.only, args.set)
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
              "cost_usd": t["cost_usd"], "error_code": t["error_code"], "error": t["error"],
              "ran_on": json.loads(t["usage"] or "{}").get("ran_on")}
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
            hosts.add(args.name, args.dest, args.root, args.user, args.port, args.identity)
        elif args.action == "probe":   # a connection test: nothing is added, nothing changes on the host
            dest = args.dest or args.name
            if not dest:
                raise HostError("hosts probe USER@HOST")
            caps, root = probe_dest(dest, args.root, args.user, args.port, args.identity)
            found = {"dest": dest, "root": root, "caps": {**caps.__dict__, "accelerators": sorted(caps.accelerators)}}
            if args.json:
                emit("host_probe", **found)
            else:
                print(f"{dest}: {caps.platform}, {caps.cpu_cores} cores, {caps.memory_gb} GB, {'/'.join(sorted(caps.accelerators))}")
            return 0
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
        elif args.action == "clean":   # what stays on the host between runs: inputs, outputs, code (never a running attempt's folder)
            host = hosts.load().get(args.name)
            if host is None:
                raise HostError(f"no host {args.name!r}")
            import shlex
            import subprocess
            q = shlex.quote
            root = q(host.root)
            # not while a node runs there (an attempt's pid that is still that attempt's node); folders a cancel, a lost
            # connection or a stop mark left are removed with the inputs and code. Only perception's own root
            # (hosts.MARKER), and inputs are read-only there (made writable to be removed).
            script = (f'[ -e {root}/.hyperstudy-perception ] || {{ echo "{host.root} is not marked as perception\'s" >&2; exit 4; }}\n'
                      f'for f in {root}/work/*/*/*/pid; do [ -f "$f" ] || continue; p=$(cat "$f"); '
                      f'if ps -ww -p "$p" -o command= 2>/dev/null | grep -qF -- "${{f%/pid}}/request.json"; then '
                      f'echo "a node is running there" >&2; exit 4; fi; done\n'
                      f"chmod -R u+w {root}/objects 2>/dev/null; rm -rf {root}/objects {root}/code {root}/work")
            out = subprocess.run([*host.ssh(), host.dest, f"sh -c {q(script)}"], capture_output=True, text=True, timeout=600)
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


def cmd_cloud(args):
    """Whether nodes with a CUDA runner are offered a cloud GPU machine (started for the node, deleted after it): on AWS
    (SkyPilot, your AWS credentials) or RunPod (an API key). `on` checks that the provider can be used first."""
    cloud = Cloud(home())
    problem = None
    if args.action == "check":   # whether it could be used, changing nothing (a test)
        problem = runpod_problem() if args.provider == "runpod" else aws_problem()
        if args.json:
            emit("cloud_check", provider=args.provider, ok=problem is None, detail=problem or f"{args.provider} can be used")
        else:
            print(problem or f"{args.provider} can be used")
        return 0
    if args.action == "on":
        problem = runpod_problem() if args.provider == "runpod" else aws_problem()
        if problem is None:
            cloud.set(True, args.provider)
    elif args.action == "off":
        # a node on a cloud machine could no longer be stopped (cancel finds its place by its kind), nor its machine deleted
        kind = "runpod" if args.provider == "runpod" else "cloud"
        busy = Store(home()).db.execute("SELECT run_id, node_id FROM tasks WHERE state IN ('running', 'leased') AND runner LIKE ?",
                                        (f"{kind}:%",)).fetchone()
        if busy:
            problem = f"run {busy['run_id']} has {busy['node_id']} on a cloud machine: cancel or finish it first (perception cancel {busy['run_id']})"
        else:
            cloud.set(False, args.provider)
    if problem:
        if args.json:
            emit("error", problems=[problem])
        else:
            print(f"perception cloud: {problem}", file=sys.stderr)
        return 1
    aws, rp = CloudBackend(), RunPodBackend()
    if args.json:
        emit("cloud", enabled=cloud.enabled, machine=aws.machine_title, region=aws.region, hourly_usd=aws.hourly_usd,
             runpod={"enabled": cloud.runpod, "machine": rp.machine_title, "region": rp.region, "hourly_usd": rp.hourly_usd})
    else:
        print(f"aws    {'on ' if cloud.enabled else 'off'}  {aws.machine_title}, {aws.region}, ${aws.hourly_usd}/h")
        print(f"runpod {'on ' if cloud.runpod else 'off'}  {rp.machine_title}; {rp.region}, at most ${rp.hourly_usd}/h")
    return 0


def aws_problem() -> str | None:
    import re
    import shutil
    import subprocess
    sky = os.environ.get("PERCEPTION_SKY") or shutil.which("sky")
    if not sky:
        return "SkyPilot is not installed (uv tool install 'skypilot[aws]')"
    check = subprocess.run([sky, "check", "aws"], capture_output=True, text=True, timeout=300)
    if check.returncode or not re.search(r"\bAWS: enabled", re.sub(r"\x1b\[[0-9;]*m", "", check.stdout)):
        return f"SkyPilot cannot use AWS with your credentials yet (sky check aws): {(check.stdout + check.stderr).strip()[-400:]}"
    return None


def runpod_problem() -> str | None:
    """A RunPod API key, which RunPod accepts (listing your pods: nothing is created)."""
    from .runpod import RunPod, RunPodError
    key = load_secrets().get("RUNPOD_API_KEY")
    if not key:
        return ("no RunPod API key: create one at https://console.runpod.io/user/settings and put RUNPOD_API_KEY=... in "
                "~/.config/hyperstudy/runpod.env (chmod 600)")
    try:
        RunPod(key).pods()
    except RunPodError as e:
        return f"RunPod did not accept the key in RUNPOD_API_KEY ({e})" if e.status in (401, 403) else f"RunPod could not be checked: {e}"
    return None


def cmd_keys(args):
    """Your keys for nodes and cloud places (keys.py). `set` reads the value from the terminal (not echoed) or from stdin,
    never from the command line (shell history, the process list); `list` gives names and where each comes from."""
    import getpass
    from .keys import Keys
    keys = Keys()
    try:
        if args.action == "set":
            if not args.name:
                raise ValueError("keys set NAME: which key?")
            value = getpass.getpass(f"{args.name}: ") if sys.stdin.isatty() else sys.stdin.readline()
            where = keys.set(args.name, value)
            result = {"name": args.name, "stored": where}
        elif args.action == "remove":
            if not args.name:
                raise ValueError("keys remove NAME: which key?")
            result = {"name": args.name, "removed": keys.remove(args.name)}
        elif args.action == "check":   # whether its provider accepts it: a refused key is an answer, not a failure
            if not args.name:
                raise ValueError("keys check NAME: which key?")
            from .keycheck import check, hugging_face
            value = keys.load().values.get(args.name)
            if args.name == "HF_TOKEN":   # and each gated model the nodes use: whether its account has access
                r = hugging_face(value, registry(args))
                (ok, detail), extra = (r["ok"], r["detail"]), {"account": r["account"], "models": r["models"]}
            else:
                (ok, detail), extra = check(args.name, value), {}
            if args.json:
                emit("key_check", name=args.name, ok=ok, detail=detail, **extra)
            else:
                print(detail)
                for m in extra.get("models", []):
                    how = f", {m['gating']} approval" if m["gating"] else ""
                    print(f"  {m['repo']:45s} {m['access']}{how}{': ' + m['reason'] if m['reason'] else ''}")
            return 0
        else:
            found = keys.load()
            result = {"keys": [{"name": k, "source": found.sources[k]} for k in sorted(found.values)], "problems": found.problems}
    except ValueError as e:
        if args.json:
            emit("error", problems=[str(e)])
        else:
            print(f"perception keys: {e}", file=sys.stderr)
        return 1
    if args.json:
        emit("keys", **result)
    elif args.action == "set":
        print(f"{args.name} stored in {result['stored']}")
    elif args.action == "remove":
        print(f"{args.name} removed from {', '.join(result['removed'])}" if result["removed"] else f"{args.name} was not stored by perception keys")
    else:
        for k in result["keys"]:
            print(f"{k['name']:28s} {k['source']}")
        for p in result["problems"]:
            print(f"refused: {p}")
    return 0


def cmd_setup(args):
    """Everything the nodes need here, ready before a run (setup.py): each environment synced from its lock, each node's
    weights fetched. With a graph, the nodes it would run here (--only, --place as for run); without,
    every node that can run here. A node placed on another machine is set up there when it runs."""
    from . import setup
    reg = registry(args)
    coord = coordinator(args)
    if args.graph:
        try:
            graph, _ = read_graph(graph_path(args.graph), reg, places=args.place, chosen=args.only)
        except GraphError as e:
            return graph_error(args, e)
        chosen = [(n.manifest, *coord.place(n.manifest, n.spec.backend, n.spec.place, graph.spec.policy), n.config)
                  for n in graph.nodes.values()]
    else:   # every node: those that cannot run here are simply not this machine's to set up
        chosen = [(m, coord.place(m)[0], "", None) for m in reg]
    result = setup.Setup()

    def report(step):
        result.steps.append(step)
        if args.json:
            emit("setup.step", node=step.node, what=step.what, ok=step.ok, seconds=step.seconds, message=step.message)
        else:
            print(f"  {step.what:12s} {step.node or 'this machine':32s} {'ok' if step.ok else 'FAILED':6s} {step.seconds:6.1f} s  {step.message}")

    for m, placement, why, config in chosen:
        if m.service:
            continue   # runs on its API's machines: nothing to set up here
        if placement is None:
            if args.graph:   # the graph would not run either: say so, not ready
                report(setup.Step(m.ref, "place", False, 0.0, f"cannot run here: {why}"))
            continue
        if placement.backend.remote:
            if not args.json:
                print(f"  {'':12s} {m.ref:32s} runs on {placement.backend.title}: set up there when it runs")
            continue
        for step in setup.setup_node(m, placement.runner, config=config, secrets=coord.secrets):
            report(step)
    if args.json:
        emit("setup", ok=result.ok, steps=[s.__dict__ for s in result.steps])
    else:
        print("ready" if result.ok else "not ready: see the steps marked FAILED")
    return 0 if result.ok else 1


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
    sub.add_parser("graphs", help="the pipeline's graphs, which run and plan take by name").set_defaults(fn=cmd_graphs)
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
    h.add_argument("action", choices=["add", "list", "check", "remove", "clean", "probe"],
                   help="clean: remove the inputs, the outputs of nodes that ran there and the code kept on a host")
    h.add_argument("name", nargs="?", help="probe: the ssh destination")
    h.add_argument("dest", nargs="?", help="add: an ssh destination (user@host, or a ~/.ssh/config alias)")
    h.add_argument("--root", default=".cache/hyperstudy-perception", help="add: where on the host its files go (under its home if relative)")
    h.add_argument("--user", help="add, probe: the user to log in as, over what ~/.ssh/config says")
    h.add_argument("--port", type=int, help="add, probe: the ssh port, over what ~/.ssh/config says")
    h.add_argument("--identity", help="add, probe: a private key file on this machine, over what ~/.ssh/config says")
    h.set_defaults(fn=cmd_hosts)
    c = sub.add_parser("cloud", help="offer a cloud GPU machine to nodes that can use one")
    c.add_argument("action", choices=["on", "off", "status", "check"])
    c.add_argument("provider", nargs="?", choices=Cloud.providers, default="aws")
    c.set_defaults(fn=cmd_cloud)
    k = sub.add_parser("keys", help="your API keys and tokens: store one in the keychain, list where each comes from")
    k.add_argument("action", choices=["set", "list", "remove", "check"])
    k.add_argument("name", nargs="?")
    k.set_defaults(fn=cmd_keys)
    st = sub.add_parser("setup", help="install what nodes need here before a run: environments and weights")
    st.add_argument("graph", nargs="?")
    st.add_argument("--only", action="append", metavar="NODE[,NODE]", help="just these nodes and what they need")
    st.add_argument("--place", action="append", metavar="NODE=BACKEND:ACCELERATOR", help="where a node runs (as for run)")
    st.set_defaults(fn=cmd_setup)
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
