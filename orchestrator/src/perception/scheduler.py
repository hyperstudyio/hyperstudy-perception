"""The coordinator: one per run, state in SQLite, so a run survives the coordinator.

Task states: pending (waiting on inputs or a retry delay) -> ready (inputs committed) -> running (submitted, leased
to this coordinator) -> succeeded | cached | failed | canceled. (`leased` is the brief state between choosing a
backend and the submit returning.)

Guarantees (spec, "Scheduler and execution policy"):
- A node is scheduled only when every input is committed.
- At-least-once execution, effectively-once publication: outputs are validated, then published under their content
  hash, then the task row is updated, in that order. A crash between the steps leaves either nothing or an
  unreferenced object, never a half-written artifact.
- A restarted coordinator (resume) looks at every task the previous one left running: if its process is still
  alive it is adopted and waited for; if it finished, its response is collected; otherwise it is retried.
- Transient errors (and a process that died without a response) are retried with backoff, up to policy.retries.
  input / schema / privacy / licence / internal errors are terminal.
- Independent nodes run in parallel within CPU, memory and GPU budgets; `gpu: exclusive` nodes run one at a time.
- A hard cost cap: a billable node is not submitted if its estimate would exceed what the run has left.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePath

from . import artifacts
from .backends.base import Backend, child_env
from .ir import CompiledNode, Graph, Policy, compile_graph
from .manifests import Manifest, Registry, Runner
from .store import Artifact, Store, dumps

TERMINAL = {"succeeded", "cached", "failed", "canceled"}
DONE = {"succeeded", "cached"}
LICENCE_ORDER = ["commercial-ok", "api-terms", "needs-review", "academic-only"]


def output_path(out_dir: Path, rel) -> tuple[Path | None, str]:
    """Where output `rel` is, checked without trusting anything the node could have changed: `rel` must be a plain
    relative path (checked before touching the disk), and no step from `out_dir` itself down to it may be a symlink.
    Nothing is resolved: a node (or a container) can replace any of these entries with a link to a host path."""
    parts = PurePath(rel).parts if isinstance(rel, str) and rel else ()
    if not parts or PurePath(rel).is_absolute() or PurePath(rel).drive or any(p in (".", "..") for p in rel.replace("\\", "/").split("/")):
        return None, f"{rel!r} is not a plain relative path"
    steps = [out_dir]
    for part in parts:
        steps.append(steps[-1] / part)
    for i, current in enumerate(steps):
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            return None, f"{rel} is missing"
        if stat.S_ISLNK(mode):
            return None, f"{current.name} is a symlink"
        if i < len(steps) - 1 and not stat.S_ISDIR(mode):
            return None, f"{current.name} is not a directory"
    return steps[-1], ""


def manifest_cost(m: Manifest, config: dict, inputs: dict[str, Artifact]) -> float:
    """A billable node's upper estimate: its manifest's price per unit (for this config) times the units in its input."""
    if m.cost is None:
        return 0.0
    if m.cost.per == "run":
        return m.cost.unit_usd(config)
    doc = json.loads(inputs[m.cost.input].path.read_text())
    units = len(doc["shots"]) if m.cost.per == "shot" else doc["duration"] / 60
    return round(m.cost.unit_usd(config) * units, 4)


class LeaseHeld(RuntimeError):
    pass


@dataclass
class Placement:
    backend: Backend
    runner: Runner

    @property
    def label(self) -> str:
        """Where it runs, as a graph's `place` and a task's runner name it: `native:mps`, `native:cpu`..."""
        return f"{self.backend.kind}:{self.runner.accelerator}"


class Coordinator:
    def __init__(self, store: Store, registry: Registry, backends: list[Backend], secrets: dict[str, str] | None = None,
                 lease_s: float = 30.0, tick_s: float = 0.2, backoff_s: float = 2.0):
        self.store, self.registry = store, registry
        self.backends = {b.kind: b for b in backends}
        self.secrets = secrets or {}
        self.lease_s, self.tick_s, self.backoff_s = lease_s, tick_s, backoff_s
        self.owner = uuid.uuid4().hex

    # --- runs -----------------------------------------------------------------------------------------------

    def create_run(self, graph: Graph, graph_doc: dict, allow_uploads: bool = False) -> str:
        """A run of `graph`. `allow_uploads`: its nodes may send data off the machine (within the graph's
        policy.allow_upload); without it a node that would is failed before it starts, whenever it comes to run."""
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        inputs = {name: self._graph_input(graph, name).to_json() for name in graph.spec.inputs}
        with self.store.tx() as db:
            db.execute("INSERT INTO runs (id, pipeline, graph, graph_path, inputs, status, created_at, max_cost_usd, allow_uploads) VALUES (?,?,?,?,?,?,?,?,?)",
                       (run_id, graph.spec.pipeline, dumps(graph_doc), str(graph.source) if graph.source else None,
                        dumps(inputs), "running", time.time(), graph.spec.policy.max_cost_usd, int(allow_uploads)))
            for node_id in graph.nodes:
                db.execute("INSERT INTO tasks (run_id, node_id, state) VALUES (?,?,?)", (run_id, node_id, "pending"))
        self.store.event(run_id, None, "run.created", f"{len(graph.nodes)} nodes")
        return run_id

    def allow_uploads(self, run_id: str):
        """Let the run's nodes send data off the machine from now on (resume or rerun --allow-uploads)."""
        with self.store.tx() as db:
            db.execute("UPDATE runs SET allow_uploads=1 WHERE id=?", (run_id,))

    def graph_of(self, run_id: str) -> Graph:
        row = self.store.db.execute("SELECT graph, graph_path FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"no run {run_id}")
        graph = compile_graph(json.loads(row["graph"]), self.registry)
        graph.source = Path(row["graph_path"]) if row["graph_path"] else None
        return graph

    def rerun(self, run_id: str, from_node: str, use_cache: bool = True):
        """Reset `from_node` and everything downstream of it. Without the cache (`use_cache=False`) only `from_node`
        is forced to run: downstream nodes run again only if its new outputs differ, and otherwise come from the
        cache (spec: rerun only nodes whose declared inputs change)."""
        graph = self.graph_of(run_id)
        if from_node not in graph.nodes:
            raise KeyError(f"run {run_id} has no node {from_node}")
        reset = {from_node} | graph.descendants(from_node)
        with self.store.tx() as db:
            for node_id in reset:
                db.execute("""UPDATE tasks SET state='pending', first_attempt=attempt+1, cache_key=NULL, runner=NULL, lease_owner=NULL,
                              lease_expires=NULL, handle=NULL, not_before=NULL, started_at=NULL, finished_at=NULL,
                              error_code=NULL, error=NULL, cost_usd=0, usage=NULL, outputs=NULL, skip_cache=?
                              WHERE run_id=? AND node_id=?""", (0 if use_cache or node_id != from_node else 1, run_id, node_id))
            db.execute("UPDATE runs SET status='running', finished_at=NULL WHERE id=?", (run_id,))
        self.store.event(run_id, None, "run.rerun", f"from {from_node}: {', '.join(sorted(reset))}")

    def cancel(self, run_id: str):
        """Record the cancel first, then kill: once a task is canceled, a coordinator that sees its process die reports
        nothing (_attempt_failed ignores tasks no longer running), so the kill is charged once, here."""
        with self.store.tx() as db:
            running = []
            for t in self._tasks(run_id).values():
                if t["state"] == "leased" and t["runner"]:   # perhaps started before its coordinator recorded it running
                    handle = self.backends[t["runner"].split(":")[0]].recorded_handle(
                        self.store.root / "work" / run_id / t["node_id"] / str(t["attempt"]))
                    if handle:
                        t = {**t, "handle": json.dumps(handle)}
                if t["state"] in ("running", "leased") and t["handle"]:
                    running.append(t)
            # a killed attempt may have spent up to its reservation: it is charged that, as a crash is, and the
            # reservation is used up (cost_usd 0), so nothing else charges it again
            db.execute("UPDATE runs SET spent_usd = spent_usd + ? WHERE id=?", (sum(t["cost_usd"] or 0.0 for t in running), run_id))
            for t in running:
                db.execute("UPDATE tasks SET cost_usd=0 WHERE run_id=? AND node_id=?", (run_id, t["node_id"]))
            db.execute("UPDATE tasks SET state='canceled', finished_at=? WHERE run_id=? AND state NOT IN ('succeeded','cached','failed')",
                       (time.time(), run_id))
            db.execute("UPDATE runs SET status='canceled', finished_at=? WHERE id=?", (time.time(), run_id))
        for t in running:
            self.backends[t["runner"].split(":")[0]].cancel(json.loads(t["handle"]))
        self.store.event(run_id, None, "run.canceled")

    # --- the loop -------------------------------------------------------------------------------------------

    async def run(self, run_id: str) -> str:
        graph = self.graph_of(run_id)
        self._recover(run_id, graph)
        last_heartbeat = 0.0
        while True:
            if self._canceled(run_id):
                return self._stop_canceled(run_id)
            tasks = self._tasks(run_id)
            for t in tasks.values():
                if t["state"] == "running" and t["lease_owner"] == self.owner:
                    self._check_running(run_id, graph, t)
            tasks = self._tasks(run_id)
            self._promote(run_id, graph, tasks)
            tasks = self._tasks(run_id)
            for node_id in graph.nodes:          # topological order: upstream nodes are placed first
                if tasks[node_id]["state"] == "ready":
                    self._start(run_id, graph, graph.nodes[node_id], tasks)
                    tasks = self._tasks(run_id)
            if time.time() - last_heartbeat > self.lease_s / 3:
                self.store.db.execute("UPDATE tasks SET lease_expires=? WHERE run_id=? AND lease_owner=? AND state='running'",
                                      (time.time() + self.lease_s, run_id, self.owner))
                last_heartbeat = time.time()
            if all(t["state"] in TERMINAL for t in tasks.values()):
                if self._canceled(run_id):   # canceled since the top of this pass
                    return self._stop_canceled(run_id)
                status = "succeeded" if all(t["state"] in DONE for t in tasks.values()) else "failed"
                self.store.db.execute("UPDATE runs SET status=?, finished_at=? WHERE id=?", (status, time.time(), run_id))
                self.store.event(run_id, None, f"run.{status}")
                return status
            await asyncio.sleep(self.tick_s)   # the loop's own pace: node processes run independently of it

    def _canceled(self, run_id: str) -> bool:
        return self.store.db.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()["status"] == "canceled"

    def _stop_canceled(self, run_id: str) -> str:
        """The run was canceled, perhaps by another process (cancel() recorded it and said so). This coordinator works from
        snapshots, so it may have retried or started a node since: stop what it has running, and leave every unfinished
        task canceled."""
        killed = 0.0
        for t in self._tasks(run_id).values():
            if t["state"] in ("running", "leased") and t["lease_owner"] == self.owner and t["handle"]:
                self.backends[t["runner"].split(":")[0]].cancel(json.loads(t["handle"]))
                killed += t["cost_usd"] or 0.0   # charged as cancel() charges what it kills
        with self.store.tx():
            self.store.db.execute("UPDATE runs SET spent_usd = spent_usd + ? WHERE id=?", (killed, run_id))
            self.store.db.execute("UPDATE tasks SET state='canceled', finished_at=COALESCE(finished_at, ?) WHERE run_id=? AND state NOT IN ('succeeded','cached','failed','canceled')",
                                  (time.time(), run_id))
        return "canceled"

    def _tasks(self, run_id: str) -> dict[str, dict]:
        return {r["node_id"]: dict(r) for r in self.store.db.execute("SELECT * FROM tasks WHERE run_id=?", (run_id,))}

    def _recover(self, run_id: str, graph: Graph):
        """Take over a run another coordinator left (it crashed, or was stopped). Only expired leases are taken: a
        coordinator that is still alive renews its leases every lease_s / 3."""
        tasks = self._tasks(run_id)
        live = [t for t in tasks.values() if t["state"] in ("leased", "running") and t["lease_owner"] not in (None, self.owner)
                and (t["lease_expires"] or 0) > time.time()]
        if live:
            wait = max(t["lease_expires"] for t in live) - time.time()
            raise LeaseHeld(f"run {run_id} is held by another coordinator (its lease on {live[0]['node_id']} runs "
                            f"{wait:.0f} s more); if that process is gone, resume after the lease expires")
        self.store.db.execute("UPDATE runs SET status='running', finished_at=NULL WHERE id=? AND status!='canceled'", (run_id,))
        for t in tasks.values():
            if t["state"] == "leased":
                # chosen; perhaps started (its process recorded a handle) before the coordinator could record it running
                handle = self.backends[t["runner"].split(":")[0]].recorded_handle(
                    self.store.root / "work" / run_id / t["node_id"] / str(t["attempt"])) if t["runner"] else None
                if handle is None:   # never submitted
                    self._set(run_id, t["node_id"], state="pending", lease_owner=None, lease_expires=None)
                    continue
                self._set(run_id, t["node_id"], state="running", handle=dumps(handle), started_at=time.time())
                t = self._tasks(run_id)[t["node_id"]]
            if t["state"] == "running" and t["lease_owner"] != self.owner:
                backend = self.backends[t["runner"].split(":")[0]]
                handle = json.loads(t["handle"])
                still = backend.poll(handle)
                self._set(run_id, t["node_id"], lease_owner=self.owner, lease_expires=time.time() + self.lease_s)
                self.store.event(run_id, t["node_id"], "task.adopted", f"attempt {t['attempt']} {'still running' if still == 'running' else 'finished while no coordinator watched'}")

    def _promote(self, run_id: str, graph: Graph, tasks: dict[str, dict]):
        now = time.time()
        for node_id, node in graph.nodes.items():
            t = tasks[node_id]
            if t["state"] != "pending" or (t["not_before"] and t["not_before"] > now):
                continue
            ups = [tasks[u]["state"] for u in node.upstream]
            if any(s in ("failed", "canceled") for s in ups):
                self._set(run_id, node_id, state="canceled", error_code="upstream", error="an upstream node failed", finished_at=now)
                self.store.event(run_id, node_id, "task.canceled", "upstream failed")
            elif all(s in DONE for s in ups):
                self._set(run_id, node_id, state="ready")

    # --- starting a task ------------------------------------------------------------------------------------

    def _inputs(self, run_id: str, graph: Graph, node: CompiledNode, tasks: dict[str, dict]) -> dict[str, Artifact]:
        run_inputs = json.loads(self.store.db.execute("SELECT inputs FROM runs WHERE id=?", (run_id,)).fetchone()["inputs"])
        out = {}
        for name, ref in node.inputs.items():
            if ref.node is None:
                out[name] = Artifact.from_json(run_inputs[ref.name])
            else:
                out[name] = Artifact.from_json(json.loads(tasks[ref.node]["outputs"])[ref.name])
        return out

    def placements(self, manifest: Manifest, wanted: str = "auto", policy: Policy | None = None) -> tuple[list[Placement], str]:
        """Every place this node can run, best first: this machine's (accelerated runners before CPU ones), then other
        machines' (lab servers, cloud), which a graph's policy must allow: they receive the node's inputs. Why none,
        when none."""
        options, reasons = self._placements(manifest, wanted, policy)
        return options, "" if options else "; ".join(f"{k}: {v}" for k, v in reasons.items()) or \
            f"no runner of kind {wanted if wanted != 'auto' else 'native or self'}"

    def _placements(self, manifest: Manifest, wanted: str, policy: Policy | None) -> tuple[list[Placement], dict[str, str]]:
        """The places, and why each backend that offers none cannot run it (backends without its kind of runner aside)."""
        local = [b for b in self.backends.values() if not b.remote]
        remote = [b for b in self.backends.values() if b.remote]
        chosen = [self.backends[wanted]] if wanted != "auto" and wanted in self.backends else local + remote if wanted == "auto" else []
        options, reasons = [], {}
        for backend in chosen:
            runners, why = backend.runners_for(manifest)
            # Apple silicon's memory is shared by CPU and GPU: a node that runs on the GPU there has no reason to be
            # offered the CPU too (a CUDA machine's CPU fallback stays a choice: its GPU memory is its own)
            if any(r.accelerator == "mps" for r in runners):
                runners = [r for r in runners if r.accelerator != "cpu"]
            if runners and backend.remote and (refusal := self.remote_policy(manifest, backend, policy)):
                runners, why = [], refusal
            options += [Placement(backend, r) for r in runners]
            if not runners and not why.startswith("no "):
                reasons[backend.kind] = why
        return options, reasons

    @staticmethod
    def remote_policy(manifest: Manifest, backend: Backend, policy: Policy | None) -> str:
        """Why the graph's policy keeps this node off another machine, or "". Without a graph (doctor, nodes), only
        what could run there is asked, not what may."""
        if policy is None:
            return ""
        if policy.placement == "local" or policy.privacy == "local-only":
            return "the graph keeps its nodes on this machine (policy.placement / policy.privacy)"
        if not_allowed := [k for k in backend.sends(manifest) if k not in policy.allow_upload]:
            return f"its inputs hold {', '.join(not_allowed)}, which policy.allow_upload does not let leave this machine"
        return ""

    def place(self, manifest: Manifest, wanted: str = "auto", choice: str | None = None,
              policy: Policy | None = None) -> tuple[Placement | None, str]:
        """Where this node runs: the graph's choice (`place`, as `native:cpu` or `ssh-gpu1:cuda`), else the best place
        on this machine. Another machine is only ever chosen, never a default: it receives the node's inputs."""
        options, reasons = self._placements(manifest, wanted, policy)
        why = "" if options else "; ".join(f"{k}: {v}" for k, v in reasons.items()) or f"no runner of kind {wanted}"
        if choice is None:
            here = [p for p in options if not p.backend.remote]
            if here:
                return here[0], ""
            if options:
                return None, f"it cannot run on this machine; it can run on {', '.join(p.label for p in options)} (choose one with --place)"
            return None, why
        chosen = next((p for p in options if p.label == choice), None)
        if chosen is None:
            listed = f" ({', '.join(p.label for p in options)})" if options else ""
            reason = reasons.get(choice.split(":")[0]) or (why if not options else "")
            return None, f"{choice} is not one of the places it can run{listed}" + (f": {reason}" if reason else "")
        return chosen, ""

    def uploads(self, manifest: Manifest, placement: Placement) -> tuple[list[str], str | None]:
        """The kinds of data that leave this machine when the node runs at `placement`, and where they go: to the API it
        calls (its manifest's privacy), or with its inputs to the other machine it runs on."""
        if manifest.privacy.external_upload:
            return list(manifest.privacy.external_upload), manifest.service
        sent = placement.backend.sends(manifest)
        return sent, placement.backend.title if sent else None

    def cache_key(self, node: CompiledNode, placement: Placement, inputs: dict[str, Artifact]) -> str:
        m = node.manifest
        key = {
            "node": m.ref,
            "runner": placement.backend.identity(m, placement.runner),
            "config": node.config,
            "inputs": {name: [a.type, a.hash] for name, a in sorted(inputs.items())},
            "outputs": m.outputs,
            "partition": None,
        }
        return hashlib.sha256(artifacts.canonical(key).encode()).hexdigest()

    def _start(self, run_id: str, graph: Graph, node: CompiledNode, tasks: dict[str, dict]):
        t, m = tasks[node.id], node.manifest
        placement, why = self.place(m, node.spec.backend, node.spec.place, graph.spec.policy)
        if placement is None:
            self._fail(run_id, node.id, "input", f"no backend can run {m.ref}: {why}")
            return
        inputs = self._inputs(run_id, graph, node, tasks)
        key = self.cache_key(node, placement, inputs)

        if not t["skip_cache"]:
            hit = self.store.db.execute("SELECT outputs, usage FROM cache WHERE key=?", (key,)).fetchone()
            if hit:
                outs = {k: Artifact.from_json(v) for k, v in json.loads(hit["outputs"]).items()}
                if all(self.store.has(a) for a in outs.values()):
                    self._set(run_id, node.id, state="cached", cache_key=key, outputs=dumps({k: a.to_json() for k, a in outs.items()}),
                              usage=hit["usage"], finished_at=time.time())
                    self.store.event(run_id, node.id, "task.cached", key[:12])
                    return

        # consent to send data off is the run's (--allow-uploads), checked when the node is about to run: a resume or a
        # rerun, or a cache that is gone, cannot make a node upload in a run that was not allowed to
        sent, to = self.uploads(m, placement)
        if sent and not self.store.db.execute("SELECT allow_uploads FROM runs WHERE id=?", (run_id,)).fetchone()[0]:
            self._fail(run_id, node.id, "policy", f"{m.ref} sends {', '.join(sent)} off this machine (to {to}), "
                                                  "and this run may not (--allow-uploads)")
            return
        if not self._fits(run_id, graph, m, placement, tasks):
            return   # stays ready; started when running nodes free their share
        estimate = placement.backend.estimate(m, placement.runner, inputs)
        estimate.cost_usd += manifest_cost(m, node.config, inputs)
        run = self.store.db.execute("SELECT max_cost_usd, spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()
        reserved = sum(x["cost_usd"] or 0.0 for x in tasks.values() if x["state"] in ("leased", "running"))
        if estimate.cost_usd > 0 and run["spent_usd"] + reserved + estimate.cost_usd > run["max_cost_usd"]:
            self._fail(run_id, node.id, "budget", f"estimated ${estimate.cost_usd:.2f} would exceed the run's cap of ${run['max_cost_usd']:.2f}")
            return

        if self._canceled(run_id):   # the last moment before anything starts or is reserved
            return
        attempt = t["attempt"] + 1
        attempt_dir = self.store.root / "work" / run_id / node.id / str(attempt)
        attempt_dir.mkdir(parents=True, exist_ok=False)
        # reserved from the moment it is chosen: a coordinator that dies before recording it running leaves the
        # reservation with the task that the next one adopts
        self._set(run_id, node.id, state="leased", attempt=attempt, cache_key=key, lease_owner=self.owner,
                  lease_expires=time.time() + self.lease_s, runner=placement.label,
                  cost_usd=estimate.cost_usd)
        request = {
            "protocol": 1, "run": run_id, "node": node.id, "uses": m.ref, "attempt": attempt, "partition": None,
            "accelerator": placement.runner.accelerator,
            "inputs": placement.backend.stage_inputs(attempt_dir, inputs), "config": node.config,
            "out_dir": str(placement.backend.output_dir(attempt_dir)), "secrets": m.secrets,
            "producer": {"stage": m.stage or m.id, "stage_version": m.version, "models": [x.model_dump(exclude={"size_gb", "gated"}) for x in m.models]},
        }
        (attempt_dir / "request.json").write_text(json.dumps(request, indent=1))
        try:
            handle = placement.backend.submit(m, placement.runner, attempt_dir, child_env(m.secrets, self.secrets))
        except Exception as e:   # could not even start (a missing tool): transient, the retry policy decides
            self._attempt_failed(run_id, graph, node.id, attempt, "transient", f"submit failed: {e}")
            return
        with self.store.tx():   # recorded running only if nothing (a cancel) changed the task while it was submitted
            now = self._tasks(run_id)[node.id]
            if now["state"] != "leased" or self._canceled(run_id):
                placement.backend.cancel(handle)   # canceled meanwhile: stop it now; it may have begun spending
                # charged what is left of its reservation (cancel() charges it, and uses it up, when it saw the process)
                self.store.db.execute("UPDATE runs SET spent_usd = spent_usd + ? WHERE id=?", (now["cost_usd"] or 0.0, run_id))
                self._set(run_id, node.id, handle=dumps(handle), cost_usd=0)
                return
            self._set(run_id, node.id, state="running", handle=dumps(handle), started_at=time.time(), cost_usd=estimate.cost_usd)
        self.store.event(run_id, node.id, "task.started", f"attempt {attempt} on {placement.backend.kind} ({placement.runner.accelerator})")

    def _fits(self, run_id: str, graph: Graph, m: Manifest, placement: Placement, tasks: dict[str, dict]) -> bool:
        """Within the CPU, memory and GPU of the machine it would run on, beside the nodes already running there."""
        machine = placement.backend.machine
        if machine is None:
            return True   # a machine of its own
        caps = placement.backend.probe_capabilities()
        running = [graph.nodes[i].manifest.resources for i, x in tasks.items() if x["state"] in ("running", "leased")
                   and x["runner"] and self.backends[x["runner"].split(":")[0]].machine == machine]
        if not running:
            return True   # a node bigger than the machine still runs, alone
        r = m.resources
        if r.gpu == "exclusive" and any(x.gpu != "none" for x in running):
            return False
        if r.gpu != "none" and any(x.gpu == "exclusive" for x in running):
            return False
        cpu = sum(x.cpu_cores for x in running) + r.cpu_cores
        mem = sum(x.memory_gb for x in running) + r.memory_gb
        return cpu <= caps.cpu_cores and mem <= caps.memory_gb * 0.8

    # --- finishing a task -----------------------------------------------------------------------------------

    def _check_running(self, run_id: str, graph: Graph, t: dict):
        backend = self.backends[t["runner"].split(":")[0]]
        handle = json.loads(t["handle"])
        if backend.poll(handle) == "running":
            return
        attempt_dir = self.store.root / "work" / run_id / t["node_id"] / str(t["attempt"])
        response = backend.collect(attempt_dir)
        backend.cleanup(attempt_dir)
        node = graph.nodes[t["node_id"]]
        if response is None:
            # it reported nothing, so it may have spent up to what was reserved for it: charge that
            self._attempt_failed(run_id, graph, node.id, t["attempt"], "transient",
                                 f"the node exited without a response (crashed or killed); see {attempt_dir / 'log.txt'}",
                                 usd=t["cost_usd"] or 0.0)
            return
        spent = float(response.get("usage", {}).get("usd", 0.0))   # what a failed attempt spent is still spent
        if not response["ok"]:
            err = response["error"]
            self._attempt_failed(run_id, graph, node.id, t["attempt"], err["code"], err["message"], usd=spent)
            return
        problems, outs = self._publish(node, backend.output_dir(attempt_dir), response)
        if problems:
            self._attempt_failed(run_id, graph, node.id, t["attempt"], problems[0][0], "; ".join(p for _, p in problems), usd=spent)
            return
        usage = {**response.get("usage", {}), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
        cost = float(usage.get("usd", 0.0))
        out_json = dumps({k: a.to_json() for k, a in outs.items()})
        with self.store.tx() as db:
            now = self._tasks(run_id)[node.id]
            if now["state"] != "running" or now["attempt"] != t["attempt"]:
                return   # canceled (and charged its reservation) while it finished: it stays canceled
            db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)", (t["cache_key"], node.manifest.ref, out_json, dumps(usage), time.time()))
            db.execute("""UPDATE tasks SET state='succeeded', outputs=?, usage=?, cost_usd=?, finished_at=?, lease_owner=NULL,
                          lease_expires=NULL, error_code=NULL, error=NULL WHERE run_id=? AND node_id=?""",
                       (out_json, dumps(usage), cost, time.time(), run_id, node.id))
            db.execute("UPDATE runs SET spent_usd = spent_usd + ? WHERE id=?", (cost, run_id))
        self.store.event(run_id, node.id, "task.succeeded", f"{usage.get('seconds', '?')} s")

    def _publish(self, node: CompiledNode, out_dir: Path, response: dict) -> tuple[list[tuple[str, str]], dict[str, Artifact]]:
        """Validate every declared output, check the models it reports, then publish. Nothing is published unless all
        of it is valid."""
        m, problems, ready = node.manifest, [], {}
        declared = {x.id: x.licence_mode for x in m.models}
        for used in response.get("models", []):
            if used["id"] not in declared:
                problems.append(("licence", f"the node used {used['id']}, which its manifest does not declare"))
            elif LICENCE_ORDER.index(used["licence_mode"]) > LICENCE_ORDER.index(declared[used["id"]]):
                problems.append(("licence", f"{used['id']} reported {used['licence_mode']}, stricter than the manifest's {declared[used['id']]}"))
        for name, type_id in m.outputs.items():
            entry = response["outputs"].get(name)
            if entry is None:
                problems.append(("schema", f"output {name} was not written"))
                continue
            path, why = output_path(out_dir, entry.get("path"))
            if path is None:
                problems.append(("schema", f"output {name}: {why}"))
                continue
            special = artifacts.special_files(path)
            if special:
                problems.append(("schema", f"output {name}: contains a symlink or special file: {', '.join(special[:5])}"))
                continue
            for p in artifacts.get(type_id).validate(path):
                problems.append(("schema", f"output {name} ({type_id}): {p}"))
            ready[name] = (path, type_id)
        if problems:
            return problems, {}
        return [], {name: self.store.publish(path, type_id) for name, (path, type_id) in ready.items()}

    def _attempt_failed(self, run_id: str, graph: Graph, node_id: str, attempt: int, code: str, message: str, usd: float = 0.0):
        """Retry a transient failure within the policy, else fail. `usd`, what the attempt spent, is charged in the same
        transaction: a coordinator that dies in between must not charge it again on resume."""
        retries = graph.spec.policy.retries
        with self.store.tx():
            now = self._tasks(run_id)[node_id]
            if now["state"] not in ("running", "leased") or now["attempt"] != attempt:
                return   # a stale report: the task was canceled (and charged) or has moved on since it was read
            first = now["first_attempt"]
            tried = attempt - first + 1
            if usd:
                self.store.db.execute("UPDATE runs SET spent_usd = spent_usd + ? WHERE id=?", (usd, run_id))
            if code == "transient" and tried <= retries:
                delay = self.backoff_s * 2 ** (tried - 1)
                self._set(run_id, node_id, state="pending", not_before=time.time() + delay, lease_owner=None, lease_expires=None,
                          handle=None, error_code=code, error=message)
                self.store.event(run_id, node_id, "task.retry", f"attempt {attempt} failed ({message}); retry in {delay:.0f} s")
            else:
                self._fail(run_id, node_id, code, message)

    def _fail(self, run_id: str, node_id: str, code: str, message: str):
        self._set(run_id, node_id, state="failed", error_code=code, error=message, finished_at=time.time(), lease_owner=None, lease_expires=None)
        self.store.event(run_id, node_id, "task.failed", f"{code}: {message}")

    def _set(self, run_id: str, node_id: str, **fields):
        cols = ", ".join(f"{k}=?" for k in fields)
        self.store.db.execute(f"UPDATE tasks SET {cols} WHERE run_id=? AND node_id=?", (*fields.values(), run_id, node_id))

    def _graph_input(self, graph: Graph, name: str) -> Artifact:
        spec = graph.spec.inputs[name]
        path = graph.input_path(name)
        if not path.exists():
            raise FileNotFoundError(f"graph input {name}: {path} does not exist")
        kind = artifacts.get(spec.type)
        return self.store.source(path, spec.type) if kind.shape == "source" else self.store.publish(path, spec.type)
