"""The coordinator against the spec's acceptance criteria, with test nodes (tests/nodes) run as real processes."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from perception import artifacts
from perception.backends.base import Estimate
from perception.backends.native import SelfBackend
from perception.ir import GraphError, compile_graph
from perception.manifests import Registry
from perception.scheduler import Coordinator, LeaseHeld
from perception.store import Store

NODES = Path(__file__).parent / "nodes"


def coordinator(root: Path, backend=None, lease_s: float = 30) -> Coordinator:
    return Coordinator(Store(root / "store"), Registry.load(NODES), [backend or SelfBackend()],
                       lease_s=lease_s, tick_s=0.02, backoff_s=0.01)


def graph_doc(root: Path, nodes: list[dict], **policy) -> dict:
    seed = root / "seed.json"
    seed.write_text('{"value": 0}')
    return {"schema_version": 1, "pipeline": "test", "inputs": {"seed": {"type": "json.v1", "uri": str(seed)}},
            "policy": {"retries": 2, **policy}, "nodes": nodes}


def work(id: str, a: str = "$inputs.seed", uses: str = "test.work@1.0.0", **config) -> dict:
    return {"id": id, "uses": uses, "inputs": {"a": a}, "config": config}


def start(coord: Coordinator, doc: dict, allow_uploads: bool = True) -> str:
    graph = compile_graph(doc, coord.registry)
    return coord.create_run(graph, doc, allow_uploads=allow_uploads)


def run(coord: Coordinator, doc: dict, allow_uploads: bool = True) -> tuple[str, str]:
    run_id = start(coord, doc, allow_uploads)
    return run_id, asyncio.run(coord.run(run_id))


def executions(record: Path, node: str | None = None) -> list[dict]:
    """One entry per node process that started, with its start and end times."""
    out = []
    for p in sorted(record.glob("*")):
        if p.suffix == ".end":
            continue
        name, attempt, pid = p.name.rsplit("-", 2)
        if node and name != node:
            continue
        end = p.with_name(p.name + ".end")
        out.append({"node": name, "attempt": int(attempt), **json.loads(p.read_text()),
                    "end": json.loads(end.read_text())["end"] if end.exists() else None})
    return out


def output(coord: Coordinator, run_id: str, node: str) -> dict:
    t = coord._tasks(run_id)[node]
    return json.loads(Path(json.loads(t["outputs"])["out"]["path"]).read_text())


@pytest.fixture
def rec(tmp_path):
    return str(tmp_path / "record")


# --- graph checks -----------------------------------------------------------------------------------------------

def test_graph_errors_are_all_reported(tmp_path, rec):
    reg = Registry.load(NODES)
    doc = graph_doc(tmp_path, [
        work("a", record_dir=rec, colour="red"),                        # unknown config
        {"id": "b", "uses": "test.work@1.0.0", "inputs": {}},          # input not connected
        work("c", a="$nodes.a.nothing", record_dir=rec),               # no such output
        work("d", a="$nodes.e.out", record_dir=rec), work("e", a="$nodes.d.out", record_dir=rec),   # cycle
        {"id": "f", "uses": "test.layer@1.0.0", "inputs": {"video": "$inputs.seed"}},              # wrong type
        {"id": "g", "uses": "test.work@9.0.0", "inputs": {}},          # unknown version
    ])
    with pytest.raises(GraphError) as e:
        compile_graph(doc, reg)
    text = "\n".join(e.value.problems)
    assert "has no config 'colour'" in text
    assert "input 'a' (json.v1) is not connected" in text
    assert "has no output 'nothing'" in text
    assert "cycle:" in text
    assert "wants video.v1 but $inputs.seed is json.v1" in text
    assert "no node test.work@9.0.0 (available: test.work@1.0.0)" in text


# --- running ----------------------------------------------------------------------------------------------------

def test_independent_nodes_run_in_parallel_and_dependent_ones_wait(tmp_path, rec):
    coord = coordinator(tmp_path)
    doc = graph_doc(tmp_path, [work("a", record_dir=rec), work("b", "$nodes.a.out", record_dir=rec, sleep=0.6),
                               work("c", "$nodes.a.out", record_dir=rec, sleep=0.6, add=10),
                               {"id": "d", "uses": "test.work@1.0.0", "inputs": {"a": "$nodes.b.out"}, "config": {"record_dir": rec}}])
    run_id, status = run(coord, doc)
    assert status == "succeeded"
    ex = {e["node"]: e for e in executions(Path(rec))}
    assert ex["b"]["start"] < ex["c"]["end"] and ex["c"]["start"] < ex["b"]["end"]     # b and c overlapped
    assert ex["b"]["start"] >= ex["a"]["end"] and ex["d"]["start"] >= ex["b"]["end"]   # each waited for its input
    assert output(coord, run_id, "d") == {"value": 3}


def test_gpu_exclusive_nodes_never_overlap(tmp_path, rec):
    coord = coordinator(tmp_path)
    doc = graph_doc(tmp_path, [work(f"g{i}", uses="test.gpu@1.0.0", record_dir=rec, sleep=0.3, add=i) for i in range(3)])
    _, status = run(coord, doc)
    assert status == "succeeded"
    spans = sorted((e["start"], e["end"]) for e in executions(Path(rec)))
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))


def test_unchanged_inputs_come_from_the_cache_and_a_change_reruns_only_descendants(tmp_path, rec):
    coord = coordinator(tmp_path)
    nodes = lambda c_add: [work("a", record_dir=rec), work("b", "$nodes.a.out", record_dir=rec),
                           work("c", "$nodes.a.out", record_dir=rec, add=c_add), work("d", "$nodes.c.out", record_dir=rec)]
    run(coord, graph_doc(tmp_path, nodes(1)))
    assert len(executions(Path(rec))) == 4

    run_id, status = run(coord, graph_doc(tmp_path, nodes(1)))   # same graph again: nothing runs
    assert status == "succeeded"
    assert len(executions(Path(rec))) == 4
    assert {t["state"] for t in coord._tasks(run_id).values()} == {"cached"}

    run_id, _ = run(coord, graph_doc(tmp_path, nodes(5)))        # c's config changed: c and d run, a and b do not
    states = {k: t["state"] for k, t in coord._tasks(run_id).items()}
    assert states == {"a": "cached", "b": "cached", "c": "succeeded", "d": "succeeded"}
    assert output(coord, run_id, "d") == {"value": 7}


def test_rerun_without_cache_forces_only_the_named_node(tmp_path, rec):
    """b runs again; its output is unchanged, so c comes from the cache instead of running a second time."""
    coord = coordinator(tmp_path)
    run_id, _ = run(coord, graph_doc(tmp_path, [work("a", record_dir=rec), work("b", "$nodes.a.out", record_dir=rec),
                                                work("c", "$nodes.b.out", record_dir=rec)]))
    coord.rerun(run_id, "b", use_cache=False)
    assert asyncio.run(coord.run(run_id)) == "succeeded"
    counts = {n: len(executions(Path(rec), n)) for n in "abc"}
    assert counts == {"a": 1, "b": 2, "c": 1}
    assert {k: t["state"] for k, t in coord._tasks(run_id).items()} == {"a": "succeeded", "b": "succeeded", "c": "cached"}


# --- failures ---------------------------------------------------------------------------------------------------

def test_a_transient_failure_is_retried_until_it_succeeds(tmp_path, rec):
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph_doc(tmp_path, [work("a", record_dir=rec, fail="transient", fail_until=2)]))
    assert status == "succeeded"
    assert [e["attempt"] for e in executions(Path(rec))] == [1, 2, 3]


def test_a_crash_without_a_response_is_retried(tmp_path, rec):
    coord = coordinator(tmp_path)
    _, status = run(coord, graph_doc(tmp_path, [work("a", record_dir=rec, fail="crash", fail_until=1)]))
    assert status == "succeeded"
    assert len(executions(Path(rec))) == 2


@pytest.mark.parametrize("kind, code", [("schema", "schema"), ("internal", "internal"), ("input", "input")])
def test_terminal_failures_are_not_retried_and_cancel_descendants(tmp_path, rec, kind, code):
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph_doc(tmp_path, [work("a", record_dir=rec, fail=kind, fail_until=9),
                                                     work("b", "$nodes.a.out", record_dir=rec)]))
    assert status == "failed"
    tasks = coord._tasks(run_id)
    assert (tasks["a"]["state"], tasks["a"]["error_code"]) == ("failed", code)
    assert tasks["b"]["state"] == "canceled"
    assert len(executions(Path(rec))) == 1
    published = [p.name for p in (coord.store.root / "objects").rglob("*") if p.is_file()]
    assert published == ["seed.json"]                                   # the graph input only: nothing a failed
                                                                        # attempt wrote was published


def test_running_out_of_memory_fails_once_and_names_the_setting_to_change(tmp_path, rec):
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph_doc(tmp_path, [work("a", record_dir=rec, fail="oom", fail_until=9)], retries=2))
    t = coord._tasks(run_id)["a"]
    assert status == "failed" and t["error_code"] == "resources"
    assert "batch_size" in t["error"] and "CUDA out of memory" in t["error"]
    assert len(executions(Path(rec))) == 1                                # the same input would run out again: no retry


def test_a_run_canceled_from_another_process_ends_canceled_not_failed(tmp_path, rec):
    coord = coordinator(tmp_path)
    run_id = start(coord, graph_doc(tmp_path, [work("a", record_dir=rec, sleep=30)]))

    async def go():
        running = asyncio.create_task(coord.run(run_id))
        deadline = time.time() + 20
        while coord._tasks(run_id)["a"]["state"] != "running":   # waiting for the node to start
            assert time.time() < deadline, "the node never started"
            await asyncio.sleep(0.02)
        coordinator(tmp_path).cancel(run_id)                     # another process (perception cancel) on the same store
        return await running

    assert asyncio.run(go()) == "canceled"
    assert coord.store.db.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()[0] == "canceled"
    kinds = [e["kind"] for e in coord.store.events(run_id)]
    assert "run.canceled" in kinds and "run.failed" not in kinds


def test_after_a_cancel_nothing_more_runs_even_a_node_that_looked_ready_or_retryable(tmp_path, rec):
    """The coordinator works from snapshots: a task it read before the cancel (a killed node's transient failure, a
    node it found ready) must not be retried or started after it."""
    coord = coordinator(tmp_path)
    run_id = start(coord, graph_doc(tmp_path, [work("a", record_dir=rec, sleep=30), work("b", "$nodes.a.out", record_dir=rec)]))
    coordinator(tmp_path).cancel(run_id)                          # canceled before this coordinator looked at all
    coord._set(run_id, "a", state="pending")                      # as a stale snapshot's retry would write it
    assert asyncio.run(coord.run(run_id)) == "canceled"
    assert executions(Path(rec)) == []
    assert {t["state"] for t in coord._tasks(run_id).values()} == {"canceled"}


def test_retries_stop_at_the_policy_limit(tmp_path, rec):
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph_doc(tmp_path, [work("a", record_dir=rec, fail="transient", fail_until=9)], retries=1))
    assert status == "failed"
    assert len(executions(Path(rec))) == 2


def test_a_billable_node_over_the_cost_cap_is_not_submitted(tmp_path, rec):
    class Billable(SelfBackend):
        def estimate(self, manifest, runner, inputs):
            return Estimate(5.0)
    coord = coordinator(tmp_path, Billable())
    run_id, status = run(coord, graph_doc(tmp_path, [work("a", record_dir=rec)], max_cost_usd=1.0))
    assert status == "failed"
    assert coord._tasks(run_id)["a"]["error_code"] == "budget"
    assert executions(Path(rec)) == []


# --- crash recovery ---------------------------------------------------------------------------------------------

COORDINATOR_SCRIPT = """
import asyncio, sys
from pathlib import Path
from perception.backends.native import SelfBackend
from perception.manifests import Registry
from perception.scheduler import Coordinator
from perception.store import Store
root, nodes, run_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
c = Coordinator(Store(root / "store"), Registry.load(nodes), [SelfBackend()], lease_s=1.0, tick_s=0.02)
asyncio.run(c.run(run_id))
"""


def test_killing_the_coordinator_mid_run_recovers_without_rerunning_or_corrupting(tmp_path, rec):
    coord = coordinator(tmp_path, lease_s=1.0)
    run_id = start(coord, graph_doc(tmp_path, [work("a", record_dir=rec, sleep=1.5), work("b", "$nodes.a.out", record_dir=rec)]))
    proc = subprocess.Popen([sys.executable, "-c", COORDINATOR_SCRIPT, str(tmp_path), str(NODES), run_id])
    deadline = time.time() + 20
    while not executions(Path(rec), "a") and time.time() < deadline:   # wait until node a's process has started
        time.sleep(0.05)
    assert executions(Path(rec), "a"), "node a never started"
    proc.send_signal(signal.SIGKILL)                                    # the coordinator dies; node a keeps running
    proc.wait()

    with pytest.raises(LeaseHeld):                                      # its lease is still live: no second coordinator
        asyncio.run(coordinator(tmp_path, lease_s=1.0).run(run_id))
    time.sleep(1.1)                                                      # the dead coordinator's lease runs out
    status = asyncio.run(coordinator(tmp_path, lease_s=1.0).run(run_id))
    assert status == "succeeded"
    assert len(executions(Path(rec), "a")) == 1                         # adopted, not started again
    assert output(coord, run_id, "b") == {"value": 2}
    for obj in (coord.store.root / "objects").rglob("*"):               # every published object matches its hash
        if obj.is_file():
            assert artifacts.sha256_file(obj) == obj.parent.name
    assert not list((coord.store.root / "tmp").iterdir())


def test_a_coordinator_that_dies_just_after_starting_a_node_leaves_it_to_be_adopted_not_run_twice(tmp_path, rec):
    """The gap between starting a node's process and recording it running: a coordinator killed there must not leave
    a task that the next coordinator starts again while the first process is still working (twice the spend for an
    API node). Seen on Linux CI, where processes start faster than on macOS."""
    class Crash(BaseException):
        pass
    coord = coordinator(tmp_path, lease_s=1.0)
    run_id = start(coord, graph_doc(tmp_path, [work("a", record_dir=rec, sleep=1.0)]))
    backend = coord.backends["self"]
    real_submit = backend.submit

    def submit_then_die(*a):
        real_submit(*a)
        raise Crash()                                    # the coordinator is gone before it records "running"
    backend.submit = submit_then_die
    with pytest.raises(Crash):
        asyncio.run(coord.run(run_id))
    time.sleep(1.1)                                      # its lease runs out
    assert asyncio.run(coordinator(tmp_path, lease_s=1.0).run(run_id)) == "succeeded"
    assert len(executions(Path(rec), "a")) == 1          # adopted, not started again


def test_a_live_node_is_recognised_however_long_its_request_path(tmp_path):
    """Recovery checks a pid's command line for the attempt's request path. procps (Linux) truncates `ps` columns to 80
    characters when not writing to a terminal, so a long path made a live node look dead and it was started again."""
    from perception.backends.native import _alive
    request = tmp_path / ("x" * 200) / "request.json"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", str(request)])
    try:
        assert _alive(proc.pid, str(request))
    finally:
        proc.kill()
        proc.wait()


def test_a_recorded_handle_is_trusted_only_for_its_own_attempt(tmp_path):
    """The node can write in its attempt folder: a handle naming another request or a strange pid steers nothing."""
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    own = str(attempt / "request.json")
    backend = SelfBackend()
    for forged in [{"pid": 1, "request": "/etc/other/request.json"}, {"pid": 1, "request": ""}, {"pid": -5, "request": own},
                   {"pid": "1", "request": own}, ["not", "a", "handle"], {"pid": 2 ** 70, "request": own}]:
        (attempt / "handle.json").write_text(json.dumps(forged))
        assert backend.recorded_handle(attempt) is None, forged
    (attempt / "handle.json").write_bytes(b"\xff\xfe not text")
    assert backend.recorded_handle(attempt) is None
    (attempt / "handle.json").write_text(json.dumps({"pid": 4242, "request": own}))
    assert backend.recorded_handle(attempt) == {"pid": 4242, "request": own}


def _crash_after_submit(tmp_path, doc):
    """Start a run whose coordinator dies just after starting its first billable node, before recording it running."""
    class Crash(BaseException):
        pass
    coord = coordinator(tmp_path, lease_s=1.0)
    run_id = start(coord, doc)
    backend = coord.backends["self"]
    real_submit = backend.submit

    def submit_then_die(m, *a):
        handle = real_submit(m, *a)
        if m.id == "test.refine":
            raise Crash()
        return handle
    backend.submit = submit_then_die
    with pytest.raises(Crash):
        asyncio.run(coord.run(run_id))
    return coord, run_id


def test_a_node_left_leased_by_a_dead_coordinator_keeps_its_reservation_and_cancel_stops_it(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    coord, run_id = _crash_after_submit(tmp_path, refine_graph(video, {"sleep": 30}, max_cost_usd=1))
    t = coord._tasks(run_id)["captions"]
    assert t["state"] == "leased" and t["cost_usd"] == pytest.approx(0.5)   # reserved from the moment it was chosen
    handle = coord.backends["self"].recorded_handle(coord.store.root / "work" / run_id / "captions" / str(t["attempt"]))
    coordinator(tmp_path).cancel(run_id)
    deadline = time.time() + 10
    while coord.backends["self"].poll(handle) == "running":                 # its process was found and stopped
        assert time.time() < deadline, "the process it had started is still running"
        time.sleep(0.05)
    spent = coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert spent == pytest.approx(0.5)


class TwoDevices(SelfBackend):
    """A machine with a CUDA GPU as well as a CPU, as far as placement can tell (the test node runs on the CPU)."""
    def probe_capabilities(self):
        import dataclasses
        return dataclasses.replace(super().probe_capabilities(), accelerators={"cpu", "cuda"}, vram_gb=24)


class AppleSilicon(SelfBackend):
    """A Mac: an Apple GPU sharing its memory with the CPU, as far as placement can tell."""
    def probe_capabilities(self):
        import dataclasses
        return dataclasses.replace(super().probe_capabilities(), accelerators={"cpu", "mps"})


def test_on_apple_silicon_a_node_that_runs_on_the_gpu_is_not_offered_the_cpu_too(tmp_path):
    coord = coordinator(tmp_path, AppleSilicon())
    assert [p.label for p in coord.placements(coord.registry.get("test.two@1.0.0"))[0]] == ["self:mps"]   # unified memory
    assert [p.label for p in coord.placements(coord.registry.get("test.work@1.0.0"))[0]] == ["self:cpu"]   # CPU only: still runs


def test_a_node_runs_where_it_is_placed_among_the_places_it_can_run_here(tmp_path, rec):
    coord = coordinator(tmp_path, TwoDevices())
    m = coord.registry.get("test.two@1.0.0")
    assert [p.label for p in coord.placements(m)[0]] == ["self:cuda", "self:cpu"]     # accelerated first
    node = lambda place=None: {**work("a", uses="test.two@1.0.0", record_dir=rec), **({"place": place} if place else {})}
    run_id, _ = run(coord, graph_doc(tmp_path, [node()]))
    assert coord._tasks(run_id)["a"]["runner"] == "self:cuda"
    run_id, _ = run(coord, graph_doc(tmp_path, [node("self:cpu")]))   # same config: another place is another result
    assert coord._tasks(run_id)["a"]["runner"] == "self:cpu"
    placement, why = coord.place(m, "auto", "self:mps")
    assert placement is None and why == "self:mps is not one of the places it can run (self:cuda, self:cpu)"


def test_an_estimate_counts_the_seconds_a_node_ran_not_time_no_coordinator_watched(tmp_path):
    from perception.estimates import estimate
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    coord = coordinator(tmp_path)
    run_id, _ = run(coord, refine_graph(video, {"sleep": 0.2}, max_cost_usd=5))
    # as if the coordinator had been down for an hour before it collected the node: its own seconds are what count
    coord.store.db.execute("UPDATE tasks SET started_at = started_at - 3600 WHERE run_id=? AND node_id='captions'", (run_id,))
    seconds, basis = estimate(coord.store, "test.refine@1.0.0", "self:cpu", 2.0)
    assert seconds is not None and seconds < 10, basis


def test_a_plan_estimates_each_node_from_its_measured_runs_here(tmp_path):
    from perception.plan import plan as make_plan
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    coord = coordinator(tmp_path)
    doc = refine_graph(video, {"sleep": 0.3}, max_cost_usd=5)
    fresh = {n.id: n for n in make_plan(coord, compile_graph(doc, coord.registry)).nodes}
    assert fresh["captions"].estimate_s is None
    assert fresh["captions"].estimate_basis == "no runs of test.refine@1.0.0 on self:cpu here yet"
    for _ in range(2):
        run(coord, refine_graph(video, {"sleep": 0.3, "usd": 0.01 + _ / 100}, max_cost_usd=5))   # a new config: runs again
    planned = {n.id: n for n in make_plan(coord, compile_graph(refine_graph(video, {"sleep": 0.3, "usd": 0.05}, max_cost_usd=5), coord.registry)).nodes}
    est = planned["captions"]
    assert est.estimate_basis == "median of 2 runs of test.refine@1.0.0 on self:cpu here"
    assert 0.3 <= est.estimate_s < 5                 # about a run's seconds: the same 2 s video each time
    assert est.estimates == {"self:cpu": {"estimate_s": est.estimate_s, "estimate_basis": est.estimate_basis, "cost_usd": 0.0,
                                          "uploads": ["frames"], "upload_to": "an API"}}   # test.refine declares frames, names no service


def test_a_plan_estimates_every_place_a_node_could_be_moved_to(tmp_path, rec):
    from perception.plan import plan as make_plan
    coord = coordinator(tmp_path, TwoDevices())
    node = {**work("a", uses="test.two@1.0.0", record_dir=rec), "place": "self:cpu"}
    planned = make_plan(coord, compile_graph(graph_doc(tmp_path, [node]), coord.registry)).nodes[0]
    assert planned.place == "self:cpu"
    assert planned.estimates == {p: {"estimate_s": None, "estimate_basis": f"no runs of test.two@1.0.0 on {p} here yet",
                                     "cost_usd": 0.0, "uploads": [], "upload_to": None}
                                 for p in ("self:cuda", "self:cpu")}   # the place not chosen too


# --- export -----------------------------------------------------------------------------------------------------

def test_a_run_exports_as_a_bundle_with_licences_carried_downstream(tmp_path):
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    doc = {"schema_version": 1, "pipeline": "test", "inputs": {"video": {"type": "video.v1", "uri": str(video)}},
           "nodes": [{"id": "shots", "uses": "test.layer@1.0.0", "inputs": {"video": "$inputs.video"}}]}
    run_id, status = run(coord, doc)
    assert status == "succeeded", coord._tasks(run_id)
    out = export.bundle(coord, run_id, tmp_path / "bundle")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["video"]["sha256"] == artifacts.sha256_file(video)
    assert manifest["layers"]["shots"]["licence_mode"] == "academic-only"
    assert manifest["layers"]["shots"]["sha256"] == artifacts.sha256_file(out / "layers" / "shots.json")
    validate = Path(__file__).resolve().parents[2] / "pipeline" / "bundle" / "validate.py"
    result = subprocess.run(["uv", "run", "--project", str(validate.parent), "python", str(validate), str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_an_output_with_a_symlink_fails_and_publishes_nothing(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph_doc(tmp_path, [{"id": "t", "uses": "test.tree@1.0.0", "inputs": {"a": "$inputs.seed"},
                                                      "config": {"link_to": str(secret)}}]))
    assert status == "failed"
    task = coord._tasks(run_id)["t"]
    assert task["error_code"] == "schema" and "symlink" in task["error"]
    assert not list((coord.store.root / "trees").rglob("*.jpg"))


def test_a_response_that_is_a_symlink_is_not_read(tmp_path):
    from perception.backends.native import SelfBackend
    secret = tmp_path / "secret.json"
    secret.write_text('{"ok": true, "outputs": {}}')
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "response.json").symlink_to(secret)
    assert SelfBackend().collect(attempt) is None


def _publish_case(tmp_path, out_dir, rel):
    """Ask the coordinator to publish output `out` of a test.work node from `out_dir`, as named by `rel`."""
    coord = coordinator(tmp_path)
    graph = compile_graph(graph_doc(tmp_path, [work("a", record_dir=str(tmp_path / "rec"))]), coord.registry)
    return coord._publish(graph.nodes["a"], out_dir, {"ok": True, "outputs": {"out": {"path": rel}}, "models": []})


def test_an_output_folder_replaced_by_a_symlink_is_not_published(tmp_path):
    host = tmp_path / "host"
    host.mkdir()
    (host / "out.json").write_text('{"value": "host secret"}')
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "out").symlink_to(host)                 # what a container can do to /work/out
    problems, outs = _publish_case(tmp_path, attempt / "out", "out.json")
    assert outs == {} and problems and "symlink" in problems[0][1]


def test_a_symlinked_folder_inside_the_output_path_is_not_followed(tmp_path):
    host = tmp_path / "host"
    host.mkdir()
    (host / "out.json").write_text('{"value": "host secret"}')
    out = tmp_path / "attempt" / "out"
    out.mkdir(parents=True)
    (out / "sub").symlink_to(host)
    problems, outs = _publish_case(tmp_path, out, "sub/out.json")
    assert outs == {} and problems and "symlink" in problems[0][1]


@pytest.mark.parametrize("rel", ["../out.json", "/etc/hosts", "./out.json", ""])
def test_output_paths_must_be_plain_relative_paths(tmp_path, rel):
    out = tmp_path / "attempt" / "out"
    out.mkdir(parents=True)
    problems, outs = _publish_case(tmp_path, out, rel)
    assert outs == {} and problems and "plain relative path" in problems[0][1]


def test_rerunning_a_node_with_the_same_inputs_gives_byte_identical_layers(tmp_path):
    """Run time and timings must not be part of a stored layer: if they were, every rerun of an upstream node would
    change its outputs' hashes and every downstream node would miss the cache (acceptance criterion 2)."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    doc = {"schema_version": 1, "pipeline": "test", "inputs": {"video": {"type": "video.v1", "uri": str(video)}},
           "nodes": [{"id": "shots", "uses": "test.layer@1.0.0", "inputs": {"video": "$inputs.video"}}]}
    run_id, _ = run(coord, doc)
    first = json.loads(coord._tasks(run_id)["shots"]["outputs"])["shots"]["hash"]
    time.sleep(1.1)                                     # a different second: a timestamp in the layer would differ
    coord.rerun(run_id, "shots", use_cache=False)
    asyncio.run(coord.run(run_id))
    task = coord._tasks(run_id)["shots"]
    assert task["attempt"] == 2 and task["state"] == "succeeded"
    assert json.loads(task["outputs"])["shots"]["hash"] == first


def test_export_puts_the_run_time_and_timings_back_into_the_bundle(tmp_path):
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    doc = {"schema_version": 1, "pipeline": "test", "inputs": {"video": {"type": "video.v1", "uri": str(video)}},
           "nodes": [{"id": "shots", "uses": "test.layer@1.0.0", "inputs": {"video": "$inputs.video"}}]}
    run_id, _ = run(coord, doc)
    out = export.bundle(coord, run_id, tmp_path / "bundle")
    producer = json.loads((out / "layers" / "shots.json").read_text())["producer"]
    assert producer["run_at"] and producer["timings"].get("seconds") is not None


def refine_graph(video: Path, config: dict | None = None, **policy) -> dict:
    return {"schema_version": 1, "pipeline": "test", "inputs": {"video": {"type": "video.v1", "uri": str(video)}},
            "policy": {"privacy": "local-preferred", "allow_upload": ["frames"], **policy},
            "nodes": [{"id": "shots", "uses": "test.layer@1.0.0", "inputs": {"video": "$inputs.video"}},
                      {"id": "captions", "uses": "test.refine@1.0.0", "inputs": {"shots": "$nodes.shots.shots"},
                       "config": config or {}}]}


def test_a_run_not_allowed_to_upload_never_starts_a_node_that_would_even_on_rerun(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    coord = coordinator(tmp_path)
    allowed, status = run(coord, refine_graph(video, max_cost_usd=1))
    assert status == "succeeded"                                     # captions is in the cache now
    run_id, status = run(coord, refine_graph(video, max_cost_usd=1), allow_uploads=False)
    assert status == "succeeded" and coord._tasks(run_id)["captions"]["state"] == "cached"   # nothing was sent
    coord.rerun(run_id, "captions", use_cache=False)                 # would send frames again
    assert asyncio.run(coord.run(run_id)) == "failed"
    t = coord._tasks(run_id)["captions"]
    assert t["error_code"] == "policy" and "may not (--allow-uploads)" in t["error"] and t["attempt"] == 0   # never started
    coord.allow_uploads(run_id)                                      # rerun --allow-uploads
    coord.rerun(run_id, "captions", use_cache=False)
    assert asyncio.run(coord.run(run_id)) == "succeeded"


def test_uploads_need_the_data_kind_allowed_by_name(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    reg = Registry.load(NODES)
    compile_graph(refine_graph(video), reg)                                       # frames allowed: valid
    with pytest.raises(GraphError, match="uploads frames, which policy.allow_upload does not permit"):
        compile_graph(refine_graph(video, allow_upload=[]), reg)
    with pytest.raises(GraphError, match="privacy policy is local-only"):
        compile_graph(refine_graph(video, privacy="local-only"), reg)


def test_a_billable_node_is_held_to_its_estimate_and_charged_what_it_spent(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, refine_graph(video, max_cost_usd=0.4))           # 1 shot x $0.5 estimate > $0.4
    assert status == "failed" and coord._tasks(run_id)["captions"]["error_code"] == "budget"
    run_id, status = run(coord, refine_graph(video, max_cost_usd=0.6))
    assert status == "succeeded"
    spent = coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert spent == pytest.approx(0.01)                                          # what the node reported, not the estimate


def test_the_estimate_follows_the_config_the_price_depends_on(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, refine_graph(video, {"model": "dear"}, max_cost_usd=0.6))   # 1 shot x $20 > $0.6
    assert status == "failed" and coord._tasks(run_id)["captions"]["error_code"] == "budget"


def test_a_cost_table_must_price_every_value_of_its_config_key():
    from perception.manifests import Manifest
    doc = {"id": "x", "version": "1.0.0", "outputs": {}, "runners": [{"kind": "self", "entry": "node.py"}],
           "config": {"model": {"enum": ["cheap", "dear"], "default": "cheap"}},
           "cost": {"usd": {"cheap": 0.5}, "by": "model", "per": "run"}}
    with pytest.raises(ValueError, match="no price for model 'dear'"):
        Manifest.model_validate(doc)


def test_a_cost_per_unit_must_name_the_input_it_counts():
    from perception.manifests import Manifest
    doc = {"id": "x", "version": "1.0.0", "inputs": {"shots": "bundle.shots@0.1"}, "outputs": {},
           "runners": [{"kind": "self", "entry": "node.py"}], "cost": {"usd": 0.1, "per": "shot"}}
    with pytest.raises(ValueError, match="cost.input None is not one of this node's inputs"):
        Manifest.model_validate(doc)
    Manifest.model_validate({**doc, "cost": {"usd": 0.1, "per": "shot", "input": "shots"}})
    with pytest.raises(ValueError, match="per minute cannot be counted in shots"):
        Manifest.model_validate({**doc, "cost": {"usd": 0.1, "per": "minute", "input": "shots"}})


def test_a_failed_attempt_is_charged_what_it_spent(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, refine_graph(video, {"fail_until": 1}, max_cost_usd=1, retries=1))
    assert status == "succeeded"
    spent = coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert spent == pytest.approx(0.02)                                          # the failed attempt and the retry


def test_a_billable_node_that_dies_without_a_response_is_charged_its_reservation(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    coord = coordinator(tmp_path)
    # 1 shot x $0.5: the first attempt dies, so the retry would need $1.0 of a $0.6 cap
    run_id, status = run(coord, refine_graph(video, {"fail_until": 1, "crash": True}, max_cost_usd=0.6, retries=1))
    assert status == "failed" and coord._tasks(run_id)["captions"]["error_code"] == "budget"
    spent = coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert spent == pytest.approx(0.5)


def test_a_billable_node_killed_by_a_cancel_is_charged_its_reservation(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    coord = coordinator(tmp_path)
    run_id = start(coord, refine_graph(video, {"sleep": 30}, max_cost_usd=1))   # 1 shot x $0.5 reserved

    async def go():
        running = asyncio.create_task(coord.run(run_id))
        deadline = time.time() + 20
        while coord._tasks(run_id)["captions"]["state"] != "running":
            assert time.time() < deadline, "the node never started"
            await asyncio.sleep(0.02)
        coordinator(tmp_path).cancel(run_id)
        return await running

    assert asyncio.run(go()) == "canceled"
    spent = coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert spent == pytest.approx(0.5)   # it may have spent what was reserved before it was killed


def test_a_killed_attempt_is_charged_once_even_if_a_stale_snapshot_reports_its_death(tmp_path):
    """cancel() charges the reservation of what it kills. The coordinator running the loop may have read that task as
    running just before, then see the dead process and report a crashed attempt: that report must change nothing."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    coord = coordinator(tmp_path)
    doc = refine_graph(video, {"sleep": 30}, max_cost_usd=1)
    run_id = start(coord, doc)
    graph = compile_graph(doc, coord.registry)

    async def go():
        running = asyncio.create_task(coord.run(run_id))
        deadline = time.time() + 20
        while coord._tasks(run_id)["captions"]["state"] != "running":
            assert time.time() < deadline, "the node never started"
            await asyncio.sleep(0.02)
        stale = coord._tasks(run_id)["captions"]          # the loop's snapshot, taken before the cancel
        coordinator(tmp_path).cancel(run_id)
        coord._attempt_failed(run_id, graph, "captions", stale["attempt"], "transient", "exited without a response", usd=stale["cost_usd"])
        return await running

    assert asyncio.run(go()) == "canceled"
    spent = coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert spent == pytest.approx(0.5)                     # once, not twice
    assert coord._tasks(run_id)["captions"]["state"] == "canceled"


def test_a_cancel_during_submission_stops_the_node_and_keeps_it_canceled(tmp_path):
    """The window between deciding to start a node and recording it running: a cancel there must win."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    coord = coordinator(tmp_path)
    run_id = start(coord, refine_graph(video, {"sleep": 30}, max_cost_usd=1))
    backend = coord.backends["self"]
    real_submit = backend.submit

    def submit_then_cancel(m, runner, attempt_dir, env):
        handle = real_submit(m, runner, attempt_dir, env)
        if m.id == "test.refine":
            coordinator(tmp_path).cancel(run_id)                 # lands after the submit, before "running" is recorded
        return handle
    backend.submit = submit_then_cancel
    assert asyncio.run(coord.run(run_id)) == "canceled"
    t = coord._tasks(run_id)["captions"]
    assert t["state"] == "canceled"
    deadline = time.time() + 10                                     # the process it had started was stopped (SIGTERM
    while backend.poll(json.loads(t["handle"])) == "running":       # lands asynchronously: a few ms on Linux CI)
        assert time.time() < deadline, "the process it had started is still running"
        time.sleep(0.05)
    spent = coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert spent == pytest.approx(0.5)                             # charged its reservation, once


def test_a_cancel_while_a_finished_node_is_collected_keeps_it_canceled_and_charged_once(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    coord = coordinator(tmp_path)
    doc = refine_graph(video, {"sleep": 1}, max_cost_usd=1)
    run_id = start(coord, doc)
    graph = compile_graph(doc, coord.registry)
    real_check = coord._check_running
    hold = {"on": True}
    # the loop does not collect captions for now (shots, upstream, is collected as usual)
    coord._check_running = lambda run_id, g, t: None if hold["on"] and t["node_id"] == "captions" else real_check(run_id, g, t)

    async def go():
        running = asyncio.create_task(coord.run(run_id))
        deadline = time.time() + 20
        while coord._tasks(run_id)["captions"]["state"] != "running":
            assert time.time() < deadline, "the node never started"
            await asyncio.sleep(0.02)
        stale = coord._tasks(run_id)["captions"]
        while coord.backends["self"].poll(json.loads(stale["handle"])) == "running":   # it finishes, with a response
            assert time.time() < deadline, "the node never finished"
            await asyncio.sleep(0.05)
        coordinator(tmp_path).cancel(run_id)                     # charges the reservation
        real_check(run_id, graph, stale)                         # the loop collects from its earlier snapshot
        hold["on"] = False
        return await running

    assert asyncio.run(go()) == "canceled"
    assert coord._tasks(run_id)["captions"]["state"] == "canceled"
    spent = coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert spent == pytest.approx(0.5)                             # the reservation; not also the reported $0.01


def test_a_node_that_refines_a_layer_supersedes_it_in_the_bundle(tmp_path):
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    coord = coordinator(tmp_path)
    run_id, _ = run(coord, refine_graph(video, max_cost_usd=1))
    out = export.bundle(coord, run_id, tmp_path / "bundle")
    shots = json.loads((out / "layers" / "shots.json").read_text())["shots"]
    assert shots[0]["caption"] == {"description": "shot 0"}
    # the strictest licence up the chain: the upstream shots node is academic-only, so the refined layer is too
    assert json.loads((out / "manifest.json").read_text())["layers"]["shots"]["licence_mode"] == "academic-only"


def faces_graph(video: Path, **config) -> dict:
    return {"schema_version": 1, "pipeline": "test", "inputs": {"video": {"type": "video.v1", "uri": str(video)}},
            "nodes": [{"id": "shots", "uses": "test.layer@1.0.0", "inputs": {"video": "$inputs.video"}},
                      {"id": "faces", "uses": "test.faces@1.0.0", "inputs": {"video": "$inputs.video"}, "config": config}]}


def test_a_layers_side_files_go_into_the_bundle_with_it(tmp_path):
    """face-track's detail, mesh and embeddings outputs become layers/faces_detail.json, layers/faces_mesh.json and
    layers/faces_embeddings.json, listed under the faces layer's side_files, which is where the annotator looks for the
    files the layer's detail.file, mesh.file and embeddings.file name (and the only bundle files it reads by path)."""
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, faces_graph(video))
    assert status == "succeeded", coord._tasks(run_id)
    out = export.bundle(coord, run_id, tmp_path / "bundle")
    manifest = json.loads((out / "manifest.json").read_text())
    outputs = json.loads(coord._tasks(run_id)["faces"]["outputs"])
    detail, mesh = out / "layers" / "faces_detail.json", out / "layers" / "faces_mesh.json"
    embeddings = out / "layers" / "faces_embeddings.json"
    assert detail.read_bytes() == Path(outputs["detail"]["path"]).read_bytes()
    assert mesh.read_bytes() == Path(outputs["mesh"]["path"]).read_bytes()
    assert embeddings.read_bytes() == Path(outputs["embeddings"]["path"]).read_bytes()
    assert manifest["layers"]["faces"]["side_files"] == [
        {"file": "layers/faces_detail.json", "sha256": artifacts.sha256_file(detail), "bytes": detail.stat().st_size},
        {"file": "layers/faces_embeddings.json", "sha256": artifacts.sha256_file(embeddings),
         "bytes": embeddings.stat().st_size},
        {"file": "layers/faces_mesh.json", "sha256": artifacts.sha256_file(mesh), "bytes": mesh.stat().st_size},
        *({"file": f"layers/faces_{role}.parquet", "sha256": artifacts.sha256_file(NODES / "faces" / f"faces_{role}.parquet"),
           "bytes": (NODES / "faces" / f"faces_{role}.parquet").stat().st_size} for role in ("pyfeat", "untracked"))]
    layer = json.loads((out / "layers" / "faces.json").read_text())
    assert layer["mesh"] == {"file": "layers/faces_mesh.json"} and layer["detail"]["file"] == "layers/faces_detail.json"
    assert layer["embeddings"] == {"file": "layers/faces_embeddings.json"}
    assert "side_files" not in manifest["layers"]["shots"]
    validate = Path(__file__).resolve().parents[2] / "pipeline" / "bundle" / "validate.py"
    result = subprocess.run(["uv", "run", "--project", str(validate.parent), "python", str(validate), str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("pointer", ["detail", "mesh", "embeddings"])
def test_a_layer_pointing_at_a_side_file_its_node_did_not_write_is_not_exported(tmp_path, pointer):
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, faces_graph(video, **{f"{pointer}_file": "layers/faces_other.json"}))
    assert status == "succeeded", coord._tasks(run_id)
    with pytest.raises(SystemExit, match=rf"points at layers/faces_other.json \({pointer}.file\)"):
        export.bundle(coord, run_id, tmp_path / "bundle")


def test_the_faces_detail_type_checks_its_shape(tmp_path):
    check = artifacts.get("bundle.faces_detail@0.1").validate
    good = {"au_names": ["AU01"], "emotion_names": ["Neutral"],
            "tracks": {"face:3": [{"t": 1.0, "lmk": [0.0] * 136, "au": [0.1], "emo": [1.0], "va": [0, 0], "head": [0, 0, 0], "gaze": [0, 0]}]}}
    path = tmp_path / "faces_detail.json"
    path.write_text(json.dumps(good))
    assert check(path) == []
    path.write_text(json.dumps(good | {"tracks": {"3": good["tracks"]["face:3"]}}))       # not a face track id
    assert check(path)
    path.write_text(json.dumps(good | {"tracks": {"face:3": [good["tracks"]["face:3"][0] | {"lmk": [0.0] * 68}]}}))   # x only
    assert check(path)
    assert artifacts.get("bundle.faces_detail@0.1").side_of == "faces" and artifacts.get("bundle.faces_detail@0.1").layer is None


MESH_SAMPLE = {"t": 1.0, "xy": [12] * 956}


@pytest.mark.parametrize("doc, ok", [
    ({"points": 478, "layout": "all x, then all y", "tracks": {"face:3": [MESH_SAMPLE]}}, True),
    ({"points": 478, "tracks": {}}, True),                                                    # no tracks; layout optional
    ({"points": 478, "tracks": {"3": [MESH_SAMPLE]}}, False),                                 # not a face track id
    ({"points": 478, "tracks": {"face:3": [MESH_SAMPLE | {"xy": [12] * 478}]}}, False),       # x only
    ({"points": 478, "tracks": {"face:3": [MESH_SAMPLE | {"xy": [12.5] * 956}]}}, False),     # not whole pixels
    ({"points": 468, "tracks": {"face:3": [MESH_SAMPLE]}}, False),                            # another mesh
    ({"points": 478, "tracks": {"face:3": [{"t": 1.0}]}}, False),                             # no xy
    ({"tracks": {"face:3": [MESH_SAMPLE]}}, False),                                           # no points
])
def test_the_faces_mesh_type_checks_its_shape(tmp_path, doc, ok):
    kind = artifacts.get("bundle.faces_mesh@0.1")
    path = tmp_path / "faces_mesh.json"
    path.write_text(json.dumps(doc))
    assert (kind.validate(path) == []) is ok, kind.validate(path)
    assert kind.side_of == "faces" and kind.layer is None and kind.bundle_file == "layers/faces_mesh.json"
    assert kind.kinds == ("faces",)


EMB_SAMPLE = {"t": 1.0, "v": [-0.5] * 512}
EMB = {"model": "ArcFace", "dims": 512, "normalised": False}


@pytest.mark.parametrize("doc, ok", [
    (EMB | {"tracks": {"face:3": [EMB_SAMPLE, EMB_SAMPLE | {"t": 1.125}]}}, True),
    (EMB | {"tracks": {}}, True),                                                          # no tracks
    (EMB | {"normalised": True, "tracks": {"face:3": [EMB_SAMPLE]}}, True),
    (EMB | {"tracks": {"3": [EMB_SAMPLE]}}, False),                                       # not a face track id
    (EMB | {"tracks": {"face:3": [EMB_SAMPLE | {"v": [0.1] * 511}]}}, False),            # a value short
    (EMB | {"tracks": {"face:3": [EMB_SAMPLE | {"v": [0.1] * 513}]}}, False),            # a value over
    (EMB | {"tracks": {"face:3": [EMB_SAMPLE | {"v": ["0.1"] * 512}]}}, False),        # not numbers
    (EMB | {"dims": 128, "tracks": {"face:3": [EMB_SAMPLE | {"v": [0.1] * 128}]}}, False),   # another model
    (EMB | {"tracks": {"face:3": [{"t": 1.0}]}}, False),                                # no v
    ({"model": "ArcFace", "dims": 512, "tracks": {}}, False),                          # normalisation unstated
    ({"dims": 512, "normalised": False, "tracks": {}}, False),                           # model unnamed
])
def test_the_faces_embeddings_type_checks_its_shape(tmp_path, doc, ok):
    kind = artifacts.get("bundle.faces_embeddings@0.1")
    path = tmp_path / "faces_embeddings.json"
    path.write_text(json.dumps(doc))
    assert (kind.validate(path) == []) is ok, kind.validate(path)
    assert kind.side_of == "faces" and kind.layer is None and kind.bundle_file == "layers/faces_embeddings.json"


def test_the_faces_embeddings_are_identity_data_and_need_consent_for_embeddings_to_leave_the_machine():
    """A step that reads the embeddings on another machine sends them there: the policy's allow_upload and the run's
    consent must name embeddings, not only faces."""
    from perception.backends.base import Backend
    kind = artifacts.get("bundle.faces_embeddings@0.1")
    assert "embeddings" in kind.kinds and set(kind.kinds) == {"faces", "embeddings"}
    remote = type("Remote", (), {"remote": True})()
    reader = type("Manifest", (), {"inputs": {"embeddings": "bundle.faces_embeddings@0.1"}})()
    assert Backend.sends(remote, reader) == ["embeddings", "faces"]


TURN = {"start": 0.03096875, "end": 1.0265937500000002, "voice": "voice:SPEAKER_00"}
DIAR = {"model": "pyannote/speaker-diarization-community-1", "voices": ["voice:SPEAKER_00", "voice:SPEAKER_01"]}


@pytest.mark.parametrize("doc, ok", [
    (DIAR | {"turns": [TURN, TURN | {"voice": "voice:SPEAKER_01"}], "exclusive_turns": [TURN]}, True),
    ({"model": "p", "voices": [], "turns": [], "exclusive_turns": []}, True),                # no speech
    (DIAR | {"turns": [TURN | {"voice": "SPEAKER_00"}], "exclusive_turns": []}, False),     # not a voice id
    (DIAR | {"turns": [TURN | {"start": -0.1}], "exclusive_turns": []}, False),            # before the video
    (DIAR | {"turns": [{"start": 0.0, "end": 1.0}], "exclusive_turns": []}, False),        # no voice
    (DIAR | {"voices": ["voice:SPEAKER_00"] * 2, "turns": [], "exclusive_turns": []}, False),   # a voice twice
    (DIAR | {"turns": [TURN]}, False),                                                     # no exclusive turns
])
def test_the_transcript_diarization_type_checks_its_shape(tmp_path, doc, ok):
    kind = artifacts.get("bundle.transcript_diarization@0.1")
    path = tmp_path / "transcript_diarization.json"
    path.write_text(json.dumps(doc))
    assert (kind.validate(path) == []) is ok, kind.validate(path)
    assert kind.side_of == "transcript" and kind.layer is None and kind.bundle_file == "layers/transcript_diarization.json"
    assert kind.kinds == ("transcript",)


VOICE_EMB = {"model": "WeSpeaker ResNet34", "dims": 256, "normalised": False}


@pytest.mark.parametrize("doc, ok", [
    (VOICE_EMB | {"voices": {"voice:SPEAKER_00": [0.1] * 256, "voice:SPEAKER_01": None}}, True),   # null: no centroid
    (VOICE_EMB | {"voices": {}}, True),
    (VOICE_EMB | {"voices": {"SPEAKER_00": [0.1] * 256}}, False),                         # not a voice id
    (VOICE_EMB | {"voices": {"voice:SPEAKER_00": [0.1] * 255}}, False),                   # a value short
    (VOICE_EMB | {"dims": 192, "voices": {"voice:SPEAKER_00": [0.1] * 192}}, False),      # another model
    ({"model": "m", "dims": 256, "voices": {}}, False),                                   # normalisation unstated
])
def test_the_transcript_voice_embeddings_type_checks_its_shape(tmp_path, doc, ok):
    kind = artifacts.get("bundle.transcript_voice_embeddings@0.1")
    path = tmp_path / "transcript_voice_embeddings.json"
    path.write_text(json.dumps(doc))
    assert (kind.validate(path) == []) is ok, kind.validate(path)
    assert kind.side_of == "transcript" and kind.layer is None
    assert kind.bundle_file == "layers/transcript_voice_embeddings.json"


def test_voice_embeddings_are_identity_data_but_the_turns_are_not():
    """A step reading the voice embeddings on another machine needs consent to embeddings; one reading only the turns
    does not."""
    from perception.backends.base import Backend
    remote = type("Remote", (), {"remote": True})()
    embeddings = type("Manifest", (), {"inputs": {"e": "bundle.transcript_voice_embeddings@0.1"}})()
    turns = type("Manifest", (), {"inputs": {"d": "bundle.transcript_diarization@0.1"}})()
    assert Backend.sends(remote, embeddings) == ["embeddings"]
    assert Backend.sends(remote, turns) == ["transcript"]


FULL_SAMPLE = {"t": 1.0, "lmk": [0.0] * 136, "au": [0.1], "emo": [1.0], "va": [0, 0], "head": [0, 0, 0], "gaze": [0, 0]}


@pytest.mark.parametrize("sample", [{"t": 1.0},                                                 # no fields but t
                                    FULL_SAMPLE | {"au": [0.1, 0.2]}, FULL_SAMPLE | {"emo": []}])   # not one per name
def test_a_faces_detail_sample_must_hold_every_field_one_value_per_name(tmp_path, sample):
    """The annotator indexes au by au_names and emo by emotion_names; gestures reads lmk."""
    path = tmp_path / "faces_detail.json"
    path.write_text(json.dumps({"au_names": ["AU01"], "emotion_names": ["Neutral"], "tracks": {"face:3": [sample]}}))
    assert artifacts.get("bundle.faces_detail@0.1").validate(path)


def bundle_validation(tmp_path, damage, file="layers/faces_detail.json") -> subprocess.CompletedProcess:
    """validate.py on an exported faces bundle after `damage(path of file)`, with the manifest's checksums updated
    (export records whatever bytes it copied)."""
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    run_id, _ = run(coord, faces_graph(video))
    out = export.bundle(coord, run_id, tmp_path / "bundle")
    damage(out / file)
    manifest = json.loads((out / "manifest.json").read_text())
    for entry in manifest["layers"].values():
        for f in [entry, *entry.get("side_files", [])]:
            f.update(sha256=artifacts.sha256_file(out / f["file"]), bytes=(out / f["file"]).stat().st_size)
    (out / "manifest.json").write_text(json.dumps(manifest))
    validate = Path(__file__).resolve().parents[2] / "pipeline" / "bundle" / "validate.py"
    return subprocess.run(["uv", "run", "--project", str(validate.parent), "python", str(validate), str(out)],
                          capture_output=True, text=True)


def _with(change):
    def damage(path):
        doc = json.loads(path.read_text())
        change(doc)
        path.write_text(json.dumps(doc))
    return damage


DETAIL, MESH, EMBEDDINGS = "layers/faces_detail.json", "layers/faces_mesh.json", "layers/faces_embeddings.json"


@pytest.mark.parametrize("file, damage, problem", [
    (DETAIL, lambda p: p.write_text("{not json"), "not JSON"),
    (DETAIL, _with(lambda d: d["tracks"].update({"face:0": ["not a sample", 3]})), "face:0"),
    (DETAIL, _with(lambda d: d["tracks"]["face:0"][0].update(au=[0.1, 0.2])), "au has 2 values for 1 au_names"),
    (DETAIL, _with(lambda d: d["tracks"].pop("face:0")), "does not match the layer's track samples"),
    (MESH, lambda p: p.write_text("{not json"), "faces_mesh.json: not JSON"),
    (MESH, _with(lambda d: d["tracks"]["face:0"][0].update(xy=[1] * 955)), "faces_mesh.json: $.tracks"),
    (MESH, _with(lambda d: d["tracks"].pop("face:0")), "faces_mesh.json does not match the layer's track samples"),
    (MESH, _with(lambda d: d["tracks"]["face:0"][1].update(t=0.25)), "faces_mesh.json does not match the layer's track samples"),
    ("layers/faces.json", _with(lambda d: d.update(mesh={"file": "layers/faces_other.json"})),
     "mesh.file layers/faces_other.json is not one of the layer's side files"),
    (EMBEDDINGS, lambda p: p.write_text("{not json"), "faces_embeddings.json: not JSON"),
    (EMBEDDINGS, _with(lambda d: d["tracks"]["face:0"][0].update(v=[0.25] * 511)), "faces_embeddings.json: $.tracks"),
    (EMBEDDINGS, _with(lambda d: d.update(dims=511)), "faces_embeddings.json: $.dims"),
    (EMBEDDINGS, _with(lambda d: d["tracks"].pop("face:0")), "faces_embeddings.json does not match the layer's track samples"),
    (EMBEDDINGS, _with(lambda d: d["tracks"]["face:0"][1].update(t=0.25)),
     "faces_embeddings.json does not match the layer's track samples"),
    (EMBEDDINGS, _with(lambda d: d["tracks"]["face:0"].pop()), "faces_embeddings.json does not match the layer's track samples"),
    ("layers/faces.json", _with(lambda d: d.update(embeddings={"file": "layers/faces_other.json"})),
     "embeddings.file layers/faces_other.json is not one of the layer's side files"),
])
def test_validate_reports_a_bad_side_file_as_a_problem(tmp_path, file, damage, problem):
    result = bundle_validation(tmp_path, damage, file)
    assert result.returncode == 1 and "Traceback" not in result.stderr, result.stdout + result.stderr
    assert "INVALID" in result.stdout and problem in result.stdout, result.stdout


# --- Parquet side files (the gaze layer's heatmaps) -------------------------------------------------------------

HEATMAPS_FIXTURE = NODES / "gaze" / "gaze_heatmaps.parquet"   # written by vision.gaze's writer (make_fixture.py)
VALIDATE = Path(__file__).resolve().parents[2] / "pipeline" / "bundle" / "validate.py"


def gaze_graph(video: Path, uses: str = "test.gaze@1.0.0", **config) -> dict:
    doc = faces_graph(video)
    doc["nodes"].append({"id": "gaze", "uses": uses, "inputs": {"video": "$inputs.video"}, "config": config})
    return doc


def test_every_side_file_is_described_by_the_bundle_schema():
    """A JSON side file validates against #/$defs/side/<name>, and a Parquet one's columns are #/$defs/parquet/<name>
    (what validate.py checks with pyarrow): a step that adds a side file adds both, and nothing there is unregistered."""
    defs = json.loads(artifacts.BUNDLE_SCHEMA.read_text())["$defs"]
    for fmt, where in (("json", "side"), ("parquet", "parquet")):
        assert set(defs[where]) == {n for n, s in artifacts.SIDE_FILES.items() if s.format == fmt}
    for name, side in artifacts.SIDE_FILES.items():
        kind = artifacts.get(f"bundle.{name}@0.1")
        assert (kind.side_of, kind.bundle_file, kind.kinds) == (side.layer, f"layers/{name}.{side.format}", side.kinds)
        assert side.layer in artifacts.BUNDLE_LAYERS and kind.layer is None


def test_the_parquet_side_type_checks_the_format_not_the_columns(tmp_path):
    """The orchestrator has no pyarrow: a Parquet side file is checked by its PAR1 magic at both ends and a footer length
    that fits. Its columns are the writing node's to test (nodes/vision/gaze/test_heatmaps.py) and validate.py's."""
    check = artifacts.get("bundle.gaze_heatmaps@0.1").validate
    good = HEATMAPS_FIXTURE.read_bytes()
    path = tmp_path / "gaze_heatmaps.parquet"
    path.write_bytes(good)
    assert check(path) == []
    footer = int.from_bytes(good[-8:-4], "little")
    for bad, problem in [(b"", "0 bytes"), (b"PAR1PAR1", "8 bytes"), (b"{}" + good[2:], "starts with"),
                         (good[:-4] + b"PARE", "ends with"), (good[:-1], "ends with"),
                         (good[:-8] + (len(good)).to_bytes(4, "little") + b"PAR1", "does not fit"),
                         (good[:-8] + (0).to_bytes(4, "little") + b"PAR1", "does not fit"),
                         (good[:4] + good[-(footer + 7):], "does not fit"), (b'{"tracks": {}}', "not Parquet")]:
        path.write_bytes(bad)
        assert any(problem in p for p in check(path)), (bad[:16], check(path))


def test_the_gaze_heatmaps_count_as_frames_when_a_step_reads_them_elsewhere():
    """Gaze-LLE's heatmaps are computed from the whole frame: together a coarse picture of where people and things are
    in it. A step that reads them on another machine needs the policy to allow frames (not faces: no face image or
    geometry is in them)."""
    from perception.backends.base import Backend
    assert artifacts.get("bundle.gaze_heatmaps@0.1").kinds == ("frames",)
    remote = type("Remote", (), {"remote": True})()
    reader = type("Manifest", (), {"inputs": {"heatmaps": "bundle.gaze_heatmaps@0.1"}})()
    assert Backend.sends(remote, reader) == ["frames"]


def test_a_layers_parquet_side_file_goes_into_the_bundle_byte_for_byte(tmp_path):
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, gaze_graph(video))
    assert status == "succeeded", coord._tasks(run_id)
    out = export.bundle(coord, run_id, tmp_path / "bundle")
    side = out / "layers" / "gaze_heatmaps.parquet"
    assert side.read_bytes() == HEATMAPS_FIXTURE.read_bytes()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["layers"]["gaze"]["side_files"] == [
        {"file": "layers/gaze_heatmaps.parquet", "sha256": artifacts.sha256_file(side), "bytes": side.stat().st_size}]
    layer = json.loads((out / "layers" / "gaze.json").read_text())
    assert layer["raw"]["heatmaps"] | {"description": None} == {"file": "layers/gaze_heatmaps.parquet", "format": "parquet",
                                                                 "description": None}
    result = subprocess.run(["uv", "run", "--project", str(VALIDATE.parent), "python", str(VALIDATE), str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "note:" not in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("config, problem", [
    ({"heatmaps_bytes": "not parquet at all"}, "output heatmaps (bundle.gaze_heatmaps@0.1): not Parquet"),
    ({"heatmaps_format": "json"}, "raw.heatmaps: layers/gaze_heatmaps.parquet is not named as a json file"),
    ({"heatmaps_file": "layers/gaze_heatmaps.json", "heatmaps_format": "json"},
     "raw.heatmaps: gaze_heatmaps is a parquet side file, not json"),
    ({"heatmaps_file": "layers/gaze_other.parquet"},
     "raw.heatmaps: layers/gaze_other.parquet is not a side file of the gaze layer"),
    ({"heatmaps_file": "layers/faces_mesh.json", "heatmaps_format": "json"},
     "raw.heatmaps: layers/faces_mesh.json is not a side file of the gaze layer"),
    ({"heatmaps_file": "../gaze_heatmaps.parquet"}, "$.raw.heatmaps.file"),
    ({"heatmaps_format": "arrow"}, "$.raw.heatmaps.format"),
])
def test_a_bad_parquet_side_file_or_raw_pointer_is_not_published(tmp_path, config, problem):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, gaze_graph(video, **config))
    task = coord._tasks(run_id)["gaze"]
    assert status == "failed" and task["error_code"] == "schema", task
    assert problem in task["error"], task["error"]


def test_a_raw_pointer_at_a_side_file_its_node_did_not_write_is_not_exported(tmp_path):
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, gaze_graph(video, uses="test.gaze-no-side@1.0.0"))
    assert status == "succeeded", coord._tasks(run_id)
    with pytest.raises(SystemExit, match=r"points at layers/gaze_heatmaps.parquet \(raw.heatmaps.file\)"):
        export.bundle(coord, run_id, tmp_path / "bundle")


def refined_gaze_graph(video: Path, uses: str = "test.gaze@1.0.0", **config) -> dict:
    doc = gaze_graph(video, uses)
    doc["nodes"].append({"id": "refine", "uses": "test.gaze-refine@1.0.0", "inputs": {"gaze": "$nodes.gaze.gaze"},
                         "config": config})
    return doc


def test_a_refined_layer_keeps_the_side_files_its_upstream_writer_wrote(tmp_path):
    """A node that refines a layer and keeps its raw pointers (audio.diarize writes asr's transcript again, with its
    raw.tokens) is exported with the side files those pointers name, from the upstream node of that layer that wrote
    them: the refined layer supersedes the upstream one, its side files go with it."""
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, refined_gaze_graph(video))
    assert status == "succeeded", coord._tasks(run_id)
    out = export.bundle(coord, run_id, tmp_path / "bundle")
    assert json.loads((out / "layers" / "gaze.json").read_text())["producer"]["stage"] == "test.gaze-refine"
    side = out / "layers" / "gaze_heatmaps.parquet"
    assert side.read_bytes() == HEATMAPS_FIXTURE.read_bytes()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["layers"]["gaze"]["side_files"] == [
        {"file": "layers/gaze_heatmaps.parquet", "sha256": artifacts.sha256_file(side), "bytes": side.stat().st_size}]
    result = subprocess.run(["uv", "run", "--project", str(VALIDATE.parent), "python", str(VALIDATE), str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "note:" not in result.stdout, result.stdout + result.stderr


def test_a_refined_layers_pointer_at_a_side_file_no_writer_of_it_wrote_is_not_exported(tmp_path):
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, refined_gaze_graph(video, uses="test.gaze-no-side@1.0.0"))
    assert status == "succeeded", coord._tasks(run_id)
    with pytest.raises(SystemExit, match=r"the gaze layer from refine points at layers/gaze_heatmaps.parquet "
                                         r"\(raw.heatmaps.file\), which neither refine nor the gaze writers upstream"):
        export.bundle(coord, run_id, tmp_path / "bundle")


def test_a_refined_layers_side_file_from_two_unrelated_upstream_writers_is_not_exported(tmp_path):
    """Two upstream writers of the layer, neither downstream of the other, both wrote the file the refined layer points
    at: which one it means is ambiguous, so export stops rather than take the first."""
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    doc = gaze_graph(video)
    doc["nodes"] += [{"id": "gaze2", "uses": "test.gaze@1.0.0", "inputs": {"video": "$inputs.video"}},
                     {"id": "merge", "uses": "test.gaze-merge@1.0.0",
                      "inputs": {"gaze": "$nodes.gaze.gaze", "other": "$nodes.gaze2.gaze"}}]
    run_id, status = run(coord, doc)
    assert status == "succeeded", coord._tasks(run_id)
    with pytest.raises(SystemExit, match=r"the gaze layer from merge points at layers/gaze_heatmaps.parquet "
                                         r"\(raw.heatmaps.file\), which unrelated upstream writers of it wrote: gaze, gaze2"):
        export.bundle(coord, run_id, tmp_path / "bundle")


def gaze_bundle(tmp_path, damage, file, graph=None) -> Path:
    """An exported gaze bundle (or `graph(video)`'s) after `damage(path of file)`, with the manifest's checksums
    updated."""
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, (graph or gaze_graph)(video))
    assert status == "succeeded", coord._tasks(run_id)
    out = export.bundle(coord, run_id, tmp_path / "bundle")
    damage(out / file)
    manifest = json.loads((out / "manifest.json").read_text())
    for entry in manifest["layers"].values():
        for f in [entry, *entry.get("side_files", [])]:
            f.update(sha256=artifacts.sha256_file(out / f["file"]), bytes=(out / f["file"]).stat().st_size)
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


HEATMAPS, GAZE = "layers/gaze_heatmaps.parquet", "layers/gaze.json"


@pytest.mark.parametrize("file, damage, problem", [
    (HEATMAPS, lambda p: p.write_bytes(p.read_bytes()[:-1]), "gaze_heatmaps.parquet: not Parquet"),
    (HEATMAPS, lambda p: p.write_bytes(p.read_bytes()[:4] + bytes(64) + p.read_bytes()[68:]), "not readable as Parquet"),
    (GAZE, _with(lambda d: d["tracks"][0]["samples"][1].update(t=0.25)), "does not match the layer's samples"),
    (GAZE, _with(lambda d: d["tracks"][0]["samples"].pop()), "does not match the layer's samples: 2 rows for 1 samples"),
    (GAZE, _with(lambda d: d["tracks"][0].update(face="face:1")), "does not match the layer's samples"),
    (GAZE, _with(lambda d: d["raw"]["heatmaps"].update(file="layers/gaze_other.parquet")),
     "raw.heatmaps.file layers/gaze_other.parquet is not one of the layer's side files"),
    (GAZE, _with(lambda d: d["raw"]["heatmaps"].update(format="json")),
     "raw.heatmaps.file layers/gaze_heatmaps.parquet is not a json file"),
    (GAZE, _with(lambda d: d["raw"]["heatmaps"].pop("description")), "gaze: $.raw.heatmaps"),
    # a sample with no time is the schema's error to report, not a crash sorting the samples for the rows check
    (GAZE, _with(lambda d: d["tracks"][0]["samples"][1].pop("t")), "gaze: $.tracks[0].samples[1]: 't' is a required"),
])
def test_validate_reports_a_bad_parquet_side_file_as_a_problem(tmp_path, file, damage, problem):
    out = gaze_bundle(tmp_path, damage, file)
    result = subprocess.run(["uv", "run", "--project", str(VALIDATE.parent), "python", str(VALIDATE), str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 1 and "Traceback" not in result.stderr, result.stdout + result.stderr
    assert "INVALID" in result.stdout and problem in result.stdout, result.stdout


def bodies_bundle(tmp_path, damage=lambda p: None, file="layers/bodies.json") -> Path:
    """An exported bodies bundle (test.bodies: two body tracks, their scores and a duplicate) after `damage(path of
    file)`, with the manifest's checksums updated."""
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    doc = {"schema_version": 1, "pipeline": "test", "inputs": {"video": {"type": "video.v1", "uri": str(video)}},
           "nodes": [{"id": "shots", "uses": "test.layer@1.0.0", "inputs": {"video": "$inputs.video"}},
                     {"id": "bodies", "uses": "test.bodies@1.0.0", "inputs": {"video": "$inputs.video"}}]}
    run_id, status = run(coord, doc)
    assert status == "succeeded", coord._tasks(run_id)
    out = export.bundle(coord, run_id, tmp_path / "bundle")
    damage(out / file)
    manifest = json.loads((out / "manifest.json").read_text())
    for entry in manifest["layers"].values():
        for f in [entry, *entry.get("side_files", [])]:
            f.update(sha256=artifacts.sha256_file(out / f["file"]), bytes=(out / f["file"]).stat().st_size)
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


@pytest.mark.parametrize("damage, problems", [
    (lambda p: None, []),
    # body:1 loses its sample: the scores no longer match, and the duplicate names a sample the layer lacks
    (_with(lambda d: d["tracks"][1]["samples"].clear()),
     ["bodies_scores.parquet does not match the layer's samples: 3 rows for 2 samples",
      "bodies_duplicates.parquet: 1 rows name no sample of the layer, first row 0"]),
    (_with(lambda d: d["tracks"][0]["samples"][1].update(t=0.25)), ["bodies_scores.parquet does not match the layer's samples"]),
    # the rows are in the layer's track order at each time; body:1's sample is still there for the duplicate
    (_with(lambda d: d["tracks"].reverse()), ["bodies_scores.parquet does not match the layer's samples"]),
])
def test_validate_checks_the_bodies_side_files_rows_against_the_layers_samples(tmp_path, damage, problems):
    """bodies_scores holds a row per sample of the bodies layer, its track column body holding the tracks' id (not face,
    as gaze_heatmaps'); every bodies_duplicates row names one of the layer's samples (of_samples)."""
    out = bodies_bundle(tmp_path, damage)
    assert sorted(p.name for p in (out / "layers").iterdir()) == ["bodies.json", "bodies_duplicates.parquet",
                                                                  "bodies_scores.parquet", "shots.json"]
    for name in ("scores", "duplicates"):
        assert (out / f"layers/bodies_{name}.parquet").read_bytes() == (NODES / "bodies" / f"bodies_{name}.parquet").read_bytes()
    result = subprocess.run(["uv", "run", "--project", str(VALIDATE.parent), "python", str(VALIDATE), str(out)],
                            capture_output=True, text=True)
    assert "Traceback" not in result.stderr and "note:" not in result.stdout, result.stdout + result.stderr
    assert result.returncode == (1 if problems else 0), result.stdout + result.stderr
    assert all(p in result.stdout for p in problems), result.stdout


def test_validate_without_pyarrow_says_it_skipped_the_parquet_content_check(tmp_path):
    """pyarrow is validate.py's dev dependency: without it the bundle can still be valid, but the output says the
    Parquet columns were not checked rather than passing in silence. Parquet's magic is checked either way."""
    out = gaze_bundle(tmp_path, lambda p: None, HEATMAPS)
    bare = ["uv", "run", "--no-project", "--with", "jsonschema>=4.23", "--with", "referencing>=0.35",
            "python", "-I", str(VALIDATE), str(out)]
    result = subprocess.run(bare, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "note: gaze: layers/gaze_heatmaps.parquet: Parquet content not checked (pyarrow is not installed" in result.stdout
    assert "3 checks skipped" in result.stdout   # the heatmaps, and test.faces' faces_pyfeat and faces_untracked
    (out / HEATMAPS).write_bytes((out / HEATMAPS).read_bytes()[:-1])
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["layers"]["gaze"]["side_files"][0].update(sha256=artifacts.sha256_file(out / HEATMAPS),
                                                       bytes=(out / HEATMAPS).stat().st_size)
    (out / "manifest.json").write_text(json.dumps(manifest))
    result = subprocess.run(bare, capture_output=True, text=True)
    assert result.returncode == 1 and "gaze_heatmaps.parquet: not Parquet" in result.stdout, result.stdout


def transcript_graph(video: Path) -> dict:
    return {"schema_version": 1, "pipeline": "test", "inputs": {"video": {"type": "video.v1", "uri": str(video)}},
            "nodes": [{"id": "shots", "uses": "test.layer@1.0.0", "inputs": {"video": "$inputs.video"}},
                      {"id": "asr", "uses": "test.transcript@1.0.0", "inputs": {"video": "$inputs.video"}}]}


TOKENS, TRANSCRIPT = "layers/transcript_tokens.parquet", "layers/transcript.json"


def _validate(out: Path):
    return subprocess.run(["uv", "run", "--project", str(VALIDATE.parent), "python", str(VALIDATE), str(out)],
                          capture_output=True, text=True)


def test_the_transcript_tokens_go_into_the_bundle_and_index_its_words(tmp_path):
    out = gaze_bundle(tmp_path, lambda p: None, TOKENS, graph=transcript_graph)
    manifest = json.loads((out / "manifest.json").read_text())
    assert [f["file"] for f in manifest["layers"]["transcript"]["side_files"]] == [TOKENS]
    assert (out / TOKENS).read_bytes() == (NODES / "transcript" / "transcript_tokens.parquet").read_bytes()
    result = _validate(out)
    assert result.returncode == 0 and "note:" not in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("damage, problem", [
    # a refiner that dropped a word but kept asr's tokens: the last word's tokens index past the layer's words
    (_with(lambda d: d["words"].pop()), "row 4's word 2 is not one of the layer's 2 words (1 rows)"),
    (_with(lambda d: d["words"][1].update(end=0.8)),
     "does not match the layer's words: 1 words with no token or another start or end, e.g. word 1: its tokens span "
     "[0.48, 0.72], the layer's word [0.48, 0.8]"),
    # a word inserted before the others: every word's index is off by one
    (_with(lambda d: d["words"].insert(0, {"w": "Oh", "start": 0.0, "end": 0.08, "voice": None})),
     "does not match the layer's words: 4 words"),
    # a word without times is the schema's to report, not a crash in the words check
    (_with(lambda d: d["words"][0].pop("start")), "transcript: $.words[0]: 'start' is a required property"),
])
def test_validate_reports_tokens_that_do_not_index_the_layers_words(tmp_path, damage, problem):
    out = gaze_bundle(tmp_path, damage, TRANSCRIPT, graph=transcript_graph)
    result = _validate(out)
    assert result.returncode == 1 and "Traceback" not in result.stderr, result.stdout + result.stderr
    assert "INVALID" in result.stdout and problem in result.stdout, result.stdout


# --- the sounds layer's tags and queries (Parquet side files whose scores are fixed-size lists) -----------------------

def sounds_bundle(tmp_path, **config) -> Path:
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    doc = faces_graph(video)   # the bundle names its video from the decode stand-in's media
    doc["nodes"].append({"id": "sounds", "uses": "test.sounds@1.0.0", "inputs": {"video": "$inputs.video"}, "config": config})
    run_id, status = run(coord, doc)
    assert status == "succeeded", coord._tasks(run_id)
    return export.bundle(coord, run_id, tmp_path / "bundle")


def test_the_sounds_scores_count_as_audio_when_a_step_reads_them_elsewhere():
    """Every tagger score every 0.32 s and every query's probability every 40 ms are a dense measure of the soundtrack,
    speech classes included, where the sounds layer keeps labelled events: audio, as the waveform is."""
    from perception.backends.base import Backend
    remote = type("Remote", (), {"remote": True})()
    for name in ("sounds_tags", "sounds_queries"):
        assert artifacts.get(f"bundle.{name}@0.1").kinds == ("audio",)
        reader = type("Manifest", (), {"inputs": {"scores": f"bundle.{name}@0.1"}})()
        assert Backend.sends(remote, reader) == ["audio"]


def test_the_sounds_side_files_go_into_the_bundle_and_validate(tmp_path):
    out = sounds_bundle(tmp_path)
    for name in ("sounds_tags", "sounds_queries"):
        assert (out / "layers" / f"{name}.parquet").read_bytes() == (NODES / "sounds" / f"{name}.parquet").read_bytes()
    manifest = json.loads((out / "manifest.json").read_text())
    assert [f["file"] for f in manifest["layers"]["sounds"]["side_files"]] == [
        "layers/sounds_queries.parquet", "layers/sounds_tags.parquet"]
    result = subprocess.run(["uv", "run", "--project", str(VALIDATE.parent), "python", str(VALIDATE), str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0 and "note:" not in result.stdout, result.stdout + result.stderr


def test_validate_refuses_a_score_that_is_not_finite_in_a_list_column(tmp_path):
    """The scores are fixed-size lists of float32: their values are checked for finiteness like a float column's."""
    out = sounds_bundle(tmp_path, queries_fixture="sounds_queries_nan.parquet")
    result = subprocess.run(["uv", "run", "--project", str(VALIDATE.parent), "python", str(VALIDATE), str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 1 and "Traceback" not in result.stderr, result.stdout + result.stderr
    assert "sounds_queries.parquet: column probs holds a value that is not finite" in result.stdout, result.stdout






# --- the faces layer's raw Parquet side files (py-feat's output beyond the layer, and the faces in no track) ----------

def test_the_faces_raw_side_files_say_what_they_hold():
    """faces_pyfeat is face geometry and expression per sample (no identity vector); faces_untracked holds every value
    py-feat gave for a face no track holds, its ArcFace embedding among them, so reading it elsewhere needs consent for
    embeddings, as faces_embeddings does."""
    from perception.backends.base import Backend
    assert artifacts.get("bundle.faces_pyfeat@0.1").kinds == ("faces",)
    assert set(artifacts.get("bundle.faces_untracked@0.1").kinds) == {"faces", "embeddings"}
    remote = type("Remote", (), {"remote": True})()
    reader = type("Manifest", (), {"inputs": {"untracked": "bundle.faces_untracked@0.1"}})()
    assert Backend.sends(remote, reader) == ["embeddings", "faces"]


def _parquet_damage(code: str):
    """A damage for bundle_validation that rewrites a Parquet file with pyarrow (validate.py's dev group): `code` gets
    the table as t and pyarrow as pa and pyarrow.compute as pc, and binds the new table to t."""
    def damage(path):
        script = ("import sys, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq\n"
                  f"t = pq.read_table(sys.argv[1])\n{code}\npq.write_table(t, sys.argv[1])\n")
        subprocess.run(["uv", "run", "--project", str(VALIDATE.parent), "python", "-c", script, str(path)], check=True)
    return damage


# mesh_z (478 floats a row) with its first value NaN: what a check of scalar float columns alone would pass
NAN_IN_A_LIST = ("v = t['mesh_z'].combine_chunks()\n"
                 "flat = pc.if_else(pa.array([i == 0 for i in range(len(v.values))]), float('nan'), v.values)\n"
                 "t = t.set_column(t.schema.get_field_index('mesh_z'), t.schema.field('mesh_z'),"
                 " pa.FixedSizeListArray.from_arrays(flat.cast(pa.float32()), type=t.schema.field('mesh_z').type))")


@pytest.mark.parametrize("file, damage, problem", [
    ("layers/faces_pyfeat.parquet", _parquet_damage(NAN_IN_A_LIST),
     "faces_pyfeat.parquet: column mesh_z holds a value that is not finite"),
    ("layers/faces_pyfeat.parquet", _parquet_damage("t = t.slice(0, 1)"),
     "faces_pyfeat.parquet does not match the layer's samples: 1 rows for 2 samples"),
    ("layers/faces_untracked.parquet", _parquet_damage("t = t.drop_columns(['embedding'])"),
     "faces_untracked.parquet: columns are"),
])
def test_validate_reports_a_bad_faces_raw_side_file_as_a_problem(tmp_path, file, damage, problem):
    result = bundle_validation(tmp_path, damage, file)
    assert result.returncode == 1 and "Traceback" not in result.stderr, result.stdout + result.stderr
    assert "INVALID" in result.stdout and problem in result.stdout, result.stdout



@pytest.mark.parametrize("backend", ["NativeBackend", "SelfBackend"])
def test_changing_the_node_sdk_changes_every_nodes_runner_identity(tmp_path, monkeypatch, backend):
    from perception.backends import native
    sdk = tmp_path / "sdk" / "src" / "perception_node"
    sdk.mkdir(parents=True)
    (sdk / "__init__.py").write_text("PROTOCOL = 1\n")
    monkeypatch.setattr(native, "REPO", tmp_path)
    m = Registry.load(NODES).get("test.work@1.0.0")
    b = getattr(native, backend)()
    before = b.identity(m, m.runners[0])
    (sdk / "__init__.py").write_text("PROTOCOL = 1  # a changed response writer\n")
    assert b.identity(m, m.runners[0]) != before


def test_a_bundle_names_the_pipeline_version_that_made_it(tmp_path):
    """Not a fixed string: a bundle from 0.1.1 says 0.1.1 (the installed package's version)."""
    from importlib.metadata import version
    from perception import export
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    coord = coordinator(tmp_path)
    doc = {"schema_version": 1, "pipeline": "test", "inputs": {"video": {"type": "video.v1", "uri": str(video)}},
           "nodes": [{"id": "shots", "uses": "test.layer@1.0.0", "inputs": {"video": "$inputs.video"}}]}
    run_id, status = run(coord, doc)
    assert status == "succeeded", coord._tasks(run_id)
    manifest = json.loads((export.bundle(coord, run_id, tmp_path / "bundle") / "manifest.json").read_text())
    assert manifest["pipeline"]["version"] == version("hyperstudy-perception")
