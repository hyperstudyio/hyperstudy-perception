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


def start(coord: Coordinator, doc: dict) -> str:
    graph = compile_graph(doc, coord.registry)
    return coord.create_run(graph, doc)


def run(coord: Coordinator, doc: dict) -> tuple[str, str]:
    run_id = start(coord, doc)
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
    assert backend.poll(json.loads(t["handle"])) != "running"    # the process it had started was stopped
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
