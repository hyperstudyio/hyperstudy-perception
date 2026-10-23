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


def test_rerun_from_a_node_resets_it_and_its_descendants(tmp_path, rec):
    coord = coordinator(tmp_path)
    run_id, _ = run(coord, graph_doc(tmp_path, [work("a", record_dir=rec), work("b", "$nodes.a.out", record_dir=rec),
                                                work("c", "$nodes.b.out", record_dir=rec)]))
    coord.rerun(run_id, "b", use_cache=False)
    assert asyncio.run(coord.run(run_id)) == "succeeded"
    counts = {n: len(executions(Path(rec), n)) for n in "abc"}
    assert counts == {"a": 1, "b": 2, "c": 2}


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
