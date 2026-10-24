"""`perception --json`: what the annotator app reads when it drives the CLI as a sidecar. One JSON object per line,
each with a protocol version, so the app never parses text meant for people."""

import json
from pathlib import Path

import pytest
import yaml

from perception.cli import main

NODES = Path(__file__).parent / "nodes"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "home"))
    return tmp_path


def graph(root: Path, nodes: list[dict]) -> Path:
    (root / "seed.json").write_text('{"value": 0}')
    doc = {"schema_version": 1, "pipeline": "cli-test", "inputs": {"seed": {"type": "json.v1", "uri": str(root / "seed.json")}},
           "policy": {"retries": 0}, "nodes": nodes}
    path = root / "graph.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def work(id: str, root: Path, a: str = "$inputs.seed", **config) -> dict:
    return {"id": id, "uses": "test.work@1.0.0", "inputs": {"a": a}, "config": {"record_dir": str(root / "rec"), **config}}


def lines(capsys) -> list[dict]:
    out = capsys.readouterr().out.splitlines()
    parsed = [json.loads(line) for line in out]            # every line is JSON: nothing for people mixed in
    assert all(p["v"] == 1 for p in parsed)
    return parsed


def cli(*args: str) -> int:
    return main(["--json", "--nodes", str(NODES), *args])


def test_plan_reports_each_node_with_placement_cache_and_cost(home, capsys):
    g = graph(home, [work("a", home), work("b", home, a="$nodes.a.out")])
    assert cli("plan", str(g)) == 0
    [plan] = lines(capsys)
    assert plan["type"] == "plan" and plan["ok"] is True and plan["estimated_usd"] == 0
    assert [(n["id"], n["uses"], n["cached"]) for n in plan["nodes"]] == [("a", "test.work@1.0.0", False), ("b", "test.work@1.0.0", False)]
    assert plan["nodes"][0]["where"] and plan["nodes"][0]["downloads"] == [] and plan["nodes"][0]["missing_secrets"] == []


def test_a_run_streams_its_events_and_ends_with_its_status(home, capsys):
    g = graph(home, [work("a", home), work("b", home, a="$nodes.a.out")])
    assert cli("run", str(g)) == 0
    out = lines(capsys)
    assert out[0]["type"] == "run.created" and out[0]["run"]
    run = out[0]["run"]
    events = [(e["node"], e["kind"]) for e in out if e["type"] == "event"]
    assert ("a", "task.succeeded") in events and ("b", "task.succeeded") in events
    assert out[-1] == {"v": 1, "type": "run.finished", "run": run, "status": "succeeded"}

    assert cli("status", run) == 0
    [status] = lines(capsys)
    assert status["type"] == "status" and status["status"] == "succeeded"
    assert {t["node"]: t["state"] for t in status["tasks"]} == {"a": "succeeded", "b": "succeeded"}


def test_a_failed_node_ends_the_run_failed_with_its_error_code(home, capsys):
    g = graph(home, [work("a", home, fail="input", fail_until=9)])
    assert cli("run", str(g)) == 1
    out = lines(capsys)
    assert out[-1]["status"] == "failed"
    assert cli("status", out[0]["run"]) == 0
    [status] = lines(capsys)
    [task] = status["tasks"]
    assert (task["state"], task["error_code"]) == ("failed", "input") and "on purpose" in task["error"]


def test_an_invalid_graph_is_one_error_object(home, capsys):
    g = graph(home, [{"id": "a", "uses": "test.nope@1.0.0", "inputs": {}}])
    assert cli("run", str(g)) == 1
    [err] = lines(capsys)
    assert err["type"] == "error" and any("test.nope" in p for p in err["problems"])


def test_doctor_lists_the_machine_and_where_each_node_can_run(home, capsys):
    assert cli("doctor") == 0
    [doc] = lines(capsys)
    assert doc["type"] == "doctor" and doc["cpu_cores"] > 0 and "cpu" in doc["accelerators"]
    work_node = next(n for n in doc["nodes"] if n["ref"] == "test.work@1.0.0")
    assert work_node["where"] and work_node["why_not"] == ""


def test_a_rerun_streams_only_its_own_events_not_the_runs_history(home, capsys):
    g = graph(home, [work("a", home)])
    assert cli("run", str(g)) == 0
    run = lines(capsys)[0]["run"]
    assert cli("rerun", run, "--from", "a", "--no-cache") == 0
    out = lines(capsys)
    kinds = [e["kind"] for e in out if e["type"] == "event"]
    assert kinds[0] == "run.rerun" and kinds.count("run.succeeded") == 1 and kinds.count("task.succeeded") == 1


def test_every_command_speaks_json(home, capsys):
    g = graph(home, [work("a", home)])
    assert cli("graph", "validate", str(g)) == 0
    [v] = lines(capsys)
    assert v["type"] == "graph" and v["valid"] is True and v["nodes"] == ["a"]
    assert cli("models", "list") == 0
    [m] = lines(capsys)
    assert m["type"] == "models" and any(x["node"] == "test.work@1.0.0" for x in m["models"])
    assert cli("run", str(g)) == 0
    run = lines(capsys)[0]["run"]
    assert cli("artifacts", run) == 0
    [a] = lines(capsys)
    assert a["type"] == "artifacts" and [(x["node"], x["name"], x["type"]) for x in a["artifacts"]] == [("a", "out", "json.v1")]
    assert cli("logs", run) == 0
    out = lines(capsys)
    assert out and all(e["type"] == "event" for e in out) and out[-1]["kind"] == "run.succeeded"


@pytest.mark.parametrize("command", [["artifacts"], ["cancel"], ["logs"], ["logs", "--follow"]])
def test_an_unknown_run_is_an_error_not_an_empty_success(home, capsys, command):
    assert cli(command[0], "no-such-run", *command[1:]) == 1
    [err] = lines(capsys)
    assert err == {"v": 1, "type": "error", "problems": ["no run no-such-run"]}
