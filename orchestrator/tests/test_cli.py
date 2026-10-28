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


def test_the_plan_says_what_each_node_sends_off_this_machine(home, capsys, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    doc = {"schema_version": 1, "pipeline": "t", "inputs": {"video": {"type": "video.v1", "uri": str(video)}},
           "policy": {"privacy": "local-preferred", "allow_upload": ["frames"], "max_cost_usd": 1},
           "nodes": [{"id": "shots", "uses": "test.layer@1.0.0", "inputs": {"video": "$inputs.video"}},
                     {"id": "captions", "uses": "test.refine@1.0.0", "inputs": {"shots": "$nodes.shots.shots"}}]}
    g = tmp_path / "g.yaml"
    g.write_text(yaml.safe_dump(doc))
    assert cli("plan", str(g)) == 1                      # cannot run as it stands: the upload is not allowed yet
    [plan] = lines(capsys)
    assert {n["id"]: n["uploads"] for n in plan["nodes"]} == {"shots": [], "captions": ["frames"]}
    assert plan["ok"] is False


def uploading_graph(tmp_path) -> Path:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    doc = {"schema_version": 1, "pipeline": "t", "inputs": {"video": {"type": "video.v1", "uri": str(video)}},
           "policy": {"privacy": "local-preferred", "allow_upload": ["frames"], "max_cost_usd": 30},
           "nodes": [{"id": "shots", "uses": "test.layer@1.0.0", "inputs": {"video": "$inputs.video"}},
                     {"id": "captions", "uses": "test.refine@1.0.0", "inputs": {"shots": "$nodes.shots.shots"}}]}
    g = tmp_path / "g.yaml"
    g.write_text(yaml.safe_dump(doc))
    return g


def test_nothing_leaves_the_machine_on_a_run_without_allow_uploads(home, capsys, tmp_path):
    g = uploading_graph(tmp_path)
    assert cli("run", str(g)) == 1                       # the graph may send frames; this run did not say it may
    [plan] = lines(capsys)
    assert plan["type"] == "plan" and plan["ok"] is False
    assert cli("run", str(g), "--only", "shots") == 0    # leaving the uploading node out needs no consent
    capsys.readouterr()
    assert cli("run", str(g), "--allow-uploads") == 0
    last = lines(capsys)[-1]
    assert last["type"] == "run.finished" and last["status"] == "succeeded"


def test_a_setting_is_chosen_per_run_from_the_choices_the_plan_lists(home, capsys, tmp_path):
    g = uploading_graph(tmp_path)
    cli("plan", str(g))
    [plan] = lines(capsys)
    captions = next(n for n in plan["nodes"] if n["id"] == "captions")
    assert captions["settings"] == [{"key": "model", "description": "", "value": "cheap", "choices": ["cheap", "dear"],
                                     "prices": {"cheap": 0.5, "dear": 20.0}, "per": "shot"}]
    cli("plan", str(g), "--set", "captions.model=dear")
    [plan] = lines(capsys)
    assert next(n for n in plan["nodes"] if n["id"] == "captions")["settings"][0]["value"] == "dear"
    assert cli("plan", str(g), "--set", "captions.model=free") == 1      # not one of the choices
    [err] = lines(capsys)
    assert err["type"] == "error" and any("config model" in p for p in err["problems"])
    assert cli("plan", str(g), "--only", "shots", "--set", "captions.model=dear") == 0   # a setting for a node left out goes with it
    lines(capsys)
    assert cli("plan", str(g), "--set", "captions=dear") == 1
    [err] = lines(capsys)
    assert err["problems"] == ["--set captions=dear: expected NODE.KEY=VALUE"]


def test_the_full_graph_is_valid_with_the_real_nodes_and_every_node_has_a_title(capsys):
    root = Path(__file__).parents[2]
    assert main(["graph", "validate", str(root / "graphs" / "full.yaml")]) == 0
    for m in (root / "nodes").glob("*/*/manifest.yaml"):
        assert yaml.safe_load(m.read_text()).get("title"), f"{m} has no title (the app's feature list shows it)"
    captions = yaml.safe_load((root / "nodes" / "vlm" / "shot-captions" / "manifest.yaml").read_text())
    assert captions["service"] == "Anthropic API"   # it runs there: the app shows the API, not this machine's CPU


def test_nodes_lists_what_each_node_takes_makes_costs_sends_and_where_it_can_run(home, capsys):
    assert cli("nodes") == 0
    [doc] = lines(capsys)
    work_node = next(n for n in doc["nodes"] if n["ref"] == "test.work@1.0.0")
    assert work_node["inputs"] == {"a": "json.v1"} and work_node["outputs"] == {"out": "json.v1"}
    assert work_node["places"] == ["self:cpu"] and work_node["why_not"] == ""
    refine = next(n for n in doc["nodes"] if n["ref"] == "test.refine@1.0.0")
    assert refine["uploads"] == ["frames"] and refine["cost"]["per"] == "shot"
    assert refine["models"] == [{"id": "test-captioner", "licence_mode": "api-terms", "source": None, "size_gb": None, "gated": False}]


def test_place_chooses_where_a_node_runs_and_a_place_it_cannot_run_is_a_plan_problem(home, capsys):
    g = graph(home, [work("a", home)])
    assert cli("plan", str(g), "--place", "a=self:cpu") == 0
    [plan] = lines(capsys)
    assert plan["nodes"][0]["place"] == "self:cpu" and plan["nodes"][0]["options"] == ["self:cpu"]
    assert cli("plan", str(g), "--place", "a=self:mps") == 1
    [plan] = lines(capsys)
    assert plan["ok"] is False and "self:mps is not one of the places it can run (self:cpu)" in plan["nodes"][0]["why_not"]
    assert cli("plan", str(g), "--place", "zz=self:cpu") == 1
    [err] = lines(capsys)
    assert err["type"] == "error" and "the graph has no node 'zz'" in err["problems"][0]


def test_only_keeps_the_chosen_nodes_and_what_they_need_and_drops_the_rest(home, capsys):
    # a -> b -> c, and d on its own
    g = graph(home, [work("a", home), work("b", home, a="$nodes.a.out"), work("c", home, a="$nodes.b.out"), work("d", home)])
    assert cli("plan", str(g)) == 0
    [plan] = lines(capsys)
    assert {n["id"]: n["needs"] for n in plan["nodes"]} == {"a": [], "b": ["a"], "c": ["b"], "d": []}   # for a stage picker
    assert cli("plan", str(g), "--only", "b") == 0
    [plan] = lines(capsys)
    assert [n["id"] for n in plan["nodes"]] == ["a", "b"]
    assert cli("plan", str(g), "--only", "c", "--only", "d") == 0
    [plan] = lines(capsys)
    assert sorted(n["id"] for n in plan["nodes"]) == ["a", "b", "c", "d"]
    assert cli("plan", str(g), "--only", "a,d") == 0
    [plan] = lines(capsys)
    assert sorted(n["id"] for n in plan["nodes"]) == ["a", "d"]
    assert cli("plan", str(g), "--only", "zz") == 1
    [err] = lines(capsys)
    assert err["type"] == "error" and "the graph has no node 'zz'" in err["problems"][0]


def test_a_run_with_only_runs_just_those_nodes_and_keeps_the_choice_for_resume(home, capsys):
    g = graph(home, [work("a", home), work("b", home, a="$nodes.a.out"), work("d", home)])
    assert cli("run", str(g), "--only", "b") == 0
    out = lines(capsys)
    run = out[0]["run"]
    assert {e["node"] for e in out if e["type"] == "event" and e["kind"] == "task.succeeded"} == {"a", "b"}
    assert cli("status", run) == 0
    [status] = lines(capsys)
    assert sorted(t["node"] for t in status["tasks"]) == ["a", "b"]   # d is not part of this run at all


def test_only_leaves_an_optional_input_unconnected_unless_its_producer_is_chosen(home, capsys):
    # c reads a, and b when b runs (test.optional's optional input b): turning b off must not turn c off
    c = {"id": "c", "uses": "test.optional@1.0.0", "inputs": {"a": "$nodes.a.out", "b": "$nodes.b.out"},
         "config": {"record_dir": str(home / "rec")}}
    g = graph(home, [work("a", home), work("b", home), c])
    assert cli("plan", str(g)) == 0
    [plan] = lines(capsys)
    assert {n["id"]: (n["needs"], n["optional_needs"]) for n in plan["nodes"]} == {"a": ([], []), "b": ([], []), "c": (["a"], ["b"])}
    assert cli("plan", str(g), "--only", "c") == 0
    [plan] = lines(capsys)
    assert [(n["id"], n["optional_needs"]) for n in plan["nodes"]] == [("a", []), ("c", [])]   # b neither runs nor is waited for
    assert cli("plan", str(g), "--only", "c,b") == 0
    [plan] = lines(capsys)
    assert {n["id"]: n["optional_needs"] for n in plan["nodes"]} == {"a": [], "b": [], "c": ["b"]}   # chosen: still connected
    assert cli("run", str(g), "--only", "c") == 0
    out = lines(capsys)
    assert {e["node"] for e in out if e["type"] == "event" and e["kind"] == "task.succeeded"} == {"a", "c"}
    g = graph(home, [work("a", home), {**c, "inputs": {"a": "$nodes.a.out", "b": "$nodes.zz.out"}}])
    assert cli("plan", str(g), "--only", "c") == 1   # a producer the graph lacks is still an error, not left out
    [err] = lines(capsys)
    assert err["type"] == "error" and "no node 'zz'" in err["problems"][0]


# --- perception setup ----------------------------------------------------------------------------------------------

@pytest.fixture
def bare_path(monkeypatch):
    """The PATH of an app started from the Finder: no ffmpeg or ffprobe, which no node needs (they read video with PyAV)."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")


def test_setup_prepares_what_a_graph_would_run_here_and_says_what_failed(home, capsys, tmp_path, bare_path, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "never-for-a-node")   # a credential the node did not declare
    rec = tmp_path / "rec"
    g = graph(home, [work("a", home)])
    doc = yaml.safe_load(g.read_text())
    doc["nodes"][0]["config"] = {"record_dir": str(rec), "add": 3}
    g.write_text(yaml.safe_dump(doc))
    assert cli("setup", str(g)) == 0
    out = lines(capsys)
    steps = [o for o in out if o["type"] == "setup.step"]
    assert [(s["what"], s["ok"]) for s in steps] == [("models", True)]   # a self runner: no environment
    assert out[-1]["type"] == "setup" and out[-1]["ok"] is True
    assert json.loads((rec / "prepared").read_text())["add"] == 3   # with the graph's settings for it (one may choose the model)
    assert json.loads((rec / "environment").read_text()).get("AWS_SECRET_ACCESS_KEY") is None   # as a run: no undeclared key
    doc["nodes"][0]["config"] = {"record_dir": str(rec), "fail": "prepare"}
    g.write_text(yaml.safe_dump(doc))
    assert cli("setup", str(g)) == 1
    out = lines(capsys)
    failed = [o for o in out if o["type"] == "setup.step" and not o["ok"]]
    assert len(failed) == 1 and "could not fetch the weights" in failed[0]["message"]


def test_setup_syncs_a_native_nodes_environment_from_its_lock_then_runs_its_prepare(tmp_path):
    from perception import setup
    from perception.manifests import Registry
    calls = tmp_path / "calls"
    uv = tmp_path / "uv"
    uv.write_text(f'#!/bin/sh\necho "$*" >> {calls}\n[ -e {tmp_path}/sync-fails ] && [ "$1" = sync ] && {{ echo "lock out of date" >&2; exit 2; }}\nexit 0\n')
    uv.chmod(0o755)
    m = Registry.load(Path(__file__).resolve().parents[2] / "nodes").get("vision.face-track@1.4.0")
    runner = next(r for r in m.runners if r.kind == "native")
    steps = setup.setup_node(m, runner, uv=str(uv))
    assert [(s.what, s.ok) for s in steps] == [("environment", True), ("models", True)]
    sync, run = calls.read_text().splitlines()
    assert sync.startswith("sync --project ") and sync.endswith("envs/faces --frozen --no-dev")
    assert run.startswith("run --project ") and "prepare.py" in run
    (tmp_path / "sync-fails").touch()
    steps = setup.setup_node(m, runner, uv=str(uv))
    assert [(s.what, s.ok) for s in steps] == [("environment", False)] and "lock out of date" in steps[0].message


def test_setup_for_a_graph_with_a_node_that_cannot_run_here_is_not_ready(home, capsys, bare_path):
    g = graph(home, [{**work("a", home), "place": "self:cuda"}])        # test.work has a CPU runner only
    assert cli("setup", str(g)) == 1
    out = lines(capsys)
    [failed] = [o for o in out if o["type"] == "setup.step" and not o["ok"]]
    assert failed["what"] == "place" and "cannot run here" in failed["message"] and out[-1]["ok"] is False


def test_cloud_check_says_whether_a_provider_can_be_used_and_changes_nothing(home, capsys, monkeypatch):
    from perception import cli as cli_mod
    monkeypatch.setattr(cli_mod, "runpod_problem", lambda: "RunPod did not accept the key in RUNPOD_API_KEY (401)")
    assert cli("cloud", "check", "runpod") == 0
    [out] = lines(capsys)
    assert out == {"v": 1, "type": "cloud_check", "provider": "runpod", "ok": False, "detail": "RunPod did not accept the key in RUNPOD_API_KEY (401)"}
    monkeypatch.setattr(cli_mod, "runpod_problem", lambda: None)
    assert cli("cloud", "check", "runpod") == 0
    assert lines(capsys)[0]["ok"] is True
    assert cli("cloud", "status", "runpod") == 0
    assert lines(capsys)[0]["runpod"]["enabled"] is False                         # checking did not turn it on


def test_models_list_gives_each_node_its_title_service_keys_and_choices_with_prices(home, capsys):
    assert main(["--json", "models", "list"]) == 0   # the repo's own nodes
    [out] = lines(capsys)
    captions = next(n for n in out["nodes"] if n["node"] == "vlm.shot-captions@1.0.0")
    assert captions["service"] == "Anthropic API" and captions["secrets"] == ["ANTHROPIC_API_KEY"]
    model = next(s for s in captions["settings"] if s["key"] == "model")
    assert model["value"] == "claude-haiku-5-5" and model["choices"] == ["claude-haiku-5-5", "claude-opus-5-5"]
    assert set(model["prices"]) == set(model["choices"]) and model["per"]
    asr = next(n for n in out["nodes"] if n["node"] == "audio.asr@1.1.0")
    assert asr["service"] is None and asr["secrets"] == []
