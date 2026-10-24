"""Nodes on a cloud machine (backends/cloud.py and the driver's cloud steps), against stand-ins for sky, aws, ssh and
guard.sh that record what they were asked to do (tmp/host/events). The node really runs, "on the machine" (a separate
home on this one, which reports an NVIDIA GPU through a stand-in nvidia-smi)."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

from perception.backends.cloud import CloudBackend, cluster_name
from perception.backends.native import SelfBackend
from perception.ir import compile_graph
from perception.manifests import Registry
from perception.plan import plan as make_plan
from perception.scheduler import Coordinator
from perception.store import Store

NODES = Path(__file__).parent / "nodes"

FAKE_SSH = r"""#!/bin/sh
while [ $# -gt 0 ]; do case "$1" in -o) shift 2;; -*) shift;; *) break;; esac; done
echo "ssh $1: $2" | head -c 300 >> "$FAKE_HOST/events"; echo >> "$FAKE_HOST/events"
shift
cd "$FAKE_HOST/home" && HOME="$FAKE_HOST/home" exec /bin/sh -c "$*"
"""
FAKE_SKY = r"""#!/bin/sh
echo "sky $* (config $SKYPILOT_PROJECT_CONFIG)" >> "$FAKE_HOST/events"
case "$1" in
  launch) [ -e "$FAKE_HOST/launch-fails" ] && { echo "no capacity"; exit 1; }; echo "launched $3"; exit 0;;
  down) [ -e "$FAKE_HOST/down-fails" ] && exit 1; exit 0;;
  check) printf '\033[32m  AWS: enabled [compute, storage]\033[0m\n'; exit 0;;
esac
"""
FAKE_AWS = r"""#!/bin/sh
echo "aws $*" >> "$FAKE_HOST/events"
case "$*" in *describe-instances*) echo "i-0abc";; esac
"""
GUARD_OK = 'echo "guard $1" >> "$FAKE_HOST/events"; echo "guard: ok"\n'
GUARD_REFUSES = 'echo "guard: the instance profile is skypilot-v1, not perception-node" >&2; exit 1\n'


def script(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def machine(tmp_path, monkeypatch):
    fake = tmp_path / "host"
    (fake / "home" / ".local" / "bin").mkdir(parents=True)
    script(fake / "home" / ".local" / "bin" / "nvidia-smi", "#!/bin/sh\necho 23034\n")
    monkeypatch.setenv("FAKE_HOST", str(fake))
    monkeypatch.setenv("PERCEPTION_SSH", str(script(tmp_path / "ssh", FAKE_SSH)))
    monkeypatch.setenv("PERCEPTION_SKY", str(script(tmp_path / "sky", FAKE_SKY)))
    monkeypatch.setenv("PERCEPTION_AWS", str(script(tmp_path / "aws", FAKE_AWS)))
    return fake


def events(fake: Path) -> list[str]:
    return (fake / "events").read_text().splitlines() if (fake / "events").exists() else []


def coordinator(tmp_path, guard=GUARD_OK, deadline_s=1800) -> Coordinator:
    cloud = CloudBackend(deadline_s=deadline_s, runner_kind="self", python=sys.executable,
                         guard=script(tmp_path / "guard.sh", guard), task=tmp_path / "node-gpu.yaml")
    return Coordinator(Store(tmp_path / "store"), Registry.load(NODES), [SelfBackend(), cloud], tick_s=0.02, backoff_s=0.01)


def graph(tmp_path, rec, uses="test.two@1.0.0", place="cloud:cuda", max_cost=5.0, **config) -> dict:
    seed = tmp_path / "seed.json"
    seed.write_text('{"value": 0}')
    return {"schema_version": 1, "pipeline": "test", "inputs": {"seed": {"type": "json.v1", "uri": str(seed)}},
            "policy": {"retries": 0, "placement": "hybrid", "privacy": "local-preferred", "max_cost_usd": max_cost,
                       "allow_upload": ["video", "audio", "frames", "faces", "transcript", "embeddings"]},   # json.v1: any
            "nodes": [{"id": "faces", "uses": uses, "inputs": {"a": "$inputs.seed"}, "place": place,
                       "config": {"record_dir": rec, **config}}]}


def run(coord, doc) -> tuple[str, str]:
    run_id = coord.create_run(compile_graph(doc, coord.registry), doc, allow_uploads=True)
    return run_id, asyncio.run(coord.run(run_id))


def log_of(coord, run_id) -> str:
    return (coord.store.root / "work" / run_id / "faces" / "1" / "log.txt").read_text()


@pytest.fixture
def rec(tmp_path):
    return str(tmp_path / "record")


def test_a_node_runs_on_a_machine_started_for_it_checked_before_anything_is_sent_and_deleted_after(tmp_path, machine, rec):
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "succeeded", log_of(coord, run_id)
    cluster = cluster_name(run_id, "faces", 1)
    assert cluster.startswith("perception-")   # the only machines the perception-node role can end
    ev = events(machine)
    launch = next(i for i, e in enumerate(ev) if e.startswith(f"sky launch -c {cluster} -y --down -i 15"))
    assert ev[launch].endswith(f"(config {CloudBackend.project_config})")   # no credentials, only perception-node
    guard = ev.index("guard 29")                 # the shutdown deadline: whole minutes left of the 30, counted from the launch
    first_send = next(i for i, e in enumerate(ev) if "objects" in e or "code/" in e)
    down = next(i for i, e in enumerate(ev) if e.startswith(f"sky down -y {cluster}"))
    assert launch < guard < first_send < down == len(ev) - 1
    assert any("--instance-initiated-shutdown-behavior terminate" in e for e in ev)
    t = coord._tasks(run_id)["faces"]
    usage = json.loads(t["usage"])
    assert t["runner"] == "cloud:cuda" and usage["machine_usd"] > 0 and t["cost_usd"] == usage["usd"]
    spent = coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert spent == pytest.approx(usage["usd"])


def test_a_machine_the_guard_refuses_gets_nothing_and_is_deleted(tmp_path, machine, rec):
    coord = coordinator(tmp_path, guard=GUARD_REFUSES)
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "failed"
    t = coord._tasks(run_id)["faces"]
    assert t["error_code"] == "privacy" and "guard.sh refused" in t["error"]
    ev = events(machine)
    assert not any("objects" in e or "code/" in e for e in ev)
    assert ev[-1].startswith("sky down -y perception-")
    assert "not perception-node" in log_of(coord, run_id)


def test_a_machine_that_could_not_start_is_deleted_and_the_attempt_is_a_crash(tmp_path, machine, rec):
    (machine / "launch-fails").touch()
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "failed"
    t = coord._tasks(run_id)["faces"]
    assert t["error_code"] == "transient"            # no capacity now may be capacity later: the policy's retries apply
    assert events(machine)[-1].startswith("sky down -y perception-")   # a half-started machine too


def test_cancel_stops_the_node_and_deletes_the_machine(tmp_path, machine, rec):
    coord = coordinator(tmp_path)
    doc = graph(tmp_path, rec, sleep=60)
    run_id = coord.create_run(compile_graph(doc, coord.registry), doc, allow_uploads=True)

    async def go():
        task = asyncio.create_task(coord.run(run_id))
        pidfile = machine / "home" / ".cache" / "hyperstudy-perception" / "work" / run_id / "faces" / "1" / "pid"
        while not pidfile.exists():   # started there
            await asyncio.sleep(0.05)
        coord.cancel(run_id)
        return await task
    assert asyncio.run(go()) == "canceled"
    deadline = time.time() + 20
    while not events(machine)[-1].startswith("sky down"):   # the driver deletes it after the coordinator has moved on
        assert time.time() < deadline, events(machine)[-3:]
        time.sleep(0.1)
    assert "[remote] stopping it on the cloud machine" in log_of(coord, run_id)


def test_a_node_still_running_near_the_deadline_is_stopped_and_fails_without_a_retry(tmp_path, machine, rec):
    coord = coordinator(tmp_path, deadline_s=8)       # stopped 2 s before: a quarter of the deadline, under 5 minutes
    doc = graph(tmp_path, rec, sleep=60)
    doc["policy"]["retries"] = 2
    run_id, status = run(coord, doc)
    assert status == "failed"
    t = coord._tasks(run_id)["faces"]
    assert (t["error_code"], t["attempt"]) == ("resources", 1)
    assert "deadline" in t["error"] and events(machine)[-1].startswith("sky down")
    spent = coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert spent > 0                                   # the machine's time is charged on a failure too


def test_a_machine_that_could_not_be_deleted_is_charged_its_whole_deadline(tmp_path, machine, rec):
    (machine / "down-fails").touch()
    coord = coordinator(tmp_path, deadline_s=3600)
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "succeeded"                       # the results are good; the warning is in the log
    usage = json.loads(coord._tasks(run_id)["faces"]["usage"])
    assert usage["machine_usd"] == CloudBackend.hourly_usd
    assert "could not delete perception-" in log_of(coord, run_id)


def test_only_gpu_nodes_are_offered_a_cloud_machine_at_its_price_for_the_deadline(tmp_path, machine):
    coord = coordinator(tmp_path)
    work = coord.registry.get("test.work@1.0.0")      # a CPU runner only
    assert [p.label for p in coord.placements(work)[0]] == ["self:cpu"]
    placement, why = coord.place(work, choice="cloud:cuda")
    assert placement is None and "no CUDA runner" in why
    planned = make_plan(coord, compile_graph(graph(tmp_path, "r"), coord.registry))
    node = planned.nodes[0]
    assert node.options[-1] == "cloud:cuda" and all(o.startswith("self:") for o in node.options[:-1])   # this machine's first
    assert node.place == "cloud:cuda"
    assert node.cost_usd == node.estimates["cloud:cuda"]["cost_usd"] == round(CloudBackend.hourly_usd / 2, 4)   # 30 min
    assert planned.places["cloud:cuda"]["title"] == "Cloud GPU (AWS g6.2xlarge, NVIDIA L4)"
    assert planned.max_cost_usd == 5.0                # the graph's cap: what the app raises by a cloud place's price


def test_a_run_whose_cap_cannot_cover_the_machine_never_starts_it(tmp_path, machine, rec):
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec, max_cost=0.10))
    assert status == "failed" and coord._tasks(run_id)["faces"]["error_code"] == "budget"
    assert events(machine) == []


def test_the_deadline_follows_the_videos_length():
    cloud = CloudBackend()
    assert cloud.deadline(None) == 3600
    assert cloud.deadline(60) == 1800                  # at least 30 minutes
    assert cloud.deadline(7200) == 20 * 60 + 7200      # 20 minutes to start, plus the film
    assert cloud.deadline(7201) == cloud.deadline(7290)   # whole 5 minutes: plan and run agree


def test_the_cli_turns_cloud_places_on_after_checking_aws(tmp_path, machine, monkeypatch, capsys):
    from perception.cli import backends, main
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "home"))
    assert not any(b.kind == "cloud" for b in backends())
    assert main(["--json", "cloud", "on"]) == 0
    [event] = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert event["enabled"] is True and event["hourly_usd"] == CloudBackend.hourly_usd
    assert any(b.kind == "cloud" for b in backends())
    assert main(["--json", "cloud", "off"]) == 0
    assert not any(b.kind == "cloud" for b in backends())
