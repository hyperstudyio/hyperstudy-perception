"""Nodes on a cloud machine (backends/cloud.py and the driver's cloud steps), against stand-ins for sky, aws, ssh and
guard.sh that record what they were asked to do (tmp/host/events). The node really runs, "on the machine" (a separate
home on this one, which reports an NVIDIA GPU through a stand-in nvidia-smi)."""

import asyncio
import json
import os
import subprocess
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
# the machine's clock 10 minutes fast, while $FAKE_HOST/skew exists
[ -e "$FAKE_HOST/skew" ] && PATH="$FAKE_HOST/skewbin:$PATH" && export PATH
# a slow link for commands that contain $FAKE_HOST/slow's text
if [ -e "$FAKE_HOST/slow" ]; then case "$*" in *"$(cat "$FAKE_HOST/slow")"*) sleep 8;; esac; fi
cd "$FAKE_HOST/home" && HOME="$FAKE_HOST/home" exec /bin/sh -c "$*"
"""
FAKE_SKY = r"""#!/bin/sh
# SkyPilot's CLI as the driver uses it: a launch is submitted to its API server (--async) and followed by its request id
echo "sky $* (config $SKYPILOT_PROJECT_CONFIG)" >> "$FAKE_HOST/events"
R=11111111-2222-3333-4444-555555555555
case "$1 $2" in
  "launch --async") date +%s > "$FAKE_HOST/launched"; echo "$4" > "$FAKE_HOST/cluster"
                    [ -e "$FAKE_HOST/launch-garbled" ] && { echo "Request submitted (see the dashboard)"; exit 0; }   # no id to read
                    printf '\033[32mSubmitted sky.launch request: %s\033[0m\n' "$R"; exit 0;;
  "api status")
    if [ "$3" = "-v" ]; then   # every active request, with its cluster
      [ -e "$FAKE_HOST/list-garbled" ] && { echo "Internal Server Error"; exit 0; }
      [ -e "$FAKE_HOST/canceled" ] && { echo "[]"; exit 0; }
      printf '[{"request_id": "%s", "name": "sky.launch", "status": "RUNNING", "cluster_name": "%s"}, {"request_id": "99999999-0000-0000-0000-000000000000", "name": "sky.launch", "status": "RUNNING", "cluster_name": "someone-elses"}]\n' "$R" "$(cat "$FAKE_HOST/cluster")"; exit 0
    fi
    if [ -e "$FAKE_HOST/canceled" ]; then st=CANCELLED
    elif [ -e "$FAKE_HOST/launch-fails" ]; then st=FAILED
    elif [ -e "$FAKE_HOST/launch-hangs" ]; then st=RUNNING            # a setup stuck on an apt lock
    elif [ -e "$FAKE_HOST/launch-slow" ] && [ $(( $(date +%s) - $(cat "$FAKE_HOST/launched") )) -lt 5 ]; then st=RUNNING
    else st=SUCCEEDED; fi
    printf 'The flag is ignored because the server does not support it yet.\n[{"request_id": "%s", "name": "sky.launch", "status": "%s"}]\n' "$R" "$st"; exit 0;;
  "api logs") echo "launch logs"; exit 0;;
  "api cancel") [ -e "$FAKE_HOST/cancel-hangs" ] && sleep 60; touch "$FAKE_HOST/canceled"; exit 0;;
esac
case "$1" in
  down) [ -e "$FAKE_HOST/down-fails" ] && exit 1
        [ -e "$FAKE_HOST/down-hangs" ] && { sleep 60; exit 0; }   # a SkyPilot server that does not answer
        # like SkyPilot, a program with its own signal handling: an ignored SIGTERM inherited from the driver does not protect it
        [ -e "$FAKE_HOST/down-slow" ] && { python3 -c 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_DFL); time.sleep(3)' || exit 1
                                           echo "sky down finished" >> "$FAKE_HOST/events"; }; exit 0;;
  check) printf '\033[32m  AWS: enabled [compute, storage]\033[0m\n'; exit 0;;
esac
"""
FAKE_AWS = r"""#!/bin/sh
echo "aws $*" >> "$FAKE_HOST/events"
case "$*" in *terminate-instances*) [ -e "$FAKE_HOST/terminate-fails" ] && exit 254
                                    [ -e "$FAKE_HOST/terminate-hangs" ] && sleep 60; exit 0;; esac
case "$*" in *describe-instances*) echo "i-0abc";; esac
"""
GUARD_OK = 'echo "guard $1" >> "$FAKE_HOST/events"; echo "guard: ok"\n'
SETUP_OK = 'echo "setup" >> "$FAKE_HOST/events"; echo "$HOME/.perception-uv/uv"\n'
GUARD_REFUSES = 'echo "guard: the instance profile is skypilot-v1, not perception-node" >&2; exit 1\n'


def script(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def machine(tmp_path, monkeypatch):
    fake = tmp_path / "host"
    (fake / "home" / ".local" / "bin").mkdir(parents=True)
    script(fake / "home" / ".local" / "bin" / "nvidia-smi", "#!/bin/sh\necho \"0, 23034, NVIDIA L4\"\n")
    monkeypatch.setenv("FAKE_HOST", str(fake))
    monkeypatch.setenv("PERCEPTION_SSH", str(script(tmp_path / "ssh", FAKE_SSH)))
    monkeypatch.setenv("PERCEPTION_SKY", str(script(tmp_path / "sky", FAKE_SKY)))
    monkeypatch.setenv("PERCEPTION_AWS", str(script(tmp_path / "aws", FAKE_AWS)))
    return fake


def events(fake: Path) -> list[str]:
    return (fake / "events").read_text().splitlines() if (fake / "events").exists() else []


def coordinator(tmp_path, guard=GUARD_OK, deadline_s=1800, launch_s=25 * 60, setup=SETUP_OK, down_s=900, aws_s=120) -> Coordinator:
    cloud = CloudBackend(deadline_s=deadline_s, runner_kind="self", python=sys.executable, launch_s=launch_s, down_s=down_s, aws_s=aws_s,
                         guard=script(tmp_path / "guard.sh", guard), setup=script(tmp_path / "node-setup.sh", setup),
                         task=tmp_path / "node-gpu.yaml")
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
    launch = next(i for i, e in enumerate(ev) if e.startswith(f"sky launch --async -c {cluster} -y --down -i 15 --env DEADLINE_EPOCH="))
    at = int(ev[launch].split("DEADLINE_EPOCH=")[1].split()[0])
    assert -70 < at - (time.time() + 1800) <= -55                    # the deadline as a moment, a minute early (clock skew)
    assert ev[launch].endswith(f"(config {CloudBackend.project_config})")   # no credentials, only perception-node
    guard = ev.index("guard 29")                 # the shutdown deadline: whole minutes left of the 30, counted from the launch
    setup = ev.index("setup")                    # ffmpeg and uv, installed and waited for (SkyPilot's setup is detached)
    first_send = next(i for i, e in enumerate(ev) if "objects" in e or "code/" in e)
    down = next(i for i, e in enumerate(ev) if e.startswith(f"sky down -y {cluster}"))
    assert launch < guard < setup < first_send < down == len(ev) - 1
    assert any("--instance-initiated-shutdown-behavior terminate" in e for e in ev)
    t = coord._tasks(run_id)["faces"]
    usage = json.loads(t["usage"])
    assert t["runner"] == "cloud:cuda" and usage["machine_usd"] > 0 and t["cost_usd"] == usage["usd"]
    spent = coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert spent == pytest.approx(usage["usd"])
    # where the time went, in the usage and the log
    assert set(usage["phases"]) >= {"launch", "setup", "code", "inputs", "run", "fetch", "teardown", "startup"}
    assert "[remote] time: launch" in log_of(coord, run_id)


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


def test_a_machine_that_could_not_start_is_deleted_retried_and_charged_only_its_time(tmp_path, machine, rec):
    (machine / "launch-fails").touch()
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "failed"
    t = coord._tasks(run_id)["faces"]
    assert t["error_code"] == "transient"            # no capacity now may be capacity later: the policy's retries apply
    assert events(machine)[-1].startswith("sky down -y perception-")   # a half-started machine too
    spent = coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert spent < 0.01                              # seconds of machine, not the 30 min reserved


def test_a_launch_that_hangs_is_stopped_and_the_machine_set_to_terminate_while_it_was_starting(tmp_path, machine, rec):
    (machine / "launch-hangs").touch()
    coord = coordinator(tmp_path, launch_s=4)
    started = time.time()
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "failed" and time.time() - started < 30
    t = coord._tasks(run_id)["faces"]
    assert t["error_code"] == "transient" and "not ready after" in t["error"]
    ev = events(machine)
    assert any("--instance-initiated-shutdown-behavior terminate" in e for e in ev)   # set during the launch
    assert "sky api cancel -y 11111111-2222-3333-4444-555555555555" in " ".join(ev)   # stopped on SkyPilot's server too
    assert ev[-1].startswith("sky down -y perception-")


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


def test_a_cancel_during_the_deletion_does_not_stop_it(tmp_path, machine, rec):
    import signal
    (machine / "down-slow").touch()
    coord = coordinator(tmp_path)
    doc = graph(tmp_path, rec)
    run_id = coord.create_run(compile_graph(doc, coord.registry), doc, allow_uploads=True)

    async def go():
        task = asyncio.create_task(coord.run(run_id))
        while not any(e.startswith("sky down") for e in events(machine)):
            await asyncio.sleep(0.05)
        await asyncio.sleep(1)                                # sky down is under way, with its own signal handling
        driver = json.loads(coord._tasks(run_id)["faces"]["handle"])["pid"]
        os.killpg(driver, signal.SIGTERM)                     # what a cancel sends: the driver's whole group
        return await task
    asyncio.run(go())
    deadline = time.time() + 15
    while "sky down finished" not in events(machine):
        assert time.time() < deadline, "the deletion was stopped"
        time.sleep(0.1)


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


def test_results_that_take_the_deadlines_margin_to_come_back_are_kept(tmp_path, machine, rec):
    coord = coordinator(tmp_path, deadline_s=12)          # stops nodes after 9 s; the last 3 s are for results
    (machine / "slow").write_text("--sender")              # bringing out/ back takes 8 s, past that
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "succeeded", log_of(coord, run_id)


def test_the_machine_is_set_to_terminate_again_once_the_launch_is_done(tmp_path, machine, rec):
    (machine / "launch-slow").touch()                      # long enough for the watcher to see it while it starts
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "succeeded"
    ev = events(machine)
    terminate = [i for i, e in enumerate(ev) if "--instance-initiated-shutdown-behavior terminate" in e]
    done = next(i for i, e in enumerate(ev) if e.startswith("ssh "))   # the first command on the started machine
    assert len(terminate) >= 2 and terminate[-1] < done   # during the launch, and after it (a replaced instance too)


def test_a_machine_is_never_charged_past_its_deadline(tmp_path, machine, rec):
    (machine / "slow").write_text("--sender")              # results take 8 s to come back, past a 4 s deadline
    coord = coordinator(tmp_path, deadline_s=4)
    run_id, status = run(coord, graph(tmp_path, rec))
    usage = json.loads(coord._tasks(run_id)["faces"]["usage"])
    assert usage["machine_usd"] <= round(CloudBackend.hourly_usd * 4 / 3600, 4)


def test_a_deletion_that_does_not_answer_is_given_up_and_charged_the_whole_deadline(tmp_path, machine, rec):
    (machine / "down-hangs").touch()
    (machine / "terminate-fails").touch()                    # nor could EC2 end it
    coord = coordinator(tmp_path, deadline_s=3600, down_s=2)
    started = time.time()
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "succeeded" and time.time() - started < 40      # not held by the deletion
    assert json.loads(coord._tasks(run_id)["faces"]["usage"])["machine_usd"] == CloudBackend.hourly_usd
    assert "sky down: timed out" in log_of(coord, run_id)


def test_a_machine_sky_down_could_not_delete_is_terminated_through_ec2(tmp_path, machine, rec):
    (machine / "launch-hangs").touch()                       # no deadline set on it yet
    (machine / "down-hangs").touch()
    coord = coordinator(tmp_path, launch_s=1, down_s=2)
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "failed"
    assert any("ec2 terminate-instances --instance-ids i-0abc" in e for e in events(machine))
    cluster = cluster_name(run_id, "faces", 1)
    # only this attempt's cluster (or SkyPilot's name for it, with a suffix): never perception-...-10 for -1
    lookups = [e for e in events(machine) if "describe-instances" in e]
    assert lookups and all(f"Values={cluster},{cluster}-*" in e for e in lookups)
    assert coord._tasks(run_id)["faces"]["error_code"] == "transient"   # it is gone: another attempt may start one


def test_a_machine_with_no_deadline_that_could_not_be_deleted_at_all_stops_the_retries(tmp_path, machine, rec):
    for f in ("launch-hangs", "down-hangs", "terminate-fails"):
        (machine / f).touch()
    coord = coordinator(tmp_path, launch_s=1, down_s=2)
    doc = graph(tmp_path, rec)
    doc["policy"]["retries"] = 2
    run_id, status = run(coord, doc)
    t = coord._tasks(run_id)["faces"]
    assert (status, t["error_code"], t["attempt"]) == ("failed", "internal", 1)   # no second machine
    assert "may still be running" in t["error"] and "AWS console" in t["error"]


def test_a_machine_sky_down_and_ec2_could_not_delete_is_charged_its_whole_deadline(tmp_path, machine, rec):
    (machine / "down-fails").touch()
    (machine / "terminate-fails").touch()
    coord = coordinator(tmp_path, deadline_s=3600)
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "succeeded"                       # the results are good; its deadline was set; the warning is in the log
    usage = json.loads(coord._tasks(run_id)["faces"]["usage"])
    assert usage["machine_usd"] == CloudBackend.hourly_usd
    assert "could not delete perception-" in log_of(coord, run_id)


def test_a_machine_sky_down_could_not_delete_but_ec2_did_is_charged_its_time(tmp_path, machine, rec):
    (machine / "down-fails").touch()
    coord = coordinator(tmp_path, deadline_s=3600)
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "succeeded" and any("terminate-instances" in e for e in events(machine))
    assert json.loads(coord._tasks(run_id)["faces"]["usage"])["machine_usd"] < 0.01


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
    assert planned.places["cloud:cuda"]["title"] == "Cloud GPU (AWS L4)"
    assert planned.max_cost_usd == 5.0                # the graph's cap: what the app raises by a cloud place's price


def test_a_run_whose_cap_cannot_cover_the_machine_never_starts_it(tmp_path, machine, rec):
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec, max_cost=0.10))
    assert status == "failed" and coord._tasks(run_id)["faces"]["error_code"] == "budget"
    assert events(machine) == []


def test_nodes_whose_shortened_ids_agree_get_machines_of_their_own():
    assert cluster_name("20261024-1-abc123", "face_track_a", 1) != cluster_name("20261024-1-abc123", "face-track-a", 1)
    assert cluster_name("r-abc123", "face_track_aaaaaaaaaaa1", 1) != cluster_name("r-abc123", "face_track_aaaaaaaaaaa2", 1)
    assert cluster_name("r-abc123", "faces", 2).startswith("perception-abc123-faces-")


def test_a_launch_whose_request_could_not_be_read_is_canceled_by_its_cluster_only(tmp_path, machine, rec):
    (machine / "launch-garbled").touch()
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "failed" and coord._tasks(run_id)["faces"]["error_code"] == "transient"
    ev = " ".join(events(machine))
    assert "sky api cancel -y 11111111-2222-3333-4444-555555555555" in ev     # this attempt's launch
    assert "99999999" not in ev.split("sky api cancel")[-1]                  # never another cluster's
    assert events(machine)[-1].startswith("sky down -y perception-")


def test_a_machine_whose_clock_disagrees_is_not_used(tmp_path, machine, rec):
    (machine / "skewbin").mkdir()
    script(machine / "skewbin" / "date", '#!/bin/sh\n[ "$1" = "+%s" ] && { echo $(( $(/bin/date +%s) + 600 )); exit 0; }\nexec /bin/date "$@"\n')
    (machine / "skew").touch()
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec))
    t = coord._tasks(run_id)["faces"]
    assert (status, t["error_code"]) == ("failed", "internal") and "clock" in t["error"]
    assert not any("objects" in e or "code/" in e for e in events(machine))   # nothing was sent
    assert events(machine)[-1].startswith("sky down -y perception-")


def test_a_machine_whose_deadline_was_never_set_is_charged_all_its_time(tmp_path, machine, rec):
    (machine / "launch-hangs").touch()                       # stopped before setup or guard.sh ran
    (machine / "down-slow").touch()                          # and its deletion takes 3 s, past a 2 s deadline
    coord = coordinator(tmp_path, deadline_s=2, launch_s=1)
    run_id, status = run(coord, graph(tmp_path, rec))
    spent = coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    assert spent > round(CloudBackend.hourly_usd * 2 / 3600, 4)   # every second it existed, not capped at the deadline


def test_ec2_not_answering_while_ending_a_machine_with_no_deadline_still_stops_the_retries(tmp_path, machine, rec):
    for f in ("launch-hangs", "down-hangs", "terminate-hangs"):
        (machine / f).touch()
    coord = coordinator(tmp_path, launch_s=1, down_s=2, aws_s=2)
    doc = graph(tmp_path, rec)
    doc["policy"]["retries"] = 2
    run_id, status = run(coord, doc)
    t = coord._tasks(run_id)["faces"]
    assert (status, t["error_code"], t["attempt"]) == ("failed", "internal", 1) and "may still be running" in t["error"]


def test_a_cancel_that_hangs_on_skypilots_server_does_not_keep_the_machine_from_being_deleted(tmp_path, machine, rec):
    for f in ("launch-hangs", "cancel-hangs"):
        (machine / f).touch()
    coord = coordinator(tmp_path, launch_s=1, aws_s=2)
    started = time.time()
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "failed" and time.time() - started < 40
    ev = events(machine)
    down = next(i for i, e in enumerate(ev) if e.startswith("sky down -y perception-"))
    assert any("terminate-instances" in e for e in ev[down:])   # and, the cancel unconfirmed, through EC2 too


def test_a_launch_that_could_not_be_stopped_is_not_counted_as_deleted_and_stops_the_retries(tmp_path, machine, rec):
    # SkyPilot's server hangs: the cancel times out and sky down too; EC2 shows no instance (not created yet)
    for f in ("launch-hangs", "cancel-hangs", "down-hangs"):
        (machine / f).touch()
    script(Path(os.environ["PERCEPTION_AWS"]), FAKE_AWS.replace('echo "i-0abc"', ':'))   # describe finds nothing
    coord = coordinator(tmp_path, launch_s=1, down_s=2, aws_s=2)
    doc = graph(tmp_path, rec)
    doc["policy"]["retries"] = 2
    run_id, status = run(coord, doc)
    t = coord._tasks(run_id)["faces"]
    assert (status, t["error_code"], t["attempt"]) == ("failed", "internal", 1)   # no second machine
    assert "still be created" in t["error"]


def test_a_launch_that_could_not_be_stopped_stops_the_retries_even_when_sky_down_says_it_is_gone(tmp_path, machine, rec):
    for f in ("launch-hangs", "cancel-hangs"):                # sky down exits 0: "Cluster(s) not found", not created yet
        (machine / f).touch()
    coord = coordinator(tmp_path, launch_s=1, aws_s=2)
    doc = graph(tmp_path, rec)
    doc["policy"]["retries"] = 2
    run_id, status = run(coord, doc)
    t = coord._tasks(run_id)["faces"]
    assert (status, t["error_code"], t["attempt"]) == ("failed", "internal", 1) and "still be created" in t["error"]
    assert any("terminate-instances" in e for e in events(machine))   # what EC2 can see is ended all the same


def test_a_launch_with_no_request_id_that_could_not_be_listed_stops_the_retries(tmp_path, machine, rec):
    for f in ("launch-garbled", "list-garbled"):            # no request id, and SkyPilot's request list unreadable
        (machine / f).touch()
    script(Path(os.environ["PERCEPTION_AWS"]), FAKE_AWS.replace('echo "i-0abc"', ':'))   # EC2 shows nothing yet
    coord = coordinator(tmp_path)
    doc = graph(tmp_path, rec)
    doc["policy"]["retries"] = 2
    run_id, status = run(coord, doc)
    t = coord._tasks(run_id)["faces"]
    assert (status, t["error_code"], t["attempt"]) == ("failed", "internal", 1) and "still be created" in t["error"]


def test_setups_shutdown_is_the_minutes_left_and_never_later():
    import re
    import yaml
    setup = yaml.safe_load(CloudBackend.task.read_text())["setup"]
    line = next(l for l in setup.splitlines() if "left=" in l) + "\n" + next(l for l in setup.splitlines() if "shutdown -h" in l)

    def shutdown(epoch):
        out = subprocess.run(["bash", "-c", "sudo() { echo \"$@\"; }\n" + line], capture_output=True, text=True,
                             env={**os.environ, **({"DEADLINE_EPOCH": str(epoch)} if epoch is not None else {})})
        return re.search(r"shutdown -h (\S+)", out.stdout).group(1)
    assert shutdown(int(time.time()) + 1799) == "+29"         # rounded down: never after the deadline
    assert shutdown(int(time.time()) - 30) == "+0"            # already past (setup ran late): now, not tomorrow
    assert shutdown(None) == "+0"                             # no deadline given: now


def test_a_machine_that_cannot_be_set_to_terminate_is_not_used(tmp_path, machine, rec):
    script(Path(os.environ["PERCEPTION_AWS"]), FAKE_AWS.replace('case "$*" in', 'case "$*" in *modify-instance-attribute*) exit 254;; esac\ncase "$*" in'))
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec))
    assert status == "failed"
    t = coord._tasks(run_id)["faces"]
    assert (t["error_code"], t["attempt"]) == ("internal", 1) and "terminate on shutdown" in t["error"]
    assert not any("objects" in e or "code/" in e for e in events(machine))   # nothing was sent to it
    assert events(machine)[-1].startswith("sky down -y perception-")


def test_a_machine_whose_setup_fails_is_a_launch_failure_charged_its_time(tmp_path, machine, rec):
    coord = coordinator(tmp_path, setup='echo "E: Could not get lock /var/lib/dpkg/lock-frontend" >&2; exit 100\n')
    run_id, status = run(coord, graph(tmp_path, rec))
    t = coord._tasks(run_id)["faces"]
    assert (status, t["error_code"]) == ("failed", "transient") and "setting up" in t["error"]
    assert not any("objects" in e or "code/" in e for e in events(machine))   # nothing sent to a machine not set up
    assert "Could not get lock" in log_of(coord, run_id)
    assert coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0] < 0.01


def test_a_machine_with_no_gpu_fails_once_charged_its_time(tmp_path, machine, rec):
    (machine / "home" / ".local" / "bin" / "nvidia-smi").unlink()
    coord = coordinator(tmp_path)
    doc = graph(tmp_path, rec)
    doc["policy"]["retries"] = 2
    run_id, status = run(coord, doc)
    t = coord._tasks(run_id)["faces"]
    assert (status, t["error_code"], t["attempt"]) == ("failed", "resources", 1)
    assert coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0] < 0.01


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
    capsys.readouterr()
    from perception.store import Store
    store = Store(tmp_path / "home")
    store.db.execute("INSERT INTO runs (id, pipeline, graph, inputs, status, created_at, max_cost_usd) VALUES ('r1', 't', '{}', '{}', 'running', 0, 1)")
    store.db.execute("INSERT INTO tasks (run_id, node_id, state, runner) VALUES ('r1', 'faces', 'running', 'cloud:cuda')")
    assert main(["--json", "cloud", "off"]) == 1                 # it could no longer be stopped, nor its machine deleted
    assert "run r1 has faces on a cloud machine" in capsys.readouterr().out
    store.db.execute("UPDATE tasks SET state='succeeded'")
    assert main(["--json", "cloud", "off"]) == 0
    assert not any(b.kind == "cloud" for b in backends())


def test_every_real_nodes_cuda_runner_fits_the_cloud_gpus_it_would_be_placed_on():
    """Placement refuses a runner whose minimum_vram_gb is over the place's vram_gb: a CUDA runner asking for more than
    the L4 (22) or RunPod's cards (20) report is never placed in the cloud, and a Linux run falls back to its CPU runner."""
    from perception.backends.cloud import RunPodBackend
    nodes = Path(__file__).resolve().parents[2] / "nodes"
    for m in Registry.load(nodes):
        for r in m.runners:
            if r.accelerator == "cuda":
                assert r.minimum_vram_gb <= min(CloudBackend.caps.vram_gb, RunPodBackend.caps.vram_gb), m.ref
