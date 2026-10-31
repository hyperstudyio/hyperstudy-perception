"""A whole run on a RunPod pod made for it (pod_run.py, `perception remote run runpod`): RunPod's API stood in
(test_runpod.FakeRunPod), and the pod a stand-in host as in test_remote_run (each ssh command runs here, in the pod's
own home, with nothing of this environment but PATH). Everything else is real: the pod's checks, the pipeline's copy,
the coordinator there, the run, the import, the pod's removal."""

import json
import sys
import time
from pathlib import Path

import pytest

from perception import pod_run, remote_run
from perception.backends.cloud import Cloud
from perception.pod_session import removal_at_boot

from test_cloud import GUARD_OK, GUARD_REFUSES, SETUP_OK, script
from test_remote_run import KEY, files_holding, graph_file, keyed, of, perception, starts, wait_for
from test_remote_run import work as remote_work
from test_runpod import KEY as RUNPOD_KEY, FakeRunPod

# test_remote_run's ssh stand-in, which also takes a pod's -F none, and gives the coordinator there where the pod's
# removal reads its time (pod_session.REMOVE_AT, at / on a pod)
FAKE_SSH = r"""#!/bin/sh
while [ $# -gt 0 ]; do case "$1" in -o|-l|-p|-i|-F) shift 2;; -*) shift;; *) break;; esac; done
echo "ssh $1: $2" | head -c 300 >> "$FAKE_HOST/events"; echo >> "$FAKE_HOST/events"
shift
printf '%s\n' "$*" >> "$FAKE_HOST/argv.log"
# the pod out of reach (sshd gone, the network down): every command, while $FAKE_HOST/unreachable exists
if [ -e "$FAKE_HOST/unreachable" ]; then echo "ssh: connect to host pod: Connection timed out" >&2; exit 255; fi
case "$*" in
  *"remote_coordinator start"*)
    tee -a "$FAKE_RECORD/start.stdin" | env -i FAKE_HOST="$FAKE_HOST" HOME="$FAKE_HOST/home" PATH="$FAKE_HOST/home/.local/bin:$PATH" PERCEPTION_REMOVE_AT="$FAKE_HOST/remove-at" /bin/sh -c "cd \"\$HOME\" && $*" > "$FAKE_HOST/start.out"
    code=$?
    # its answer lost on the way back, the run started there all the same
    if [ -e "$FAKE_HOST/lose-start-answer" ]; then echo "Connection reset by peer" >&2; exit 255; fi
    cat "$FAKE_HOST/start.out"; exit $code;;
esac
cd "$FAKE_HOST/home" && exec env -i FAKE_HOST="$FAKE_HOST" HOME="$FAKE_HOST/home" PATH="$FAKE_HOST/home/.local/bin:$PATH" PERCEPTION_REMOVE_AT="$FAKE_HOST/remove-at" /bin/sh -c "$*"
"""


@pytest.fixture
def runpod(tmp_path, monkeypatch):
    """RunPod on (its key set here), a pod that is a stand-in host (tmp_path/host/home), this machine's store in
    tmp_path/mac and TEST_API_KEY set here."""
    fake = tmp_path / "host"
    (fake / "home" / ".local" / "bin").mkdir(parents=True)
    script(fake / "home" / ".local" / "bin" / "nvidia-smi", "#!/bin/sh\necho \"0, 23034, NVIDIA L4\"\n")
    record = tmp_path.parent / f"{tmp_path.name}-record"   # outside tmp_path: the searches for the key skip it
    record.mkdir()
    monkeypatch.setenv("PERCEPTION_SSH", str(script(tmp_path / "ssh", FAKE_SSH)))
    monkeypatch.setenv("FAKE_HOST", str(fake))
    monkeypatch.setenv("FAKE_RECORD", str(record))
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "mac"))
    monkeypatch.setenv("TEST_API_KEY", KEY)
    monkeypatch.setenv("RUNPOD_API_KEY", RUNPOD_KEY)
    api = FakeRunPod(fake)
    monkeypatch.setenv("PERCEPTION_RUNPOD_API", api.url)
    # the pod's Python: this interpreter (it has the orchestrator's dependencies), not an environment made there
    monkeypatch.setattr(remote_run.Remote, "interpreter", property(lambda self: sys.executable))
    real_init = remote_run.Remote.__init__
    monkeypatch.setattr(remote_run.Remote, "__init__",
                        lambda self, host, nodes=None, python=sys.executable: real_init(self, host, nodes, python))
    real_spec = pod_run.spec   # the guard and setup stand-ins (test_cloud's), the time limits short
    guard, setup = script(tmp_path / "guard.sh", GUARD_OK), script(tmp_path / "node-setup.sh", SETUP_OK)
    monkeypatch.setattr(pod_run, "spec", lambda deadline, gpus=None: {**real_spec(deadline, gpus), "guard": str(guard), "setup": str(setup),
                                                           "launch_s": 60, "down_s": 60, "unknown_s": 5})
    Cloud(tmp_path / "mac").set(True, "runpod")
    yield api
    api.server.shutdown()


def work(tmp_path, node_id, a="$inputs.seed", **config):
    rec = tmp_path / "work-record"
    rec.mkdir(exist_ok=True)
    return remote_work(node_id, a, record_dir=str(rec), **config)


def events(fake: Path) -> list[str]:
    return (fake / "events").read_text().splitlines() if (fake / "events").exists() else []


def record_of(run: str) -> dict:
    return json.loads(remote_run.record(run).read_text())


def test_a_run_on_a_pod_made_for_it_is_imported_and_the_pod_deleted(tmp_path, runpod, capsys):
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a"), work(tmp_path, "b", "$nodes.a.out")])),
                           "--allow-uploads", "--max-cost-usd", "5")
    assert code == 0, [(o.get("node"), o.get("kind"), o.get("message")) for o in out if o["type"] in ("event", "error")]
    run = of(out, "run.created")[0]["run"]
    [made] = of(out, "pod")
    # the pod: the cheaper GPUs, its own disk, its removal at its deadline set from its first moment (in its name too)
    [body] = runpod.bodies
    assert set(body["gpuTypeIds"]) == set(pod_run.GPUS) and max(pod_run.GPUS.values()) <= 0.60
    assert body["containerDiskInGb"] == pod_run.DISK_GB and body["volumeInGb"] == 0 and body["cloudType"] == "SECURE"
    assert body["name"] == f"perception-{run.rsplit('-', 1)[-1]}-run-d{made['deadline_at']}"
    assert body["dockerStartCmd"] == ["bash", "-c", removal_at_boot(made["deadline_at"] - 60, pod_run.RunPodBackend.start)]
    # checked (guard.sh) before anything was sent, then the run there, imported, and the pod deleted
    ev = events(tmp_path / "host")
    assert ev.index("guard 580") < ev.index("setup") < next(i for i, e in enumerate(ev) if "rsync" in e or "remote_coordinator" in e)
    # the uv the setup installed (or the image has) is the one the run's environment is made with: it is not on PATH
    assert record_of(run)["pod"]["caps"]["tools"]["uv"] == str(tmp_path / "host" / "home" / ".perception-uv" / "uv")
    assert of(out, "fetched")[0]["run"] == run and of(out, "run.finished")[-1]["status"] == "succeeded"
    [deleted] = of(out, "pod_deleted")
    assert deleted["deleted"] and f"api DELETE /pods/{made['pod']}" in ev and not runpod.pods
    rec = record_of(run)["pod"]
    assert rec["deleted"] and rec["fetched"] and rec["cost_usd"] == deleted["cost_usd"] >= 0
    # the run here, with what the pod cost in its spend
    code, out = perception(capsys, "status", run)
    assert of(out, "status")[0]["status"] == "succeeded" and of(out, "status")[0]["spent_usd"] >= rec["cost_usd"]
    # and the coordinator there set the pod's removal for the end of its linger window when the run ended
    removal = int((tmp_path / "host" / "remove-at").read_text())
    assert abs(removal - (time.time() + 30 * 60)) < 120
    # the pod is gone: asked again, nothing is reached
    code, out = perception(capsys, "remote", "status", "runpod", run)
    assert code == 1 and "deleted after its run was fetched" in of(out, "error")[0]["problems"][0]


def test_the_coordinator_there_gets_the_runs_cap_less_the_pods_reservation(tmp_path, runpod, capsys):
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a")])),
                           "--allow-uploads", "--max-cost-usd", "5", "--linger-min", "10")
    assert code == 0, out
    [spec] = starts(tmp_path)
    deadline = pod_run.deadline_s(None, 600)
    assert spec["graph"]["policy"]["max_cost_usd"] == pytest.approx(5 - pod_run.reservation_usd(deadline))
    assert spec["linger_s"] == 600
    assert deadline == max(pod_run.MIN_DEADLINE_S, pod_run.SETUP_S + 3600 + 600)   # no video: an hour for the steps


def test_a_run_whose_cap_would_not_cover_its_pod_is_refused_before_a_pod_is_made(tmp_path, runpod, capsys):
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a")])),
                           "--allow-uploads", "--max-cost-usd", "0.5")
    assert code == 1 and not runpod.bodies
    [plan] = of(out, "remote_plan")
    assert not plan["ok"] and plan["pod"]["reserved_usd"] > 0.5
    assert any("--max-cost-usd" in p and "reserved" in p for p in plan["problems"])


def test_a_dry_run_says_what_the_pod_would_cost_and_makes_none(tmp_path, runpod, capsys):
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a")])),
                           "--allow-uploads", "--max-cost-usd", "5", "--dry-run")
    assert code == 0 and not runpod.bodies
    pod = of(out, "remote_plan")[0]["pod"]
    assert pod["hourly_usd"] == pod_run.hourly_usd() and pod["reserved_usd"] == pod_run.reservation_usd(pod["deadline_s"])
    assert pod["api_cap_usd"] == pytest.approx(5 - pod["reserved_usd"]) and pod["disk_gb"] == pod_run.DISK_GB


def test_runpod_off_makes_no_pod(tmp_path, runpod, capsys):
    Cloud(tmp_path / "mac").set(False, "runpod")
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a")])),
                           "--allow-uploads", "--max-cost-usd", "5")
    assert code == 1 and not runpod.bodies
    assert any("perception cloud on runpod" in p for p in of(out, "remote_plan")[0]["problems"])


def test_a_pods_run_gets_keys_only_with_their_own_yes_and_never_the_accounts(tmp_path, runpod, capsys):
    rec = tmp_path / "keyed-record"
    rec.mkdir()
    graph = str(graph_file(tmp_path, [keyed("k", "$inputs.seed", str(rec))]))
    code, out = perception(capsys, "remote", "run", "runpod", graph, "--allow-uploads", "--max-cost-usd", "5")
    assert code == 1 and not runpod.bodies   # refused here, before a pod is made
    assert any("perception cloud keys runpod on" in p for p in of(out, "remote_plan")[0]["problems"])
    code, out = perception(capsys, "cloud", "keys", "runpod", "on")
    assert code == 0 and of(out, "cloud")[0]["runpod"]["keys"] is True
    code, out = perception(capsys, "remote", "run", "runpod", graph, "--allow-uploads", "--max-cost-usd", "5")
    assert code == 0, out
    [spec] = starts(tmp_path)
    assert spec["keys"] == {"TEST_API_KEY": KEY}   # the step's key, and never the RunPod account's
    stdin = (tmp_path.parent / f"{tmp_path.name}-record" / "start.stdin").read_text()
    assert RUNPOD_KEY not in stdin and not files_holding(tmp_path / "host", RUNPOD_KEY)
    assert RUNPOD_KEY not in (tmp_path / "host" / "argv.log").read_text()


def test_a_pod_the_guard_refuses_gets_nothing_and_is_deleted(tmp_path, runpod, capsys, monkeypatch):
    spec = pod_run.spec
    guard = script(tmp_path / "refuses.sh", GUARD_REFUSES)
    monkeypatch.setattr(pod_run, "spec", lambda deadline, gpus=None: {**spec(deadline, gpus), "guard": str(guard)})
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a")])),
                           "--allow-uploads", "--max-cost-usd", "5")
    assert code == 1 and "guard.sh refused" in of(out, "error")[0]["problems"][0]
    ev = events(tmp_path / "host")
    assert not any("rsync" in e or "remote_coordinator" in e for e in ev) and "api DELETE /pods/pod1" in ev
    assert not runpod.pods


def test_a_detached_run_is_reached_by_its_pod_and_canceling_it_deletes_the_pod(tmp_path, runpod, capsys):
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a", sleep=30)])),
                           "--allow-uploads", "--max-cost-usd", "5", "--detach")
    assert code == 0, out
    run = of(out, "run.created")[0]["run"]
    assert runpod.pods and not of(out, "pod_deleted")
    code, out = perception(capsys, "remote", "status", "runpod")
    assert [p["run"] for p in of(out, "pods")[0]["pods"]] == [run]
    code, out = perception(capsys, "remote", "status", "runpod", run)
    assert code == 0 and of(out, "status")[0]["status"] == "running"
    code, out = perception(capsys, "remote", "cancel", "runpod", run)
    assert code == 0 and of(out, "canceled")[0]["canceled"] and of(out, "pod_deleted")[0]["deleted"]
    assert not runpod.pods and record_of(run)["pod"]["deleted"]


def test_a_run_whose_pod_is_gone_is_never_run_again_on_another(tmp_path, runpod, capsys):
    """Review of #53: `--run-id` on a run already fetched (its pod deleted) is refused, as a lab server answers that
    it ended: a new pod would run every step again, API spend included, past the cap that saw one reservation."""
    graph = str(graph_file(tmp_path, [work(tmp_path, "a")]))
    code, out = perception(capsys, "remote", "run", "runpod", graph, "--allow-uploads", "--max-cost-usd", "5")
    assert code == 0, out
    run = of(out, "run.created")[0]["run"]
    code, out = perception(capsys, "remote", "run", "runpod", graph, "--allow-uploads", "--max-cost-usd", "5", "--run-id", run)
    assert code == 1 and len(runpod.bodies) == 1
    assert "pod was deleted" in of(out, "error")[0]["problems"][0]


def test_a_run_that_fails_on_its_pod_is_fetched_and_its_pod_deleted(tmp_path, runpod, capsys):
    """Review of #53: any end (failed, canceled), not only success, brings the run back and deletes the pod, so it
    stops billing and its cost is in the run's spend; the exit code still says it failed."""
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a", fail="input", fail_until=5)])),
                           "--allow-uploads", "--max-cost-usd", "5")
    assert code == 1 and of(out, "run.finished")[-1]["status"] == "failed"
    assert of(out, "fetched") and of(out, "pod_deleted")[0]["deleted"] and not runpod.pods


def test_the_imported_run_keeps_the_cap_it_was_given(tmp_path, runpod, capsys):
    """Review of #53: its spend here includes the pod, so its cap here is the run's own, not the one the coordinator
    there had (that less the reservation)."""
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a")])),
                           "--allow-uploads", "--max-cost-usd", "5")
    run = of(out, "run.created")[0]["run"]
    code, out = perception(capsys, "status", run)
    assert of(out, "status")[0]["max_cost_usd"] == 5


def test_a_pods_gpus_have_the_memory_the_graphs_steps_need():
    """Only GPUs with at least the largest minimum_vram_gb of the graph's CUDA runners, up to $0.60/h: the sounds step
    needs 23 GB for its longest piece (measured on an RTX 3090, 2026-10-31), so the 20 GB cards and the L4 are not
    offered for the full graph; reserved at the dearest of those left."""
    from perception.backends.cloud import RunPodBackend
    assert set(RunPodBackend.gpus) == set(RunPodBackend.gpu_vram_gb)
    gpus = pod_run.gpus_for(23)
    assert gpus and all(RunPodBackend.gpu_vram_gb[g] >= 23 and usd <= 0.60 for g, usd in gpus.items())
    assert not {"NVIDIA RTX A4500", "NVIDIA RTX 4000 Ada Generation", "NVIDIA L4"} & set(gpus)
    assert set(pod_run.gpus_for(0)) == set(pod_run.GPUS) and pod_run.hourly_usd(gpus) == max(gpus.values())
    assert pod_run.gpus_for(500) == {}


def test_the_full_graph_needs_what_its_sounds_step_does():
    from perception.manifests import Registry
    reg = Registry.load(Path(pod_run.__file__).resolve().parents[3] / "nodes")
    sounds = reg.get("audio.sounds@1.1.0")
    assert max(r.minimum_vram_gb for r in sounds.runners if r.accelerator == "cuda") == 23



def detached(tmp_path, capsys, sleep=30) -> str:
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a", sleep=sleep)])),
                           "--allow-uploads", "--max-cost-usd", "5", "--detach")
    assert code == 0, out
    return of(out, "run.created")[0]["run"]


def test_canceling_a_pod_run_that_cannot_be_reached_still_deletes_its_pod(tmp_path, runpod, capsys):
    """Review of #53: cancel is "stop spending", and RunPod's API deletes the pod without reaching it."""
    run = detached(tmp_path, capsys)
    (tmp_path / "host" / "unreachable").touch()
    code, out = perception(capsys, "remote", "cancel", "runpod", run)
    assert of(out, "pod_deleted")[0]["deleted"] and not runpod.pods and record_of(run)["pod"]["deleted"]
    assert code == 1 and "deleting it ended the run" in of(out, "error")[0]["problems"][0]


def test_a_canceled_pod_run_is_imported_with_what_it_spent(tmp_path, runpod, capsys):
    """Review of #53: the run there has ended, so it is brought back (no bundle) before its pod goes, and what it and
    its pod cost are this store's."""
    run = detached(tmp_path, capsys)
    code, out = perception(capsys, "remote", "cancel", "runpod", run)
    assert code == 0 and of(out, "canceled")[0]["canceled"] and of(out, "fetched")[0]["run"] == run
    pod = of(out, "pod_deleted")[0]
    code, out = perception(capsys, "status", run)
    [st] = of(out, "status")
    assert st["status"] == "canceled" and st["spent_usd"] >= pod["cost_usd"] and st["max_cost_usd"] == 5
    assert record_of(run)["pod"]["charged_usd"] == pod["cost_usd"]


def test_every_line_a_json_pod_run_prints_is_json(tmp_path, runpod, capsys):
    """Review of #53: the pod's progress (its launch, guard.sh, setup, deletion) goes to stderr, never into the --json
    stream (docs/cli-json.md: every line on stdout is one JSON object)."""
    from perception import cli
    code = cli.main(["--json", "--nodes", str(Path(__file__).parent / "nodes"), "remote", "run", "runpod",
                     str(graph_file(tmp_path, [work(tmp_path, "a")])), "--allow-uploads", "--max-cost-usd", "5"])
    captured = capsys.readouterr()
    assert code == 0
    for line in captured.out.splitlines():
        json.loads(line)
    assert "starting a RunPod pod" in captured.err and "guard: ok" in captured.err


def test_a_start_whose_answer_was_lost_keeps_its_pod_for_the_run_there(tmp_path, runpod, capsys):
    """Review of #53: the run may have started there; its pod is left to it (attach follows it, cancel ends it), never
    deleted with it half done."""
    (tmp_path / "host" / "lose-start-answer").touch()
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a")])),
                           "--allow-uploads", "--max-cost-usd", "5")
    run = of(out, "run.created")[0]["run"]
    assert code == 1 and not of(out, "pod_deleted") and runpod.pods
    assert "remote attach runpod" in of(out, "error")[0]["problems"][-1]
    (tmp_path / "host" / "lose-start-answer").unlink()
    code, out = perception(capsys, "remote", "attach", "runpod", run)   # it ran there: followed, imported, deleted
    assert code == 0 and of(out, "fetched") and of(out, "pod_deleted")[0]["deleted"] and not runpod.pods


def test_a_step_that_needs_a_key_a_pod_is_never_given_is_refused_before_a_pod(tmp_path, runpod, capsys, monkeypatch):
    """Review of #53: refused up front, naming the step and the key, not started without it on a pod already paid for."""
    monkeypatch.setattr(pod_run, "NEVER_SENT", (*pod_run.NEVER_SENT, "TEST_API_KEY"))
    perception(capsys, "cloud", "keys", "runpod", "on")
    rec = tmp_path / "keyed-record"
    rec.mkdir()
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [keyed("k", "$inputs.seed", str(rec))])),
                           "--allow-uploads", "--max-cost-usd", "5")
    assert code == 1 and not runpod.bodies
    assert any("k needs TEST_API_KEY" in p and "never sent to a pod" in p for p in of(out, "remote_plan")[0]["problems"])


def test_api_steps_whose_units_are_not_known_yet_are_estimated_from_the_videos_length():
    """Review of #53: a fresh run's shots and utterances are known only after its steps run; the check before a pod is
    made counts them from the video (more than OOO's: 0.30-0.36 shots and 0.55-0.72 utterances a second)."""
    from perception.manifests import Cost
    assert pod_run.unknown_units_usd(Cost(usd=0.01, per="shot", input="shots"), {}, 100) == pytest.approx(0.5)
    assert pod_run.unknown_units_usd(Cost(usd=0.008, per="utterance", input="transcript"), {}, 100) == pytest.approx(0.8)
    assert pod_run.unknown_units_usd(Cost(usd=0.04, per="minute", input="media"), {}, 120) == pytest.approx(0.08)
    assert pod_run.unknown_units_usd(Cost(usd={"a": 0.1}, by="model", per="shot", input="shots"), {"model": "a"}, 10) == pytest.approx(0.5)


def test_a_pod_that_could_not_be_deleted_is_charged_to_its_run_once(tmp_path, runpod, capsys, monkeypatch):
    """Review of #53: its whole deadline is charged when RunPod would not delete it; a fetch asked again adds nothing."""
    spec = pod_run.spec
    monkeypatch.setattr(pod_run, "spec", lambda deadline, gpus=None: {**spec(deadline, gpus), "down_s": 2})
    run = detached(tmp_path, capsys, sleep=0)
    runpod.flags.add("delete-fails")
    assert wait_for(lambda: perception(capsys, "remote", "status", "runpod", run)[1][0].get("status") == "succeeded")
    code, out = perception(capsys, "remote", "fetch", "runpod", run)
    assert code == 0 and not of(out, "pod_deleted")[0]["deleted"]
    spent = of(perception(capsys, "status", run)[1], "status")[0]["spent_usd"]
    code, out = perception(capsys, "remote", "fetch", "runpod", run)
    assert of(perception(capsys, "status", run)[1], "status")[0]["spent_usd"] == spent
    runpod.flags.discard("delete-fails")
    perception(capsys, "remote", "cancel", "runpod", run)   # clean up: deleted now
    assert not runpod.pods


def test_a_run_id_recorded_on_another_host_is_refused_before_a_pod_is_made(tmp_path, runpod, capsys):
    """Review of #53: a run's id names one run (its host, graph and inputs); the record's refusal comes before a pod
    is made, not after one is paid for."""
    run = "20261031-120000-abc123"
    remote_run.record(run).parent.mkdir(parents=True, exist_ok=True)
    remote_run.record(run).write_text(json.dumps({"host": "lab", "inputs": {}, "input_sha256": {}, "graph_doc": {}}))
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a")])),
                           "--allow-uploads", "--max-cost-usd", "5", "--run-id", run)
    assert code == 1 and not runpod.bodies
    assert "another graph" in of(out, "error")[0]["problems"][0]


@pytest.mark.parametrize("linger", ["-5", "nan", "inf", "100000"])
def test_a_linger_window_that_is_not_a_number_of_minutes_is_refused(tmp_path, runpod, capsys, linger):
    """Review of #53: a negative window would shorten the deadline below what the run needs; nan and inf are no time."""
    with pytest.raises(SystemExit):
        perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a")])),
                   "--allow-uploads", "--max-cost-usd", "5", "--linger-min", linger)
    assert not runpod.bodies


def test_without_the_videos_length_api_units_are_estimated_for_the_hour_the_deadline_assumes():
    """Review of #53: no length (an input that is not a video) is not $0 of API steps: the hour deadline_s assumes."""
    from perception.manifests import Cost
    assert pod_run.unknown_units_usd(Cost(usd=0.01, per="shot", input="shots"), {}, None) == pytest.approx(0.01 * 0.5 * 3600)


def test_a_pod_whose_launch_failed_and_could_not_be_deleted_is_recorded_for_cancel(tmp_path, runpod, capsys, monkeypatch):
    """Review of #53: refused by guard.sh (its removal at the deadline not confirmed) and then not deleted, the pod
    is the run's record's, the error says it may still be running, and cancel deletes it."""
    spec = pod_run.spec
    guard = script(tmp_path / "refuses.sh", GUARD_REFUSES)
    monkeypatch.setattr(pod_run, "spec", lambda deadline, gpus=None: {**spec(deadline, gpus), "guard": str(guard), "down_s": 2})
    runpod.flags.add("delete-fails")
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a")])),
                           "--allow-uploads", "--max-cost-usd", "5")
    run = of(out, "run.created")[0]["run"]
    assert code == 1 and runpod.pods
    problems = " ".join(of(out, "error")[0]["problems"])
    assert "may still be running" in problems and f"remote cancel runpod {run}" in problems
    assert record_of(run)["pod"]["id"] == "pod1" and not record_of(run)["pod"]["deleted"]
    code, out = perception(capsys, "remote", "status", "runpod")
    assert [p["pod"] for p in of(out, "pods")[0]["pods"]] == ["pod1"]
    # its removal at the deadline was never confirmed: a cancel that cannot delete it says so (review of #53)
    code, out = perception(capsys, "remote", "cancel", "runpod", run)
    assert code == 1 and "may still be running" in of(out, "pod_deleted")[0]["may_still_run"]
    # a start asked again with its id is refused, not a traceback (review of #53)
    code, out = perception(capsys, "remote", "run", "runpod", str(graph_file(tmp_path, [work(tmp_path, "a")])),
                           "--allow-uploads", "--max-cost-usd", "5", "--run-id", run)
    assert code == 1 and "failed to start" in of(out, "error")[0]["problems"][0] and len(runpod.bodies) == 1
    runpod.flags.discard("delete-fails")
    code, out = perception(capsys, "remote", "cancel", "runpod", run)
    assert code == 0 and of(out, "pod_deleted")[0]["deleted"] and not runpod.pods


def test_a_pods_cost_is_charged_once_when_a_cancel_follows_a_fetch_whose_delete_failed(tmp_path, runpod, capsys, monkeypatch):
    """Review of #53: what was charged already (charged_usd) is not charged again by a cancel that imports nothing."""
    spec = pod_run.spec
    monkeypatch.setattr(pod_run, "spec", lambda deadline, gpus=None: {**spec(deadline, gpus), "down_s": 2})
    run = detached(tmp_path, capsys, sleep=0)
    assert wait_for(lambda: perception(capsys, "remote", "status", "runpod", run)[1][0].get("status") == "succeeded")
    runpod.flags.add("delete-fails")
    perception(capsys, "remote", "fetch", "runpod", run)
    spent = of(perception(capsys, "status", run)[1], "status")[0]["spent_usd"]
    runpod.flags.discard("delete-fails")
    runpod.pods.clear()   # it removed itself at the end of its linger window
    (tmp_path / "host" / "unreachable").touch()
    perception(capsys, "remote", "cancel", "runpod", run)
    assert of(perception(capsys, "status", run)[1], "status")[0]["spent_usd"] == spent



def test_a_start_that_failed_before_it_was_sent_can_be_asked_again(tmp_path, runpod, capsys, monkeypatch):
    """Review of #53: its pod was deleted, but nothing ran there: --run-id starts it, on a new pod."""
    real = remote_run.Remote.send_input
    calls = []

    def fails_once(self, *a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise remote_run.RemoteError("the video could not be sent")
        return real(self, *a, **kw)
    monkeypatch.setattr(remote_run.Remote, "send_input", fails_once)
    graph = str(graph_file(tmp_path, [work(tmp_path, "a")]))
    code, out = perception(capsys, "remote", "run", "runpod", graph, "--allow-uploads", "--max-cost-usd", "5")
    run = of(out, "run.created")[0]["run"]
    assert code == 1 and of(out, "pod_deleted")[0]["deleted"] and not runpod.pods
    code, out = perception(capsys, "remote", "run", "runpod", graph, "--allow-uploads", "--max-cost-usd", "5", "--run-id", run)
    assert code == 0, out
    assert of(out, "run.finished")[-1]["status"] == "succeeded" and len(runpod.bodies) == 2 and not runpod.pods


def test_a_pod_that_removed_itself_is_seen_gone(tmp_path, runpod, capsys):
    """Review of #53: nobody fetched it, and it went at the end of its linger window: status and fetch say so, from
    RunPod's answer, with what it cost (at most its deadline) in its record."""
    run = detached(tmp_path, capsys, sleep=0)
    runpod.pods.clear()   # it removed itself
    code, out = perception(capsys, "remote", "status", "runpod")
    [row] = of(out, "pods")[0]["pods"]
    assert row["deleted"] and row["cost_usd"] is not None
    assert record_of(run)["pod"]["removed_itself"]
    code, out = perception(capsys, "remote", "fetch", "runpod", run)
    assert code == 1 and "removed itself" in of(out, "error")[0]["problems"][0]
