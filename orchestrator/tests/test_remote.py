"""Nodes on another machine (backends/remote.py, remote_driver.py), against a stand-in for ssh that runs each command on
this machine with a separate home (the "host"). Everything else is real: the driver process, rsync, the shell scripts
run on the host, the node process there, cancel and the coordinator's checks."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from perception.backends.base import Capabilities, this_platform
from perception.backends.native import SelfBackend
from perception.backends.remote import SshBackend
from perception.hosts import Host, HostError, Hosts
from perception.ir import compile_graph
from perception.manifests import Registry
from perception.plan import plan as make_plan
from perception.scheduler import Coordinator
from perception.store import Store

NODES = Path(__file__).parent / "nodes"

FAKE_SSH = r"""#!/bin/sh
# ssh [-o OPTION]... DEST COMMAND: runs COMMAND here, in the fake host's home. Exits 255 (ssh's "no connection") while
# $FAKE_HOST/unreachable exists: for every command, or, if the file holds some text, for commands that contain it.
while [ $# -gt 0 ]; do case "$1" in -o) shift 2;; -*) shift;; *) break;; esac; done
shift
if [ -e "$FAKE_HOST/unreachable" ]; then
  pattern=$(cat "$FAKE_HOST/unreachable")
  case "$*" in *"$pattern"*) echo "ssh: connect to host lab: Network is unreachable" >&2; exit 255;; esac
fi
# a slow link for commands that contain $FAKE_HOST/slow's text
if [ -e "$FAKE_HOST/slow" ]; then case "$*" in *"$(cat "$FAKE_HOST/slow")"*) sleep 8;; esac; fi
cd "$FAKE_HOST/home" && HOME="$FAKE_HOST/home" exec /bin/sh -c "$*"
"""


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A fake host, `lab`: ssh is the stand-in, and the host's home is tmp_path/host/home."""
    fake = tmp_path / "host"
    (fake / "home").mkdir(parents=True)
    ssh = tmp_path / "ssh"
    ssh.write_text(FAKE_SSH)
    ssh.chmod(0o755)
    monkeypatch.setenv("PERCEPTION_SSH", str(ssh))
    monkeypatch.setenv("FAKE_HOST", str(fake))
    return fake


@pytest.fixture
def rec(tmp_path):
    return str(tmp_path / "record")


def lab(fake: Path, **caps) -> Host:
    c = Capabilities(platform=this_platform(), cpu_cores=8, memory_gb=32, accelerators={"cpu"}, vram_gb=0,
                     tools={"uv": None, "ffmpeg": None, "ffprobe": None})
    for k, v in caps.items():
        setattr(c, k, v)
    return Host("lab", "lab", str(fake / "home" / ".cache" / "hp"), c, time.time())


def coordinator(tmp_path, fake, lost_after_s=600) -> Coordinator:
    # the test nodes are `self` runners: on the host they run with this interpreter, which has the node SDK
    remote = SshBackend(lab(fake), runner_kind="self", python=sys.executable, lost_after_s=lost_after_s)
    return Coordinator(Store(tmp_path / "store"), Registry.load(NODES), [SelfBackend(), remote],
                       tick_s=0.02, backoff_s=0.01)


ALL_KINDS = ["video", "audio", "frames", "faces", "transcript", "embeddings"]   # a json.v1 input could hold any of them
VIDEO_POLICY = {"placement": "hybrid", "privacy": "local-preferred", "allow_upload": ["video"]}


def video_graph(tmp_path, place=None, **policy) -> dict:
    video = tmp_path / "my clip (final).mp4"   # a name a shell would split
    video.write_bytes(b"not really a video")
    node = {"id": "shots", "uses": "test.layer@1.0.0", "inputs": {"video": "$inputs.video"}}
    if place:
        node["place"] = place
    return {"schema_version": 1, "pipeline": "test", "inputs": {"video": {"type": "video.v1", "uri": str(video)}},
            "policy": {"retries": 0, **(policy or VIDEO_POLICY)}, "nodes": [node]}


def work_graph(tmp_path, rec, place="ssh-lab:cpu", **config) -> dict:
    seed = tmp_path / "seed.json"
    seed.write_text('{"value": 0}')
    return {"schema_version": 1, "pipeline": "test", "inputs": {"seed": {"type": "json.v1", "uri": str(seed)}},
            "policy": {"retries": 0, "placement": "hybrid", "privacy": "local-preferred", "allow_upload": ALL_KINDS},
            "nodes": [{"id": "a", "uses": "test.work@1.0.0", "inputs": {"a": "$inputs.seed"}, "place": place,
                       "config": {"record_dir": rec, **config}}]}


def start(coord, doc, allow_uploads=True) -> str:
    return coord.create_run(compile_graph(doc, coord.registry), doc, allow_uploads=allow_uploads)


def log_of(coord, run_id, node, attempt=1) -> str:
    return (coord.store.root / "work" / run_id / node / str(attempt) / "log.txt").read_text()


# --- running there ------------------------------------------------------------------------------------------------

def test_a_node_placed_on_a_host_runs_there_and_its_outputs_come_back(tmp_path, host):
    coord = coordinator(tmp_path, host)
    run_id = start(coord, video_graph(tmp_path, place="ssh-lab:cpu"))
    assert asyncio.run(coord.run(run_id)) == "succeeded", coord._tasks(run_id)
    t = coord._tasks(run_id)["shots"]
    assert t["runner"] == "ssh-lab:cpu"
    shots = json.loads(Path(json.loads(t["outputs"])["shots"]["path"]).read_text())   # published here, from the host's out/
    assert shots["layer"] == "shots"
    root = host / "home" / ".cache" / "hp"
    sent = [p for p in (root / "objects").rglob("*") if p.is_file() and not p.name.startswith(".")]
    assert [p.name for p in sent] == ["data"]                   # by content; the node got it under its own name
    assert not any((root / "work").iterdir())                     # the attempt's folder there is gone, and the run's
    code = [p.relative_to(root / "code").parts[1:] for p in (root / "code").rglob("*.py")]
    assert ("orchestrator", "tests", "nodes", "layer", "node.py") in code
    assert not any("work" in c for c in code)                       # only this node's files (and the SDK)
    log = log_of(coord, run_id, "shots")
    assert "[remote] copying video" in log and "[remote] started on lab" in log


def test_the_same_input_and_code_are_sent_to_a_host_once(tmp_path, host, rec):
    coord = coordinator(tmp_path, host)
    for _ in range(2):
        run_id = start(coord, work_graph(tmp_path, rec))
        assert asyncio.run(coord.run(run_id)) == "succeeded", coord._tasks(run_id)
        coord.store.db.execute("DELETE FROM cache")
    assert "copying" not in log_of(coord, run_id, "a")   # the second run found both there


def test_a_node_reads_inputs_another_remote_node_made_through_this_machine(tmp_path, host, rec):
    coord = coordinator(tmp_path, host)
    doc = work_graph(tmp_path, rec)
    doc["nodes"].append({"id": "b", "uses": "test.work@1.0.0", "inputs": {"a": "$nodes.a.out"}, "place": "ssh-lab:cpu",
                         "config": {"record_dir": rec, "add": 10}})
    run_id = start(coord, doc)
    assert asyncio.run(coord.run(run_id)) == "succeeded", coord._tasks(run_id)
    out = json.loads(Path(json.loads(coord._tasks(run_id)["b"]["outputs"])["out"]["path"]).read_text())
    assert out == {"value": 11}


def test_a_node_that_fails_on_a_host_fails_with_its_own_error(tmp_path, host, rec):
    coord = coordinator(tmp_path, host)
    run_id = start(coord, work_graph(tmp_path, rec, fail="input", fail_until=9))
    assert asyncio.run(coord.run(run_id)) == "failed"
    t = coord._tasks(run_id)["a"]
    assert (t["error_code"], t["error"]) == ("input", "failing on purpose (input)")


# --- where it may run ---------------------------------------------------------------------------------------------

def test_a_host_is_only_ever_chosen_never_a_default(tmp_path, host):
    coord = coordinator(tmp_path, host)
    graph = compile_graph(video_graph(tmp_path), coord.registry)
    m = graph.nodes["shots"].manifest
    options, _ = coord.placements(m, policy=graph.spec.policy)
    assert [p.label for p in options] == ["self:cpu", "ssh-lab:cpu"]      # this machine's first
    placement, _ = coord.place(m, policy=graph.spec.policy)
    assert placement.label == "self:cpu"


def test_a_host_is_offered_only_when_the_graph_lets_its_inputs_leave(tmp_path, host):
    coord = coordinator(tmp_path, host)
    for policy, why in [({"privacy": "local-only"}, "keeps its nodes on this machine"),
                        ({"placement": "local", "privacy": "local-preferred", "allow_upload": ["video"]}, "keeps its nodes on this machine"),
                        ({"placement": "hybrid", "privacy": "local-preferred", "allow_upload": ["frames"]}, "its inputs hold video, which policy.allow_upload does not")]:
        graph = compile_graph(video_graph(tmp_path, **policy), coord.registry)
        options, _ = coord.placements(graph.nodes["shots"].manifest, policy=graph.spec.policy)
        assert [p.label for p in options] == ["self:cpu"]
        placement, reason = coord.place(graph.nodes["shots"].manifest, choice="ssh-lab:cpu", policy=graph.spec.policy)
        assert placement is None and why in reason, reason


def test_a_node_on_a_host_without_the_runs_consent_never_starts(tmp_path, host):
    coord = coordinator(tmp_path, host)
    run_id = start(coord, video_graph(tmp_path, place="ssh-lab:cpu"), allow_uploads=False)
    assert asyncio.run(coord.run(run_id)) == "failed"
    t = coord._tasks(run_id)["shots"]
    assert t["error_code"] == "policy" and "sends video off this machine (to lab (SSH))" in t["error"]
    assert t["attempt"] == 0 and not (host / "home" / ".cache").exists()   # nothing was sent


def test_nodes_that_call_an_api_or_need_secrets_stay_on_this_machine(tmp_path, host):
    coord = coordinator(tmp_path, host)
    refine = coord.registry.get("test.refine@1.0.0")   # sends frames to an API
    options, _ = coord.placements(refine)
    assert all(not p.backend.remote for p in options)
    placement, why = coord.place(refine, choice="ssh-lab:cpu")
    assert placement is None and "it sends frames to" in why


def test_a_plan_says_what_each_place_sends_and_where(tmp_path, host):
    coord = coordinator(tmp_path, host)
    planned = make_plan(coord, compile_graph(video_graph(tmp_path, place="ssh-lab:cpu"), coord.registry))
    node = planned.nodes[0]
    assert (node.place, node.uploads, node.upload_to) == ("ssh-lab:cpu", ["video"], "lab (SSH)")
    assert node.estimates["self:cpu"]["uploads"] == [] and node.estimates["ssh-lab:cpu"]["uploads"] == ["video"]
    assert planned.places["ssh-lab:cpu"] == {"title": "lab (SSH)", "remote": True, "accelerator": "cpu"}
    assert planned.places["self:cpu"]["remote"] is False
    assert not planned.ok   # it would send, and the plan was not given consent


def test_a_remote_node_does_not_take_this_machines_gpu(tmp_path, host, rec):
    coord = coordinator(tmp_path, host)
    seed = tmp_path / "seed.json"
    seed.write_text('{"value": 0}')
    gpu = lambda id, place: {"id": id, "uses": "test.gpu@1.0.0", "inputs": {"a": "$inputs.seed"}, "place": place,
                             "config": {"record_dir": rec, "sleep": 1.5}}
    doc = {"schema_version": 1, "pipeline": "test", "inputs": {"seed": {"type": "json.v1", "uri": str(seed)}},
           "policy": {"retries": 0, "placement": "hybrid", "privacy": "local-preferred", "allow_upload": ALL_KINDS},
           "nodes": [gpu("here", "self:cpu"), gpu("there", "ssh-lab:cpu")]}
    run_id = start(coord, doc)
    assert asyncio.run(coord.run(run_id)) == "succeeded", coord._tasks(run_id)
    t = coord._tasks(run_id)
    assert t["there"]["started_at"] < t["here"]["finished_at"] and t["here"]["started_at"] < t["there"]["finished_at"]   # overlapped


# --- stopping and losing it ---------------------------------------------------------------------------------------

def remote_pid(host, run_id, node="a", attempt=1, wait_s=30) -> int:
    pidfile = host / "home" / ".cache" / "hp" / "work" / run_id / node / str(attempt) / "pid"
    deadline = time.time() + wait_s
    while not pidfile.exists():
        assert time.time() < deadline, "the node never started on the host"
        time.sleep(0.05)
    time.sleep(0.1)
    return int(pidfile.read_text())


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


def test_cancel_stops_the_node_on_the_host(tmp_path, host, rec):
    coord = coordinator(tmp_path, host)
    run_id = start(coord, work_graph(tmp_path, rec, sleep=60))

    async def go():
        task = asyncio.create_task(coord.run(run_id))
        pid = await asyncio.to_thread(remote_pid, host, run_id)
        assert alive(pid)
        coord.cancel(run_id)
        return pid, await task
    pid, status = asyncio.run(go())
    assert status == "canceled"
    deadline = time.time() + 15
    while alive(pid):
        assert time.time() < deadline, "the node is still running on the host"
        time.sleep(0.1)
    assert "[remote] stopping it on lab" in log_of(coord, run_id, "a")


def test_a_host_lost_for_too_long_ends_the_attempt_as_a_crash_and_its_watchdog_ends_the_node(tmp_path, host, rec):
    coord = coordinator(tmp_path, host, lost_after_s=1)
    run_id = start(coord, work_graph(tmp_path, rec, sleep=60))

    async def go():
        task = asyncio.create_task(coord.run(run_id))
        pid = await asyncio.to_thread(remote_pid, host, run_id)
        (host / "unreachable").write_text("echo running")   # the driver's checks fail; the host itself carries on
        return pid, await task
    pid, status = asyncio.run(go())
    assert status == "failed"
    t = coord._tasks(run_id)["a"]
    assert t["error_code"] == "transient" and "without a response" in t["error"]
    assert "could not be reached for 1 s" in log_of(coord, run_id, "a")
    attempt = host / "home" / ".cache" / "hp" / "work" / run_id / "a" / "1"
    deadline = time.time() + 20
    while alive(pid) or attempt.parent.parent.exists():   # no driver asked about it for 2 s: stopped, its folder (outputs) gone
        assert time.time() < deadline, "the node was left running on the host"
        time.sleep(0.2)


def test_a_new_attempt_stops_one_a_lost_connection_left_running_there(tmp_path, host, rec):
    coord = coordinator(tmp_path, host, lost_after_s=1)
    doc = work_graph(tmp_path, rec, sleep=60)
    doc["policy"]["retries"] = 1
    run_id = start(coord, doc)

    async def go():
        task = asyncio.create_task(coord.run(run_id))
        first = await asyncio.to_thread(remote_pid, host, run_id)
        (host / "unreachable").write_text("echo running")   # watching fails; starting the next attempt does not
        while coord._tasks(run_id)["a"]["attempt"] < 2 and coord._tasks(run_id)["a"]["state"] != "failed":
            await asyncio.sleep(0.05)
        (host / "unreachable").unlink()                   # back for the retry
        second = await asyncio.to_thread(remote_pid, host, run_id, "a", 2)
        coord.cancel(run_id)
        await task
        return first, second
    first, second = asyncio.run(go())
    deadline = time.time() + 15
    while alive(first) or alive(second):
        assert time.time() < deadline, "an attempt is still running on the host"
        time.sleep(0.1)


def test_running_on_a_host_always_needs_the_runs_consent_and_unknown_inputs_count_as_every_kind(tmp_path, host, rec):
    coord = coordinator(tmp_path, host)
    doc = work_graph(tmp_path, rec)
    graph = compile_graph({**doc, "policy": {**doc["policy"], "allow_upload": ["video"]}}, coord.registry)
    placement, why = coord.place(graph.nodes["a"].manifest, choice="ssh-lab:cpu", policy=graph.spec.policy)
    assert placement is None and "audio, embeddings, faces, frames, transcript" in why   # a json.v1 input could hold any
    run_id = start(coord, doc, allow_uploads=False)
    assert asyncio.run(coord.run(run_id)) == "failed"
    t = coord._tasks(run_id)["a"]
    assert t["error_code"] == "policy" and t["attempt"] == 0


def test_a_node_whose_inputs_hold_none_of_the_kinds_still_needs_consent_to_run_on_a_host(tmp_path, host):
    coord = coordinator(tmp_path, host)
    media = tmp_path / "media.json"
    media.write_text(json.dumps({"sha256": "0" * 64, "duration": 2.0, "width": 4, "height": 4, "fps": 25.0,
                                 "time_base": "1/12800", "variable_frame_rate": False}))
    doc = {"schema_version": 1, "pipeline": "test", "inputs": {"m": {"type": "media.v1", "uri": str(media)}},
           "policy": {"retries": 0, "placement": "hybrid", "privacy": "local-preferred"},   # no kinds allowed: none needed
           "nodes": [{"id": "a", "uses": "test.meta@1.0.0", "inputs": {"m": "$inputs.m"}, "place": "ssh-lab:cpu"}]}
    planned = make_plan(coord, compile_graph(doc, coord.registry)).nodes[0]
    assert (planned.uploads, planned.upload_to) == ([], "lab (SSH)")
    run_id = start(coord, doc, allow_uploads=False)
    assert asyncio.run(coord.run(run_id)) == "failed"
    t = coord._tasks(run_id)["a"]
    assert t["error_code"] == "policy" and "sends its inputs off this machine (to lab (SSH))" in t["error"]


def test_the_same_video_under_another_name_is_not_sent_again_and_the_node_gets_its_new_name(tmp_path, host):
    coord = coordinator(tmp_path, host)
    run_id = start(coord, video_graph(tmp_path, place="ssh-lab:cpu"))
    assert asyncio.run(coord.run(run_id)) == "succeeded"
    coord.store.db.execute("DELETE FROM cache")          # the same content would be a cache hit
    doc = video_graph(tmp_path, place="ssh-lab:cpu")
    renamed = tmp_path / "renamed.mp4"
    renamed.write_bytes(b"not really a video")
    doc["inputs"]["video"]["uri"] = str(renamed)
    run_id = start(coord, doc)
    assert asyncio.run(coord.run(run_id)) == "succeeded", log_of(coord, run_id, "shots")
    assert "copying video" not in log_of(coord, run_id, "shots")


@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/dash"])
def test_publishing_on_a_host_never_replaces_or_nests_into_what_is_there(tmp_path, shell):
    from perception.remote_driver import PUBLISH
    if not Path(shell).exists():
        pytest.skip(f"no {shell}")
    first, second, target = tmp_path / "a", tmp_path / "b", tmp_path / "objects" / "x"
    for d, text in ((first, "first"), (second, "second")):
        d.mkdir()
        (d / "data").write_text(text)
        (d / ".complete").touch()
    target.parent.mkdir()
    sh = lambda a, b: subprocess.run([shell, "-c", f'{PUBLISH}\npublish "$1" "$2"', "sh", str(a), str(b)]).returncode
    assert sh(first, target) == 0 and (target / "data").read_text() == "first"
    assert sh(second, target) == 0                       # another upload of the same content: dropped, not nested
    assert sorted(p.name for p in target.iterdir()) == [".complete", "data"] and (target / "data").read_text() == "first"
    assert not second.exists()


def test_two_nodes_sending_the_same_input_to_a_host_at_once_both_run(tmp_path, host, rec):
    coord = coordinator(tmp_path, host)
    doc = work_graph(tmp_path, rec)
    doc["nodes"] += [{**doc["nodes"][0], "id": f"n{i}"} for i in range(3)]
    run_id = start(coord, doc)
    assert asyncio.run(coord.run(run_id)) == "succeeded", coord._tasks(run_id)
    objects = host / "home" / ".cache" / "hp" / "objects"
    stored = [p for p in objects.iterdir() if not p.name.startswith(".incoming")]
    assert len(stored) == 1 and sorted(x.name for x in stored[0].iterdir()) == [".complete", "data"]


def test_a_node_that_replaces_its_out_folder_with_a_link_there_has_nothing_fetched(tmp_path, host, rec):
    coord = coordinator(tmp_path, host)
    run_id = start(coord, work_graph(tmp_path, rec, sleep=2))
    elsewhere = host / "home" / "elsewhere"

    async def go():
        task = asyncio.create_task(coord.run(run_id))
        await asyncio.to_thread(remote_pid, host, run_id)
        out = host / "home" / ".cache" / "hp" / "work" / run_id / "a" / "1" / "out"
        elsewhere.mkdir()
        out.rmdir()
        out.symlink_to(elsewhere)                        # the node then writes its outputs through the link
        return await task
    assert asyncio.run(go()) == "failed"
    t = coord._tasks(run_id)["a"]
    assert t["error_code"] == "schema" and "out.json is missing" in t["error"]
    assert (elsewhere / "out.json").exists()             # written there; never fetched through the link


def test_a_host_with_a_node_on_it_is_neither_removed_nor_cleaned(tmp_path, host, monkeypatch, capsys, rec):
    from perception.cli import main
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "home"))
    assert main(["hosts", "add", "lab", "lab"]) == 0
    store = Store(tmp_path / "home")
    store.db.execute("INSERT INTO runs (id, pipeline, graph, inputs, status, created_at, max_cost_usd) VALUES ('r1', 't', '{}', '{}', 'running', 0, 0)")
    store.db.execute("INSERT INTO tasks (run_id, node_id, state, runner) VALUES ('r1', 'faces', 'running', 'ssh-lab:cuda')")
    assert main(["hosts", "remove", "lab"]) == 1
    assert "run r1 has faces on lab" in capsys.readouterr().err
    store.db.execute("UPDATE tasks SET state='succeeded'")
    work = host / "home" / ".cache" / "hyperstudy-perception" / "work"
    stale = work / "r1" / "a" / "1"
    stale.mkdir(parents=True)
    (stale / "stop").touch()                             # a cancel's mark, from a start that never came: not a node
    busy = work / "r2" / "a" / "1"
    busy.mkdir(parents=True)
    # its command names the request (a second command, so sh does not replace itself with sleep)
    node = subprocess.Popen(["sh", "-c", "sleep 30; :", str(busy / "request.json")])
    (busy / "pid").write_text(str(node.pid))
    try:
        assert main(["hosts", "clean", "lab"]) == 1
        assert "a node is running there" in capsys.readouterr().err
    finally:
        node.kill()
        node.wait()
    assert main(["hosts", "clean", "lab"]) == 0
    assert not work.exists()                            # the stale folders went with the inputs
    assert main(["hosts", "remove", "lab"]) == 0


def test_a_finished_nodes_results_survive_a_connection_dropped_while_they_are_fetched(tmp_path, host, rec):
    import threading
    coord = coordinator(tmp_path, host)
    run_id = start(coord, work_graph(tmp_path, rec, sleep=1))

    async def go():
        task = asyncio.create_task(coord.run(run_id))
        await asyncio.to_thread(remote_pid, host, run_id)
        (host / "unreachable").write_text("! -L out")       # only the fetch's first command, for 5 s
        threading.Timer(5, (host / "unreachable").unlink).start()
        return await task
    assert asyncio.run(go()) == "succeeded", log_of(coord, run_id, "a")
    assert coord._tasks(run_id)["a"]["attempt"] == 1          # not run a second time
    assert "cannot be reached" in log_of(coord, run_id, "a")


def test_a_slow_fetch_keeps_the_heartbeat_so_the_watchdog_leaves_the_results(tmp_path, host, rec):
    coord = coordinator(tmp_path, host, lost_after_s=3)       # the watchdog would act after 6 s without a heartbeat
    (host / "slow").write_text("--sender")                     # rsync bringing out/ back takes 8 s
    run_id = start(coord, work_graph(tmp_path, rec))
    assert asyncio.run(coord.run(run_id)) == "succeeded", log_of(coord, run_id, "a")
    assert coord._tasks(run_id)["a"]["attempt"] == 1


def test_a_linked_video_is_sent_as_its_content(tmp_path, host):
    coord = coordinator(tmp_path, host)
    doc = video_graph(tmp_path, place="ssh-lab:cpu")
    real = Path(doc["inputs"]["video"]["uri"])
    link = tmp_path / "linked.mp4"
    link.symlink_to(real)                                  # a graph input that points at the video (an external drive)
    doc["inputs"]["video"]["uri"] = str(link)
    run_id = start(coord, doc)
    assert asyncio.run(coord.run(run_id)) == "succeeded", log_of(coord, run_id, "shots")
    [stored] = (host / "home" / ".cache" / "hp" / "objects").glob("*/data")
    assert not stored.is_symlink() and stored.read_bytes() == real.read_bytes()
    media = json.loads(Path(json.loads(coord._tasks(run_id)["shots"]["outputs"])["media"]["path"]).read_text())
    assert media["filename"] == "linked.mp4"               # the name it has here, as a local run gives it, not the target's


def test_an_input_that_changed_since_the_run_read_it_is_not_filed_under_the_old_hash(tmp_path, host):
    coord = coordinator(tmp_path, host)
    doc = video_graph(tmp_path, place="ssh-lab:cpu")
    run_id = start(coord, doc)                             # the run has the video's sha256 now
    Path(doc["inputs"]["video"]["uri"]).write_bytes(b"another cut of the film")
    assert asyncio.run(coord.run(run_id)) == "failed"
    assert "arrived as" in log_of(coord, run_id, "shots") and "it changed since the run read it" in log_of(coord, run_id, "shots")
    objects = host / "home" / ".cache" / "hp" / "objects"
    assert not [p for p in objects.iterdir() if not p.name.startswith(".")]   # nothing kept under that hash


def test_a_host_unreachable_when_the_attempt_starts_is_waited_for(tmp_path, host, rec):
    import threading
    coord = coordinator(tmp_path, host)
    (host / "unreachable").write_text("")                 # every command, for the first 4 s
    threading.Timer(4, (host / "unreachable").unlink).start()
    run_id = start(coord, work_graph(tmp_path, rec))
    assert asyncio.run(coord.run(run_id)) == "succeeded", log_of(coord, run_id, "a")
    assert coord._tasks(run_id)["a"]["attempt"] == 1 and "cannot be reached" in log_of(coord, run_id, "a")


def test_inputs_kept_on_a_host_are_read_only_and_still_cleaned(tmp_path, host, monkeypatch, capsys):
    from perception.cli import main
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "home"))
    assert main(["hosts", "add", "lab", "lab"]) == 0
    coord = Coordinator(Store(tmp_path / "home"), Registry.load(NODES),
                        [SelfBackend(), SshBackend(Hosts(tmp_path / "home").load()["lab"], runner_kind="self", python=sys.executable)],
                        tick_s=0.02, backoff_s=0.01)
    run_id = start(coord, video_graph(tmp_path, place="ssh-lab:cpu"))
    assert asyncio.run(coord.run(run_id)) == "succeeded"
    objects = host / "home" / ".cache" / "hyperstudy-perception" / "objects"
    [stored] = objects.glob("*/data")
    for path in (stored, stored.parent):                    # a node cannot change it through its link, as in the store
        assert path.stat().st_mode & 0o222 == 0, path
    assert main(["hosts", "clean", "lab"]) == 0
    assert not objects.exists()


def test_a_root_that_already_holds_other_files_is_never_taken_changed_or_cleaned(tmp_path, host, monkeypatch, capsys):
    from perception.cli import main
    mine = host / "home" / "projects"
    (mine / "code").mkdir(parents=True)
    (mine / "code" / "analysis.py").write_text("my own work")
    mine.chmod(0o750)
    with pytest.raises(HostError, match="already holds other files"):
        Hosts(tmp_path / "home").add("lab", "lab", "projects")
    assert mine.stat().st_mode & 0o777 == 0o750
    # nor cleaned, if hosts.json names it some other way
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "home"))
    Hosts(tmp_path / "home").save({"lab": Host("lab", "lab", str(mine), lab(host).caps, 0)})
    assert main(["hosts", "clean", "lab"]) == 1
    assert "not marked as perception's" in capsys.readouterr().err
    assert (mine / "code" / "analysis.py").read_text() == "my own work"


def test_an_input_that_changed_fails_once_as_an_input_error_not_a_crash_to_retry(tmp_path, host):
    coord = coordinator(tmp_path, host)
    doc = video_graph(tmp_path, place="ssh-lab:cpu")
    doc["policy"]["retries"] = 2
    run_id = start(coord, doc)
    Path(doc["inputs"]["video"]["uri"]).write_bytes(b"another cut of the film")
    assert asyncio.run(coord.run(run_id)) == "failed"
    t = coord._tasks(run_id)["shots"]
    assert (t["error_code"], t["attempt"]) == ("input", 1) and "it changed since the run read it" in t["error"]


def test_a_remote_identity_is_the_code_that_is_sent(tmp_path, host, monkeypatch):
    import perception.backends.remote as remote
    backend = SshBackend(lab(host), runner_kind="self", python=sys.executable)
    m = Registry.load(NODES).get("test.gpu@1.0.0")                # its entry is ../work/node.py
    runner = m.runners[0]
    assert "orchestrator/tests/nodes/work/node.py" in remote.code_files(m, runner)
    before = backend.identity(m, runner)
    real = remote.code_hash
    monkeypatch.setattr(remote, "code_hash", lambda files: real(files) + "-changed")   # what is sent changed
    assert backend.identity(m, runner) != before


def test_on_another_machine_every_weight_a_node_uses_counts_as_a_download(tmp_path, host, monkeypatch):
    import perception.plan as plan_mod
    from perception.manifests import ModelRef
    monkeypatch.setattr(plan_mod, "hf_cached", lambda source: True)          # all on this machine
    coord = coordinator(tmp_path, host)
    graph = compile_graph(video_graph(tmp_path, place="ssh-lab:cpu"), coord.registry)
    m = graph.nodes["shots"].manifest
    monkeypatch.setattr(m, "models", [ModelRef(id="w", source="hf://org/weights", licence_mode="commercial-ok")])
    planned = make_plan(coord, graph, allow_uploads=True)
    assert [d.id for d in planned.nodes[0].downloads] == ["w"] and not planned.ok   # the host's cache is not known
    here = make_plan(coord, compile_graph(video_graph(tmp_path), coord.registry), allow_uploads=True)
    assert here.nodes[0].downloads == []


def test_code_saved_while_it_is_being_sent_is_not_filed_under_the_old_hash(tmp_path, host, rec, monkeypatch):
    import perception.backends.remote as remote
    coord = coordinator(tmp_path, host)
    # the attempt's spec names one hash; the driver (its own process) finds the files it copies hold something else
    monkeypatch.setattr(remote, "code_hash", lambda files, root=remote.REPO: "0" * 20)
    run_id = start(coord, work_graph(tmp_path, rec))
    assert asyncio.run(coord.run(run_id)) == "failed"
    assert "the node's code changed since this attempt started" in log_of(coord, run_id, "a")
    assert not (host / "home" / ".cache" / "hp" / "code").exists() or not any((host / "home" / ".cache" / "hp" / "code").iterdir())


def test_code_with_a_link_git_tracks_is_not_sent(tmp_path):
    from perception.backends.remote import refuse_links
    (tmp_path / "node").mkdir()
    (tmp_path / "node" / "node.py").write_text("print()")
    (tmp_path / "node" / "sample.mp4").symlink_to(tmp_path / "elsewhere.mp4")   # e.g. a study video outside the repository
    refuse_links(["node/node.py"], tmp_path)
    with pytest.raises(RuntimeError, match="links are not sent"):
        refuse_links(["node/node.py", "node/sample.mp4"], tmp_path)


def test_a_node_from_outside_the_repository_is_not_offered_a_host(tmp_path, host):
    import shutil
    outside = tmp_path / "more-nodes" / "work"
    shutil.copytree(NODES / "work", outside)
    coord = Coordinator(Store(tmp_path / "store"), Registry.load(tmp_path / "more-nodes"),
                        [SelfBackend(), SshBackend(lab(host), runner_kind="self", python=sys.executable)])
    m = coord.registry.get("test.work@1.0.0")
    placement, why = coord.place(m, choice="ssh-lab:cpu")
    assert placement is None and "not in the pipeline's repository" in why
    assert [p.label for p in coord.placements(m)[0]] == ["self:cpu"]


def test_a_host_whose_home_is_not_a_plain_path_is_refused(tmp_path, host, monkeypatch):
    spaced = tmp_path / "Lab Shared"                     # the host's $HOME becomes ".../Lab Shared/home"
    (spaced / "home").mkdir(parents=True)
    monkeypatch.setenv("FAKE_HOST", str(spaced))
    with pytest.raises(HostError, match="not a plain path"):
        Hosts(tmp_path / "home").add("lab", "lab")


def watchdog(tmp_path, alive: str, lost=6, keep=3600, seconds=2.5, node=None) -> Path:
    """Run the host's watchdog over a fake attempt folder for a few seconds; the folder, or where it was."""
    from perception.remote_driver import WATCHDOG
    w = tmp_path / "work" / "r" / "a" / "1"
    w.mkdir(parents=True, exist_ok=True)
    (w / "alive").write_text(alive)
    pid = node.pid if node else 999999
    dog = subprocess.Popen(["sh", "-c", WATCHDOG, "watchdog", str(w), str(pid), str(lost), "1", str(keep)])
    time.sleep(seconds)
    dog.kill()
    dog.wait()
    return w


def test_the_watchdog_never_reads_a_heartbeat_being_written_as_long_ago(tmp_path):
    assert watchdog(tmp_path, "").exists()                  # emptied by `date > alive` at that moment


def test_the_watchdog_keeps_a_finished_nodes_results_for_a_driver_that_comes_back(tmp_path):
    old = str(int(time.time()) - 3600)                      # no driver for an hour: the laptop slept
    assert watchdog(tmp_path / "kept", old, keep=24 * 3600).exists()
    assert not watchdog(tmp_path / "gone", old, keep=60).exists()   # past keep, removed


def test_the_watchdog_stops_a_running_node_no_driver_watches(tmp_path):
    w = tmp_path / "work" / "r" / "a" / "1"
    w.mkdir(parents=True)
    node = subprocess.Popen(["sh", "-c", "sleep 60; :", str(w / "request.json")], start_new_session=True)
    try:
        assert not watchdog(tmp_path, str(int(time.time()) - 60), lost=6, seconds=8, node=node).exists()
        assert node.wait(timeout=5) is not None
    finally:
        node.kill()


def test_a_cancel_during_an_upload_leaves_no_partial_copy_on_the_host(tmp_path, host, rec):
    coord = coordinator(tmp_path, host)
    (host / "slow").write_text("--server")                 # rsync sending to the host: 8 s before it starts
    run_id = start(coord, work_graph(tmp_path, rec))
    objects = host / "home" / ".cache" / "hp" / "objects"

    async def go():
        task = asyncio.create_task(coord.run(run_id))
        while not (objects.exists() and any(p.name.startswith(".incoming") for p in objects.iterdir())):
            await asyncio.sleep(0.05)                     # the upload's folder exists: the copy is on its way
        coord.cancel(run_id)
        return await task
    assert asyncio.run(go()) == "canceled"
    deadline = time.time() + 20
    while any(p.name.startswith(".incoming") for p in objects.iterdir()):
        assert time.time() < deadline, "a partial upload stayed on the host"
        time.sleep(0.1)


def test_a_node_is_prepared_on_the_host_before_it_starts_and_the_time_is_reported(tmp_path, host, rec):
    coord = coordinator(tmp_path, host)
    run_id = start(coord, work_graph(tmp_path, rec, add=5))
    assert asyncio.run(coord.run(run_id)) == "succeeded", log_of(coord, run_id, "a")
    prepared = json.loads((Path(rec) / "prepared").read_text())
    assert prepared["add"] == 5                            # with the node's own settings
    phases = json.loads(coord._tasks(run_id)["a"]["usage"])["phases"]
    assert {"code", "inputs", "prepare", "run", "fetch", "startup"} <= set(phases)


def test_a_node_whose_prepare_fails_on_the_host_never_starts_and_fails_as_transient(tmp_path, host, rec):
    coord = coordinator(tmp_path, host)
    run_id = start(coord, work_graph(tmp_path, rec, fail="prepare"))
    assert asyncio.run(coord.run(run_id)) == "failed"
    t = coord._tasks(run_id)["a"]
    assert t["error_code"] == "transient" and "preparing the node failed" in t["error"]
    assert "could not fetch the weights" in log_of(coord, run_id, "a")
    assert not list(Path(rec).glob("a-*"))                 # the node itself never ran


def test_prosody_counts_as_transcript_too():
    from perception.artifacts import get
    assert set(get("bundle.prosody@0.1").kinds) == {"audio", "transcript"}   # each utterance's words


# --- hosts --------------------------------------------------------------------------------------------------------

def test_a_host_is_probed_once_when_added_and_its_places_come_from_that(tmp_path, host):
    hosts = Hosts(tmp_path / "home")
    added = hosts.add("lab", "me@lab.example.edu")
    assert added.caps.platform == this_platform() and added.caps.cpu_cores == os.cpu_count()
    assert added.root == f"{host}/home/.cache/hyperstudy-perception"   # absolute, under the host's home
    assert added.caps.tools["uv"]                                        # found on the PATH a login shell has
    assert Hosts(tmp_path / "home").load()["lab"].caps == added.caps
    (host / "unreachable").touch()
    with pytest.raises(HostError, match="could not reach me@lab.example.edu"):
        hosts.check("lab")


def test_a_hosts_root_is_private_to_the_user_and_one_someone_else_owns_is_refused(tmp_path, host, rec):
    root = host / "home" / ".cache" / "hyperstudy-perception"
    Hosts(tmp_path / "home").add("lab", "lab")
    assert root.stat().st_mode & 0o777 == 0o700
    with pytest.raises(HostError, match="already holds other files"):
        Hosts(tmp_path / "home").add("lab2", "lab", "/usr")   # root's, not ours: never used, never changed

    coord = coordinator(tmp_path, host)
    hp = host / "home" / ".cache" / "hp"
    hp.mkdir(parents=True)
    hp.chmod(0o755)                     # opened up since it was added: made private again before anything is sent
    run_id = start(coord, work_graph(tmp_path, rec))
    assert asyncio.run(coord.run(run_id)) == "succeeded", coord._tasks(run_id)
    assert hp.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("name,dest,root", [("Lab", "lab", "x"), ("native", "lab", "x"), ("lab", "-oProxyCommand=x", "x"),
                                            ("lab", "lab", "../etc"), ("lab", "lab", "/"), ("lab", "lab", "~"), ("lab", "lab", "a b")])
def test_a_host_with_a_name_destination_or_root_that_could_do_harm_is_refused(tmp_path, host, name, dest, root):
    with pytest.raises(HostError):
        Hosts(tmp_path / "home").add(name, dest, root)
    assert not (tmp_path / "home" / "hosts.json").exists()


def test_the_cli_adds_lists_and_plans_with_a_host(tmp_path, host, monkeypatch, capsys):
    from perception.cli import main
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "home"))
    assert main(["--json", "hosts", "add", "lab", "me@lab"]) == 0
    [event] = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert event["type"] == "hosts" and [h["name"] for h in event["hosts"]] == ["lab"]
    assert main(["--json", "hosts", "add", "lab2", "--", "-oProxyCommand=x"]) == 1
    capsys.readouterr()
    assert main(["--json", "hosts", "clean", "lab"]) == 0
    capsys.readouterr()
    assert main(["--json", "hosts", "remove", "lab"]) == 0
    [event] = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert event["hosts"] == []
