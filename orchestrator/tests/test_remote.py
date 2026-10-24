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
            "policy": {"retries": 0, "placement": "hybrid", "privacy": "local-preferred"},
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
    assert [p.name for p in sent] == ["my clip (final).mp4"]   # the input, under its own name
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
           "policy": {"retries": 0, "placement": "hybrid", "privacy": "local-preferred"},
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


def test_a_host_lost_for_too_long_ends_the_attempt_as_a_crash(tmp_path, host, rec):
    coord = coordinator(tmp_path, host, lost_after_s=1)
    run_id = start(coord, work_graph(tmp_path, rec, sleep=60))

    async def go():
        task = asyncio.create_task(coord.run(run_id))
        pid = await asyncio.to_thread(remote_pid, host, run_id)
        (host / "unreachable").touch()
        return pid, await task
    pid, status = asyncio.run(go())
    try:
        assert status == "failed"
        t = coord._tasks(run_id)["a"]
        assert t["error_code"] == "transient" and "without a response" in t["error"]
        assert "could not be reached for 1 s" in log_of(coord, run_id, "a")
    finally:
        os.killpg(pid, signal.SIGTERM)   # still running "there": nothing could reach it


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
    with pytest.raises(HostError, match="belongs to someone else"):
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
