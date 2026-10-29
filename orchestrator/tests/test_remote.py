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
    # by content: the video (the node got it under its own name), and the two outputs it made, kept for nodes after it
    assert [p.name for p in sent] == ["data", "data", "data"]
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


def test_a_node_reads_what_another_node_made_on_the_same_host_without_it_being_sent_again(tmp_path, host, rec):
    coord = coordinator(tmp_path, host)
    doc = work_graph(tmp_path, rec)
    doc["nodes"].append({"id": "b", "uses": "test.work@1.0.0", "inputs": {"a": "$nodes.a.out"}, "place": "ssh-lab:cpu",
                         "config": {"record_dir": rec, "add": 10}})
    run_id = start(coord, doc)
    assert asyncio.run(coord.run(run_id)) == "succeeded", coord._tasks(run_id)
    out = json.loads(Path(json.loads(coord._tasks(run_id)["b"]["outputs"])["out"]["path"]).read_text())
    assert out == {"value": 11}
    assert "[remote] kept out on lab" in log_of(coord, run_id, "a")
    assert "copying" not in log_of(coord, run_id, "b")   # a's output was already there, and so was the code


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
    # the input, and the one output the four nodes all made (the same content)
    assert len(stored) == 2 and all(sorted(x.name for x in p.iterdir()) == [".complete", "data"] for p in stored)


# --- outputs kept on the host -------------------------------------------------------------------------------------

def finished(tmp_path, host, make, inputs=None, node="n"):
    """A driver for an attempt whose node has finished on the fake host: `make(out)` writes its out/ there, brought
    back here as fetch brings it (rsync -rlt), with a response naming each entry of out/ as an output. What fetch
    leaves for keep_outputs."""
    from perception.remote_driver import Driver
    root = host / "home" / ".cache" / "hp"
    there = root / "work" / "r1" / node / "1" / "out"
    there.mkdir(parents=True)
    (root / "objects").mkdir(exist_ok=True)
    make(there)
    local = tmp_path / f"attempt-{node}"
    local.mkdir()
    subprocess.run(["rsync", "-rlt", f"{there}/", f"{local}/out/"], check=True)
    outputs = {p.name: {"path": p.name} for p in sorted(there.iterdir())}
    (local / "response.json.part").write_text(json.dumps({"ok": True, "outputs": outputs}))
    (local / "request.json").write_text(json.dumps({"run": "r1", "node": node, "attempt": 1, "inputs": inputs or {}}))
    spec = {"dest": "lab", "root": str(root), "host": "lab", "ssh": lab(host).ssh(), "code": "c", "lost_after_s": 600}
    return Driver(local / "request.json", spec), root, there


def outputs_of_every_shape(out: Path):
    """A layer, and a tree whose files sort one way as paths and another as strings ("a/b" before "a-c" as paths, after
    it as strings), with a hidden file, a name with spaces and accents, an empty folder and a larger file."""
    (out / "layer.json").write_bytes(b'{"layer": "faces"}')
    tree = out / "assets"
    for rel, data in {"a/b": b"1", "a-c": b"2", ".hidden": b"3", "crops/ü é.jpg": b"4", "crops/deep/x": b"5",
                      "big.bin": os.urandom(3 << 20)}.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_bytes(data)
    (tree / "empty").mkdir()


def test_outputs_kept_on_a_host_are_filed_under_the_hash_this_machines_store_gives_them(tmp_path, host, capsys):
    from perception import artifacts
    driver, root, there = finished(tmp_path, host, outputs_of_every_shape)
    files = [f for f, _ in artifacts.tree_files(driver.local / "out" / "assets")]
    assert files.index("a/b") < files.index("a-c") and sorted(files) != files   # sorted as paths, not as strings
    driver.keep_outputs()
    assert "[remote] kept assets, layer.json on lab" in capsys.readouterr().out
    store = Store(tmp_path / "store")
    published = {name: store.publish(driver.local / "out" / name, "x") for name in ("layer.json", "assets")}
    for name, a in published.items():
        kept = root / "objects" / a.hash
        assert sorted(p.name for p in kept.iterdir()) == [".complete", "data"], name
        assert artifacts.content_hash(kept / "data") == a.hash, name            # the same bytes there as in the store
        assert (there / name).exists()                  # a copy: nothing the node's processes still hold is kept
        assert (kept / "data").stat().st_mode & 0o222 == 0                     # read-only, as inputs are
    assert sorted(p.name for p in (root / "objects").iterdir()) == sorted(a.hash for a in published.values())
    # the next node on that host is not sent them: they are its inputs, by the store's hashes
    inputs = {name: {"type": "x", "sha256": a.hash, "path": str(a.path)} for name, a in published.items()}
    after, _, _ = finished(tmp_path, host, lambda out: None, inputs=inputs, node="m")
    sent = after.send_inputs()
    assert "copying" not in capsys.readouterr().out
    assert sent["assets"]["object"] == f"{root}/objects/{published['assets'].hash}/data"
    # another node that makes the same layer: it is there already, and said so
    again, _, _ = finished(tmp_path, host, lambda out: (out / "layer.json").write_bytes(b'{"layer": "faces"}'), node="k")
    again.keep_outputs()
    assert "[remote] layer.json: on lab already (the same content)" in capsys.readouterr().out


def test_an_output_that_changed_on_the_host_after_it_came_back_is_not_kept(tmp_path, host, capsys):
    driver, root, there = finished(tmp_path, host, outputs_of_every_shape)
    (there / "layer.json").write_bytes(b'{"layer": "other"}')   # a process the node left behind, still writing
    (there / "assets" / "late.jpg").write_bytes(b"")            # a file more than came back
    driver.keep_outputs()
    out = capsys.readouterr().out
    assert "layer.json is not kept on lab" in out and "assets is not kept on lab" in out and "[remote] kept" not in out
    assert list((root / "objects").iterdir()) == []             # nothing filed, and nothing of this attempt's left


def test_an_output_with_a_link_in_it_is_never_kept_and_no_link_is_followed(tmp_path, host, capsys):
    from perception import artifacts
    secret = host / "home" / "secret.txt"
    secret.write_text("private")

    def make(out):
        outputs_of_every_shape(out)
        (out / "linked.json").symlink_to(secret)                # an output that is a link
    driver, root, there = finished(tmp_path, host, make)
    (there / "assets" / "crops" / "b.jpg").symlink_to(secret)  # and one made in a tree after it came back
    driver.keep_outputs()
    assert "assets is not kept on lab" in capsys.readouterr().out
    [kept] = (root / "objects").iterdir()                       # only the layer
    assert kept.name == artifacts.sha256_file(driver.local / "out" / "layer.json")
    assert secret.read_text() == "private" and (there / "linked.json").is_symlink()


def test_what_a_process_the_node_left_or_a_hard_link_writes_later_never_reaches_a_kept_output(tmp_path, host):
    """The node has exited, but a process it left (a worker, an ffmpeg child) still holds an output open for writing,
    and another output is a hard link to a file outside out/: whatever is written through either after the output is
    kept must not change it (it would be read under a hash its content no longer has), nor make that file read-only."""
    from perception import artifacts
    outside = host / "home" / "notes.txt"
    outside.write_bytes(b"mine")
    holder = {}

    def make(out):
        (out / "layer.json").write_bytes(b'{"layer": "faces"}')
        holder["f"] = open(out / "layer.json", "r+b")           # held open, as by the process the node left
        os.link(outside, out / "notes.json")                      # the same file as one outside out/
    driver, root, there = finished(tmp_path, host, make)
    driver.keep_outputs()
    layer, notes = (artifacts.sha256_file(driver.local / "out" / n) for n in ("layer.json", "notes.json"))
    holder["f"].write(b'{"layer": "later"}')
    holder["f"].close()
    with open(outside, "r+b") as f:
        f.write(b"MINE")
    for h in (layer, notes):
        kept = root / "objects" / h / "data"
        assert (kept.parent / ".complete").exists() and artifacts.sha256_file(kept) == h
    assert outside.read_bytes() == b"MINE" and outside.stat().st_mode & 0o200   # still the user's own, writable


def test_the_drivers_cleanup_running_while_the_host_still_keeps_an_output_never_publishes_part_of_it(tmp_path, host, monkeypatch):
    """ssh dropped while the host was keeping an output: the driver gave up on it and removes this attempt's unfinished
    uploads (discard_incoming, by its names: .incoming-<tag>-*) while the host's keep goes on. Its cleanup, run part way
    just before the host publishes, must not leave an output with files missing filed under its whole hash, with
    .complete. Keep stages in a folder of its own (.keeping-*) that the cleanup's names never match, so here it removes
    nothing; staged under the attempt's names (as before), the layer was filed with no data."""
    import perception.remote_driver as rd
    from perception import artifacts
    driver, root, there = finished(tmp_path, host, outputs_of_every_shape)
    # the driver's cleanup (its glob), having removed one file of each of this attempt's uploads when publish runs
    cleanup = (f'for d in {root}/objects/.incoming-{driver.tag}-*; do [ -d "$d" ] || continue; chmod -R u+w "$d"; '
               'f=$(find "$d" -type f ! -name .complete | head -n 1); [ -n "$f" ] && rm -f "$f"; done')
    monkeypatch.setattr(rd, "PUBLISH", rd.PUBLISH.replace("publish() {", "real_publish() {", 1)
                        + f'\npublish() {{\n  {cleanup}\n  real_publish "$@"\n}}')
    driver.keep_outputs()
    filed = [p for p in (root / "objects").iterdir() if not p.name.startswith(".")]
    assert len(filed) == 2                                       # the layer and the tree, each whole, under its own hash
    for p in filed:
        assert (p / ".complete").exists() and artifacts.content_hash(p / "data") == p.name, p.name
    assert list((root / "objects").glob(".keeping-*")) == []    # and nothing of the keep's own left


def test_a_keep_a_host_never_finished_is_swept_after_a_day_and_one_under_way_is_left(tmp_path, host):
    driver, root, there = finished(tmp_path, host, lambda out: (out / "layer.json").write_bytes(b"{}"))
    old, new = root / "objects" / ".keeping-old", root / "objects" / ".keeping-new"
    for d in (old, new):
        (d / "obj" / "data").mkdir(parents=True)
        (d / "obj" / "data" / "x").write_bytes(b"part")
    (old / "obj" / "data" / "x").chmod(0o444)
    for p in (old / "obj" / "data" / "x", old / "obj" / "data", old / "obj", old):
        os.utime(p, (time.time() - 2 * 86400,) * 2)
    driver.keep_outputs()
    assert not old.exists() and new.exists()


def test_an_output_the_host_could_not_keep_says_why(tmp_path, host, capsys):
    driver, root, there = finished(tmp_path, host, lambda out: (out / "layer.json").write_bytes(b"{}"))
    (root / "objects").chmod(0o500)                              # as a full disk would: nothing can be written there
    try:
        driver.keep_outputs()
    finally:
        (root / "objects").chmod(0o700)
    assert "[remote] layer.json is not kept on lab (exit 1" in capsys.readouterr().out
    assert list((root / "objects").iterdir()) == []


def test_a_failed_node_or_a_cloud_machine_keeps_nothing(tmp_path, host):
    driver, root, there = finished(tmp_path, host, outputs_of_every_shape)
    (driver.local / "response.json.part").write_text(json.dumps({"ok": False, "outputs": {"layer.json": {"path": "layer.json"}}}))
    driver.keep_outputs()
    driver.spec["cloud"] = {"provider": "runpod"}               # deleted after the attempt: nothing to keep it for
    (driver.local / "response.json.part").write_text(json.dumps({"ok": True, "outputs": {"layer.json": {"path": "layer.json"}}}))
    driver.keep_outputs()
    assert list((root / "objects").iterdir()) == [] and (there / "layer.json").exists()


def test_a_host_without_the_output_kept_there_is_sent_it(tmp_path, host, rec):
    """A host from before outputs were kept, or one `perception hosts clean` emptied: the next node there is sent it."""
    coord = coordinator(tmp_path, host)
    doc = work_graph(tmp_path, rec)
    doc["nodes"].append({"id": "b", "uses": "test.work@1.0.0", "inputs": {"a": "$nodes.a.out"}, "place": "ssh-lab:cpu",
                         "config": {"record_dir": rec, "add": 10}})
    run_id = start(coord, doc)
    assert asyncio.run(coord.run(run_id)) == "succeeded", coord._tasks(run_id)
    kept = host / "home" / ".cache" / "hp" / "objects" / json.loads(coord._tasks(run_id)["a"]["outputs"])["out"]["hash"]
    subprocess.run(["chmod", "-R", "u+w", str(kept)], check=True)
    subprocess.run(["rm", "-rf", str(kept)], check=True)
    coord.store.db.execute("DELETE FROM cache WHERE key=?", (coord._tasks(run_id)["b"]["cache_key"],))   # b runs again; a is cached
    run_id = start(coord, doc)
    assert asyncio.run(coord.run(run_id)) == "succeeded", coord._tasks(run_id)
    assert coord._tasks(run_id)["a"]["state"] == "cached"
    assert "[remote] copying a " in log_of(coord, run_id, "b")


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
    sha = coord.store.source(link, "video.v1").hash
    stored = host / "home" / ".cache" / "hp" / "objects" / sha / "data"
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


def test_inputs_and_outputs_kept_on_a_host_are_read_only_and_still_cleaned(tmp_path, host, monkeypatch, capsys):
    from perception.cli import main
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "home"))
    assert main(["hosts", "add", "lab", "lab"]) == 0
    coord = Coordinator(Store(tmp_path / "home"), Registry.load(NODES),
                        [SelfBackend(), SshBackend(Hosts(tmp_path / "home").load()["lab"], runner_kind="self", python=sys.executable)],
                        tick_s=0.02, backoff_s=0.01)
    run_id = start(coord, video_graph(tmp_path, place="ssh-lab:cpu"))
    assert asyncio.run(coord.run(run_id)) == "succeeded"
    objects = host / "home" / ".cache" / "hyperstudy-perception" / "objects"
    stored = list(objects.glob("*/data"))
    assert len(stored) == 3                                 # the video, and the two outputs the node made there
    for path in (*stored, *(p.parent for p in stored)):     # a node cannot change it through its link, as in the store
        assert path.stat().st_mode & 0o222 == 0, path
    envs = objects.parent / "envs" / "0123abcd"                  # an environment kept there (make_env)
    (envs / ".venv").mkdir(parents=True)
    assert main(["hosts", "clean", "lab"]) == 0
    assert not objects.exists() and not envs.parent.exists()


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


def test_a_result_made_on_a_host_is_reused_on_any_machine_of_its_kind_and_says_where_it_was_made(tmp_path, host, rec):
    """The cache identity names no machine (Backend.identity): the same machine added under a second name is served the
    result, and so is this one (same platform, the same `self` runner); a machine of another platform is not."""
    from perception.cli import run_json
    again = Host("lab-again", "lab", str(host / "home" / ".cache" / "hp"), lab(host).caps, time.time())
    other = lab(host, platform="linux/arm64" if this_platform() != "linux/arm64" else "linux/amd64")
    other.name = "arm"
    coord = Coordinator(Store(tmp_path / "store"), Registry.load(NODES),
                        [SelfBackend(), *(SshBackend(h, runner_kind="self", python=sys.executable) for h in (lab(host), again, other))],
                        tick_s=0.02, backoff_s=0.01)
    run_id = start(coord, work_graph(tmp_path, rec, place="ssh-lab:cpu"))
    assert asyncio.run(coord.run(run_id)) == "succeeded", coord._tasks(run_id)
    made = {"place": "ssh-lab:cpu", "title": "lab (SSH)", "platform": this_platform()}
    assert json.loads(coord._tasks(run_id)["a"]["usage"])["ran_on"] == made
    for place, cached in [("ssh-lab-again:cpu", True), ("self:cpu", True), ("ssh-arm:cpu", False)]:
        planned = make_plan(coord, compile_graph(work_graph(tmp_path, rec, place=place), coord.registry), allow_uploads=True)
        assert (planned.nodes[0].place, planned.nodes[0].cached) == (place, cached)
    again_id = start(coord, work_graph(tmp_path, rec, place="ssh-lab-again:cpu"))
    assert asyncio.run(coord.run(again_id)) == "succeeded"
    (task,) = run_json(coord.store, coord.store.db.execute("SELECT * FROM runs WHERE id=?", (again_id,)).fetchone())["tasks"]
    assert task["state"] == "cached" and task["ran_on"] == made   # where the result it was served was made
    assert len([p for p in Path(rec).glob("a-*") if p.suffix != ".end"]) == 1, "the node ran once"


def test_a_result_cached_under_the_old_identity_is_made_again_not_misread(tmp_path, host, rec):
    """Before 2026-10-28 a result's identity began with the backend's kind (`ssh-lab:...`); a store holding one keeps
    it, but no key made now is that key, so the node runs once more under the new rule."""
    import hashlib
    from perception import artifacts
    from perception.store import dumps
    coord = coordinator(tmp_path, host)
    doc = work_graph(tmp_path, rec)
    graph = compile_graph(doc, coord.registry)
    node = graph.nodes["a"]
    seed = coord._graph_input(graph, "seed")
    stale = coord.store.publish(Path(doc["inputs"]["seed"]["uri"]), "json.v1")   # some object the store has
    old = hashlib.sha256(artifacts.canonical({
        "node": node.manifest.ref, "runner": "ssh-lab:0123456789abcdef", "config": node.config,
        "inputs": {"a": [seed.type, seed.hash]}, "outputs": node.manifest.outputs, "partition": None}).encode()).hexdigest()
    with coord.store.tx() as db:
        db.execute("INSERT INTO cache VALUES (?,?,?,?,?)", (old, node.manifest.ref, dumps({"out": stale.to_json()}), "{}", time.time()))
    assert not make_plan(coord, graph, allow_uploads=True).nodes[0].cached
    run_id = start(coord, doc)
    assert asyncio.run(coord.run(run_id)) == "succeeded"
    t = coord._tasks(run_id)["a"]
    assert t["state"] == "succeeded" and t["cache_key"] != old


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


def test_a_weight_only_another_runner_loads_is_no_download(tmp_path, host, monkeypatch):
    """A model declared for other runners' environments (audio.asr: the MLX weights, the NeMo ones) is not the placed
    runner's: never a download for it, here or on another machine, and not among the models the plan says it uses."""
    import perception.plan as plan_mod
    from perception.manifests import ModelRef
    monkeypatch.setattr(plan_mod, "hf_cached", lambda source: False)         # nothing on this machine
    coord = coordinator(tmp_path, host)
    graph = compile_graph(video_graph(tmp_path, place="ssh-lab:cpu"), coord.registry)
    m = graph.nodes["shots"].manifest
    monkeypatch.setattr(m, "models", [ModelRef(id="w", source="hf://org/weights", licence_mode="commercial-ok"),
                                      ModelRef(id="other", source="hf://org/other", licence_mode="commercial-ok",
                                               envs=["envs/another-runners"])])
    there = make_plan(coord, graph, allow_uploads=True)
    assert [d.id for d in there.nodes[0].downloads] == ["w"] and [x.id for x in there.nodes[0].models] == ["w"]
    here = make_plan(coord, compile_graph(video_graph(tmp_path), coord.registry), allow_uploads=True)
    assert [d.id for d in here.nodes[0].downloads] == ["w"] and [x.id for x in here.nodes[0].models] == ["w"]


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


def test_a_node_whose_prepare_fails_on_the_host_never_starts_and_is_not_retried(tmp_path, host, rec):
    coord = coordinator(tmp_path, host)
    doc = work_graph(tmp_path, rec, fail="prepare")
    doc["policy"]["retries"] = 2
    run_id = start(coord, doc)
    assert asyncio.run(coord.run(run_id)) == "failed"
    t = coord._tasks(run_id)["a"]
    assert (t["error_code"], t["attempt"]) == ("internal", 1) and "preparing the node failed" in t["error"]
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


# --- user, port and identity over ~/.ssh/config; a probe that changes nothing -----------------------------------

LOGGING_SSH = FAKE_SSH.replace('while [ $# -gt 0 ]; do case "$1" in -o) shift 2;; -*) shift;; *) break;; esac; done',
                               'while [ $# -gt 0 ]; do case "$1" in -o) case "$2" in Control*) echo "$2" >> "$FAKE_HOST/control";; esac; shift 2;; '
                               '-l|-p|-i) echo "$1 $2" >> "$FAKE_HOST/options"; shift 2;; -*) shift;; *) break;; esac; done')


@pytest.fixture
def logging_host(host, tmp_path):
    """The fake host, its ssh stand-in also writing each -l, -p and -i it is given to host/options."""
    (tmp_path / "ssh").write_text(LOGGING_SSH)
    return host


def test_a_hosts_user_port_and_key_go_with_every_command_sent_to_it(tmp_path, logging_host, monkeypatch, capsys):
    from perception.cli import main
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "home"))
    key = tmp_path / "id_lab"
    key.write_text("not really a key")
    assert main(["hosts", "add", "lab", "lab.example.org", "--user", "ana", "--port", "2222", "--identity", str(key)]) == 0
    sent = (logging_host / "options").read_text().splitlines()
    assert sent == ["-l ana", "-p 2222", f"-i {key.resolve()}"]                      # the probe when it was added
    saved = Hosts(tmp_path / "home").load()["lab"]
    assert (saved.user, saved.port, saved.identity) == ("ana", 2222, str(key.resolve()))
    assert main(["hosts", "check", "lab"]) == 0
    assert main(["hosts", "clean", "lab"]) == 0
    sent = (logging_host / "options").read_text().splitlines()
    assert sent == ["-l ana", "-p 2222", f"-i {key.resolve()}"] * 3                  # check and clean too
    remote = SshBackend(saved, runner_kind="self", python=sys.executable)
    assert remote.host.ssh()[-6:] == ["-p", "2222", "-i", str(key.resolve()), "-o", "IdentitiesOnly=yes"]


def test_a_host_saved_before_overrides_existed_loads_without_them(tmp_path):
    hosts = Hosts(tmp_path)
    old = lab(tmp_path).to_json()
    for k in ("user", "port", "identity"):
        old.pop(k)
    hosts.path.write_text(json.dumps({"lab": old}))
    h = hosts.load()["lab"]
    assert (h.user, h.port, h.identity) == (None, None, None) and "-l" not in h.ssh()


def test_overrides_ssh_would_misread_are_refused(tmp_path, host):
    hosts = Hosts(tmp_path / "home")
    with pytest.raises(HostError, match="give the user once"):
        hosts.add("lab", "ana@lab", user="bob")
    with pytest.raises(HostError, match="not a port"):
        hosts.add("lab", "lab", port=70000)
    with pytest.raises(HostError, match="no key file"):
        hosts.add("lab", "lab", identity=str(tmp_path / "missing"))
    with pytest.raises(HostError, match="not a user name"):
        hosts.add("lab", "lab", user="-oProxyCommand=x")
    for odd in ("id_%u", "id_${HOME}"):                       # ssh expands these in a key file's path: not the file checked
        (tmp_path / odd).write_text("not really a key")
        with pytest.raises(HostError, match="ssh would expand"):
            hosts.add("lab", "lab", identity=str(tmp_path / odd))
    assert hosts.load() == {}


def test_a_probe_says_what_a_destination_has_and_changes_nothing_there_or_here(tmp_path, logging_host, monkeypatch, capsys):
    from perception.cli import main
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "home"))
    assert main(["--json", "hosts", "probe", "lab.example.org", "--port", "2222"]) == 0
    [found] = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert found["type"] == "host_probe" and found["dest"] == "lab.example.org"
    assert found["caps"]["cpu_cores"] > 0 and found["root"].endswith("/.cache/hyperstudy-perception")
    assert not (logging_host / "home" / ".cache").exists()                          # no folder made on the host
    assert not (tmp_path / "home" / "hosts.json").exists()                          # and no host added here
    assert (logging_host / "options").read_text().splitlines() == ["-p 2222"]
    (logging_host / "unreachable").touch()
    assert main(["--json", "hosts", "probe", "lab.example.org"]) == 1
    [err] = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert err["type"] == "error" and "could not reach lab.example.org" in err["problems"][0]


def test_a_connection_test_never_rides_a_shared_connection_and_a_hosts_key_shares_only_with_itself(tmp_path, logging_host, monkeypatch):
    """ssh shares a connection by host, port and user, not by key: a test that rode one opened with another key would
    say yes to a key the server refuses."""
    from perception.cli import main
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "home"))
    key = tmp_path / "id_lab"
    key.write_text("not really a key")
    assert main(["hosts", "probe", "lab", "--identity", str(key)]) == 0
    assert main(["hosts", "add", "lab", "lab", "--identity", str(key)]) == 0
    assert main(["hosts", "check", "lab"]) == 0
    tests = (logging_host / "control").read_text().splitlines()
    assert tests == ["ControlMaster=no", "ControlPath=none"] * 3                     # probe, add, check: each a fresh connection
    shared = Hosts(tmp_path / "home").load()["lab"].ssh()
    path = next(o for o in shared if o.startswith("ControlPath="))
    other = Host("lab", "lab", "/r", lab(tmp_path).caps, 0, identity=str(tmp_path / "id_other")).ssh()
    plain = Host("lab", "lab", "/r", lab(tmp_path).caps, 0).ssh()
    assert path.endswith("%C-" + __import__("hashlib").sha256(str(key.resolve()).encode()).hexdigest()[:12])
    assert len({path, next(o for o in other if o.startswith("ControlPath=")), next(o for o in plain if o.startswith("ControlPath="))}) == 3


# --- environments on a host, by their lock ---------------------------------------------------------------------------

FAKE_UV = r"""#!/bin/sh
# uv sync|run --project PROJECT [FLAG]... [python ARG...], on the fake host: each call recorded in $FAKE_HOST/uv.log with
# the SDK path it was given. `sync` makes PROJECT/.venv, a real virtual environment of the tests' interpreter ("made"),
# or finds it there ("found"). The one it makes holds a stale SDK, as an editable install from an earlier version of the
# code would leave it (a .pth naming that version's sdk/src): a node must never import it. `run` runs ARGs with the
# environment's Python.
cmd=$1; shift
project=; prev=
for a in "$@"; do [ "$prev" = --project ] && project=$a; prev=$a; done
case "$cmd" in
  sync) if [ -d "$project/.venv" ]; then made=found
        else
          @PYTHON@ -m venv --without-pip "$project/.venv" || exit 1
          mkdir -p "$project/.venv/stale-sdk/perception_node"
          echo 'raise ImportError("the stale SDK an earlier version of the code installed")' > "$project/.venv/stale-sdk/perception_node/__init__.py"
          site=$("$project/.venv/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
          echo "$project/.venv/stale-sdk" > "$site/_perception_node.pth"
          made=made
        fi
        echo "sync $made $project | $*" >> "$FAKE_HOST/uv.log";;
  run) echo "run $project ${PYTHONPATH-} | $*" >> "$FAKE_HOST/uv.log"
       while [ "$1" != python ]; do shift; done; shift
       exec "$project/.venv/bin/python" "$@";;
  *) exit 2;;
esac
"""


def native_coordinator(tmp_path, fake) -> Coordinator:
    """A host whose nodes run in uv environments (`native` runners), with a stand-in for uv there."""
    uv = tmp_path / "uv"
    uv.write_text(FAKE_UV.replace("@PYTHON@", sys.executable))
    uv.chmod(0o755)
    remote = SshBackend(lab(fake, tools={"uv": str(uv), "ffmpeg": None, "ffprobe": None}))
    return Coordinator(Store(tmp_path / "store"), Registry.load(NODES), [SelfBackend(), remote], tick_s=0.02, backoff_s=0.01)


def test_nodes_with_the_same_lock_share_one_environment_on_a_host_and_each_imports_its_own_codes_sdk(tmp_path, host, rec):
    """Two nodes with different code and the same uv project (envs/core): one environment on the host, made by the first
    sync and found by the second, and each node runs in it with the SDK from its own code's folder. Never one the
    environment holds: the host never installs the SDK there (--no-install-package; --no-sync when the node runs), and
    even an SDK an editable install left there (a .pth, which the stand-in plants) is not what the node imports, since
    PYTHONPATH comes before site-packages. The SDK a node imports is then the one its cache key hashed."""
    from perception.backends.remote import code_files, code_hash, env_key
    coord = native_coordinator(tmp_path, host)
    seed = tmp_path / "seed.json"
    seed.write_text('{"value": 0}')
    node = lambda id, uses, a: {"id": id, "uses": uses, "inputs": {"a": a}, "place": "ssh-lab:cpu", "config": {"record_dir": rec}}
    doc = {"schema_version": 1, "pipeline": "test", "inputs": {"seed": {"type": "json.v1", "uri": str(seed)}},
           "policy": {"retries": 0, "placement": "hybrid", "privacy": "local-preferred", "allow_upload": ALL_KINDS},
           "nodes": [node("a", "test.native-a@1.0.0", "$inputs.seed"), node("b", "test.native-b@1.0.0", "$nodes.a.out")]}
    run_id = start(coord, doc)
    assert asyncio.run(coord.run(run_id)) == "succeeded", log_of(coord, run_id, "a") + log_of(coord, run_id, "b")
    root = host / "home" / ".cache" / "hp"
    codes, keys = {}, set()
    for id, ref in (("a", "test.native-a@1.0.0"), ("b", "test.native-b@1.0.0")):
        m = coord.registry.get(ref)
        files = code_files(m, m.runners[0])
        codes[id] = code_hash(files)
        keys.add(env_key(files, "envs/core"))
    assert codes["a"] != codes["b"]                         # two versions of the code
    [key] = keys                                            # one environment's key
    [env] = [p for p in (root / "envs").iterdir()]          # one environment, under its lock's key
    assert env.name == key and sorted(p.name for p in (root / "code").iterdir()) == sorted(codes.values())
    assert sorted(str(p.relative_to(env)) for p in env.rglob("*") if ".venv/" not in str(p)) == [
        ".complete", "envs", "envs/core", "envs/core/.venv", "envs/core/pyproject.toml", "envs/core/uv.lock", "sdk", "sdk/pyproject.toml"]
    assert not list((root / "code").glob("*/envs/core/.venv"))   # none in the code's folders
    project = f"{env}/envs/core"
    calls = [line.split(" | ")[0].split() + [line.split(" | ")[1]] for line in (host / "uv.log").read_text().splitlines()]
    syncs = [c for c in calls if c[0] == "sync"]
    assert [c[1:3] for c in syncs] == [["made", project], ["found", project]]
    assert all("--frozen --no-dev --no-install-package perception-node" in c[3] for c in syncs)
    runs = [c for c in calls if c[0] == "run"]               # prepare and the node, for each
    assert {c[1] for c in runs} == {project} and all("--frozen --no-dev --no-sync python" in c[3] for c in runs)
    assert sorted({c[2] for c in runs}) == sorted(f"{root}/code/{h}/sdk/src" for h in codes.values())
    sdks = {e["node"]: e["sdk"] for e in executions(rec)}
    assert sdks == {id: f"{root}/code/{codes[id]}/sdk/src/perception_node/__init__.py" for id in ("a", "b")}


def executions(rec: str) -> list[dict]:
    out = []
    for p in sorted(Path(rec).glob("*-*-*")):
        if p.suffix != ".end":
            name, attempt, pid = p.name.rsplit("-", 2)
            out.append({"node": name, "attempt": int(attempt), **json.loads(p.read_text())})
    return out


def test_an_environments_key_is_its_lock_not_the_nodes_code_nor_the_sdks(tmp_path):
    """A change to a node's code or to the SDK's code reuses the host's environment; a change to the lock (or to the SDK's
    dependencies) is another environment."""
    import shutil
    from perception.backends.native import REPO
    from perception.backends.remote import code_files, env_files, env_key
    m = Registry.load(NODES).get("test.native-a@1.0.0")
    files = code_files(m, m.runners[0])
    assert env_files(files, "envs/core") == ["envs/core/pyproject.toml", "envs/core/uv.lock", "sdk/pyproject.toml"]
    assert env_key(files, None) is None                     # no uv project (a `self` runner): no environment
    for f in files:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, tmp_path / f)
    key = env_key(files, "envs/core", tmp_path)
    assert key == env_key(files, "envs/core")
    for edited, same in (("orchestrator/tests/nodes/work/node.py", True), ("sdk/src/perception_node/__init__.py", True),
                         ("envs/core/uv.lock", False), ("sdk/pyproject.toml", False)):
        before = (tmp_path / edited).read_bytes()
        (tmp_path / edited).write_bytes(before + b"\n# an edit\n")
        assert (env_key(files, "envs/core", tmp_path) == key) is same, edited
        (tmp_path / edited).write_bytes(before)


def test_every_environment_has_the_sdk_that_a_host_leaves_out_and_imports_from_the_code():
    """The SDK a host leaves out of shared environments (uv sync --no-install-package) is each environment's path
    dependency on sdk/, and what it imports from the code's folder (PYTHONPATH) is where the package is."""
    import tomllib
    from perception.backends.native import REPO
    from perception.backends.remote import SDK_PACKAGE, SDK_PROJECT, SDK_SOURCE
    assert tomllib.loads((REPO / SDK_PROJECT).read_text())["project"]["name"] == SDK_PACKAGE
    assert (REPO / SDK_SOURCE / "perception_node" / "__init__.py").is_file()
    envs = sorted((REPO / "envs").glob("*/pyproject.toml"))
    assert len(envs) >= 9
    for p in envs:
        doc = tomllib.loads(p.read_text())
        assert SDK_PACKAGE in doc["project"]["dependencies"], p
        assert doc["tool"]["uv"]["sources"][SDK_PACKAGE] == {"path": "../../sdk", "editable": True}, p


# --- a host's GPUs ---------------------------------------------------------------------------------------------------

FAKE_SMI = r"""#!/bin/sh
# nvidia-smi --query-gpu=index,memory.total,name --format=csv,noheader,nounits, as liquidswords2 answers it
echo "0, 24576, NVIDIA GeForce RTX 3090"
echo "1, 97887, NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
"""


def test_nvidia_smis_cards_are_read_and_a_card_it_could_not_fill_is_left_out():
    from perception.backends.base import Gpu
    from perception.backends.probe import parse_gpus
    text = "1, 97887, NVIDIA RTX PRO 6000, a name with a comma\n0, 24576, NVIDIA GeForce RTX 3090\n2, [N/A], NVIDIA A100\n\n"
    assert parse_gpus(text) == [Gpu(0, "NVIDIA GeForce RTX 3090", 24.0), Gpu(1, "NVIDIA RTX PRO 6000, a name with a comma", 95.6)]


def test_a_host_is_probed_for_every_gpu_and_one_saved_before_cards_were_recorded_counts_as_one(tmp_path, host, monkeypatch):
    from perception.backends.base import Gpu
    smi = host / "home" / ".local" / "bin" / "nvidia-smi"   # on the PATH the probe adds
    smi.parent.mkdir(parents=True)
    smi.write_text(FAKE_SMI)
    smi.chmod(0o755)
    hosts = Hosts(tmp_path / "home")
    added = hosts.add("lab", "lab")
    assert added.caps.gpus == [Gpu(0, "NVIDIA GeForce RTX 3090", 24.0), Gpu(1, "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 95.6)]
    assert "cuda" in added.caps.accelerators and added.caps.vram_gb == 95.6   # the largest card: what runners are checked against
    assert hosts.load()["lab"].caps == added.caps
    # as hosts.json held it before the cards were recorded: no `gpus`, the largest card's memory alone
    old = json.loads(hosts.path.read_text())
    old["lab"]["caps"].pop("gpus")
    hosts.path.write_text(json.dumps(old))
    caps = hosts.load()["lab"].caps
    assert caps.gpus == [] and caps.vram_gb == 95.6 and caps.devices() == [Gpu(None, "", 95.6)]
    assert hosts.check("lab").caps.gpus == added.caps.gpus   # recorded again by a check, not a new `add`
    # a host whose own environment keeps CUDA off a card: only the cards it leaves can be given (probe.visible_gpus)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")             # (the stand-in ssh passes this one's environment on)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    assert [g.index for g in hosts.check("lab").caps.gpus] == [1]
    monkeypatch.delenv("CUDA_DEVICE_ORDER")                       # CUDA's own order: not nvidia-smi's, so no card
    assert hosts.check("lab").caps.gpus == []


def test_two_exclusive_gpu_nodes_on_a_host_with_two_cards_run_together_each_on_its_own_card(tmp_path, host, rec):
    from perception.backends.base import Gpu
    remote = SshBackend(lab(host, accelerators={"cpu", "cuda"}, vram_gb=95.6,
                            gpus=[Gpu(0, "RTX 3090", 24.0), Gpu(1, "RTX PRO 6000", 95.6)]), runner_kind="self", python=sys.executable)
    coord = Coordinator(Store(tmp_path / "store"), Registry.load(NODES), [SelfBackend(), remote], tick_s=0.02, backoff_s=0.01)
    seed = tmp_path / "seed.json"
    seed.write_text('{"value": 0}')
    gpu = lambda id: {"id": id, "uses": "test.gpu@1.0.0", "inputs": {"a": "$inputs.seed"}, "place": "ssh-lab:cpu",
                      "config": {"record_dir": rec, "sleep": 2, "add": int(id[1])}}
    doc = {"schema_version": 1, "pipeline": "test", "inputs": {"seed": {"type": "json.v1", "uri": str(seed)}},
           "policy": {"retries": 0, "placement": "hybrid", "privacy": "local-preferred", "allow_upload": ALL_KINDS},
           "nodes": [gpu("g1"), gpu("g2"), gpu("g3")]}
    run_id = start(coord, doc)
    status = asyncio.run(coord.run(run_id))
    t = coord._tasks(run_id)
    assert status == "succeeded", {n: (x["state"], x["error_code"], x["error"], log_of(coord, run_id, n)[-1500:]) for n, x in t.items()}
    t = coord._tasks(run_id)
    first, second, third = sorted(t.values(), key=lambda x: x["started_at"])
    assert {first["gpu"], second["gpu"]} == {0, 1}                       # one on each card, at once
    assert second["started_at"] < first["finished_at"]
    assert third["started_at"] >= min(first["finished_at"], second["finished_at"])   # a third waits for a card
    seen = {e["node"]: (e["cuda"], e["order"]) for e in executions(rec)}
    assert seen == {x["node_id"]: (str(x["gpu"]), "PCI_BUS_ID") for x in t.values()}   # the node sees its card alone
    assert "GPU 1 (RTX PRO 6000)" in " ".join(e["message"] or "" for e in coord.store.db.execute(
        "SELECT message FROM events WHERE run_id=? AND kind='task.started'", (run_id,)))



def test_two_coordinators_never_put_exclusive_nodes_on_one_card_of_a_host(tmp_path, host, rec):
    """Two runs on a host's cards from two coordinators (two stores here; or this Mac and the lab server itself): each
    one's scheduler counts only its own nodes, and both choose the larger card. The card's lease on the host (GPU_LEASES)
    makes the second wait until the first node ends, and its log says why."""
    from perception.backends.base import Gpu
    seed = tmp_path / "seed.json"
    seed.write_text('{"value": 0}')
    coords, runs = [], []
    for i in (1, 2):
        remote = SshBackend(lab(host, accelerators={"cpu", "cuda"}, vram_gb=95.6,
                                gpus=[Gpu(0, "RTX 3090", 24.0), Gpu(1, "RTX PRO 6000", 95.6)]), runner_kind="self", python=sys.executable)
        coord = Coordinator(Store(tmp_path / f"store{i}"), Registry.load(NODES), [SelfBackend(), remote], tick_s=0.02, backoff_s=0.01)
        doc = {"schema_version": 1, "pipeline": "test", "inputs": {"seed": {"type": "json.v1", "uri": str(seed)}},
               "policy": {"retries": 0, "placement": "hybrid", "privacy": "local-preferred", "allow_upload": ALL_KINDS},
               "nodes": [{"id": f"g{i}", "uses": "test.gpu@1.0.0", "inputs": {"a": "$inputs.seed"}, "place": "ssh-lab:cpu",
                          "config": {"record_dir": rec, "sleep": 2, "add": i}}]}
        coords.append(coord)
        runs.append(start(coord, doc))

    async def both():
        return await asyncio.gather(*(c.run(r) for c, r in zip(coords, runs)))
    assert asyncio.run(both()) == ["succeeded", "succeeded"]
    assert [c._tasks(r)[f"g{i}"]["gpu"] for i, (c, r) in enumerate(zip(coords, runs), 1)] == [1, 1]   # both chose card 1
    first, second = sorted(executions(rec), key=lambda e: e["start"])
    end = json.loads(next(Path(rec).glob(f"{first['node']}-*.end")).read_text())["end"]
    assert second["start"] >= end                                         # never on the card at once
    logs = [log_of(c, r, f"g{i}") for i, (c, r) in enumerate(zip(coords, runs), 1)]
    waited = [log for log in logs if "another run on this machine has that GPU" in log]
    assert len(waited) == 1, logs
    run_of = {f"g{i}": r for i, r in enumerate(runs, 1)}
    assert f"(held by exclusive run {run_of[first['node']]} node {first['node']} attempt 1 pid " in waited[0]   # and by whom
    # in the user's default root on the host (whatever root it was added with), naming only the node that held it last
    lease = host / "home" / ".cache" / "hyperstudy-perception" / "gpus" / f"{os.uname().nodename}-1"
    assert lease.read_text().startswith(f"exclusive run {run_of[second['node']]} node {second['node']} attempt 1 pid ")
    assert lease.read_text().count("\n") == 1


def test_a_coordinator_on_the_host_and_one_elsewhere_lease_the_same_card_whatever_root_the_host_was_added_with(tmp_path, host, rec, monkeypatch):
    """A coordinator on the lab server itself (a native node, on its own cards) and one elsewhere that added the server with
    a root of its own (`--root`): the leases are in the user's default root either way, so the two exclusive nodes that
    both choose card 1 never share it."""
    import dataclasses
    from perception.backends.base import Gpu
    cards = [Gpu(0, "RTX 3090", 24.0), Gpu(1, "RTX PRO 6000", 95.6)]

    class TheServer(SelfBackend):   # the coordinator on the server: this machine, whose home is the fake host's
        def probe_capabilities(self):
            return dataclasses.replace(super().probe_capabilities(), accelerators={"cpu", "cuda"}, vram_gb=95.6, gpus=cards)
    monkeypatch.setenv("HOME", str(host / "home"))
    seed = tmp_path / "seed.json"
    seed.write_text('{"value": 0}')
    remote = SshBackend(lab(host, accelerators={"cpu", "cuda"}, vram_gb=95.6, gpus=cards), runner_kind="self", python=sys.executable)
    assert not remote.host.root.endswith("/.cache/hyperstudy-perception")   # added with a root of its own
    coords, runs = [], []
    for i, (backend, place) in enumerate(((TheServer(), "self:cpu"), (remote, "ssh-lab:cpu")), 1):
        coord = Coordinator(Store(tmp_path / f"store{i}"), Registry.load(NODES), [backend], tick_s=0.02, backoff_s=0.01)
        doc = {"schema_version": 1, "pipeline": "test", "inputs": {"seed": {"type": "json.v1", "uri": str(seed)}},
               "policy": {"retries": 0, "placement": "hybrid", "privacy": "local-preferred", "allow_upload": ALL_KINDS},
               "nodes": [{"id": f"g{i}", "uses": "test.gpu@1.0.0", "inputs": {"a": "$inputs.seed"}, "place": place,
                          "config": {"record_dir": rec, "sleep": 2, "add": i}}]}
        coords.append(coord)
        runs.append(start(coord, doc))

    async def both():
        return await asyncio.gather(*(c.run(r) for c, r in zip(coords, runs)))
    assert asyncio.run(both()) == ["succeeded", "succeeded"]
    assert [c._tasks(r)[f"g{i}"]["gpu"] for i, (c, r) in enumerate(zip(coords, runs), 1)] == [1, 1]
    first, second = sorted(executions(rec), key=lambda e: e["start"])
    end = json.loads(next(Path(rec).glob(f"{first['node']}-*.end")).read_text())["end"]
    assert second["start"] >= end                                         # never on the card at once


def test_a_lease_names_only_the_nodes_holding_it_now(tmp_path):
    """Shared nodes, one after another and side by side: the lease file never keeps the lines of nodes that ended, so
    it stays small on a card that only ever has shared nodes, and says who holds the card."""
    from perception.backends.base import lease_command
    path = tmp_path / f"{os.uname().nodename}-0"
    for i in range(3):   # one after another
        subprocess.run([*lease_command(str(tmp_path), 0, "shared", f"node s{i}"), "true"], check=True)
    assert path.read_text().splitlines()[0].startswith("shared node s2 pid ") and path.read_text().count("\n") == 1
    first = subprocess.Popen([*lease_command(str(tmp_path), 0, "shared", "node a"), "sleep", "1"])
    time.sleep(0.3)
    subprocess.run([*lease_command(str(tmp_path), 0, "shared", "node b"), "true"], check=True)   # beside a: both named
    assert [line.split(" pid ")[0] for line in path.read_text().splitlines()] == ["shared node a", "shared node b"]
    first.wait()
    subprocess.run([*lease_command(str(tmp_path), 0, "exclusive", "node x"), "true"], check=True)
    assert [line.split(" pid ")[0] for line in path.read_text().splitlines()] == ["exclusive node x"]
