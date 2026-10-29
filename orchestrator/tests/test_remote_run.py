"""A whole run coordinated on another machine (remote_run.py, remote_coordinator.py), against a stand-in for ssh that
runs each command on this machine, in a separate home (the "host") and with an empty environment, as ssh gives a remote
command: nothing of this process's environment (its keys above all) reaches the host but what is sent. Everything else
is real: the listed copy of the pipeline, rsync, the orchestrator run from that copy there, the detached coordinator, the
nodes it runs, and the import here."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from perception import cli, remote_run
from perception.backends.base import Capabilities, this_platform
from perception.backends.native import SelfBackend
from perception.backends.remote import SshBackend
from perception.hosts import Host, Hosts
from perception.ir import compile_graph
from perception.manifests import Registry
from perception.plan import plan as make_plan
from perception.scheduler import Coordinator
from perception.store import Store

NODES = Path(__file__).parent / "nodes"
KEY = "sk-test-" + "9f3c1d7e" * 4   # a stand-in key: what must never be found where it does not belong
ALL_KINDS = ["video", "audio", "frames", "faces", "transcript", "embeddings"]

# ssh [-o OPTION]... DEST COMMAND, run here in the host's home with nothing of this environment but PATH. Each command
# line is recorded (argv.log); stdin is recorded only for a start (start.stdin, outside the folders searched for the key)
FAKE_SSH = r"""#!/bin/sh
while [ $# -gt 0 ]; do case "$1" in -o|-l|-p|-i) shift 2;; -*) shift;; *) break;; esac; done
shift
printf '%s\n' "$*" >> "$FAKE_HOST/argv.log"
if [ -e "$FAKE_HOST/unreachable" ]; then echo "ssh: connect to host lab: Network is unreachable" >&2; exit 255; fi
case "$*" in
  *"remote_coordinator start"*)
    tee -a "$FAKE_RECORD/start.stdin" | env -i HOME="$FAKE_HOST/home" PATH="$PATH" /bin/sh -c "cd \"\$HOME\" && $*"; exit $?;;
esac
cd "$FAKE_HOST/home" && exec env -i HOME="$FAKE_HOST/home" PATH="$PATH" /bin/sh -c "$*"
"""


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A host `lab` (its home is tmp_path/host/home), this machine's store in tmp_path/mac, and TEST_API_KEY set here."""
    fake = tmp_path / "host"
    (fake / "home").mkdir(parents=True)
    record = tmp_path.parent / f"{tmp_path.name}-record"   # outside tmp_path: the searches for the key skip it
    record.mkdir()
    ssh = tmp_path / "ssh"
    ssh.write_text(FAKE_SSH)
    ssh.chmod(0o755)
    monkeypatch.setenv("PERCEPTION_SSH", str(ssh))
    monkeypatch.setenv("FAKE_HOST", str(fake))
    monkeypatch.setenv("FAKE_RECORD", str(record))
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "mac"))
    monkeypatch.setenv("TEST_API_KEY", KEY)
    # the host's Python: this interpreter (it has the orchestrator's dependencies), not an environment made there
    monkeypatch.setattr(remote_run.Remote, "interpreter", property(lambda self: sys.executable))
    caps = Capabilities(platform=this_platform(), cpu_cores=8, memory_gb=32, accelerators={"cpu"}, vram_gb=0,
                        tools={"uv": None, "ffmpeg": None, "ffprobe": None})
    host = Host("lab", "lab", str(fake / "home" / ".cache" / "hp"), caps, time.time())
    Hosts(tmp_path / "mac").save({"lab": host})
    return fake


def keys_on(tmp_path):
    Hosts(tmp_path / "mac").allow_keys("lab", True)


def graph_file(tmp_path, nodes, **policy) -> Path:
    seed = tmp_path / "seed.json"
    seed.write_text('{"value": 0}')
    doc = {"schema_version": 1, "pipeline": "test", "inputs": {"seed": {"type": "json.v1", "uri": str(seed)}},
           "policy": {"placement": "hybrid", "privacy": "local-preferred", "allow_upload": ALL_KINDS, "retries": 1, **policy},
           "nodes": nodes}
    path = tmp_path / "graph.yaml"
    path.write_text(json.dumps(doc))
    return path


def work(node_id, a="$inputs.seed", **config):
    return {"id": node_id, "uses": "test.work@1.0.0", "inputs": {"a": a}, "config": config}


def keyed(node_id, a, rec):
    return {"id": node_id, "uses": "test.keyed@1.0.0", "inputs": {"a": a}, "config": {"record_dir": rec}}


def perception(capsys, *argv) -> tuple[int, list[dict]]:
    """The CLI with --json and this repository's test nodes: its exit code and what it printed."""
    code = cli.main(["--json", "--nodes", str(NODES), *argv])
    return code, [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]


def of(lines, type_):
    return [x for x in lines if x["type"] == type_]


def host_root(fake) -> Path:
    return fake / "home" / ".cache" / "hp"


def executions(rec: Path, node: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(rec.glob(f"{node}-*")) if not p.name.endswith(".end")]


def wait_for(check, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if check():
            return True
        time.sleep(0.1)
    return False


def starts(tmp_path) -> list[dict]:
    """What each start was sent on stdin, in order (the stand-in records them one after another)."""
    text, out, at = (tmp_path.parent / f"{tmp_path.name}-record" / "start.stdin").read_text(), [], 0
    while at < len(text):
        doc, at = json.JSONDecoder().raw_decode(text, at)
        out.append(doc)
    return out


def files_holding(folder: Path, text: str) -> list[Path]:
    needle = text.encode()
    return [p for p in folder.rglob("*") if p.is_file() and not p.is_symlink() and needle in p.read_bytes()]


# --- a run there ---------------------------------------------------------------------------------------------------

def test_a_run_coordinated_on_a_host_runs_there_detached_and_is_fetched_into_this_store(tmp_path, lab, capsys):
    rec = str(tmp_path / "rec")
    graph = graph_file(tmp_path, [work("a", record_dir=rec), work("b", "$nodes.a.out", record_dir=rec, add=10)])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads")
    assert code == 0, out
    run_id = of(out, "run.created")[0]["run"]
    assert of(out, "remote_started")[0]["started"] is True
    assert of(out, "run.finished")[0]["status"] == "succeeded"
    kinds = [e["kind"] for e in of(out, "event")]
    assert kinds[0] == "run.created" and kinds[-1] == "run.succeeded" and kinds.count("task.succeeded") == 2

    root = host_root(lab)
    copies = list((root / "pipelines").iterdir())
    assert len(copies) == 1 and (copies[0] / ".complete").exists()
    listed = json.loads((copies[0] / "RELEASE.json").read_text())
    assert "nodes/core/decode/manifest.yaml" in listed["files"] and "orchestrator/tests/nodes/work/node.py" in listed["files"]
    assert not any(f.startswith(("pipeline/", "docs/", "orchestrator/tests/test_")) for f in listed["files"])
    assert not (Path(tmp_path / "mac") / "work" / run_id).exists()   # nothing ran here

    code, out = perception(capsys, "remote", "fetch", "lab", run_id)
    assert code == 0, out
    fetched = of(out, "fetched")[0]
    assert (fetched["tasks"], fetched["cache_added"], fetched["cache_kept"]) == (2, 2, 0)
    store = Store(tmp_path / "mac")
    tasks = {t["node_id"]: t for t in store.db.execute("SELECT * FROM tasks WHERE run_id=?", (run_id,))}
    assert {n: (t["state"], t["attempt"], t["runner"]) for n, t in tasks.items()} == \
        {"a": ("succeeded", 1, "ssh-lab:cpu"), "b": ("succeeded", 1, "ssh-lab:cpu")}
    b = json.loads(tasks["b"]["outputs"])["out"]
    assert json.loads(Path(b["path"]).read_text()) == {"value": 11}
    assert Path(b["path"]).is_relative_to(store.root)                 # this store's object, checked against its hash
    ran_on = json.loads(tasks["a"]["usage"])["ran_on"]
    assert (ran_on["place"], ran_on["title"], ran_on["coordinator"]) == ("ssh-lab:cpu", "lab (SSH)", "lab")
    assert (store.root / "work" / run_id / "a" / "1" / "log.txt").exists()
    run = store.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    assert run["status"] == "succeeded"
    assert json.loads(run["inputs"])["seed"]["hash"] == hashlib.sha256(b'{"value": 0}').hexdigest()
    # the status and logs commands read it as one of this store's runs
    code, out = perception(capsys, "status", run_id)
    assert code == 0 and of(out, "status")[0]["tasks"][0]["ran_on"]["place"] == "ssh-lab:cpu"
    code, out = perception(capsys, "artifacts", run_id)
    assert {a["node"]: a["hash"] for a in of(out, "artifacts")[0]["artifacts"]}["b"] == b["hash"]


def test_a_fetched_runs_bundle_is_the_one_exported_there_and_this_store_exports_the_same_layers(tmp_path, lab, capsys):
    video = tmp_path / "my clip.mp4"
    video.write_bytes(b"not really a video")
    doc = {"schema_version": 1, "pipeline": "test", "inputs": {"video": {"type": "video.v1", "uri": str(video)}},
           "policy": {"placement": "hybrid", "privacy": "local-preferred", "allow_upload": ["video", "transcript", "faces"]},
           "nodes": [{"id": "shots", "uses": "test.layer@1.0.0", "inputs": {"video": "$inputs.video"}}]}
    graph = tmp_path / "graph.yaml"
    graph.write_text(json.dumps(doc))
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads")
    assert code == 0, out
    run_id = of(out, "run.created")[0]["run"]
    code, out = perception(capsys, "remote", "fetch", "lab", run_id, "--output", str(tmp_path / "bundle"))
    assert code == 0 and of(out, "fetched")[0]["path"] == str(tmp_path / "bundle"), out
    there = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    assert there["video"]["filename"] == "my clip.mp4"                  # the name it has here, not the host's object's
    code, out = perception(capsys, "export", run_id, "--output", str(tmp_path / "here"))
    assert code == 0, out
    here = json.loads((tmp_path / "here" / "manifest.json").read_text())
    assert {k: v["sha256"] for k, v in here["layers"].items()} == {k: v["sha256"] for k, v in there["layers"].items()}
    store = Store(tmp_path / "mac")
    run = store.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    assert json.loads(run["inputs"])["video"]["path"] == str(video)    # the input where it is here


def test_a_run_survives_the_machine_that_started_it_going_away(tmp_path, lab, capsys):
    """--detach returns once it has started; the host is then unreachable from here for a while, and the run ends there
    anyway. `attach` afterwards shows it finished."""
    rec = str(tmp_path / "rec")
    graph = graph_file(tmp_path, [work("a", record_dir=rec, sleep=2)])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads", "--detach")
    assert code == 0, out
    run_id = of(out, "run.created")[0]["run"]
    # its input is held there for the run (the host's retention cleanup leaves it), and released when the run ends
    sha = hashlib.sha256(b'{"value": 0}').hexdigest()
    hold = host_root(lab) / "holds" / sha / f"{run_id}-remote-seed"
    assert hold.read_text().strip() == f"coordinator/remote/{run_id}/held"
    assert (host_root(lab) / "coordinator" / "remote" / run_id / "held").is_dir()
    (lab / "unreachable").write_text("")                              # this machine is gone
    assert wait_for(lambda: len(list(Path(rec).glob("a-*.end"))) == 1)  # the node ends there regardless
    time.sleep(1)
    (lab / "unreachable").unlink()
    code, out = perception(capsys, "remote", "attach", "lab", run_id)
    assert code == 0 and of(out, "run.finished")[0]["status"] == "succeeded", out
    code, out = perception(capsys, "remote", "status", "lab", run_id)
    assert of(out, "status")[0]["status"] == "succeeded" and of(out, "status")[0]["coordinator"] == "finished"
    assert not (host_root(lab) / "coordinator" / "remote" / run_id / "held").exists()   # the hold is stale now


def test_starting_a_run_again_attaches_to_it_instead_of_starting_a_second_coordinator(tmp_path, lab, capsys):
    """The run's id is said first, so a start whose answer was lost can be asked again with --run-id."""
    rec = str(tmp_path / "rec")
    graph = graph_file(tmp_path, [work("a", record_dir=rec, sleep=3)])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads", "--detach")
    run_id = of(out, "run.created")[0]["run"]
    assert of(out, "remote_started")[0]["started"] is True
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads", "--detach", "--run-id", run_id)
    assert code == 0, out
    again = of(out, "remote_started")[0]
    assert (again["started"], again["status"]) == (False, "running")
    code, out = perception(capsys, "remote", "attach", "lab", run_id)
    assert of(out, "run.finished")[0]["status"] == "succeeded"
    assert len(executions(Path(rec), "a")) == 1                       # one coordinator, one execution
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads", "--detach", "--run-id", run_id)
    assert (of(out, "remote_started")[0]["started"], of(out, "remote_started")[0]["status"]) == (False, "succeeded")
    # the same id with another graph is another run: refused there
    other = graph_file(tmp_path, [work("a", record_dir=rec, add=5)])
    code, out = perception(capsys, "remote", "run", "lab", str(other), "--allow-uploads", "--detach", "--run-id", run_id)
    assert code == 1 and "with another graph" in of(out, "error")[0]["problems"][0]


def test_a_run_whose_coordinator_is_gone_is_resumed_by_starting_it_again(tmp_path, lab, capsys):
    """The host restarted (here: the coordinator is killed while its node runs): attach says so, and the same start
    resumes it. The node the dead coordinator started is adopted, not run a second time."""
    rec = str(tmp_path / "rec")
    graph = graph_file(tmp_path, [work("a", record_dir=rec, sleep=4)])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads", "--detach")
    run_id = of(out, "run.created")[0]["run"]
    assert wait_for(lambda: executions(Path(rec), "a"))
    pidfile = host_root(lab) / "coordinator" / "remote" / run_id / "coordinator.pid"
    os.kill(int(pidfile.read_text()), 9)
    time.sleep(0.5)
    code, out = perception(capsys, "hosts", "clean", "lab")   # its node goes on: the host is not cleaned under it
    assert code == 1 and "a node is running there" in of(out, "error")[0]["problems"][0], out
    code, out = perception(capsys, "remote", "attach", "lab", run_id)
    assert code == 1 and "is gone" in of(out, "error")[0]["problems"][0]
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads", "--run-id", run_id)
    assert code == 0, out
    assert of(out, "remote_started")[0]["resumed"] is True and of(out, "run.finished")[0]["status"] == "succeeded"
    assert len(executions(Path(rec), "a")) == 1


# --- what may go there --------------------------------------------------------------------------------------------

def nothing_sent(fake) -> bool:
    return not (fake / "home" / ".cache").exists()


def test_a_graph_that_keeps_its_nodes_here_is_refused_before_anything_is_sent(tmp_path, lab, capsys):
    rec = str(tmp_path / "rec")
    for policy in ({"privacy": "local-only"}, {"placement": "local"}):
        code, out = perception(capsys, "remote", "run", "lab", str(graph_file(tmp_path, [work("a", record_dir=rec)], **policy)),
                               "--allow-uploads")
        assert code == 1
        plan = of(out, "remote_plan")[0]
        assert not plan["ok"] and any("keeps its nodes on this machine" in p for p in plan["problems"])
    assert nothing_sent(lab)


def test_a_run_is_refused_when_the_host_would_hold_a_kind_the_policy_does_not_let_leave(tmp_path, lab, capsys):
    """Every kind in the inputs and every kind the steps make, with the host as the destination: here a json.v1
    output could hold anything, so only every kind allowed lets it go."""
    rec = str(tmp_path / "rec")
    graph = graph_file(tmp_path, [work("a", record_dir=rec)], allow_upload=["video", "transcript"])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads")
    plan = of(out, "remote_plan")[0]
    assert code == 1 and not plan["ok"]
    assert any("would hold audio (from input seed, a)" in p and "embeddings" in p for p in plan["problems"]), plan["problems"]
    # and without the run's consent, even a policy that allows it all
    code, out = perception(capsys, "remote", "run", "lab", str(graph_file(tmp_path, [work("a", record_dir=rec)])))
    assert code == 1 and any("--allow-uploads" in p for p in of(out, "remote_plan")[0]["problems"])
    assert nothing_sent(lab)


def test_cloud_and_other_hosts_places_are_refused_here_and_there(tmp_path, lab, capsys):
    rec = str(tmp_path / "rec")
    for place, why in (("runpod:cuda", "RunPod key and AWS credentials never leave"), ("cloud:cuda", "never leave"),
                       ("ssh-other:cuda", "runs every step there")):
        node = {**work("a", record_dir=rec), "place": place}
        code, out = perception(capsys, "remote", "run", "lab", str(graph_file(tmp_path, [node])), "--allow-uploads")
        assert code == 1 and any(why in p for p in of(out, "remote_plan")[0]["problems"]), out
    assert nothing_sent(lab)
    # the coordinator there refuses them too, whatever it is sent
    remote = remote_run.Remote(Hosts(tmp_path / "mac").load()["lab"], nodes=[str(NODES)])
    seed = tmp_path / "seed.json"
    sha = Store(tmp_path / "mac").source(seed, "json.v1").hash
    remote.send_input(seed, sha, "20261029-120000-abcdef", "seed")
    doc = json.loads(graph_file(tmp_path, [{**work("a", record_dir=rec), "place": "runpod:cuda"}]).read_text())
    doc["inputs"]["seed"]["uri"] = "seed.json"
    spec = {"run": "20261029-120000-abcdef", "graph": doc, "inputs": {"seed": {"sha256": sha, "filename": "seed.json"}},
            "allow_uploads": True, "nodes": remote.nodes, "keys": {}}
    with pytest.raises(remote_run.RemoteError, match="never leave the machine that starts the run"):
        remote.call("start", data=json.dumps(spec).encode())


# --- keys ----------------------------------------------------------------------------------------------------------

def test_a_host_holds_no_keys_until_the_user_says_it_may(tmp_path, lab, capsys):
    rec = str(tmp_path / "rec")
    assert Hosts(tmp_path / "mac").load()["lab"].keys is False                  # off by default
    old = json.loads((tmp_path / "mac" / "hosts.json").read_text())
    del old["lab"]["keys"]                                                       # a hosts.json from before it was recorded
    (tmp_path / "mac" / "hosts.json").write_text(json.dumps(old))
    assert Hosts(tmp_path / "mac").load()["lab"].keys is False
    graph = graph_file(tmp_path, [work("a", record_dir=rec), keyed("k", "$nodes.a.out", rec)])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads")
    plan = of(out, "remote_plan")[0]
    assert code == 1 and plan["keys"] == {"TEST_API_KEY": ["k"]} and plan["keys_allowed"] is False
    assert any("k need TEST_API_KEY" in p and "hosts keys lab on" in p for p in plan["problems"])
    assert nothing_sent(lab)
    code, out = perception(capsys, "hosts", "keys", "lab", "on")
    assert code == 0 and of(out, "hosts")[0]["hosts"][0]["keys"] is True
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads", "--dry-run")
    assert code == 0 and of(out, "remote_plan")[0]["ok"] and nothing_sent(lab)
    code, out = perception(capsys, "hosts", "keys", "lab", "off")
    assert of(out, "hosts")[0]["hosts"][0]["keys"] is False


def test_keys_reach_only_the_steps_that_declare_them_and_no_file_command_line_or_log(tmp_path, lab, capsys):
    rec = str(tmp_path / "rec")
    keys_on(tmp_path)
    graph = graph_file(tmp_path, [work("a", record_dir=rec), keyed("k", "$nodes.a.out", rec)])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads")
    assert code == 0 and of(out, "run.finished")[0]["status"] == "succeeded", out
    run_id = of(out, "run.created")[0]["run"]
    (k,), (a,) = executions(Path(rec), "k"), executions(Path(rec), "a")
    assert k["sha256"] == hashlib.sha256(KEY.encode()).hexdigest() and k["keys"] == ["TEST_API_KEY"]   # given to k
    assert a["keys"] == []                                            # not to a, which does not declare it
    argv = (lab / "argv.log").read_text()
    assert KEY not in argv and KEY[8:] not in argv                     # no command line carried it
    started = (tmp_path.parent / f"{tmp_path.name}-record" / "start.stdin").read_text()
    assert KEY in started                                              # the start's stdin did, and only it
    assert files_holding(tmp_path, KEY[8:]) == []                      # no file here or there: stores, logs, requests
    code, out = perception(capsys, "remote", "fetch", "lab", run_id)
    assert code == 0 and files_holding(tmp_path, KEY[8:]) == []


def test_only_the_keys_the_steps_declare_are_sent(tmp_path, lab, capsys, monkeypatch):
    rec = str(tmp_path / "rec")
    keys_on(tmp_path)
    monkeypatch.setenv("OTHER_API_KEY", "sk-other-" + "5a" * 16)
    graph = graph_file(tmp_path, [work("a", record_dir=rec), keyed("k", "$nodes.a.out", rec)])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads")
    assert code == 0, out
    assert starts(tmp_path)[0]["keys"] == {"TEST_API_KEY": KEY}
    code, out = perception(capsys, "remote", "run", "lab", str(graph_file(tmp_path, [work("b", record_dir=rec)])), "--allow-uploads")
    assert code == 0
    assert starts(tmp_path)[1]["keys"] == {}                          # no step declares one: none goes


# --- import -------------------------------------------------------------------------------------------------------

def test_imported_attempts_get_this_stores_numbers_and_never_replace_its_cache_rows(tmp_path, lab, capsys):
    rec = str(tmp_path / "rec")
    # a retried on the host (its second attempt succeeds there): here it is this store's first attempt of a
    graph = graph_file(tmp_path, [work("a", record_dir=rec, fail="transient", fail_until=1), work("b", "$nodes.a.out", record_dir=rec)])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads")
    assert code == 0, out
    run_id = of(out, "run.created")[0]["run"]
    host_store = Store(host_root(lab) / "coordinator")
    host_tasks = {t["node_id"]: t for t in host_store.db.execute("SELECT * FROM tasks WHERE run_id=?", (run_id,))}
    assert host_tasks["a"]["attempt"] == 2
    # this store already has b's result under its key, from a run of its own: kept as it is
    store = Store(tmp_path / "mac")
    store.db.execute("INSERT INTO cache VALUES (?,?,?,?,?)", (host_tasks["b"]["cache_key"], "test.work@1.0.0", "{}", '{"mine": 1}', 1.0))
    code, out = perception(capsys, "remote", "fetch", "lab", run_id)
    fetched = of(out, "fetched")[0]
    assert (fetched["tasks"], fetched["cache_added"], fetched["cache_kept"]) == (2, 1, 1)
    assert store.db.execute("SELECT usage FROM cache WHERE key=?", (host_tasks["b"]["cache_key"],)).fetchone()["usage"] == '{"mine": 1}'
    tasks = {t["node_id"]: t for t in store.db.execute("SELECT * FROM tasks WHERE run_id=?", (run_id,))}
    assert (tasks["a"]["attempt"], tasks["b"]["attempt"]) == (1, 1)
    assert (store.root / "work" / run_id / "a" / "1" / "log.txt").exists()   # the attempt that succeeded there: its log
    # fetched again: nothing new, nothing renumbered
    code, out = perception(capsys, "remote", "fetch", "lab", run_id, "--output", str(tmp_path / "again"))
    assert of(out, "fetched")[0]["tasks"] == 0
    assert store.db.execute("SELECT attempt FROM tasks WHERE run_id=? AND node_id='a'", (run_id,)).fetchone()[0] == 1
    # a run of this store's own under the same id is never imported over
    store.db.execute("DELETE FROM events WHERE run_id=? AND kind='run.imported'", (run_id,))
    code, out = perception(capsys, "remote", "fetch", "lab", run_id, "--output", str(tmp_path / "third"))
    assert code == 1 and "another run" in of(out, "error")[0]["problems"][0]


def test_a_running_run_is_not_fetched(tmp_path, lab, capsys):
    rec = str(tmp_path / "rec")
    code, out = perception(capsys, "remote", "run", "lab", str(graph_file(tmp_path, [work("a", record_dir=rec, sleep=3)])),
                           "--allow-uploads", "--detach")
    run_id = of(out, "run.created")[0]["run"]
    code, out = perception(capsys, "remote", "fetch", "lab", run_id)
    assert code == 1 and "still running" in of(out, "error")[0]["problems"][0]
    perception(capsys, "remote", "attach", "lab", run_id)


def test_the_coordinator_there_computes_the_cache_keys_this_machine_does_and_imported_results_are_hits(tmp_path, lab, capsys):
    """The copy lists its files (RELEASE.json) as this checkout's git does, so a node's identity there (native: the
    host runs it itself) is the one this machine computes for it placed on that host (ssh-lab): after a fetch, a plan
    here with the nodes placed on the host finds every one cached."""
    rec = str(tmp_path / "rec")
    graph = graph_file(tmp_path, [work("a", record_dir=rec), work("b", "$nodes.a.out", record_dir=rec)])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads")
    run_id = of(out, "run.created")[0]["run"]
    perception(capsys, "remote", "fetch", "lab", run_id)
    host = Hosts(tmp_path / "mac").load()["lab"]
    coord = Coordinator(Store(tmp_path / "mac"), Registry.load(NODES),
                        [SelfBackend(), SshBackend(host, runner_kind="self", python=sys.executable)])
    doc = json.loads(graph.read_text())
    for n in doc["nodes"]:
        n["place"] = "ssh-lab:cpu"
    p = make_plan(coord, compile_graph(doc, coord.registry))
    assert [(n.id, n.cached) for n in p.nodes] == [("a", True), ("b", True)]
    host_keys = {t["node_id"]: t["cache_key"] for t in Store(host_root(lab) / "coordinator").db.execute(
        "SELECT * FROM tasks WHERE run_id=?", (run_id,))}
    local_keys = {t["node_id"]: t["cache_key"] for t in coord.store.db.execute("SELECT * FROM tasks WHERE run_id=?", (run_id,))}
    assert host_keys == local_keys


def test_a_host_is_not_cleaned_while_a_run_is_coordinated_there_and_then_its_copies_and_store_go(tmp_path, lab, capsys):
    rec = str(tmp_path / "rec")
    code, out = perception(capsys, "remote", "run", "lab", str(graph_file(tmp_path, [work("a", record_dir=rec, sleep=3)])),
                           "--allow-uploads", "--detach")
    run_id = of(out, "run.created")[0]["run"]
    code, out = perception(capsys, "hosts", "clean", "lab")
    assert code == 1 and f"a run is being coordinated there (run {run_id})" in of(out, "error")[0]["problems"][0]
    perception(capsys, "remote", "attach", "lab", run_id)
    root = host_root(lab)
    assert (root / "pipelines").exists() and (root / "coordinator").exists()
    code, out = perception(capsys, "hosts", "clean", "lab")
    assert code == 0, out
    assert not any((root / d).exists() for d in ("pipelines", "coordinator", "objects", "envs"))


# --- what a host sends back is not trusted ---------------------------------------------------------------------------

def tampered(monkeypatch, change):
    """The host's dump, changed by `change` before this machine reads it: a host where root or another of the user's
    processes rewrote the coordinator's store."""
    real = remote_run.Remote.call

    def call(self, action, *args, **kw):
        got = real(self, action, *args, **kw)
        if action == "dump":
            change(got)
        return got
    monkeypatch.setattr(remote_run.Remote, "call", call)


def finished_run(tmp_path, capsys, rec) -> str:
    graph = graph_file(tmp_path, [work("a", record_dir=rec), work("b", "$nodes.a.out", record_dir=rec)])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads")
    assert code == 0, out
    return of(out, "run.created")[0]["run"]


def nothing_imported(tmp_path, run_id) -> bool:
    store = Store(tmp_path / "mac")
    return (not store.db.execute("SELECT 1 FROM runs WHERE id=?", (run_id,)).fetchone()
            and not store.db.execute("SELECT 1 FROM cache").fetchone() and not (store.root / "work" / run_id).exists())


@pytest.mark.parametrize("change, why", [
    (lambda d: d["run"].update(id="20261029-000000-000000"), "another run"),
    (lambda d: d["tasks"][0].update(node_id="../../../../escaped"), "not a node of the run"),
    (lambda d: d["tasks"][0].update(node_id="c"), "not a node of the run"),
    (lambda d: d["tasks"].pop(), "the run's nodes"),
    (lambda d: d["tasks"][0].update(attempt="1/../../../escaped"), "attempt"),
    (lambda d: d["events"].append({**d["events"][0], "node_id": "../escaped"}), "not a node of the run"),
])
def test_a_dump_naming_another_run_or_nodes_outside_the_run_is_refused(tmp_path, lab, capsys, monkeypatch, change, why):
    rec = str(tmp_path / "rec")
    run_id = finished_run(tmp_path, capsys, rec)
    tampered(monkeypatch, change)
    code, out = perception(capsys, "remote", "fetch", "lab", run_id)
    assert code == 1 and why in of(out, "error")[0]["problems"][0], out
    assert nothing_imported(tmp_path, run_id)
    assert not list(tmp_path.parent.rglob("escaped"))


def test_a_bundle_path_the_host_names_is_never_read(tmp_path, lab, capsys, monkeypatch):
    """The bundle is exported here from the checked run: a path the host's answer names for one is not used."""
    rec = str(tmp_path / "rec")
    run_id = finished_run(tmp_path, capsys, rec)
    tampered(monkeypatch, lambda d: d.update(bundle="../../../../.."))
    code, out = perception(capsys, "remote", "fetch", "lab", run_id, "--output", str(tmp_path / "bundle"))
    assert code == 0 and of(out, "fetched")[0]["path"] is None   # test.work makes no media: no bundle, and nothing read


@pytest.mark.parametrize("change", [
    lambda d: d["logs"][0].update(path="../../../../../etc/hosts"),
    lambda d: d["logs"][0].update(path=f"work/{d['run']['id']}/a/1/../../../../../escaped"),
])
def test_bundle_and_log_paths_outside_their_layouts_are_refused(tmp_path, lab, capsys, monkeypatch, change):
    rec = str(tmp_path / "rec")
    run_id = finished_run(tmp_path, capsys, rec)
    tampered(monkeypatch, change)
    code, out = perception(capsys, "remote", "fetch", "lab", run_id, "--output", str(tmp_path / "bundle"))
    assert code == 1 and "not where" in of(out, "error")[0]["problems"][0], out
    assert nothing_imported(tmp_path, run_id) and not (tmp_path / "bundle").exists()


def test_a_link_the_host_planted_in_what_comes_back_is_not_followed(tmp_path, lab, capsys):
    """rsync brings back no links: a log or a bundle file replaced there by a link to a file of this machine's comes
    back as nothing, and is refused."""
    rec = str(tmp_path / "rec")
    run_id = finished_run(tmp_path, capsys, rec)
    secret = tmp_path / "secret.txt"
    secret.write_text("this machine's own file")
    log = host_root(lab) / "coordinator" / "work" / run_id / "a" / "1" / "log.txt"
    log.unlink()
    log.symlink_to(secret)
    code, out = perception(capsys, "remote", "fetch", "lab", run_id)
    store = Store(tmp_path / "mac")
    copied = store.root / "work" / run_id / "a" / "1" / "log.txt"
    assert not copied.exists() or "this machine's own file" not in copied.read_text()


def test_cache_rows_come_only_from_this_runs_tasks_under_keys_this_machine_computes(tmp_path, lab, capsys, monkeypatch):
    rec = str(tmp_path / "rec")
    run_id = finished_run(tmp_path, capsys, rec)
    planted = "ab" * 32

    def change(d):
        # a row for a key no task of the run has (what another step, or other inputs, would look up): an import makes
        # rows from the run's tasks alone, so a dump's own rows are never read
        b = next(t for t in d["tasks"] if t["node_id"] == "b")
        d["cache"] = [{"key": planted, "node": "test.work@1.0.0", "outputs": b["outputs"], "usage": "{}", "created_at": 1.0}]
        # and b's result under a key that is not the one its node, config and inputs give
        b["cache_key"] = "cd" * 32
    tampered(monkeypatch, change)
    code, out = perception(capsys, "remote", "fetch", "lab", run_id)
    assert code == 0, out
    fetched = of(out, "fetched")[0]
    assert (fetched["cache_added"], fetched["cache_refused"]) == (1, 1)
    store = Store(tmp_path / "mac")
    keys = {r["key"] for r in store.db.execute("SELECT key FROM cache")}
    assert planted not in keys and "cd" * 32 not in keys and len(keys) == 1
    a = store.db.execute("SELECT cache_key FROM tasks WHERE run_id=? AND node_id='a'", (run_id,)).fetchone()["cache_key"]
    assert keys == {a}


@pytest.mark.parametrize("change, why", [
    # an output claimed to be a source: a path on this machine, never an object brought back
    (lambda d: next(t for t in d["tasks"] if t["node_id"] == "a").update(
        outputs=json.dumps({"out": {"hash": "ef" * 32, "shape": "source", "type": "json.v1", "path": "/etc/hosts"}})), "not a store object"),
    # an output under another type than its node makes, or one it does not make
    (lambda d: next(t for t in d["tasks"] if t["node_id"] == "a").update(
        outputs=json.dumps({"out": {**json.loads(next(t for t in d["tasks"] if t["node_id"] == "a")["outputs"])["out"], "type": "media.v1"}})), "not what"),
    # an object the host does not have
    (lambda d: next(t for t in d["tasks"] if t["node_id"] == "a").update(
        outputs=json.dumps({"out": {**json.loads(next(t for t in d["tasks"] if t["node_id"] == "a")["outputs"])["out"], "hash": "ef" * 32}})), "objects/ef/"),
])
def test_outputs_are_only_store_objects_of_their_nodes_types_whose_bytes_match(tmp_path, lab, capsys, monkeypatch, change, why):
    rec = str(tmp_path / "rec")
    run_id = finished_run(tmp_path, capsys, rec)
    tampered(monkeypatch, change)
    code, out = perception(capsys, "remote", "fetch", "lab", run_id)
    assert code == 1 and any(why in p for p in of(out, "error")[0]["problems"]), out
    assert nothing_imported(tmp_path, run_id)


def test_an_object_whose_bytes_are_not_its_hash_is_refused(tmp_path, lab, capsys):
    rec = str(tmp_path / "rec")
    run_id = finished_run(tmp_path, capsys, rec)
    host_store = Store(host_root(lab) / "coordinator")
    a = json.loads(host_store.db.execute("SELECT outputs FROM tasks WHERE run_id=? AND node_id='a'", (run_id,)).fetchone()[0])["out"]
    obj = Path(a["path"])
    obj.chmod(0o644)
    obj.write_text('{"value": 999}')                                 # rewritten there, under its old hash
    code, out = perception(capsys, "remote", "fetch", "lab", run_id)
    assert code == 1 and "is not the file its hash names" in of(out, "error")[0]["problems"][0], out
    assert nothing_imported(tmp_path, run_id)


def test_a_run_id_given_here_must_be_one(tmp_path, lab, capsys):
    for action in ("attach", "fetch", "status"):
        code, out = perception(capsys, "remote", action, "lab", "../../etc")
        assert code == 1 and "not a run id" in of(out, "error")[0]["problems"][0]
    assert not (lab / "argv.log").exists()                             # nothing was asked of the host


def test_events_that_are_not_events_end_an_attach(tmp_path, lab, capsys, monkeypatch):
    rec = str(tmp_path / "rec")
    run_id = finished_run(tmp_path, capsys, rec)
    real = remote_run.Remote.call
    monkeypatch.setattr(remote_run.Remote, "call", lambda self, action, *a, **k: (
        {**real(self, action, *a, **k), "events": [{"id": "x"}]} if action == "events" else real(self, action, *a, **k)))
    code, out = perception(capsys, "remote", "attach", "lab", run_id)
    assert code == 1 and "not one" in of(out, "error")[0]["problems"][0]


def test_the_fetched_bundle_is_exported_here_from_the_checked_run_not_taken_from_the_host(tmp_path, lab, capsys):
    """Review of #46: a bundle the host kept could have been rewritten there after the run (a layer, a licence mode);
    the one fetch writes is exported here, from the objects it checked against their hashes."""
    video = tmp_path / "my clip.mp4"
    video.write_bytes(b"not really a video")
    doc = {"schema_version": 1, "pipeline": "test", "inputs": {"video": {"type": "video.v1", "uri": str(video)}},
           "policy": {"placement": "hybrid", "privacy": "local-preferred", "allow_upload": ["video", "transcript", "faces"]},
           "nodes": [{"id": "shots", "uses": "test.layer@1.0.0", "inputs": {"video": "$inputs.video"}}]}
    graph = tmp_path / "graph.yaml"
    graph.write_text(json.dumps(doc))
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads")
    run_id = of(out, "run.created")[0]["run"]
    code, out = perception(capsys, "remote", "fetch", "lab", run_id, "--output", str(tmp_path / "first"))
    assert code == 0, out
    # a bundle the host has (one an earlier fetch had it export, or one written there), rewritten there
    there = host_root(lab) / "coordinator" / "remote" / run_id / "bundle"
    there.mkdir(parents=True, exist_ok=True)
    (there / "layers").mkdir(exist_ok=True)
    if (there / "layers" / "shots.json").exists():
        (there / "layers" / "shots.json").chmod(0o644)
    (there / "layers" / "shots.json").write_text('{"tampered": true}')
    (there / "manifest.json").write_text('{"tampered": true}')
    code, out = perception(capsys, "remote", "fetch", "lab", run_id, "--output", str(tmp_path / "bundle"))
    assert code == 0, out
    assert "tampered" not in (tmp_path / "bundle" / "layers" / "shots.json").read_text()
    code, out = perception(capsys, "export", run_id, "--output", str(tmp_path / "here"))
    assert (tmp_path / "bundle" / "layers" / "shots.json").read_bytes() == (tmp_path / "here" / "layers" / "shots.json").read_bytes()


def test_starting_a_run_again_differently_leaves_this_machines_record_of_it(tmp_path, lab, capsys):
    """Review of #46: `remote run --run-id X` with another --set (or --only) is another run under X's id: refused
    before anything is sent, and X's record (what fetch checks the host's answer against) is left as it was."""
    rec = str(tmp_path / "rec")
    graph = graph_file(tmp_path, [work("a", record_dir=rec), work("b", "$nodes.a.out", record_dir=rec)])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads")
    run_id = of(out, "run.created")[0]["run"]
    before = remote_run.record(run_id).read_text()
    sent = (lab / "argv.log").read_text()
    for change in (["--set", "b.add=5"], ["--only", "a"]):
        code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads", "--run-id", run_id, *change)
        assert code == 1 and "another graph" in of(out, "error")[0]["problems"][0], out
    assert remote_run.record(run_id).read_text() == before
    assert (lab / "argv.log").read_text() == sent                       # nothing was asked of the host
    code, out = perception(capsys, "remote", "fetch", "lab", run_id)
    assert code == 0 and of(out, "fetched")[0]["cache_refused"] == 0, out
    # and the same start again is still an attach, which keeps what fetch recorded
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads", "--run-id", run_id, "--detach")
    assert code == 0 and of(out, "remote_started")[0]["started"] is False
    assert json.loads(remote_run.record(run_id).read_text()).get("imported_events")


def holds_naming(lab, run_id) -> list[Path]:
    return [h for h in (host_root(lab) / "holds").glob(f"*/{run_id}-remote-*")
            if (host_root(lab) / h.read_text().strip()).is_dir()]   # a hold whose folder is gone is stale


def test_a_start_that_starts_no_coordinator_leaves_no_hold_on_its_inputs(tmp_path, lab, capsys):
    """Review of #46: the input is held before it is sent; a start that ends with no coordinator running for the run
    (the host's plan refuses it, the run has ended already) must release it, or the host's retention would never remove
    the clip."""
    rec = str(tmp_path / "rec")
    # the host's own plan refuses it: a place the host does not have (here, no CUDA)
    node = {**work("a", record_dir=rec), "place": "native:cuda"}
    code, out = perception(capsys, "remote", "run", "lab", str(graph_file(tmp_path, [node])), "--allow-uploads")
    assert code == 1 and of(out, "plan") and not of(out, "plan")[0]["ok"], out
    run_id = of(out, "run.created")[0]["run"]
    assert holds_naming(lab, run_id) == []
    # started again after it ended: nothing starts, and nothing stays held
    graph = graph_file(tmp_path, [work("b", record_dir=rec)])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads")
    run_id = of(out, "run.created")[0]["run"]
    assert code == 0 and holds_naming(lab, run_id) == []
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads", "--run-id", run_id, "--detach")
    assert of(out, "remote_started")[0]["started"] is False
    assert holds_naming(lab, run_id) == []


def test_a_second_fetch_leaves_what_this_store_reran_since(tmp_path, lab, capsys):
    """Review of #46: a task rerun here after the first import is this store's own result: a later fetch keeps it."""
    rec = str(tmp_path / "rec")
    run_id = finished_run(tmp_path, capsys, rec)
    perception(capsys, "remote", "fetch", "lab", run_id)
    store = Store(tmp_path / "mac")
    store.db.execute("UPDATE tasks SET attempt=2, outputs='{}', cache_key=NULL, state='failed' WHERE run_id=? AND node_id='b'", (run_id,))
    store.db.execute("UPDATE runs SET status='failed' WHERE id=?", (run_id,))   # as a rerun here that failed leaves it
    code, out = perception(capsys, "remote", "fetch", "lab", run_id)
    assert code == 0, out
    b = store.db.execute("SELECT attempt, state FROM tasks WHERE run_id=? AND node_id='b'", (run_id,)).fetchone()
    assert tuple(b) == (2, "failed")
    assert store.db.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()[0] == "failed"


def test_a_run_id_of_a_run_made_here_is_refused_before_anything_is_sent(tmp_path, lab, capsys):
    """Review of #46: a --run-id naming one of this store's own runs would run there, spend, and never be imported."""
    rec = str(tmp_path / "rec")
    graph = graph_file(tmp_path, [work("a", record_dir=rec)])
    coord = Coordinator(Store(tmp_path / "mac"), Registry.load(NODES), [SelfBackend()])
    doc = json.loads(graph.read_text())
    mine = coord.create_run(compile_graph(doc, coord.registry), doc)
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads", "--run-id", mine)
    assert code == 1 and "a run of this machine's" in of(out, "error")[0]["problems"][0], out
    assert nothing_sent(lab) and not (lab / "argv.log").exists()


def test_a_run_is_resumed_only_with_the_inputs_it_started_with(tmp_path, lab, capsys):
    """Re-review of #46 (MAJOR): an input rewritten in place (the same path and name, other bytes) is another run's
    input. A start of the run again with it is refused here, before anything is sent, and there (a host told otherwise
    would decode the new clip under the old one's cache key); a fetch refuses a run whose input here changed."""
    rec = str(tmp_path / "rec")
    graph = graph_file(tmp_path, [work("a", record_dir=rec, sleep=4)])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads", "--detach")
    run_id = of(out, "run.created")[0]["run"]
    assert wait_for(lambda: executions(Path(rec), "a"))
    os.kill(int((host_root(lab) / "coordinator" / "remote" / run_id / "coordinator.pid").read_text()), 9)
    (tmp_path / "seed.json").write_text('{"value": 5}')                # re-exported in place
    sent = (lab / "argv.log").read_text()
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads", "--run-id", run_id)
    assert code == 1 and "other inputs" in of(out, "error")[0]["problems"][0], out
    assert (lab / "argv.log").read_text() == sent
    # the host refuses it too, told the new input's hash for the old run
    remote = remote_run.Remote(Hosts(tmp_path / "mac").load()["lab"], nodes=[str(NODES)])
    sha = Store(tmp_path / "mac").source(tmp_path / "seed.json", "json.v1").hash
    remote.send_input(tmp_path / "seed.json", sha, run_id, "seed")
    doc = json.loads(remote_run.record(run_id).read_text())["graph_doc"]
    doc["inputs"]["seed"]["uri"] = "seed.json"
    spec = {"run": run_id, "graph": doc, "inputs": {"seed": {"sha256": sha, "filename": "seed.json"}},
            "allow_uploads": True, "nodes": remote.nodes, "keys": {}}
    with pytest.raises(remote_run.RemoteError, match="other inputs"):
        remote.call("start", data=json.dumps(spec).encode())


def test_runs_that_ended_there_go_after_the_hosts_keep_days(tmp_path, lab, capsys, monkeypatch):
    """Re-review of #46 (MEDIUM): what a run coordinated on a host made (face crops, transcripts, embeddings) stays in
    the coordinator's store there only for the host's keep days, as what the host keeps for steps run there does."""
    rec = str(tmp_path / "rec")
    old = finished_run(tmp_path, capsys, rec)
    store = Store(host_root(lab) / "coordinator")
    made = [Path(a["path"]) for t in store.db.execute("SELECT outputs FROM tasks WHERE run_id=?", (old,))
            for a in json.loads(t["outputs"]).values()]
    assert all(p.exists() for p in made)
    store.db.execute("UPDATE runs SET finished_at = finished_at - 8 * 86400 WHERE id=?", (old,))   # ended 8 days ago
    graph = graph_file(tmp_path, [work("c", record_dir=rec, add=50)])   # outputs of its own: none shared with the old run
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads")   # the next run there removes it
    new = of(out, "run.created")[0]["run"]
    assert not store.db.execute("SELECT 1 FROM runs WHERE id=?", (old,)).fetchone()
    assert not store.db.execute("SELECT 1 FROM tasks WHERE run_id=?", (old,)).fetchone()
    assert not any(p.exists() for p in made)
    assert not (host_root(lab) / "coordinator" / "work" / old).exists()
    assert not (host_root(lab) / "coordinator" / "remote" / old).exists()
    code, out = perception(capsys, "remote", "fetch", "lab", new)      # the run that is still within its days
    assert code == 0, out


def test_a_host_used_once_is_swept_too(tmp_path, lab, capsys):
    """The coordinator starts the host's sweeper for these runs as it ends: an ended run past the keep period goes with
    no other start there."""
    rec = str(tmp_path / "rec")
    Hosts(tmp_path / "mac").set("lab", keep_days=3 / 86400)            # three seconds
    run_id = finished_run(tmp_path, capsys, rec)
    store = Store(host_root(lab) / "coordinator")
    assert wait_for(lambda: not store.db.execute("SELECT 1 FROM runs WHERE id=?", (run_id,)).fetchone(), timeout=20)
    assert not (host_root(lab) / "coordinator" / "work" / run_id).exists()


def test_a_run_whose_coordinator_died_goes_after_the_keep_days_too(tmp_path, lab, capsys):
    """Re-review of #46: a run left `running` by a coordinator that died (the host restarted) and never resumed is kept
    no longer than an ended one, and the hold on its input goes with it."""
    rec = str(tmp_path / "rec")
    code, out = perception(capsys, "remote", "run", "lab", str(graph_file(tmp_path, [work("a", record_dir=rec, sleep=3)])),
                           "--allow-uploads", "--detach")
    run_id = of(out, "run.created")[0]["run"]
    assert wait_for(lambda: executions(Path(rec), "a"))
    os.kill(int((host_root(lab) / "coordinator" / "remote" / run_id / "coordinator.pid").read_text()), 9)
    assert wait_for(lambda: list(Path(rec).glob("a-*.end")))            # its node ends; nothing records it
    store = Store(host_root(lab) / "coordinator")
    week = 8 * 86400
    store.db.execute("UPDATE runs SET created_at = created_at - ? WHERE id=?", (week, run_id))
    store.db.execute("UPDATE tasks SET started_at = started_at - ? WHERE run_id=?", (week, run_id))
    store.db.execute("UPDATE events SET at = at - ? WHERE run_id=?", (week, run_id))
    assert holds_naming(lab, run_id)
    graph = graph_file(tmp_path, [work("c", record_dir=rec, add=50)])
    code, out = perception(capsys, "remote", "run", "lab", str(graph), "--allow-uploads")
    assert code == 0, out
    assert not store.db.execute("SELECT 1 FROM runs WHERE id=?", (run_id,)).fetchone()
    assert holds_naming(lab, run_id) == []


def test_a_start_waits_for_a_prune_that_is_removing_objects(tmp_path, lab, capsys):
    """Re-review of #46: prune decides what no run reads, then removes it; a run starting meanwhile could have its cache
    hits or new outputs removed under it. Both take the store's prune lock: a start waits for a prune holding it."""
    import fcntl
    import threading
    rec = str(tmp_path / "rec")
    finished_run(tmp_path, capsys, rec)                                  # the store there exists
    fd = os.open(host_root(lab) / "coordinator" / ".prune", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)                                       # a prune at work
    graph = graph_file(tmp_path, [work("c", record_dir=rec, add=50)])
    done = []
    t = threading.Thread(target=lambda: done.append(cli.main(["--json", "--nodes", str(NODES), "remote", "run", "lab", str(graph),
                                                              "--allow-uploads", "--detach"])))
    t.start()
    time.sleep(4)
    store = Store(host_root(lab) / "coordinator")
    assert not store.db.execute("SELECT 1 FROM tasks WHERE node_id='c'").fetchone()   # not created while it is held
    os.close(fd)                                                         # the prune ends
    t.join(60)
    assert done == [0]
    assert store.db.execute("SELECT 1 FROM tasks WHERE node_id='c'").fetchone()
