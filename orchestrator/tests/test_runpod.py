"""Nodes on a RunPod pod (RunPodBackend and the driver's RunPod steps), against a stand-in for RunPod's REST API (a server
in this process that records each request) and the stand-ins test_cloud.py uses for ssh: the node really runs, "on the
pod" (a separate home on this machine). Also cloud/runpod/guard.sh itself, with stand-ins for what it calls."""

import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from perception.backends.cloud import RunPodBackend
from perception.backends.native import SelfBackend
from perception.scheduler import Coordinator
from perception.store import Store
from perception.manifests import Registry

from test_cloud import GUARD_OK, GUARD_REFUSES, NODES, SETUP_OK, events, graph, log_of, run, script

KEY = "rp-test-key-0123456789"
GUARD = Path(__file__).resolve().parents[2] / "cloud" / "runpod" / "guard.sh"

# test_cloud's ssh stand-in, which also takes -i KEY and -p PORT (a pod's own key and port)
FAKE_SSH = r"""#!/bin/sh
while [ $# -gt 0 ]; do case "$1" in -o|-i|-p) echo "ssh option $1 $2" >> "$FAKE_HOST/ssh-options"; shift 2;; -*) shift;; *) break;; esac; done
echo "ssh $1: $2" | head -c 300 >> "$FAKE_HOST/events"; echo >> "$FAKE_HOST/events"
shift
cd "$FAKE_HOST/home" && HOME="$FAKE_HOST/home" exec /bin/sh -c "$*"
"""


class FakeRunPod:
    """RunPod's REST API as the driver uses it. `flags`: no-capacity (create answers 500), no-answer (create makes the
    pod but drops the connection), delete-fails (500 to every DELETE)."""

    def __init__(self, fake_host: Path):
        self.fake_host, self.flags, self.pods, self.bodies, self.headers = fake_host, set(), {}, [], []
        self.seq = 0
        rp = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def answer(self, code, doc=None):
                data = json.dumps(doc).encode() if doc is not None else b""
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def handle_one(self, method):
                rp.headers.append(dict(self.headers))
                with open(rp.fake_host / "events", "a") as f:
                    f.write(f"api {method} {self.path}\n")
                if self.headers.get("Authorization") != f"Bearer {KEY}":
                    return self.answer(401, {"error": "unauthorized"})
                if method == "POST" and self.path == "/pods":
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    rp.bodies.append(body)
                    if "no-capacity" in rp.flags:
                        return self.answer(500, {"error": "create pod: There are no instances currently available"})
                    rp.seq += 1
                    pod = {"id": f"pod{rp.seq}", "name": body["name"], "costPerHr": 0.27, "polls": 0}
                    rp.pods[pod["id"]] = pod
                    if "no-answer" in rp.flags:
                        self.close_connection = True
                        self.connection.close()   # made, and never said: only its name finds it
                        return
                    return self.answer(201, {k: v for k, v in pod.items() if k != "polls"})
                if method == "GET" and self.path == "/pods":
                    return self.answer(200, [{"id": p["id"], "name": p["name"]} for p in rp.pods.values()])
                pod = rp.pods.get(self.path.removeprefix("/pods/"))
                if pod is None:
                    return self.answer(404, {"error": "pod not found"})
                if method == "GET":
                    pod["polls"] += 1
                    ready = pod["polls"] > 1   # an address from the second look on
                    return self.answer(200, {"id": pod["id"], "name": pod["name"], "costPerHr": 0.27,
                                             **({"publicIp": "127.0.0.1", "portMappings": {"22": 2222}} if ready else {})})
                if method == "DELETE":
                    if "delete-fails" in rp.flags:
                        return self.answer(500, {"error": "internal"})
                    del rp.pods[pod["id"]]
                    return self.answer(204)
                return self.answer(405)

            def do_GET(self):
                self.handle_one("GET")

            def do_POST(self):
                self.handle_one("POST")

            def do_DELETE(self):
                self.handle_one("DELETE")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def pod(tmp_path, monkeypatch):
    fake = tmp_path / "host"
    (fake / "home" / ".local" / "bin").mkdir(parents=True)
    script(fake / "home" / ".local" / "bin" / "nvidia-smi", "#!/bin/sh\necho 23034\n")
    monkeypatch.setenv("FAKE_HOST", str(fake))
    monkeypatch.setenv("PERCEPTION_SSH", str(script(tmp_path / "ssh", FAKE_SSH)))
    api = FakeRunPod(fake)
    monkeypatch.setenv("PERCEPTION_RUNPOD_API", api.url)
    yield api
    api.server.shutdown()


def coordinator(tmp_path, guard=GUARD_OK, deadline_s=1800, key=KEY, down_s=900) -> Coordinator:
    b = RunPodBackend(api_key=key, deadline_s=deadline_s, runner_kind="self", python=sys.executable, launch_s=60, down_s=down_s,
                      guard=script(tmp_path / "guard.sh", guard), setup=script(tmp_path / "node-setup.sh", SETUP_OK))
    return Coordinator(Store(tmp_path / "store"), Registry.load(NODES), [SelfBackend(), b], tick_s=0.02, backoff_s=0.01)


def rec(tmp_path):
    return str(tmp_path / "record")


def task(coord, run_id) -> tuple[dict, dict]:
    t = coord._tasks(run_id)["faces"]
    return t, json.loads(t["usage"] or "{}")


def test_a_node_runs_on_a_pod_made_for_it_checked_before_anything_is_sent_and_deleted_after(tmp_path, pod):
    coord = coordinator(tmp_path)
    started = time.time()
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    assert status == "succeeded", log_of(coord, run_id)
    body = pod.bodies[0]
    # Secure Cloud in the US, any of the GPUs (the cheapest first), its own disk only: nothing outlives it
    assert body["cloudType"] == "SECURE" and all(d.startswith("US-") for d in body["dataCenterIds"])
    assert body["gpuTypeIds"] == list(RunPodBackend.gpus) and body["gpuTypePriority"] == "custom" and body["volumeInGb"] == 0
    # named for the attempt, with its deadline (a later launch deletes it once that has passed)
    assert body["name"].startswith("perception-") and abs(int(body["name"].rsplit("-d", 1)[1]) - (started + 1800)) < 30
    # reached with a key made for the attempt: its public half is what the pod was given
    attempt = coord.store.root / "work" / run_id / "faces" / "1"
    assert body["env"] == {"PUBLIC_KEY": (attempt / "ssh" / "id_ed25519.pub").read_text().strip()}
    options = (pod.fake_host / "ssh-options").read_text()
    assert f"-i {attempt / 'ssh' / 'id_ed25519'}" in options and "-p 2222" in options
    assert all(h.get("User-Agent") == "hyperstudy-perception" for h in pod.headers)   # Cloudflare refuses Python's own
    ev = events(pod.fake_host)
    create, guard, setup = ev.index("api POST /pods"), ev.index("guard 29"), ev.index("setup")
    first_send = next(i for i, e in enumerate(ev) if "objects" in e or "code/" in e)
    delete = ev.index("api DELETE /pods/pod1")
    assert create < guard < setup < first_send < delete and ev[-1] == "api GET /pods/pod1" and not pod.pods
    t, usage = task(coord, run_id)
    assert t["runner"] == "runpod:cuda"
    # charged at the price of the pod it got ($0.27/h), for the time it existed
    assert 0 < usage["machine_usd"] <= round(0.27 * (time.time() - started) / 3600, 4) + 0.0001
    assert set(usage["phases"]) >= {"launch", "setup", "code", "inputs", "run", "fetch", "teardown"}


def test_the_account_key_is_given_to_the_driver_alone(tmp_path, pod):
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    assert status == "succeeded", log_of(coord, run_id)
    attempt = coord.store.root / "work" / run_id / "faces" / "1"
    # in no file of the attempt (remote.json above all), in nothing sent to the pod, in nothing the node saw
    for f in [*attempt.rglob("*"), *(pod.fake_host / "home").rglob("*")]:
        if f.is_file() and not f.is_symlink():
            assert KEY.encode() not in f.read_bytes(), f
    assert KEY not in json.dumps(pod.bodies) and KEY not in (pod.fake_host / "events").read_text()


def test_a_pod_the_guard_refuses_gets_nothing_and_is_deleted(tmp_path, pod):
    coord = coordinator(tmp_path, guard=GUARD_REFUSES)
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    assert status == "failed"
    ev = events(pod.fake_host)
    assert not any("objects" in e or "code/" in e for e in ev) and "api DELETE /pods/pod1" in ev and not pod.pods
    t, _ = task(coord, run_id)
    assert t["error_code"] == "privacy" and "guard.sh refused" in t["error"]
    # a failed attempt's spending is the run's: the pod's time
    assert coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0] > 0


def test_no_capacity_is_retried_and_nothing_is_deleted_or_charged(tmp_path, pod):
    pod.flags.add("no-capacity")
    coord = coordinator(tmp_path)
    doc = graph(tmp_path, rec(tmp_path), place="runpod:cuda")
    doc["policy"]["retries"] = 1
    run_id, status = run(coord, doc)
    assert status == "failed"
    assert len(pod.bodies) == 2   # another attempt: capacity comes and goes
    assert not any(e.startswith("api DELETE") for e in events(pod.fake_host))
    t, _ = task(coord, run_id)
    assert t["error_code"] == "transient"
    assert coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0] == 0
    assert "no instances currently available" in log_of(coord, run_id)


def test_a_pod_whose_creation_got_no_answer_is_found_by_its_name_and_deleted(tmp_path, pod):
    pod.flags.add("no-answer")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    assert status == "failed"
    ev = events(pod.fake_host)
    assert "api GET /pods" in ev[ev.index("api POST /pods"):] and "api DELETE /pods/pod1" in ev and not pod.pods


def test_a_launch_deletes_perception_pods_past_their_deadline_and_no_others(tmp_path, pod):
    now = int(time.time())
    pod.pods = {
        "old": {"id": "old", "name": f"perception-ab12-faces-123456-1-d{now - 3600}", "polls": 0},      # its driver never came back
        "recent": {"id": "recent", "name": f"perception-cd34-faces-123456-1-d{now - 60}", "polls": 0},  # within the grace
        "running": {"id": "running", "name": f"perception-ef56-faces-123456-1-d{now + 900}", "polls": 0},   # another run's
        "mine": {"id": "mine", "name": "my-notebook", "polls": 0},                                        # the user's own
    }
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    assert status == "succeeded", log_of(coord, run_id)
    assert set(pod.pods) == {"recent", "running", "mine"}
    assert "a perception pod past its deadline" in log_of(coord, run_id)


def test_a_pod_that_could_not_be_deleted_is_charged_its_whole_deadline(tmp_path, pod):
    pod.flags.add("delete-fails")
    coord = coordinator(tmp_path, down_s=3)
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    t, usage = task(coord, run_id)
    # it removes itself at its deadline (guard.sh), and costs at most that
    assert usage["machine_usd"] == round(0.27 * 1800 / 3600, 4)
    assert "it removes itself at its deadline" in log_of(coord, run_id)


def test_a_pod_with_no_deadline_that_could_not_be_deleted_stops_the_retries(tmp_path, pod):
    pod.flags.add("delete-fails")
    coord = coordinator(tmp_path, guard=GUARD_REFUSES, down_s=3)
    doc = graph(tmp_path, rec(tmp_path), place="runpod:cuda")
    doc["policy"]["retries"] = 2
    run_id, status = run(coord, doc)
    assert status == "failed" and len(pod.bodies) == 1   # no other attempt starts another pod
    t, usage = task(coord, run_id)
    assert t["error_code"] == "internal" and "console.runpod.io" in t["error"]


def test_no_key_creates_nothing(tmp_path, pod):
    coord = coordinator(tmp_path, key=None)
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    assert status == "failed" and not pod.bodies
    assert "RUNPOD_API_KEY" in task(coord, run_id)[0]["error"]


def test_the_cli_turns_runpod_on_after_checking_the_key(tmp_path, pod, monkeypatch, capsys):
    from perception import cli
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HOME", str(tmp_path / "user"))   # no keys of the user's own
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert cli.main(["cloud", "on", "runpod"]) == 1
    assert "RUNPOD_API_KEY" in capsys.readouterr().err
    monkeypatch.setenv("RUNPOD_API_KEY", "not-the-key")
    assert cli.main(["cloud", "on", "runpod"]) == 1
    assert "did not accept" in capsys.readouterr().err
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    assert cli.main(["cloud", "on", "runpod"]) == 0
    assert cli.main(["--json", "cloud", "status"]) == 0
    status = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert status["runpod"]["enabled"] is True and status["enabled"] is False and status["runpod"]["hourly_usd"] == 0.59
    assert [b.kind for b in cli.backends()][-1] == "runpod"
    assert cli.main(["cloud", "off", "runpod"]) == 0 and "runpod" not in [b.kind for b in cli.backends()]


# --- cloud/runpod/guard.sh ---------------------------------------------------------------------------------------

@pytest.fixture
def guard_env(tmp_path):
    """A pod's environment for guard.sh: its variables (as /proc/1/environ holds them), and stand-ins for curl (the API's
    answer to the pod's key in $T/code), runpodctl (records what it was asked) and setsid (Linux's)."""
    t = tmp_path
    (t / "home").mkdir()
    (t / "bin").mkdir()
    (t / "environ").write_bytes(b"RUNPOD_POD_ID=pod42\0RUNPOD_API_KEY=pod-scoped\0RUNPOD_DC_ID=US-TX-1\0PATH=/usr/bin\0")
    (t / "code").write_text("403")
    script(t / "bin" / "curl", f'#!/bin/sh\necho "curl $*" >> {t}/calls; cat {t}/code\n')
    script(t / "bin" / "runpodctl", f'#!/bin/sh\necho "runpodctl $* key=$RUNPOD_API_KEY" >> {t}/calls\n')
    script(t / "bin" / "nvidia-smi", '#!/bin/sh\necho "NVIDIA RTX A5000"\n')
    script(t / "bin" / "setsid", '#!/bin/sh\nexec "$@"\n')
    script(t / "bin" / "sleep", f'#!/bin/sh\necho "sleep $*" >> {t}/calls; /bin/sleep 1\n')   # a second, whatever it is asked
    return t


def guard(t: Path, minutes: str = "1") -> subprocess.CompletedProcess:
    import os
    env = {"HOME": str(t / "home"), "PATH": f"{t / 'bin'}:/usr/bin:/bin", "PERCEPTION_POD_ENVIRON": str(t / "environ")}
    return subprocess.run(["bash", "-s", minutes], input=GUARD.read_bytes(), env=env, capture_output=True, timeout=60)


def calls(t: Path) -> str:
    return (t / "calls").read_text() if (t / "calls").exists() else ""


def test_the_guard_sets_the_pods_own_removal_at_its_deadline(guard_env):
    out = guard(guard_env, "1")
    assert out.returncode == 0, out.stderr
    assert "NVIDIA RTX A5000 in US-TX-1; no key to the account; the pod removes itself in 1 min" in out.stdout.decode()
    for _ in range(50):   # detached: it runs after the guard has returned
        if "runpodctl remove pod pod42" in calls(guard_env):
            break
        time.sleep(0.1)
    assert "runpodctl remove pod pod42 key=pod-scoped" in calls(guard_env)   # the pod's own key, read when it runs
    assert "pod-scoped" not in (guard_env / "home" / ".perception-deadline.sh").read_text()


def test_the_guard_refuses_a_pod_whose_key_reaches_the_account(guard_env):
    (guard_env / "code").write_text("200")
    out = guard(guard_env)
    assert out.returncode == 1 and b"got 200" in out.stderr
    assert "runpodctl" not in calls(guard_env)   # refused before its removal is set: the driver deletes it


@pytest.mark.parametrize("config", [".runpod/config.toml", ".aws/credentials"])
def test_the_guard_refuses_a_pod_holding_an_accounts_config(guard_env, config):
    (guard_env / "home" / config).parent.mkdir(parents=True)
    (guard_env / "home" / config).write_text("apikey = x\n")
    out = guard(guard_env)
    assert out.returncode == 1 and config.encode() in out.stderr
