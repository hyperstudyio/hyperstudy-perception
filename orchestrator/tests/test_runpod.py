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
from perception.remote_driver import removal_at_boot
from perception.scheduler import Coordinator
from perception.store import Store
from perception.manifests import Registry

from test_cloud import GUARD_OK, GUARD_REFUSES, NODES, SETUP_OK, events, graph, log_of, run, script

KEY = "rp-test-key-0123456789"
GUARD = Path(__file__).resolve().parents[2] / "cloud" / "runpod" / "guard.sh"

# test_cloud's ssh stand-in, which also takes -i KEY, -p PORT, -F CONFIG (a pod's own key, port and no config) and -l USER
# (Linux's rsync passes root@host to ssh as -l root host)
FAKE_SSH = r"""#!/bin/sh
while [ $# -gt 0 ]; do case "$1" in -o|-i|-p|-F|-l) echo "ssh option $1 $2" >> "$FAKE_HOST/ssh-options"; shift 2;; -*) shift;; *) break;; esac; done
echo "ssh $1: $2" | head -c 300 >> "$FAKE_HOST/events"; echo >> "$FAKE_HOST/events"
shift
# the pod's clock 10 minutes fast, while $FAKE_HOST/skew exists
[ -e "$FAKE_HOST/skew" ] && PATH="$FAKE_HOST/skewbin:$PATH" && export PATH
cd "$FAKE_HOST/home" && HOME="$FAKE_HOST/home" exec /bin/sh -c "$*"
"""


class FakeRunPod:
    """RunPod's REST API as the driver uses it. `flags`: no-capacity (create answers 500), no-answer (create makes the
    pod but drops the connection), made-then-502 (create makes the pod, then a gateway's 502), slow-create (the pod is
    made 2 s after the create arrives, and only then answered), delete-fails (500 to every DELETE), truncated-once (the
    first DELETE's answer is cut short, and does nothing), truncated-500-once (the same for an error answer), redirect (GET /pods sends the caller to /elsewhere), pricey
    (the pod costs $9.99/h)."""

    def __init__(self, fake_host: Path):
        self.fake_host, self.flags, self.pods, self.bodies, self.headers, self.elsewhere = fake_host, set(), {}, [], [], []
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
                if self.path == "/elsewhere":   # where a redirect leads: a key must never arrive here
                    rp.elsewhere.append(self.headers.get("Authorization"))
                    return self.answer(200, [])
                if self.headers.get("Authorization") != f"Bearer {KEY}":
                    return self.answer(401, {"error": "unauthorized"})
                if method == "POST" and self.path == "/pods":
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    rp.bodies.append(body)
                    if "no-capacity" in rp.flags:
                        return self.answer(500, {"error": "create pod: There are no instances currently available"})
                    if "slow-create" in rp.flags:
                        time.sleep(2)
                    rp.seq += 1
                    pod = {"id": f"pod{rp.seq}", "name": body["name"], "costPerHr": 9.99 if "pricey" in rp.flags else 0.27, "polls": 0}
                    rp.pods[pod["id"]] = pod
                    if "made-then-502" in rp.flags:
                        return self.answer(502, {"error": "bad gateway"})
                    if "no-answer" in rp.flags:
                        self.close_connection = True
                        self.connection.close()   # made, and never said: only its name finds it
                        return
                    return self.answer(201, {k: v for k, v in pod.items() if k != "polls"})
                if method == "GET" and self.path == "/pods" and "redirect" in rp.flags:
                    self.send_response(302)
                    self.send_header("Location", f"{rp.url}/elsewhere")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
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
                    if "truncated-500-once" in rp.flags:
                        rp.flags.discard("truncated-500-once")
                        self.send_response(500)
                        self.send_header("Content-Length", "100")
                        self.end_headers()
                        self.wfile.write(b'{"error": "inter')   # and the connection closes
                        self.close_connection = True
                        return
                    if "truncated-once" in rp.flags:
                        rp.flags.discard("truncated-once")
                        self.send_response(200)
                        self.send_header("Content-Length", "100")
                        self.end_headers()
                        self.wfile.write(b'{"id": "')   # and the connection closes: 92 bytes short
                        self.close_connection = True
                        return
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


def coordinator(tmp_path, guard=GUARD_OK, deadline_s=1800, key=KEY, down_s=900, api_s=60, unknown_s=180) -> Coordinator:
    b = RunPodBackend(api_key=key, deadline_s=deadline_s, runner_kind="self", python=sys.executable, launch_s=60, down_s=down_s, api_s=api_s,
                      unknown_s=unknown_s,
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
    # Secure Cloud in the US, Canada or Europe (never elsewhere), any of the GPUs (the cheapest first), its own disk only
    assert body["cloudType"] == "SECURE" and all(d.startswith(("US-", "CA-", "EU-", "EUR-")) for d in body["dataCenterIds"])
    assert list(RunPodBackend.gpus.values()) == sorted(RunPodBackend.gpus.values())
    assert body["gpuTypeIds"] == list(RunPodBackend.gpus) and body["gpuTypePriority"] == "custom" and body["volumeInGb"] == 0
    # named for the attempt, with its deadline (a later launch deletes it once that has passed)
    assert body["name"].startswith("perception-") and abs(int(body["name"].rsplit("-d", 1)[1]) - (started + 1800)) < 30
    # reached with a key made for the attempt: its public half is what the pod was given
    attempt = coord.store.root / "work" / run_id / "faces" / "1"
    assert body["env"] == {"PUBLIC_KEY": (attempt / "ssh" / "id_ed25519.pub").read_text().strip()}
    # its removal at its deadline (a minute early) set by its start command, before the image's own start
    deadline = int(body["name"].rsplit("-d", 1)[1])
    assert body["dockerStartCmd"] == ["bash", "-c", removal_at_boot(deadline - 60, RunPodBackend.start)]
    options = (pod.fake_host / "ssh-options").read_text()
    assert f"-i {attempt / 'ssh' / 'id_ed25519'}" in options and "-p 2222" in options
    assert all(h.get("User-Agent") == "hyperstudy-perception" for h in pod.headers)   # Cloudflare refuses Python's own
    ev = events(pod.fake_host)
    # the guard is given the oldest NVIDIA driver the nodes' CUDA 13 torch runs on (RunPod's own CUDA filter is not used:
    # it would exclude hosts that report 13.2)
    assert "allowedCudaVersions" not in body
    create, guard, setup = ev.index("api POST /pods"), ev.index("guard 580"), ev.index("setup")
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
    assert status["runpod"]["enabled"] is True and status["enabled"] is False and status["runpod"]["hourly_usd"] == 1.19
    assert [b.kind for b in cli.backends()][-1] == "runpod"
    assert cli.main(["cloud", "off", "runpod"]) == 0 and "runpod" not in [b.kind for b in cli.backends()]


def test_a_pod_made_before_a_gateway_error_is_found_by_its_name_and_deleted(tmp_path, pod):
    # a 5xx proves nothing: RunPod (or Cloudflare before it) may have made the pod first
    pod.flags.add("made-then-502")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    assert status == "failed" and not pod.pods, events(pod.fake_host)
    assert "api DELETE /pods/pod1" in events(pod.fake_host)


def test_a_pod_made_after_its_create_timed_out_is_waited_for_by_its_name_and_deleted(tmp_path, pod):
    # the create's answer comes after the driver stopped waiting (1 s): the pod appears a moment after teardown first looks
    pod.flags.add("slow-create")
    coord = coordinator(tmp_path, api_s=1, down_s=30)
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    for _ in range(30):   # the stand-in makes it 2 s after the create arrived
        if pod.seq:
            break
        time.sleep(0.1)
    assert status == "failed" and pod.seq == 1 and not pod.pods, events(pod.fake_host)
    assert "api DELETE /pods/pod1" in events(pod.fake_host)


def test_an_answer_cut_short_while_deleting_is_asked_again(tmp_path, pod):
    pod.flags.add("truncated-once")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    assert status == "succeeded", log_of(coord, run_id)
    assert not pod.pods and events(pod.fake_host).count("api DELETE /pods/pod1") == 2
    assert task(coord, run_id)[1]["machine_usd"] > 0


def test_an_error_answer_cut_short_while_deleting_is_asked_again(tmp_path, pod):
    pod.flags.add("truncated-500-once")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    assert status == "succeeded", log_of(coord, run_id)
    assert not pod.pods and events(pod.fake_host).count("api DELETE /pods/pod1") == 2


def test_a_create_that_never_reached_runpod_looks_for_nothing_and_is_retried(tmp_path, pod, monkeypatch):
    # this laptop offline: nothing was sent, so no pod can exist
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()   # a port nothing listens on: connection refused
    monkeypatch.setenv("PERCEPTION_RUNPOD_API", f"http://127.0.0.1:{port}")
    coord = coordinator(tmp_path, down_s=20)
    started = time.monotonic()
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    t, _ = task(coord, run_id)
    assert status == "failed" and t["error_code"] == "transient", t["error"]
    assert time.monotonic() - started < 15   # not down_s spent looking for a pod that cannot exist
    assert coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0] == 0


def test_a_pod_whose_create_got_no_clear_answer_and_never_appears_is_taken_as_never_made(tmp_path, pod):
    pod.flags.add("no-capacity")   # stands for a create RunPod never acted on
    pod.flags.add("made-then-502")
    coord = coordinator(tmp_path, unknown_s=2, down_s=30)
    pod.flags.discard("no-capacity")

    class Never(dict):   # the create's pod never shows in the listing
        def __setitem__(self, k, v):
            pass
    pod.pods = Never()
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    t, _ = task(coord, run_id)
    assert status == "failed" and t["error_code"] == "transient"
    assert "none was made" in log_of(coord, run_id)
    assert coord.store.db.execute("SELECT spent_usd FROM runs WHERE id=?", (run_id,)).fetchone()[0] == 0


def test_a_create_refused_with_a_4xx_looks_for_nothing(tmp_path, pod):
    coord = coordinator(tmp_path, key="not-the-key")
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    assert status == "failed" and task(coord, run_id)[0]["error_code"] == "internal"
    assert events(pod.fake_host).count("api GET /pods") == 1   # the sweep's listing only: no search by name


def test_a_pod_dearer_than_what_was_reserved_is_not_used(tmp_path, pod):
    pod.flags.add("pricey")
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    assert status == "failed" and not pod.pods
    ev = events(pod.fake_host)
    assert not any("objects" in e or "code/" in e for e in ev) and "guard " not in ev
    t, _ = task(coord, run_id)
    assert t["error_code"] == "internal" and "9.99" in t["error"]


def test_the_key_never_follows_a_redirect(tmp_path, pod):
    from perception.runpod import RunPod, RunPodError
    pod.flags.add("redirect")
    with pytest.raises(RunPodError):
        RunPod(KEY, pod.url).pods()
    assert pod.elsewhere == []


def test_a_pods_ssh_forwards_nothing_and_pins_its_host_key_in_a_path_with_spaces(tmp_path, pod, monkeypatch):
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "a home"))
    coord = Coordinator(Store(tmp_path / "my store"), Registry.load(NODES),
                        [SelfBackend(), RunPodBackend(api_key=KEY, deadline_s=1800, runner_kind="self", python=sys.executable, launch_s=60,
                                                      guard=script(tmp_path / "guard.sh", GUARD_OK), setup=script(tmp_path / "node-setup.sh", SETUP_OK))],
                        tick_s=0.02, backoff_s=0.01)
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    assert status == "succeeded", log_of(coord, run_id)
    options = (pod.fake_host / "ssh-options").read_text()
    assert "-o ForwardAgent=no" in options and "-o ClearAllForwardings=yes" in options
    assert "-F none" in options   # none of the user's ssh config (its IdentityFiles above all) reaches a pod
    known = coord.store.root / "work" / run_id / "faces" / "1" / "ssh" / "known_hosts"
    # quoted: an unquoted value is "one or more files ... separated by whitespace" (ssh_config(5))
    assert f'-o UserKnownHostsFile="{known}"' in options


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
    (t / "driver").write_text("595.91.07")
    script(t / "bin" / "nvidia-smi", f'#!/bin/sh\ncase "$*" in *driver_version*) cat {t}/driver;; *) echo "NVIDIA RTX A5000";; esac\n')
    script(t / "bin" / "setsid", '#!/bin/sh\nexec "$@"\n')
    return t


def guard(t: Path, min_driver: str = "580") -> subprocess.CompletedProcess:
    env = {"HOME": str(t / "home"), "PATH": f"{t / 'bin'}:/usr/bin:/bin", "PERCEPTION_POD_ENVIRON": str(t / "environ")}
    return subprocess.run(["bash", "-s", min_driver], input=GUARD.read_bytes(), env=env, capture_output=True, timeout=60)


@pytest.fixture
def remover():
    """A process named as a pod's removal at its deadline is (removal_at_boot): what guard.sh looks for."""
    import os
    import signal
    # two commands: sh would otherwise exec sleep, and its name would go
    proc = subprocess.Popen(["sh", "-c", "sleep 60; :", "perception-deadline"], start_new_session=True)
    for _ in range(50):
        if subprocess.run(["pgrep", "-f", "perception-deadline"], capture_output=True).returncode == 0:
            break
        time.sleep(0.05)
    yield proc
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def calls(t: Path) -> str:
    return (t / "calls").read_text() if (t / "calls").exists() else ""


def test_the_guard_passes_a_pod_whose_removal_is_running_and_whose_key_reaches_nothing(guard_env, remover):
    out = guard(guard_env)
    assert out.returncode == 0, out.stderr
    assert "NVIDIA RTX A5000 (driver 595.91.07) in US-TX-1; no key to the account; its removal at its deadline is running" in out.stdout.decode()


@pytest.mark.parametrize("driver", ["570.172.08", "550.54.15", ""])
def test_the_guard_says_a_pod_with_an_older_driver_is_not_this_ones_to_use(guard_env, remover, driver):
    # the nodes' torch is built for CUDA 13 (driver 580 at least): another pod may have a newer one
    (guard_env / "driver").write_text(driver)
    out = guard(guard_env)
    assert out.returncode == 75 and b"580" in out.stderr, out.stderr


def test_the_guard_refuses_a_pod_whose_removal_is_not_running(guard_env):
    out = guard(guard_env)
    assert out.returncode == 1 and b"removal at its deadline is not running" in out.stderr


def test_the_guard_refuses_a_pod_whose_key_reaches_the_account(guard_env):
    (guard_env / "code").write_text("200")
    out = guard(guard_env)
    assert out.returncode == 1 and b"got 200" in out.stderr


def test_the_guard_refuses_a_pod_with_no_key_of_its_own(guard_env):
    # its removal at the deadline would run with no key, and fail
    (guard_env / "environ").write_bytes(b"RUNPOD_POD_ID=pod42\0RUNPOD_DC_ID=US-TX-1\0")
    (guard_env / "code").write_text("401")
    out = guard(guard_env)
    assert out.returncode == 1 and b"RUNPOD_API_KEY" in out.stderr


@pytest.mark.parametrize("code", ["000", "500", "429"])
def test_the_guard_says_its_check_could_not_be_made_apart_from_a_refusal(guard_env, code):
    # no answer, or RunPod's own trouble: the key was not shown to reach the account, nor not to (the driver retries)
    (guard_env / "code").write_text(code)
    out = guard(guard_env)
    assert out.returncode == 75, out.stderr


def test_a_guard_that_could_not_check_the_pod_is_a_launch_failure_retried(tmp_path, pod):
    coord = coordinator(tmp_path, guard='echo "guard: the pod\'s NVIDIA driver is 570.1, older than 580" >&2; exit 75\n')
    doc = graph(tmp_path, rec(tmp_path), place="runpod:cuda")
    doc["policy"]["retries"] = 1
    run_id, status = run(coord, doc)
    assert status == "failed" and len(pod.bodies) == 2 and not pod.pods
    t = task(coord, run_id)[0]
    # the guard's own reason, not a guess at it
    assert t["error_code"] == "transient" and "NVIDIA driver is 570.1, older than 580" in t["error"]



def test_a_pods_start_command_removes_it_at_its_deadline_asking_until_it_is_done(tmp_path):
    t = tmp_path
    (t / "bin").mkdir()
    # runpodctl fails the first time, saying so with exit 0 (its exit is not relied on), then "removes" it: the stand-in
    # ends the loop as RunPod would, by stopping the container (its process group)
    script(t / "bin" / "runpodctl", f'#!/bin/sh\necho "runpodctl $*" >> {t}/calls; [ -e {t}/once ] && kill -9 0; touch {t}/once; echo "Error: Unauthorized"; exit 0\n')
    script(t / "bin" / "sleep", f'#!/bin/sh\necho "sleep $*" >> {t}/calls\n')   # no waiting
    env = {"PATH": f"{t / 'bin'}:/usr/bin:/bin", "RUNPOD_POD_ID": "pod42"}
    out = subprocess.run(["bash", "-c", removal_at_boot(int(time.time()) - 1, "echo the image starts")], env=env,
                         capture_output=True, text=True, timeout=30, start_new_session=True)
    assert out.stdout.strip() == "the image starts"   # its own start runs (exec), the removal beside it
    for _ in range(50):
        if calls(t).count("runpodctl pod delete pod42") == 2:
            break
        time.sleep(0.1)
    assert calls(t).count("runpodctl pod delete pod42") == 2 and "sleep 30" in calls(t)


def test_a_pods_removal_waits_for_its_deadline(tmp_path):
    t = tmp_path
    (t / "bin").mkdir()
    script(t / "bin" / "runpodctl", f'#!/bin/sh\necho "runpodctl $*" >> {t}/calls\n')
    env = {"PATH": f"{t / 'bin'}:/usr/bin:/bin", "RUNPOD_POD_ID": "pod42"}
    subprocess.run(["bash", "-c", removal_at_boot(int(time.time()) + 3600, "true")], env=env, timeout=30)
    try:
        time.sleep(1)
        assert subprocess.run(["pgrep", "-f", "perception-deadline"], capture_output=True).returncode == 0
        assert "runpodctl" not in calls(t)
    finally:
        subprocess.run(["pkill", "-f", f"date \\+%s\\) -ge {int(time.time()) + 3600}"], capture_output=True)
        subprocess.run(["pkill", "-f", "perception-deadline"], capture_output=True)


def test_a_pod_whose_clock_disagrees_is_not_used(tmp_path, pod):
    skew = pod.fake_host / "skewbin"
    skew.mkdir()
    script(skew / "date", '#!/bin/sh\n[ "$1" = "+%s" ] && { echo $(( $(/bin/date +%s) + 600 )); exit 0; }; exec /bin/date "$@"\n')
    (pod.fake_host / "skew").touch()
    coord = coordinator(tmp_path)
    run_id, status = run(coord, graph(tmp_path, rec(tmp_path), place="runpod:cuda"))
    assert status == "failed" and not pod.pods
    t, _ = task(coord, run_id)
    assert t["error_code"] == "internal" and "clock" in t["error"]
    assert "guard " not in events(pod.fake_host)


@pytest.mark.parametrize("config", [".runpod/config.toml", ".aws/credentials"])
def test_the_guard_refuses_a_pod_holding_an_accounts_config(guard_env, config):
    (guard_env / "home" / config).parent.mkdir(parents=True)
    (guard_env / "home" / config).write_text("apikey = x\n")
    out = guard(guard_env)
    assert out.returncode == 1 and config.encode() in out.stderr


# --- the image (cloud/runpod/image) -------------------------------------------------------------------------------

IMAGE = Path(__file__).resolve().parents[2] / "cloud" / "runpod" / "image"


def image_tag_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("runpod_image_tag", IMAGE / "tag.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_backend_uses_the_image_built_from_the_repositorys_files():
    # a lock or image change needs a new build, under the tag its files give (.github/workflows/runpod-image.yml)
    assert RunPodBackend.image_tag == image_tag_module().tag(), "rebuild the image (push) and set RunPodBackend.image_tag to tag.py's"
    assert RunPodBackend.image.endswith(":" + RunPodBackend.image_tag)


def test_the_images_context_holds_no_code_of_the_repositorys(tmp_path):
    # it is public: its Dockerfile and start script, nothing else
    tag = image_tag_module()
    tag.stage(tmp_path)
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()) == ["Dockerfile", "start.sh"]


def test_the_image_starts_on_runpods_cached_base_pinned_by_its_digest():
    # its hosts keep that image: a pod then pulls only the small layers on top (an image of its own took 145 s to start)
    first = next(l for l in (IMAGE / "Dockerfile").read_text().splitlines() if l.startswith("FROM "))
    assert first.startswith("FROM runpod/pytorch:") and "@sha256:" in first
