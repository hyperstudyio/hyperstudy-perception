"""Runs one attempt of a node on another machine, for the coordinator (backends/remote.py starts it):

    python -m perception.remote_driver --request ATTEMPT/request.json      (with ATTEMPT/remote.json beside it)

1. Copies the node's code (by content: once per version) and its inputs (by sha256: once per host) to the host.
2. Writes the request there, with the host's paths, and starts the node in its own session (setsid), detached,
   so a dropped connection does not end it, with a watchdog beside it (WATCHDOG) that stops it and removes its folder
   if no driver asks about it for a while. A previous attempt of the same node still running there is stopped first.
3. Watches it every few seconds, appending its log to this attempt's log.txt. A host that cannot be reached is waited
   for (a laptop changes networks) for up to lost_after_s; after that the attempt ends without a response, which the
   coordinator treats as a crash (retried within the run's policy), and the host's watchdog ends the node.
4. Brings out/ and response.json back into this attempt's folder, then removes the attempt's folder on the host.

SIGTERM (the coordinator's cancel) stops the node on the host before the driver exits. Its output (stdout) is the
attempt's log.txt.

On a cloud machine (spec "cloud", backends/cloud.py) the driver first starts the machine (sky launch, with the
repository's project config), makes an OS shutdown terminate it, probes it like a host, and runs guard.sh there (the
perception-node role, no AWS credentials, a shutdown at the deadline) before anything is sent. The node is stopped
when the deadline nears. Whatever happens, the machine is deleted (sky down) before the driver exits, and the time it
existed is added to the response's usage.usd (all of the deadline if it could not be deleted).

On a RunPod pod (cloud "provider": "runpod") the same steps go through RunPod's API instead: create the pod (the account
key in this process's environment only), wait for its SSH port, reach it with a key made for the attempt, run
cloud/runpod/guard.sh there (no key to the account on it, its own removal at the deadline), and delete it at the end.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

POLL_S = 3.0
# exit codes the scripts here use for a refusal that would happen again: the node's error code for each
REFUSALS = {3: "input", 5: "input"}
# how long past its deadline a perception pod is left before a RunPod launch deletes it: its own driver deletes it before
# its deadline, and it removes itself at it (guard.sh)
SWEEP_GRACE_S = 10 * 60
KEEP_S = 24 * 3600   # how long a finished node's results wait on the host for a driver that lost touch (WATCHDOG)


class Canceled(Exception):
    pass


class Unreachable(Exception):
    pass


class PrepareFailed(Exception):
    """The node's environment could not be synced, or its prepare script failed. Not retried: these fail the same way
    again (no wheel for the machine, an import error, a model setting that is wrong), and on a cloud place each attempt
    pays for a machine. The log says why; a run started again tries once more."""


class Refused(Exception):
    """The host answered, and the attempt cannot go on: the same would happen again (an input that changed since the run
    read it, a root that is not the user's). It fails as the node's own error, not a crash to retry."""
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Deadline(Exception):
    pass


class GuardRefused(Exception):
    pass


class LaunchFailed(Exception):
    """The cloud machine did not start (no capacity, a setup that failed or hung): worth another attempt."""


# Publish a folder in one step: rename $1 to $2 unless $2 exists, in which case $1 is dropped (the same content, by its
# hash, arrived first). rename(2) onto a folder that is not empty fails, so a published folder is never replaced or
# nested into: GNU mv -T on Linux, perl's rename on a Mac (BSD mv has no -T and would move $1 into $2).
PUBLISH = r"""publish() {
  mv -T "$1" "$2" 2>/dev/null || perl -e 'rename($ARGV[0], $ARGV[1]) or exit 1' "$1" "$2" 2>/dev/null || { chmod -R u+w "$1"; rm -rf "$1"; [ -e "$2/.complete" ]; }
}"""

# The host's backstop for a node its driver lost (the laptop gone, a cancel that could not reach the host): every
# `every` seconds, while the attempt's folder exists, if no driver has written the heartbeat (alive) for `lost` seconds
# and the node is still running, stop its process group (only if that pid is still this attempt's node) and remove the
# folder: its outputs are partial. A node that finished keeps its folder for `keep` seconds, for a driver that comes back
# (a laptop that slept) to fetch: its results are whole, and running it again would cost the GPU time again.
WATCHDOG = r"""w=$1 pid=$2 lost=$3 every=$4 keep=$5
while [ -d "$w" ]; do
  sleep "$every"
  [ -d "$w" ] || exit 0
  last=$(cat "$w/alive" 2>/dev/null)
  case "$last" in ''|*[!0-9]*) continue;; esac   # not a time (being written): never read as 0, which is long ago
  age=$(( $(date +%s) - last ))
  if ps -ww -p "$pid" -o command= 2>/dev/null | grep -qF -- "$w/request.json"; then
    [ "$age" -gt "$lost" ] || continue
    kill -s TERM -- "-$pid" 2>/dev/null; sleep 5; kill -s KILL -- "-$pid" 2>/dev/null
  elif [ "$age" -le "$keep" ]; then
    continue
  fi
  rm -rf "$w"
  rmdir "${w%/*}" "${w%/*/*}" 2>/dev/null   # the run's folders too, once empty (as fetch does)
  exit 0
done"""


def heartbeat_sh(attempt: str) -> str:
    """sh that writes the heartbeat the watchdog reads, in one step (written beside it, then renamed): `date > alive`
    empties the file first, and a watchdog reading it then would see no time at all."""
    a = shlex.quote(attempt)
    return f"date +%s > {a}/alive.tmp 2>/dev/null && mv -f {a}/alive.tmp {a}/alive 2>/dev/null"


def removal_at_boot(at: int, start: str) -> str:
    """A pod's start command: its removal at `at` (a Unix time), set before anything else runs on it and asked again
    until RunPod does it (runpodctl, with the pod's own key, scoped to that pod), then the image's own start (sshd).
    The backstop for a pod whose driver never comes back, from its first moment; perception-deadline names it (guard.sh)."""
    # asked every 30 s until it happens, whatever runpodctl's exit says: the pod's removal ends this loop with it
    removal = (f"until [ $(date +%s) -ge {int(at)} ]; do sleep 15; done; "
               'while :; do runpodctl remove pod "$RUNPOD_POD_ID"; sleep 30; done')
    return f"nohup sh -c {shlex.quote(removal)} perception-deadline > /dev/null 2>&1 & exec {start}"


def say(msg: str):
    print(f"[remote] {msg}", flush=True)


class Driver:
    def __init__(self, request_path: Path, spec: dict):
        self.request_path, self.spec = request_path, spec
        self.local = request_path.parent
        self.request = json.loads(request_path.read_text())
        self.pid: int | None = None
        self.uploading = False   # an upload to the host began: discard_incoming has something to look for
        self.fetching = False    # the node finished and its results are coming back: within the deadline's margin
        self.unreachable_since: float | None = None
        self.started = time.monotonic()
        self.stop_at: float | None = None   # a cloud machine's deadline, less the time to bring results back and delete it
        self.extra_usd = 0.0
        self.phases: dict[str, float] = {}   # seconds each part of the attempt took (in the response's usage, and the log)
        self.deadline_set = False   # the machine's own OS shutdown is scheduled (guard.sh): it cannot outlive its deadline
        self.launch_unstopped = False   # a launch whose cancel on SkyPilot's server was not confirmed: it may still make a machine
        # a RunPod pod: its id once created, the price RunPod gave it, and whether a create was sent without an answer
        self.pod: str | None = None
        self.pod_name: str | None = None
        self.pod_usd: float | None = None
        self.pod_unknown = False
        # the account key, for this process alone: out of the environment its ssh and rsync commands inherit
        self.runpod_key = os.environ.pop("RUNPOD_API_KEY", None)
        self.at(spec["dest"], spec["root"])

    def at(self, dest: str, root: str):
        self.dest, self.root = dest, root
        r = self.request
        self.attempt = f"{root}/work/{r['run']}/{r['node']}/{r['attempt']}"
        self.node_runs = f"{root}/work/{r['run']}/{r['node']}"
        self.code = f"{root}/code/{self.spec['code']}"
        self.tag = f"{r['run']}-{r['node']}-{r['attempt']}"   # this attempt's own names for what it uploads

    # --- commands on the host -----------------------------------------------------------------------------------

    def sh(self, script: str, check: bool = True, data: bytes | None = None, timeout: float = 600) -> subprocess.CompletedProcess:
        """Run a POSIX sh script on the host (through sh: the login shell may be zsh or fish). ssh's own failures (exit
        255: no connection) raise Unreachable."""
        out = subprocess.run([*self.spec["ssh"], self.dest, f"sh -c {shlex.quote(script)}"], input=data, capture_output=True, timeout=timeout)
        if out.returncode == 255:
            raise Unreachable(out.stderr.decode(errors="replace").strip()[-300:])
        if check and out.returncode in REFUSALS:
            raise Refused(REFUSALS[out.returncode], out.stderr.decode(errors="replace").strip()[-500:])
        if check and out.returncode:
            raise RuntimeError(f"on {self.spec['host']}: {script[:120]}... exited {out.returncode}: "
                               f"{out.stderr.decode(errors='replace').strip()[-500:]}")
        return out

    def rsync(self, *args: str):
        shell = " ".join(shlex.quote(a) for a in self.spec["ssh"])
        # -rlt: files, folders, links as links (never followed: a link in out/ is refused at publication) and times;
        # no devices, permissions or owners
        out = subprocess.run(["rsync", "-rlt", "-e", shell, *args], capture_output=True, text=True, timeout=24 * 3600)
        if out.returncode:
            raise RuntimeError(f"rsync {' '.join(args[-2:])} failed ({out.returncode}): {out.stderr.strip()[-500:]}")

    # --- steps ----------------------------------------------------------------------------------------------------

    def check_root(self):
        """The root must still be the user's own private folder: the markers below are trusted only because no one
        else can write there (hosts.PRIVATE_ROOT)."""
        from .hosts import private_root_script
        self.sh(private_root_script(self.root))

    def send_code(self):
        """The code under code/<its hash>, published in one step (PUBLISH): runs on the host at the same time may send
        the same code. What is sent is a copy taken here and checked against that hash first: a file saved while the
        code is on its way must not be filed under the hash of the code before it."""
        import shutil
        from .backends.native import REPO
        from .backends.remote import code_hash
        q = shlex.quote
        if self.sh(f"test -e {q(self.code)}/.complete", check=False).returncode == 0:
            return
        snapshot = self.local / "code"
        shutil.rmtree(snapshot, ignore_errors=True)
        for f in self.spec["files"]:
            (snapshot / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / f, snapshot / f)
        if code_hash(self.spec["files"], snapshot) != self.spec["code"]:
            shutil.rmtree(snapshot)
            raise RuntimeError("the node's code changed since this attempt started: nothing was sent (the next attempt sends the new code)")
        say(f"copying the node's code to {self.spec['host']}")
        incoming = f"{self.root}/code/.incoming-{self.tag}"
        self.uploading = True
        self.sh(f"rm -rf {q(incoming)} && mkdir -p {q(incoming)}")
        listing = self.local / "code-files.txt"
        listing.write_text("".join(f + "\n" for f in self.spec["files"]))
        self.rsync("--files-from", str(listing), f"{snapshot}/", f"{self.dest}:{incoming}/")
        self.sh(f"{PUBLISH}\ntouch {q(incoming)}/.complete && publish {q(incoming)} {q(self.code)}")
        shutil.rmtree(snapshot)

    def send_inputs(self) -> dict[str, dict]:
        """Each input as objects/<sha256>/data on the host, published in one step (PUBLISH): nodes on the host at the
        same time may send the same input. The node gets it under its own name, a link in its attempt's in/ (the same
        bytes can arrive under another name: the video renamed)."""
        q = shlex.quote
        out = {}
        for name, entry in self.request["inputs"].items():
            # the content, not a link to it: a graph input is often a link (to a video on an external drive), which rsync
            # would send as a link pointing nowhere there (or somewhere else). The node gets it under the name it has
            # here, the link's own, as a local run gives it (decode records it): the target's name never goes there.
            given = Path(entry["path"])
            local = given.resolve()
            folder = f"{self.root}/objects/{entry['sha256']}"
            out[name] = {**entry, "path": f"{self.attempt}/in/{name}/{given.name}", "object": f"{folder}/data"}
            if self.sh(f"test -e {q(folder)}/.complete", check=False).returncode == 0:
                continue
            size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file()) if local.is_dir() else local.stat().st_size
            say(f"copying {name} ({size / 1e6:.1f} MB) to {self.spec['host']}")
            incoming = f"{self.root}/objects/.incoming-{self.tag}-{name}"
            self.uploading = True
            self.sh(f"rm -rf {q(incoming)} && mkdir -p {q(incoming)}")
            self.rsync(f"{local}/" if local.is_dir() else str(local), f"{self.dest}:{incoming}/data" + ("/" if local.is_dir() else ""))
            # filed under its sha256 only if that is what arrived (a file changed since the run hashed it is refused);
            # a folder comes from the store, which checked it when it was published
            verify = "" if local.is_dir() else (
                f"got=$( (sha256sum {q(incoming)}/data 2>/dev/null || shasum -a 256 {q(incoming)}/data) | cut -d' ' -f1)\n"
                f"[ \"$got\" = {q(entry['sha256'])} ] || {{ rm -rf {q(incoming)}; echo \"{name} arrived as $got, not {entry['sha256']}: "
                f"it changed since the run read it\" >&2; exit 5; }}\n")
            # read-only once published, as the local store's objects are: a node that wrote to its input (through its
            # link) would change it for every later run on this host, under the same sha256
            self.sh(f"{PUBLISH}\n{verify}touch {q(incoming)}/.complete && publish {q(incoming)} {q(folder)} && chmod -R a-w {q(folder)}")
        return out

    def start(self, inputs: dict[str, dict]):
        q, s = shlex.quote, self.spec
        request = {**self.request, "inputs": {k: {x: y for x, y in v.items() if x != "object"} for k, v in inputs.items()},
                   "out_dir": f"{self.attempt}/out"}
        if s["runner_kind"] == "self":
            node = f"{q(s['python'])} {q(s['entry'])}"
        else:
            node = f"{q(s['uv'])} run --project {q(self.code + '/' + s['env'])} --frozen --no-dev python {q(s['entry'])}"
        req = f"{self.attempt}/request.json"
        path = ":".join(s["path"])
        links = "\n".join(f"mkdir -p {q(self.attempt)}/in/{q(name)} && ln -sfn {q(v['object'])} {q(v['path'])}"
                          for name, v in inputs.items())
        lost = int(s["lost_after_s"] + min(60, s["lost_after_s"]))   # after the driver has given up
        every = max(1, min(30, lost // 4))
        # an earlier attempt of this node that a lost connection left running would hold the GPU: stop it first
        script = f"""
set -e
for f in {q(self.node_runs)}/*/pid; do
  [ -f "$f" ] || continue
  old=$(cat "$f")
  # only if that pid is still that attempt's node (pids are reused on a shared machine)
  if ps -ww -p "$old" -o command= 2>/dev/null | grep -qF -- "${{f%/pid}}/request.json"; then
    kill -s TERM -- "-$old" 2>/dev/null || true
    n=0; while [ $n -lt 10 ] && kill -0 "$old" 2>/dev/null; do sleep 1; n=$((n + 1)); done   # gone before this one starts
    kill -s KILL -- "-$old" 2>/dev/null || true
  fi
done
mkdir -p {q(self.attempt)}/out
{links}
cat > {q(req)}
{heartbeat_sh(self.attempt)}
cd {q(self.code + '/' + s['node_dir'])}
# its own session, so its whole process tree can be stopped (kill -s TERM -- -PID) and a closed connection does not
# reach it: setsid on Linux, perl's on a Mac (no setsid command there). Not a group leader in the background, so
# neither forks: $! stays the node's pid.
if command -v setsid >/dev/null 2>&1; then set -- setsid; else set -- perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV or die'; fi
PATH={q(path)}${{PATH:+:$PATH}} nohup "$@" {node} --request {q(req)} > {q(self.attempt)}/log.txt 2>&1 < /dev/null &
pid=$!
echo $pid > {q(self.attempt)}/pid
# a cancel that came while this script ran (and found no pid yet) left a stop mark: stop the node now
if [ -e {q(self.attempt)}/stop ]; then kill -s TERM -- "-$pid" 2>/dev/null || true; fi
nohup "$@" sh -c {q(WATCHDOG)} watchdog {q(self.attempt)} "$pid" {lost} {every} {KEEP_S} > /dev/null 2>&1 < /dev/null &
echo $pid
"""
        out = self.sh(script, data=json.dumps(request, indent=1).encode())
        self.pid = int(out.stdout.decode().split()[-1])
        say(f"started on {s['host']} (pid {self.pid})")

    def unreachable(self, e: Exception):
        """Wait out a host that cannot be reached, for up to lost_after_s in all since it was last reached; past that,
        give up (raise). Not past a cloud machine's deadline: that is a failure of its own (Deadline)."""
        now = time.monotonic()
        if self.stop_at is not None and not self.fetching and now > self.stop_at:
            raise Deadline()
        if self.unreachable_since is None:
            self.unreachable_since = now
            say(f"{self.spec['host']} cannot be reached ({e}); waiting up to {self.spec['lost_after_s']:.0f} s")
        elif now - self.unreachable_since > self.spec["lost_after_s"]:
            raise Unreachable(str(e))
        time.sleep(POLL_S)

    def patiently(self, step, *args):
        """A step repeated while the host cannot be reached (up to lost_after_s, as watching waits): a short outage
        must not cost an attempt, nor a finished node's results. A failed rsync is repeated only if the host is
        unreachable; any other failure is the step's own."""
        while True:
            # an upload that ran into a cloud machine's deadline; never a fetch, which the deadline keeps a margin for
            if self.stop_at is not None and not self.fetching and time.monotonic() > self.stop_at:
                raise Deadline()
            try:
                result = step(*args)
                self.unreachable_since = None
                return result
            except Unreachable as e:
                self.unreachable(e)
            except RuntimeError as e:
                try:
                    self.sh("true")
                except Unreachable:
                    self.unreachable(e)
                    continue
                raise

    @contextlib.contextmanager
    def heartbeat(self):
        """Keep writing the heartbeat while something long happens (a large out/ coming back), so the host's watchdog
        does not take this attempt for one no driver is watching."""
        done = threading.Event()
        every = min(POLL_S, max(0.2, self.spec["lost_after_s"] / 2))

        def beat():
            while not done.wait(every):
                try:
                    self.sh(heartbeat_sh(self.attempt), check=False)
                except (Unreachable, subprocess.TimeoutExpired):
                    pass   # a dropped connection: the step itself waits it out
        thread = threading.Thread(target=beat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join()

    def watch(self):
        q = shlex.quote
        req, log = f"{self.attempt}/request.json", f"{self.attempt}/log.txt"
        offset = 0
        while True:
            # the heartbeat first: the host's watchdog ends a node no driver has asked about for too long
            check = (f"{heartbeat_sh(self.attempt)}; "
                     f"if kill -0 {self.pid} 2>/dev/null && ps -ww -p {self.pid} -o command= | grep -qF -- {q(req)}; "
                     f"then echo running; else echo exited; fi; tail -c +{offset + 1} {q(log)} 2>/dev/null || true")
            try:
                out = self.sh(check).stdout
                self.unreachable_since = None
            except Unreachable as e:
                self.unreachable(e)
                continue
            state, _, text = out.partition(b"\n")
            if text:
                sys.stdout.buffer.write(text)
                sys.stdout.flush()
                offset += len(text)
            if state.strip() == b"exited":
                return
            if self.stop_at is not None and time.monotonic() > self.stop_at:
                raise Deadline()
            time.sleep(POLL_S)

    def fetch(self):
        """out/ and response.json, into this attempt's folder. No response there: the node crashed (none is written
        here either, and the coordinator sees a crash). The node has finished: a dropped connection is waited out, and
        the heartbeat kept up, rather than its results lost."""
        q = shlex.quote
        self.fetching = True
        with self.heartbeat():
            # never through a link the node made: rsync would copy what `out/` points at, which publication cannot see
            listing = self.patiently(self.sh, f"cd {q(self.attempt)} 2>/dev/null || exit 0; [ -d out ] && [ ! -L out ] && echo out; "
                                              f"[ -f response.json ] && [ ! -L response.json ] && echo response.json; true").stdout.decode().split()
            if "out" in listing:
                self.patiently(self.rsync, f"{self.dest}:{self.attempt}/out/", f"{self.local}/out/")
            if "response.json" in listing:
                self.patiently(self.rsync, f"{self.dest}:{self.attempt}/response.json", f"{self.local}/response.json.part")
        # the run's folders too once empty (another node of the run may still be using them)
        self.patiently(self.sh, f"rm -rf {q(self.attempt)}; rmdir {q(self.node_runs)} {q(self.node_runs.rsplit('/', 1)[0])} 2>/dev/null || true")

    def prepare(self):
        """What `perception setup` does for a node here, done on the host before it starts: its uv environment synced from
        its lock, then its prepare script (its weights) run in it with its settings. A failure is the attempt's own and
        final (PrepareFailed: not retried, since it would most likely fail again, at a cloud machine's cost)."""
        q, s = shlex.quote, self.spec
        path = ":".join(s["path"])
        node_dir = f"{self.code}/{s['node_dir']}"
        if s["runner_kind"] == "self":
            python = q(s["python"])
            sync = ""
        else:
            project = q(f"{self.code}/{s['env']}")
            python = f"{q(s['uv'])} run --project {project} --frozen --no-dev python"
            sync = f"{q(s['uv'])} sync --project {project} --frozen --no-dev"
        steps = [x for x in (sync, f"{python} {q(s['prepare'])} {q(json.dumps(self.request.get('config', {})))}" if s.get("prepare") else "") if x]
        if not steps:
            return
        say("preparing the node there (its environment, its weights)")
        # within a cloud machine's deadline: past it, the node could not run anyway
        limit = 3600 if self.stop_at is None else self.stop_at - time.monotonic()
        if limit <= 0:
            raise Deadline()
        try:
            done = self.patiently(self.sh, f"cd {q(node_dir)} && export PATH={q(path)}${{PATH:+:$PATH}} && " + " && ".join(steps),
                                  False, None, limit)
        except subprocess.TimeoutExpired:
            if self.stop_at is not None:
                raise Deadline()
            raise PrepareFailed("preparing the node took more than an hour")
        sys.stdout.buffer.write(done.stdout[-4000:] + done.stderr[-4000:])
        if done.returncode:
            raise PrepareFailed(f"preparing the node failed (exit {done.returncode}): see the end of the log")

    def stop(self):
        """Stop this attempt's node on the host, if it started (by its pid file: the driver may have been canceled
        between starting it and reading its pid)."""
        signal.signal(signal.SIGTERM, signal.SIG_IGN)   # a second cancel must not interrupt the stopping
        q = shlex.quote
        req, pidfile = f"{self.attempt}/request.json", f"{self.attempt}/pid"
        say(f"stopping it on {self.spec['host']}")
        try:
            # the mark first, then the pid: a start still running on the host sees the one, or this finds the other
            self.sh(f'mkdir -p {q(self.attempt)} && touch {q(self.attempt)}/stop; [ -f {q(pidfile)} ] || exit 0; p=$(cat {q(pidfile)}); '
                    f'if ps -ww -p "$p" -o command= | grep -qF -- {q(req)}; then kill -s TERM -- "-$p"; fi', check=False)
        except (Unreachable, subprocess.TimeoutExpired) as e:
            say(f"could not reach {self.spec['host']} to stop it ({e}); it ends when the next attempt of this node starts there")

    @contextlib.contextmanager
    def phase(self, name: str):
        """Time a part of the attempt (launch, setup, code, inputs, run, fetch, teardown): where its time goes."""
        start = time.monotonic()
        try:
            yield
        finally:
            self.phases[name] = round(self.phases.get(name, 0.0) + time.monotonic() - start, 1)

    def publish_response(self):
        """The node's response, now that it is final: with what the cloud machine cost added to what the node spent, and
        where the attempt's time went (usage.phases; `startup` is the run's time outside the node's own work: its imports,
        and the driver's ssh round trips and polling)."""
        if self.phases:
            say("time: " + ", ".join(f"{k} {v:.0f} s" for k, v in self.phases.items()))
        part = self.local / "response.json.part"
        if not part.exists():
            return
        response = json.loads(part.read_text())
        usage = response.setdefault("usage", {})
        if self.extra_usd:
            usage["usd"] = round(float(usage.get("usd", 0.0)) + self.extra_usd, 4)
            usage["machine_usd"] = self.extra_usd
        phases = dict(self.phases)
        if "run" in phases and isinstance(usage.get("seconds"), (int, float)):
            phases["startup"] = round(max(0.0, phases["run"] - usage["seconds"]), 1)
        usage["phases"] = phases
        part.write_text(json.dumps(response, indent=1))
        part.replace(self.local / "response.json")

    # --- a cloud machine ------------------------------------------------------------------------------------------

    def cloud_env(self, c: dict) -> dict:
        return {**os.environ, "SKYPILOT_PROJECT_CONFIG": c["project_config"]}

    def provision(self, c: dict):
        """Start the machine. Its own deadline is set from the first moment, in case this laptop never comes back: the
        task's setup schedules the OS shutdown first (DEADLINE_EPOCH), and while the launch runs a watcher makes an OS
        shutdown terminate the instance (EC2 would only stop it, its disk billing). The launch has a time limit
        (launch_s). SkyPilot runs a launch on its API server, where it goes on if its client is stopped: so it is
        submitted (--async), followed by its id, and on a timeout or a cancel canceled there before the machine is
        deleted."""
        with self.phase("launch"):
            if c.get("provider") == "runpod":
                self.runpod_launch(c)
            else:
                self.launch(c)
        with self.phase("setup"):
            self.set_up(c)

    def launch(self, c: dict):
        from .hosts import HostError, probe
        say(f"starting a cloud machine, {c['cluster']} (at most {c['deadline_s'] / 60:.0f} min at ${c['hourly_usd']}/h)")
        # a moment, not a delay: SkyPilot runs that setup detached, perhaps after guard.sh, and must not move the deadline on
        # (a minute early: the machine's clock may be up to a minute behind this one's, which is checked)
        deadline_at = int(time.time() + c["deadline_s"] - (time.monotonic() - self.started)) - 60
        request = None
        try:
            submitted = subprocess.run([c["sky"], "launch", "--async", "-c", c["cluster"], "-y", "--down", "-i", "15",
                                        "--env", f"DEADLINE_EPOCH={deadline_at}", c["task"]], env=self.cloud_env(c),
                                       capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600)
            sys.stdout.write(submitted.stdout + submitted.stderr)
            sys.stdout.flush()
            found = re.search(r"request: ([0-9a-f]{8}-[0-9a-f-]{27,})", re.sub(r"\x1b\[[0-9;]*m", "", submitted.stdout))
            if submitted.returncode or not found:
                raise LaunchFailed(f"sky launch was not submitted, or did not say its request (exit {submitted.returncode})")
            request = found.group(1)
        except FileNotFoundError:
            raise Refused("input", f"SkyPilot is not installed here ({c['sky']}): perception cloud on checks for it")
        finally:
            if request is None:   # interrupted, timed out or not understood: it may still have been submitted
                self.cancel_launches(c)
        logs = subprocess.Popen([c["sky"], "api", "logs", request], env=self.cloud_env(c), stdout=sys.stdout,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        terminating, finished = False, False
        try:
            while True:
                state = self.request_status(c, request)
                if state == "SUCCEEDED":
                    finished = True
                    break
                if state in ("FAILED", "CANCELLED"):
                    finished = True
                    raise LaunchFailed(f"sky launch {state.lower()}")
                if not terminating:
                    terminating = self.terminate_on_shutdown(c, quiet=True)
                if time.monotonic() - self.started > c["launch_s"]:
                    raise LaunchFailed(f"the machine was not ready after {c['launch_s'] / 60:.0f} min (sky launch canceled)")
                time.sleep(POLL_S)
        finally:
            if not finished:   # a timeout or a cancel: stop the launch on the API server, or sky down could miss it
                self.sky_cancel(c, [request])
            logs.terminate()
            logs.wait()
        # again now that it is up: the launch may have replaced the instance it first created (another zone). A machine
        # that would only stop at its deadline (its disk billing on) is not used.
        if not self.terminate_on_shutdown(c):
            raise Refused("internal", f"{c['cluster']} could not be set to terminate on shutdown (aws ec2 modify-instance-attribute): it is not used")
        try:
            caps, root = probe(c["cluster"])
        except HostError as e:
            raise LaunchFailed(f"{c['cluster']} started but could not be probed: {e}")
        self.at(c["cluster"], root)
        self.spec.update(uv=caps.tools.get("uv"), path=sorted({str(Path(t).parent) for t in caps.tools.values() if t}))
        if "cuda" not in caps.accelerators:
            raise Refused("resources", f"{c['cluster']} has no NVIDIA GPU that nvidia-smi can see")
        # setup schedules the shutdown at the deadline as a moment by this laptop's clock: the machine's must agree
        self.check_clock(c["cluster"])
        # its OS shuts it down (and so terminates it) at the deadline, counted from the start of the launch
        minutes = max(1, math.floor((c["deadline_s"] - (time.monotonic() - self.started)) / 60))
        guard = Path(c["guard"]).read_bytes()
        result = self.sh(f"bash -s {minutes}", check=False, data=guard)
        sys.stdout.buffer.write(result.stdout + result.stderr)
        if result.returncode:
            raise GuardRefused(f"guard.sh refused {c['cluster']} (exit {result.returncode}): nothing was sent; see the log")
        self.deadline_set = True

    def check_clock(self, machine: str):
        """The machine's own deadline is a moment by this laptop's clock (a minute early): its clock must agree, to a minute."""
        skew = int(self.sh("date +%s").stdout.decode().strip()) - time.time()
        if abs(skew) > 60:
            raise Refused("internal", f"{machine}'s clock and this machine's differ by {skew:.0f} s: set this machine's "
                                      "clock (its deadline is a moment by it)")

    def set_up(self, c: dict):
        # what the node needs, installed now and waited for (SkyPilot's own setup runs detached: node-setup.sh)
        machine = c.get("cluster") or f"pod {self.pod}"
        say("installing ffmpeg, rsync and uv on the machine")
        try:
            done = self.sh(f"bash -s", check=False, data=Path(c["setup"]).read_bytes())
        except subprocess.TimeoutExpired:
            raise LaunchFailed(f"setting up {machine} took more than 10 min")
        sys.stdout.buffer.write(done.stderr)
        if done.returncode:
            raise LaunchFailed(f"setting up {machine} failed (exit {done.returncode}): see the log")
        uv = done.stdout.decode().strip().splitlines()[-1] if done.stdout.strip() else ""
        self.spec["uv"] = uv or None
        if not uv:
            raise LaunchFailed(f"setting up {machine} did not say where uv is")
        # 5 minutes to bring results back and delete it, and 3 more: setup's shutdown may come that much early (a minute
        # set early, a minute rounding down, a minute of clock skew)
        self.stop_at = self.started + c["deadline_s"] - min(480, c["deadline_s"] / 4)

    def cancel_launches(self, c: dict):
        """Cancel, on SkyPilot's API server, any launch of this attempt's cluster still pending or running: found by its
        cluster's name, which only this attempt uses (a launch whose request id was never read)."""
        try:
            out = subprocess.run([c["sky"], "api", "status", "-v", "-o", "json"], env=self.cloud_env(c), capture_output=True,
                                 text=True, stdin=subprocess.DEVNULL, start_new_session=True, timeout=120)
        except subprocess.TimeoutExpired:
            self.launch_unstopped = True   # not known to be stopped: teardown treats it as able to make a machine still
            say(f"WARNING: SkyPilot did not list its requests in time to cancel a launch of {c['cluster']}; sky down follows")
            return
        try:
            ids = [r["request_id"] for r in json.loads(out.stdout[out.stdout.index("["):])
                   if r.get("name") == "sky.launch" and r.get("cluster_name") == c["cluster"] and r.get("status") in ("PENDING", "RUNNING")]
        except (ValueError, KeyError, TypeError):
            self.launch_unstopped = True
            say(f"WARNING: could not list SkyPilot's requests to cancel a launch of {c['cluster']}; sky down follows")
            return
        if ids:
            self.sky_cancel(c, ids)

    def sky_cancel(self, c: dict, requests: list[str]):
        """Cancel launch requests on SkyPilot's API server, within a time limit: a server that hangs must not keep the
        machine from being deleted (teardown follows whatever happens here)."""
        try:
            code = subprocess.run([c["sky"], "api", "cancel", "-y", *requests], env=self.cloud_env(c), stdout=sys.stdout,
                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                  timeout=c.get("aws_s", 120)).returncode
        except (subprocess.TimeoutExpired, OSError) as e:
            code = str(e)
        if code != 0:
            self.launch_unstopped = True
            say(f"WARNING: could not cancel the launch on SkyPilot's server ({code}); deleting the machine follows")

    def request_status(self, c: dict, request: str) -> str:
        """A request's status on SkyPilot's API server (PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLED), or "" when it
        could not be read this time."""
        try:
            out = subprocess.run([c["sky"], "api", "status", "-a", request, "-o", "json"], env=self.cloud_env(c),
                                 capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
        except subprocess.TimeoutExpired:
            return ""   # not read this time: the launch's own time limit still applies
        try:
            text = out.stdout[out.stdout.index("["):]
            return next((r["status"] for r in json.loads(text) if r["request_id"] == request), "")
        except (ValueError, KeyError, TypeError):
            return ""

    def terminate_on_shutdown(self, c: dict, quiet: bool = False) -> bool:
        """EC2 only stops an on-demand machine on an OS shutdown (its disk keeps billing): make it terminate, so its
        deadline deletes it (as launch.sh does). Whether its instance was found and changed; `quiet` while the launch is
        still creating it."""
        aws = [c["aws"], "--region", c["region"], "--output", "text"]
        ids = subprocess.run([*aws, "ec2", "describe-instances", "--query", "Reservations[].Instances[].InstanceId",
                              "--filters", f"Name=tag:skypilot-cluster-name,Values={c['cluster']},{c['cluster']}-*",
                              "Name=instance-state-name,Values=pending,running"], capture_output=True, text=True)
        found = ids.stdout.split() if ids.returncode == 0 else []
        done_all = bool(found)
        for instance in found:
            done = subprocess.run([*aws, "ec2", "modify-instance-attribute", "--instance-id", instance,
                                   "--instance-initiated-shutdown-behavior", "terminate"], capture_output=True, text=True)
            done_all &= done.returncode == 0
            say(f"{instance} terminates on OS shutdown" if done.returncode == 0 else
                f"WARNING: could not make {instance} terminate on shutdown ({done.stderr.strip()[-200:]}); its deadline only stops it")
        if not found and not quiet:
            say(f"WARNING: no instance found for {c['cluster']} ({ids.stderr.strip()[-200:]}); its deadline only stops it")
        return done_all

    def teardown(self, c: dict):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)   # deleting the machine must not be interrupted
        say(f"deleting {c['cluster']}")
        try:   # a limit too: a SkyPilot server that hangs must not hold the run
            code = subprocess.run([c["sky"], "down", "-y", c["cluster"]], env=self.cloud_env(c), stdout=sys.stdout,
                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                  timeout=c.get("down_s", 900)).returncode   # its own session: a cancel must not stop it
        except subprocess.TimeoutExpired:
            code = "timed out"
        if code == 0 and self.launch_unstopped:
            # sky down says "not found" (exit 0) for a cluster a launch has not made yet: that proves nothing either
            code = "the launch was not stopped"
        if code != 0:   # SkyPilot could not delete it: end the instance through EC2 directly
            say(f"sky down: {code}; terminating {c['cluster']}'s instances through EC2")
            # no instance found proves nothing while a launch that was not stopped may still make one
            code = 0 if self.terminate_instances(c) and not self.launch_unstopped else code
        seconds = time.monotonic() - self.started
        if code == 0:   # its time; never more than its deadline once its OS shutdown was set (it cannot outlive it then)
            self.extra_usd = round(c["hourly_usd"] * (min(seconds, c["deadline_s"]) if self.deadline_set else seconds) / 3600, 4)
        elif self.deadline_set:   # it lives until its deadline at most (guard.sh), and is charged that
            self.extra_usd = round(c["hourly_usd"] * c["deadline_s"] / 3600, 4)
            say(f"WARNING: could not delete {c['cluster']}: run `sky down {c['cluster']}`; otherwise it shuts down at its deadline")
        else:   # no deadline on it either: nothing ends it but someone; no other attempt starts another one
            self.extra_usd = round(c["hourly_usd"] * c["deadline_s"] / 3600, 4)
            message = (f"{c['cluster']} could not be deleted and has no shutdown deadline (or its launch could not be stopped): "
                       "it may still be running, or still be created. "
                       f"Run `sky down {c['cluster']}`, or terminate its instance in the AWS console ({c['region']})")
            say(f"WARNING: {message}")
            self.fail("internal", message)
        say(f"the machine cost about ${self.extra_usd:.2f}" + ("" if code == 0 else " (at least)"))

    def terminate_instances(self, c: dict) -> bool:
        """Terminate every instance of this attempt's cluster through EC2 (when SkyPilot could not): whether none is left.
        An AWS call that fails or does not answer in time counts as not terminated (teardown then says so)."""
        aws = [c["aws"], "--region", c["region"], "--output", "text"]
        limit = c.get("aws_s", 120)
        try:
            ids = subprocess.run([*aws, "ec2", "describe-instances", "--query", "Reservations[].Instances[].InstanceId",
                                  "--filters", f"Name=tag:skypilot-cluster-name,Values={c['cluster']},{c['cluster']}-*",
                                  "Name=instance-state-name,Values=pending,running,stopping,stopped"],
                                 capture_output=True, text=True, start_new_session=True, timeout=limit)
            if ids.returncode:
                say(f"could not list {c['cluster']}'s instances: {ids.stderr.strip()[-200:]}")
                return False
            found = ids.stdout.split()
            if not found:
                return True
            done = subprocess.run([*aws, "ec2", "terminate-instances", "--instance-ids", *found], capture_output=True,
                                  text=True, start_new_session=True, timeout=limit)
        except (subprocess.TimeoutExpired, OSError) as e:
            say(f"could not reach EC2 to terminate {c['cluster']}'s instances ({e})")
            return False
        if done.returncode:
            say(f"could not terminate {', '.join(found)}: {done.stderr.strip()[-200:]}")
            return False
        say(f"terminated {', '.join(found)}")
        return True

    # --- a RunPod pod ---------------------------------------------------------------------------------------------

    def runpod_api(self, c: dict):
        from .runpod import RunPod
        return RunPod(self.runpod_key or "", c["api"], timeout=c.get("api_s", 60))

    def runpod_launch(self, c: dict):
        """Create the pod (any GPU of the list, in any of the US datacenters: whichever RunPod has free), wait for its
        SSH port, probe it like a host and run guard.sh there before anything is sent. The pod's name carries its
        deadline, so a later launch can delete it if this driver never does (runpod_sweep)."""
        from .hosts import HostError, probe, ssh_command
        from .runpod import RunPodError
        if not self.runpod_key:
            raise Refused("input", "no RunPod API key (RUNPOD_API_KEY in ~/.config/hyperstudy/runpod.env): perception cloud on runpod checks for it")
        api = self.runpod_api(c)
        self.runpod_sweep(api)
        deadline_at = int(time.time() + c["deadline_s"] - (time.monotonic() - self.started))
        self.pod_name = f"{c['name']}-d{deadline_at}"
        say(f"starting a RunPod pod, {self.pod_name} (at most {c['deadline_s'] / 60:.0f} min, at most ${c['hourly_usd']}/h)")
        keys = self.local / "ssh"   # a key made for this pod alone: the user's own keys are never offered to it
        keys.mkdir(mode=0o700, exist_ok=True)
        key = keys / "id_ed25519"
        if not key.exists():
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "perception", "-f", str(key)],
                           check=True, capture_output=True, stdin=subprocess.DEVNULL)
        body = {"name": self.pod_name, "imageName": c["image"], "gpuTypeIds": c["gpus"], "gpuTypePriority": "custom",
                "gpuCount": 1, "cloudType": "SECURE", "dataCenterIds": c["datacenters"], "dataCenterPriority": "availability",
                # its own disk only: no volume, so nothing (the inputs above all) outlives the pod
                "containerDiskInGb": c["disk_gb"], "volumeInGb": 0, "ports": ["22/tcp"],
                "env": {"PUBLIC_KEY": (keys / "id_ed25519.pub").read_text().strip()},
                # no allowedCudaVersions: its values stop at 13.0 and match a host's exactly, so it excludes the hosts that
                # report 13.2 (driver 595; checked 2026-10-25). guard.sh checks the driver instead
                "dockerStartCmd": ["bash", "-c", removal_at_boot(deadline_at - 60, c["start"])]}
        self.pod_unknown = True   # until RunPod answers, a pod may exist that only its name can find
        try:
            pod = api.create_pod(body)
        except RunPodError as e:
            # only a 4xx, a create never sent (this laptop offline) and RunPod's answer when none of the GPUs is free (a
            # 500) prove no pod was made; any other failure may come after it was made (a gateway's 502): teardown looks
            # for it by its name
            if e.nothing_done or "no instances currently available" in str(e):
                self.pod_unknown = False
            if e.nothing_done and e.sent:   # RunPod refused it: the same would happen again
                raise Refused("internal", f"RunPod refused to create the pod: {e}")
            raise LaunchFailed(f"RunPod did not start a pod: {e}")
        self.pod, self.pod_unknown = pod["id"], False
        self.pod_usd = float(pod.get("costPerHr") or c["hourly_usd"])
        say(f"pod {self.pod}: ${self.pod_usd}/h")
        if self.pod_usd > c["hourly_usd"]:   # its price rose past the list's: the run's reservation would not cover it
            raise Refused("internal", f"RunPod's price for pod {self.pod} is ${self.pod_usd}/h, above the ${c['hourly_usd']}/h "
                                      "reserved for it: perception's RunPod price list needs updating")
        while True:
            try:
                p = api.pod(self.pod)
            except RunPodError as e:
                if e.status:
                    raise LaunchFailed(f"RunPod could not say how pod {self.pod} is: {e}")
                p = {}   # no answer this time: the launch's time limit still applies
            if p is None:
                raise LaunchFailed(f"pod {self.pod} went away while it was starting")
            if p.get("publicIp") and (p.get("portMappings") or {}).get("22"):
                break
            if time.monotonic() - self.started > c["launch_s"]:
                raise LaunchFailed(f"pod {self.pod} had no SSH address after {c['launch_s'] / 60:.0f} min")
            time.sleep(POLL_S)
        # its host key is new: accepted the first time, in a file of this attempt's own, and held to after that
        # (quoted: ssh reads an unquoted path with a space as two files). None of the user's ssh config applies (-F none:
        # its IdentityFiles would be offered too), and nothing of this machine's is forwarded to it (an agent, ports, X11)
        self.spec["ssh"] = [*ssh_command(), "-F", "none", "-i", str(key), "-p", str(int(p["portMappings"]["22"])), "-o", "IdentitiesOnly=yes",
                            "-o", f'UserKnownHostsFile="{keys / "known_hosts"}"', "-o", "StrictHostKeyChecking=accept-new",
                            "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes", "-o", "ForwardX11=no"]
        dest = f"root@{p['publicIp']}"
        self.dest = dest
        while True:   # its sshd starts a moment after the port is mapped
            try:
                if self.sh("true", check=False, timeout=60).returncode == 0:
                    break
            except (Unreachable, subprocess.TimeoutExpired):
                pass
            if time.monotonic() - self.started > c["launch_s"]:
                raise LaunchFailed(f"pod {self.pod}'s SSH did not answer within {c['launch_s'] / 60:.0f} min")
            time.sleep(POLL_S)
        try:
            caps, root = probe(dest, ssh=self.spec["ssh"])
        except HostError as e:
            raise LaunchFailed(f"pod {self.pod} started but could not be probed: {e}")
        self.at(dest, root)
        self.spec.update(uv=caps.tools.get("uv"), path=sorted({str(Path(t).parent) for t in caps.tools.values() if t}))
        if "cuda" not in caps.accelerators:
            raise Refused("resources", f"pod {self.pod} has no NVIDIA GPU that nvidia-smi can see")
        self.check_clock(f"pod {self.pod}")   # it removes itself at its deadline, a moment by this laptop's clock
        # its removal at the deadline is running (its start command), and it holds no key to the account
        result = self.sh(f"bash -s {int(c['min_driver'])}", check=False, data=Path(c["guard"]).read_bytes())
        sys.stdout.buffer.write(result.stdout + result.stderr)
        if result.returncode == 75:   # its checks could not be made this time: nothing known against the pod
            raise LaunchFailed(f"guard.sh could not check pod {self.pod}: see the log")
        if result.returncode:
            raise GuardRefused(f"guard.sh refused pod {self.pod} (exit {result.returncode}): nothing was sent; see the log")
        self.deadline_set = True

    def runpod_sweep(self, api):
        """Delete any perception pod past its deadline (by its name): one a driver never came back to (a laptop that died)
        whose own removal at the deadline (guard.sh) did not happen either. A pod still within its deadline may be another
        run's, and is left alone."""
        from .runpod import POD_NAME, RunPodError
        try:
            pods = api.pods()
        except RunPodError as e:
            say(f"WARNING: could not list RunPod pods to delete any left past their deadline ({e})")
            return
        for p in pods:
            m = POD_NAME.match(p.get("name") or "")
            if m and int(m.group(1)) + SWEEP_GRACE_S < time.time():
                say(f"deleting {p['name']}: a perception pod past its deadline that was never deleted")
                try:
                    api.delete_pod(p["id"])
                except RunPodError as e:
                    say(f"WARNING: could not delete {p['name']} ({e}): delete it at https://console.runpod.io/pods")

    def runpod_teardown(self, c: dict):
        """Delete the pod through the API and wait until RunPod no longer has it (404), within down_s; charged at its
        price for the time it existed. A create with no clear answer may have made one a moment later: looked for by its
        name for up to unknown_s, and taken as never made (charged nothing) if it does not appear."""
        from .runpod import RunPodError
        signal.signal(signal.SIGTERM, signal.SIG_IGN)   # deleting the pod must not be interrupted
        if self.pod is None and not self.pod_unknown:
            return   # none was made (RunPod refused, or no key): nothing to delete or charge
        api = self.runpod_api(c)
        say(f"deleting pod {self.pod or self.pod_name}")
        start = time.monotonic()
        end = start + c["down_s"]
        gone, problem = False, ""
        while not gone and time.monotonic() < end:
            try:   # deleting is idempotent: asked again until RunPod no longer has the pod or the time is up
                if self.pod is None:
                    found = [p["id"] for p in api.pods() if p.get("name") == self.pod_name]
                    if not found:
                        if time.monotonic() - start < min(c.get("unknown_s", 180), c["down_s"]):
                            time.sleep(2)
                            continue
                        say(f"no pod named {self.pod_name} appeared in {c.get('unknown_s', 180):.0f} s: none was made. If one "
                            "appears at https://console.runpod.io/pods, delete it (the next RunPod run does, after its deadline)")
                        self.pod_unknown = False
                        return
                    self.pod = found[0]   # names are the attempt's own: one at most
                api.delete_pod(self.pod)
                while time.monotonic() < end and api.pod(self.pod) is not None:
                    time.sleep(2)
                gone = api.pod(self.pod) is None
            except RunPodError as e:
                problem = str(e)
                say(f"RunPod: {e}")
                time.sleep(min(5.0, max(0.0, end - time.monotonic())))
        price = self.pod_usd or c["hourly_usd"]
        seconds = time.monotonic() - self.started
        if gone:   # its time; never more than its deadline once its own removal was set (it cannot outlive it then)
            self.extra_usd = round(price * (min(seconds, c["deadline_s"]) if self.deadline_set else seconds) / 3600, 4)
        elif self.deadline_set:   # it removes itself at its deadline at most (guard.sh), and is charged that
            self.extra_usd = round(price * c["deadline_s"] / 3600, 4)
            say(f"WARNING: could not delete pod {self.pod} ({problem or 'still there'}): it removes itself at its deadline; "
                "or delete it at https://console.runpod.io/pods")
        else:   # nothing ends it but someone: no other attempt starts another one
            self.extra_usd = round(price * c["deadline_s"] / 3600, 4)
            message = (f"pod {self.pod or self.pod_name} could not be deleted ({problem or 'still there'}), and its removal at "
                       "its deadline was not confirmed: it may still be running. Delete it at https://console.runpod.io/pods")
            say(f"WARNING: {message}")
            self.fail("internal", message)
        say(f"the pod cost about ${self.extra_usd:.2f}" + ("" if gone else " (at least)"))

    def run(self) -> int:
        cloud = self.spec.get("cloud")
        try:
            if cloud:
                self.provision(cloud)
            # each repeated while the host cannot be reached: they can be (uploads go under the attempt's own names and
            # are published in one step; starting stops this attempt's node if a lost answer hid that it had started)
            with self.phase("code"):
                self.patiently(self.check_root)
                self.patiently(self.send_code)
            with self.phase("inputs"):
                inputs = self.patiently(self.send_inputs)
            with self.phase("prepare"):   # its environment synced and its weights fetched, before it starts (setup.py)
                self.prepare()
            with self.phase("run"):   # from starting the node to its end
                self.patiently(self.start, inputs)
                self.watch()
            with self.phase("fetch"):
                self.fetch()
            return 0
        except Canceled:
            self.stop()
            return 143
        except LaunchFailed as e:
            say(str(e))
            # retried within the policy, charged what the machine cost until it was deleted (not the whole deadline)
            self.fail("transient", str(e))
            return 1
        except GuardRefused as e:
            say(str(e))
            # the machine is not set up as this repository requires (its role, credentials on it): not worth retrying
            self.fail("privacy", str(e))
            return 1
        except Deadline:
            minutes = cloud["deadline_s"] / 60
            say(f"stopping it: the machine's deadline ({minutes:.0f} min) is near")
            self.stop()
            # a failure of its own, not a crash: another attempt would only reach the same deadline, at the same cost
            if not (self.local / "response.json.part").exists():   # never over results already brought back
                self.fail("resources", f"stopped at the cloud machine's deadline ({minutes:.0f} min); a longer video needs a longer deadline")
            return 124
        except Unreachable as e:
            say(f"{self.spec['host']} could not be reached for {self.spec['lost_after_s']:.0f} s ({e}): giving up on this attempt")
            return 75
        except Refused as e:
            say(str(e))
            self.fail(e.code, str(e))
            return 1
        except PrepareFailed as e:
            say(str(e))
            self.fail("internal", str(e))   # not retried (see PrepareFailed); charged what it took
            return 1
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)   # a cancel now must not skip deleting the machine or charging it
            self.discard_incoming()
            if cloud:
                with self.phase("teardown"):
                    if cloud.get("provider") == "runpod":
                        self.runpod_teardown(cloud)
                    else:
                        self.teardown(cloud)
            self.publish_response()

    def discard_incoming(self):
        """This attempt's own uploads that never finished (a cancel, a failure): a partial copy of an input must not stay
        on the host. One that finished was renamed into place, so nothing published is touched."""
        if not self.uploading:
            return   # nothing was sent (a machine the guard refused gets nothing, not even this)
        q = shlex.quote
        try:
            self.sh(f"chmod -R u+w {q(self.root)}/objects/.incoming-{q(self.tag)}-* 2>/dev/null; "
                    f"rm -rf {q(self.root)}/objects/.incoming-{q(self.tag)}-* {q(self.root)}/code/.incoming-{q(self.tag)}", check=False)
        except (Unreachable, subprocess.TimeoutExpired) as e:
            say(f"could not remove this attempt's unfinished uploads on {self.spec['host']} ({e}): perception hosts clean removes them")

    def fail(self, code: str, message: str):
        """A response saying the attempt failed, as a node writes one: the coordinator does not retry it (published as
        the run ends, with what a cloud machine cost)."""
        (self.local / "response.json.part").write_text(json.dumps({"ok": False, "usage": {}, "error": {"code": code, "message": message}}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="perception.remote_driver")
    ap.add_argument("--request", required=True)
    args = ap.parse_args(argv)
    request = Path(args.request)

    def on_term(signum, frame):
        raise Canceled()
    signal.signal(signal.SIGTERM, on_term)
    driver = Driver(request, json.loads((request.parent / "remote.json").read_text()))
    return driver.run()


if __name__ == "__main__":
    sys.exit(main())
