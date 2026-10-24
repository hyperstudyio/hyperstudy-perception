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
"""

from __future__ import annotations

import argparse
import contextlib
import json
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


class Canceled(Exception):
    pass


class Unreachable(Exception):
    pass


class Refused(Exception):
    """The host answered, and the attempt cannot go on: the same would happen again (an input that changed since the run
    read it, a root that is not the user's). It fails as the node's own error, not a crash to retry."""
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# Publish a folder in one step: rename $1 to $2 unless $2 exists, in which case $1 is dropped (the same content, by its
# hash, arrived first). rename(2) onto a folder that is not empty fails, so a published folder is never replaced or
# nested into: GNU mv -T on Linux, perl's rename on a Mac (BSD mv has no -T and would move $1 into $2).
PUBLISH = r"""publish() {
  mv -T "$1" "$2" 2>/dev/null || perl -e 'rename($ARGV[0], $ARGV[1]) or exit 1' "$1" "$2" 2>/dev/null || { chmod -R u+w "$1"; rm -rf "$1"; [ -e "$2/.complete" ]; }
}"""

# The host's backstop for a node its driver lost (the laptop gone, a cancel that could not reach the host): every
# `every` seconds, while the attempt's folder exists, if no driver has written the heartbeat (alive) for `lost` seconds,
# stop the node's process group (if that pid is still this attempt's node) and remove the folder, outputs and all.
WATCHDOG = r"""w=$1 pid=$2 lost=$3 every=$4
while [ -d "$w" ]; do
  sleep "$every"
  [ -d "$w" ] || exit 0
  last=$(cat "$w/alive" 2>/dev/null || echo 0)
  if [ $(( $(date +%s) - last )) -gt "$lost" ]; then
    if ps -ww -p "$pid" -o command= 2>/dev/null | grep -qF -- "$w/request.json"; then
      kill -s TERM -- "-$pid" 2>/dev/null; sleep 5; kill -s KILL -- "-$pid" 2>/dev/null
    fi
    rm -rf "$w"
    rmdir "${w%/*}" "${w%/*/*}" 2>/dev/null   # the run's folders too, once empty (as fetch does)
    exit 0
  fi
done"""


def say(msg: str):
    print(f"[remote] {msg}", flush=True)


class Driver:
    def __init__(self, request_path: Path, spec: dict):
        self.request_path, self.spec = request_path, spec
        self.local = request_path.parent
        self.request = json.loads(request_path.read_text())
        self.dest, self.root = spec["dest"], spec["root"]
        r = self.request
        self.attempt = f"{self.root}/work/{r['run']}/{r['node']}/{r['attempt']}"
        self.node_runs = f"{self.root}/work/{r['run']}/{r['node']}"
        self.code = f"{self.root}/code/{spec['code']}"
        self.tag = f"{r['run']}-{r['node']}-{r['attempt']}"   # this attempt's own names for what it uploads
        self.pid: int | None = None
        self.unreachable_since: float | None = None

    # --- commands on the host -----------------------------------------------------------------------------------

    def sh(self, script: str, check: bool = True, data: bytes | None = None) -> subprocess.CompletedProcess:
        """Run a POSIX sh script on the host (through sh: the login shell may be zsh or fish). ssh's own failures (exit
        255: no connection) raise Unreachable."""
        out = subprocess.run([*self.spec["ssh"], self.dest, f"sh -c {shlex.quote(script)}"], input=data, capture_output=True, timeout=600)
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
            # would send as a link pointing nowhere there (or somewhere else)
            local = Path(entry["path"]).resolve()
            folder = f"{self.root}/objects/{entry['sha256']}"
            out[name] = {**entry, "path": f"{self.attempt}/in/{name}/{local.name}", "object": f"{folder}/data"}
            if self.sh(f"test -e {q(folder)}/.complete", check=False).returncode == 0:
                continue
            size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file()) if local.is_dir() else local.stat().st_size
            say(f"copying {name} ({size / 1e6:.1f} MB) to {self.spec['host']}")
            incoming = f"{self.root}/objects/.incoming-{self.tag}-{name}"
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
date +%s > {q(self.attempt)}/alive
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
nohup "$@" sh -c {q(WATCHDOG)} watchdog {q(self.attempt)} "$pid" {lost} {every} > /dev/null 2>&1 < /dev/null &
echo $pid
"""
        out = self.sh(script, data=json.dumps(request, indent=1).encode())
        self.pid = int(out.stdout.decode().split()[-1])
        say(f"started on {s['host']} (pid {self.pid})")

    def unreachable(self, e: Exception):
        """Wait out a host that cannot be reached, for up to lost_after_s in all since it was last reached; past that,
        give up (raise)."""
        now = time.monotonic()
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
                    self.sh(f"date +%s > {shlex.quote(self.attempt)}/alive 2>/dev/null", check=False)
                except Exception:
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
            check = (f"date +%s > {q(self.attempt)}/alive 2>/dev/null; "
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
            time.sleep(POLL_S)

    def fetch(self):
        """out/ and response.json, into this attempt's folder. No response there: the node crashed (none is written
        here either, and the coordinator sees a crash). The node has finished: a dropped connection is waited out, and
        the heartbeat kept up, rather than its results lost."""
        q = shlex.quote
        with self.heartbeat():
            # never through a link the node made: rsync would copy what `out/` points at, which publication cannot see
            listing = self.patiently(self.sh, f"cd {q(self.attempt)} 2>/dev/null || exit 0; [ -d out ] && [ ! -L out ] && echo out; "
                                              f"[ -f response.json ] && [ ! -L response.json ] && echo response.json; true").stdout.decode().split()
            if "out" in listing:
                self.patiently(self.rsync, f"{self.dest}:{self.attempt}/out/", f"{self.local}/out/")
            if "response.json" in listing:
                self.patiently(self.rsync, f"{self.dest}:{self.attempt}/response.json", f"{self.local}/response.json.part")
        if "response.json" in listing:
            (self.local / "response.json.part").replace(self.local / "response.json")
        # the run's folders too once empty (another node of the run may still be using them)
        self.patiently(self.sh, f"rm -rf {q(self.attempt)}; rmdir {q(self.node_runs)} {q(self.node_runs.rsplit('/', 1)[0])} 2>/dev/null || true")

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

    def run(self) -> int:
        try:
            # each repeated while the host cannot be reached: they can be (uploads go under the attempt's own names and
            # are published in one step; starting stops this attempt's node if a lost answer hid that it had started)
            self.patiently(self.check_root)
            self.patiently(self.send_code)
            inputs = self.patiently(self.send_inputs)
            self.patiently(self.start, inputs)
            self.watch()
            self.fetch()
            return 0
        except Canceled:
            self.stop()
            return 143
        except Unreachable as e:
            say(f"{self.spec['host']} could not be reached for {self.spec['lost_after_s']:.0f} s ({e}): giving up on this attempt")
            return 75
        except Refused as e:
            say(str(e))
            self.fail(e.code, str(e))
            return 1

    def fail(self, code: str, message: str):
        """A response saying the attempt failed, as a node writes one: the coordinator does not retry it."""
        tmp = self.local / "response.json.tmp"
        tmp.write_text(json.dumps({"ok": False, "usage": {}, "error": {"code": code, "message": message}}))
        tmp.replace(self.local / "response.json")


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
