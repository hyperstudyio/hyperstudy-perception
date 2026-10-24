"""Runs one attempt of a node on another machine, for the coordinator (backends/remote.py starts it):

    python -m perception.remote_driver --request ATTEMPT/request.json      (with ATTEMPT/remote.json beside it)

1. Copies the node's code (by content: once per version) and its inputs (by sha256: once per host) to the host.
2. Writes the request there, with the host's paths, and starts the node in its own session (setsid), detached,
   so a dropped connection does not end it. A previous attempt of the same node still running there is stopped first.
3. Watches it every few seconds, appending its log to this attempt's log.txt. A host that cannot be reached is waited
   for (a laptop changes networks) for up to lost_after_s; after that the attempt ends without a response, which the
   coordinator treats as a crash (retried within the run's policy).
4. Brings out/ and response.json back into this attempt's folder, then removes the attempt's folder on the host.

SIGTERM (the coordinator's cancel) stops the node on the host before the driver exits. Its output (stdout) is the
attempt's log.txt.
"""

from __future__ import annotations

import argparse
import json
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

POLL_S = 3.0


class Canceled(Exception):
    pass


class Unreachable(Exception):
    pass


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
        self.pid: int | None = None

    # --- commands on the host -----------------------------------------------------------------------------------

    def sh(self, script: str, check: bool = True, data: bytes | None = None) -> subprocess.CompletedProcess:
        """Run a POSIX sh script on the host (through sh: the login shell may be zsh or fish). ssh's own failures (exit
        255: no connection) raise Unreachable."""
        out = subprocess.run([*self.spec["ssh"], self.dest, f"sh -c {shlex.quote(script)}"], input=data, capture_output=True, timeout=600)
        if out.returncode == 255:
            raise Unreachable(out.stderr.decode(errors="replace").strip()[-300:])
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

    def send_code(self):
        q = shlex.quote
        if self.sh(f"test -e {q(self.code)}/.complete", check=False).returncode == 0:
            return
        say(f"copying the node's code to {self.spec['host']}")
        self.sh(f"mkdir -p {q(self.code)}")
        listing = self.local / "code-files.txt"
        listing.write_text("".join(f + "\n" for f in self.spec["files"]))
        from .backends.native import REPO
        self.rsync("--files-from", str(listing), f"{REPO}/", f"{self.dest}:{self.code}/")
        self.sh(f"touch {q(self.code)}/.complete")

    def send_inputs(self) -> dict[str, dict]:
        """Each input under objects/<sha256>/<its name> on the host (uploaded under a fixed name, then renamed: a file
        name can hold anything a shell would split)."""
        q = shlex.quote
        out = {}
        for name, entry in self.request["inputs"].items():
            local = Path(entry["path"])
            folder = f"{self.root}/objects/{entry['sha256']}"
            there = f"{folder}/{local.name}"
            out[name] = {**entry, "path": there}
            if self.sh(f"test -e {q(folder)}/.complete", check=False).returncode == 0:
                continue
            size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file()) if local.is_dir() else local.stat().st_size
            say(f"copying {name} ({size / 1e6:.1f} MB) to {self.spec['host']}")
            self.sh(f"rm -rf {q(folder)} && mkdir -p {q(folder)}")
            if local.is_dir():
                self.rsync(f"{local}/", f"{self.dest}:{folder}/upload/")
            else:
                self.rsync(str(local), f"{self.dest}:{folder}/upload")
            self.sh(f"mv {q(folder)}/upload {q(there)} && touch {q(folder)}/.complete")
        return out

    def start(self, inputs: dict[str, dict]):
        q, s = shlex.quote, self.spec
        request = {**self.request, "inputs": inputs, "out_dir": f"{self.attempt}/out"}
        if s["runner_kind"] == "self":
            node = f"{q(s['python'])} {q(s['entry'])}"
        else:
            node = f"{q(s['uv'])} run --project {q(self.code + '/' + s['env'])} --frozen --no-dev python {q(s['entry'])}"
        req = f"{self.attempt}/request.json"
        path = ":".join(s["path"])
        # an earlier attempt of this node that a lost connection left running would hold the GPU: stop it first
        script = f"""
set -e
for f in {q(self.node_runs)}/*/pid; do
  [ -f "$f" ] || continue
  old=$(cat "$f")
  # only if that pid is still that attempt's node (pids are reused on a shared machine)
  if ps -ww -p "$old" -o command= 2>/dev/null | grep -qF -- "${{f%/pid}}/request.json"; then kill -s TERM -- "-$old" 2>/dev/null || true; fi
done
mkdir -p {q(self.attempt)}/out
cat > {q(req)}
cd {q(self.code + '/' + s['node_dir'])}
# its own session, so its whole process tree can be stopped (kill -s TERM -- -PID) and a closed connection does not
# reach it: setsid on Linux, perl's on a Mac (no setsid command there). Not a group leader in the background, so
# neither forks: $! stays the node's pid.
if command -v setsid >/dev/null 2>&1; then set -- setsid; else set -- perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV or die'; fi
PATH={q(path)}${{PATH:+:$PATH}} nohup "$@" {node} --request {q(req)} > {q(self.attempt)}/log.txt 2>&1 < /dev/null &
echo $! > {q(self.attempt)}/pid
echo $!
"""
        out = self.sh(script, data=json.dumps(request, indent=1).encode())
        self.pid = int(out.stdout.decode().split()[-1])
        say(f"started on {s['host']} (pid {self.pid})")

    def watch(self):
        q = shlex.quote
        req, log = f"{self.attempt}/request.json", f"{self.attempt}/log.txt"
        offset, unreachable_since = 0, None
        while True:
            check = (f"if kill -0 {self.pid} 2>/dev/null && ps -ww -p {self.pid} -o command= | grep -qF -- {q(req)}; "
                     f"then echo running; else echo exited; fi; tail -c +{offset + 1} {q(log)} 2>/dev/null || true")
            try:
                out = self.sh(check).stdout
                unreachable_since = None
            except Unreachable as e:
                now = time.monotonic()
                if unreachable_since is None:
                    unreachable_since = now
                    say(f"{self.spec['host']} cannot be reached ({e}); waiting up to {self.spec['lost_after_s']:.0f} s")
                elif now - unreachable_since > self.spec["lost_after_s"]:
                    raise
                time.sleep(POLL_S)
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
        here either, and the coordinator sees a crash)."""
        q = shlex.quote
        listing = self.sh(f"cd {q(self.attempt)} && ls -A").stdout.decode().split()
        if "out" in listing:
            self.rsync(f"{self.dest}:{self.attempt}/out/", f"{self.local}/out/")
        if "response.json" in listing:
            self.rsync(f"{self.dest}:{self.attempt}/response.json", f"{self.local}/response.json.part")
            (self.local / "response.json.part").replace(self.local / "response.json")
        # the run's folders too once empty (another node of the run may still be using them)
        self.sh(f"rm -rf {q(self.attempt)}; rmdir {q(self.node_runs)} {q(self.node_runs.rsplit('/', 1)[0])} 2>/dev/null || true")

    def stop(self):
        """Stop this attempt's node on the host, if it started (by its pid file: the driver may have been canceled
        between starting it and reading its pid)."""
        signal.signal(signal.SIGTERM, signal.SIG_IGN)   # a second cancel must not interrupt the stopping
        q = shlex.quote
        req, pidfile = f"{self.attempt}/request.json", f"{self.attempt}/pid"
        say(f"stopping it on {self.spec['host']}")
        try:
            self.sh(f'[ -f {q(pidfile)} ] || exit 0; p=$(cat {q(pidfile)}); '
                    f'if ps -ww -p "$p" -o command= | grep -qF -- {q(req)}; then kill -s TERM -- "-$p"; fi', check=False)
        except (Unreachable, subprocess.TimeoutExpired) as e:
            say(f"could not reach {self.spec['host']} to stop it ({e}); it ends when the next attempt of this node starts there")

    def run(self) -> int:
        try:
            self.send_code()
            inputs = self.send_inputs()
            self.start(inputs)
            self.watch()
            self.fetch()
            return 0
        except Canceled:
            self.stop()
            return 143
        except Unreachable as e:
            say(f"{self.spec['host']} could not be reached for {self.spec['lost_after_s']:.0f} s ({e}): giving up on this attempt")
            return 75


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
