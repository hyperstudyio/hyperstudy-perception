"""Native runs: the node's script in its uv environment on this machine (kind `native`), or in the coordinator's own
interpreter (kind `self`, for tests and trivial nodes). This is the backend that reaches the Apple GPU: containers on
macOS cannot.

Each node runs in its own process group with its output going to the attempt's log.txt. The handle records the pid
and the request path, so a coordinator that restarts can tell whether an attempt is still running (and wait for it)
instead of starting a second copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from ..manifests import Manifest, Runner
from ..store import Artifact
from .base import Backend, Capabilities
from .probe import probe

REPO = Path(__file__).resolve().parents[4]


def sdk_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((REPO / "sdk" / "src").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


class NativeBackend(Backend):
    kind = "native"

    def __init__(self):
        self._children: dict[int, subprocess.Popen] = {}
        self._caps: Capabilities | None = None

    def probe_capabilities(self) -> Capabilities:
        if self._caps is None:
            self._caps = probe()
        return self._caps

    def missing_tools(self, runner: Runner) -> list[str]:
        return [] if self.probe_capabilities().tools.get("uv") else ["uv"]

    def identity(self, manifest: Manifest, runner: Runner) -> str:
        lock = REPO / runner.env / "uv.lock" if runner.env else None
        h = hashlib.sha256(manifest.code_hash().encode())
        h.update(sdk_hash().encode())   # the node SDK shapes what a node writes; it is installed editable, outside the lock
        h.update(lock.read_bytes() if lock and lock.exists() else b"no lock")
        h.update(runner.accelerator.encode())
        return f"native:{h.hexdigest()[:16]}"

    def stage_inputs(self, attempt_dir: Path, inputs: dict[str, Artifact]) -> dict[str, dict]:
        # same machine: the node reads published objects in place (they are read-only)
        return {name: {"type": a.type, "sha256": a.hash, "path": str(a.path)} for name, a in inputs.items()}

    def command(self, manifest: Manifest, runner: Runner, request: Path) -> list[str]:
        entry = str(manifest.directory / runner.entry)
        if self.kind == "self":
            return [sys.executable, entry, "--request", str(request)]
        if not runner.env:
            raise ValueError(f"{manifest.ref}: a native runner needs env (a uv project)")
        return [shutil.which("uv") or "uv", "run", "--project", str(REPO / runner.env), "--frozen", "--no-dev",
                "python", entry, "--request", str(request)]

    def submit(self, manifest: Manifest, runner: Runner, attempt_dir: Path, env: dict[str, str]) -> dict:
        request = attempt_dir / "request.json"
        log = open(attempt_dir / "log.txt", "ab")
        proc = subprocess.Popen(self.command(manifest, runner, request), cwd=manifest.directory, env=env,
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)   # its own process group: cancel stops the whole tree
        log.close()
        self._children[proc.pid] = proc
        handle = {"pid": proc.pid, "request": str(request)}
        tmp = attempt_dir / "handle.json.tmp"   # at once, before the coordinator records it: see recorded_handle
        tmp.write_text(json.dumps(handle))
        os.replace(tmp, attempt_dir / "handle.json")
        return handle

    def recorded_handle(self, attempt_dir: Path) -> dict | None:
        path = attempt_dir / "handle.json"
        if not path.is_file() or path.is_symlink():
            return None
        return json.loads(path.read_text())

    def poll(self, handle: dict) -> str:
        pid = handle["pid"]
        proc = self._children.get(pid)
        if proc is not None:   # our own child: poll() also reaps it
            return "running" if proc.poll() is None else "exited"
        return "running" if _alive(pid, handle["request"]) else "exited"   # an attempt a previous coordinator started

    def cancel(self, handle: dict) -> None:
        pid = handle["pid"]
        if self.poll(handle) == "running":
            try:
                os.killpg(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass   # already gone, or exiting: macOS reports EPERM for a group whose leader is a zombie


class SelfBackend(NativeBackend):
    kind = "self"

    def missing_tools(self, runner: Runner) -> list[str]:
        return []

    def identity(self, manifest: Manifest, runner: Runner) -> str:
        return f"self:{hashlib.sha256((manifest.code_hash() + sdk_hash()).encode()).hexdigest()[:16]}"


def _alive(pid: int, request: str) -> bool:
    """Is `pid` still the node process for this request? (pids are reused, so check its command line too.)"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    # -ww: unlimited width (procps truncates columns to 80 characters when not writing to a terminal, and a long request
    # path then made a live node look dead)
    out = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    return request in out.stdout
