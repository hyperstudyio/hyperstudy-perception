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

from ..release import REPO, env_vars


def sdk_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((REPO / "sdk" / "src").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


def unlisted_code_hash(manifest: Manifest, runner: Runner) -> str:
    """The code of a node git cannot list (outside the repository, or a checkout without git), for its identity
    (Backend.identity): every file in its directory and, as code_files takes, in the folder of a script it shares with
    another node (what that script imports beside it), its uv project's lock and the node SDK's sources. Only this
    machine runs such a node, so this never has to agree with what another machine is sent."""
    h = hashlib.sha256(manifest.code_hash().encode())
    shared = (manifest.directory / runner.entry).resolve().parent if runner.entry else None
    if shared and shared != manifest.directory.resolve() and shared.is_dir():
        for p in sorted(shared.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(shared)).encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    h.update(sdk_hash().encode())
    lock = REPO / runner.env / "uv.lock" if runner.env else None
    h.update(lock.read_bytes() if lock and lock.exists() else b"no lock")
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
        env = {**env, **(env_vars(runner.env) if self.kind == "native" and runner.env else {})}   # the package's: in the store
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
        """Trusted only as a handle for this attempt: the node can write in its folder, so a handle naming another request,
        or a pid that is not one, steers no kill or adoption."""
        path = attempt_dir / "handle.json"
        if not path.is_file() or path.is_symlink():
            return None
        try:
            h = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        # a pid in the range any OS gives (Linux's PID_MAX_LIMIT is 2^22): larger ones would overflow os.kill
        ok = (isinstance(h, dict) and set(h) == {"pid", "request"} and type(h["pid"]) is int and 1 < h["pid"] <= 2 ** 22
              and h["request"] == str(attempt_dir / "request.json"))
        return h if ok else None

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
