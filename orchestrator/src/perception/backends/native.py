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
from .base import GPU_LEASES, LEASE_FOLDER, OFFLINE, Backend, Capabilities, LeaseRefused, gpu_env, holder, lease_command
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

    def submit(self, manifest: Manifest, runner: Runner, attempt_dir: Path, env: dict[str, str], gpu: int | None = None,
               prepare_secrets: dict[str, str] | None = None) -> dict:
        request = attempt_dir / "request.json"
        env = {**env, **self.project_env(runner)}
        env.update(gpu_env(gpu))   # the card the scheduler gave it, alone (none on a Mac)
        command = self.command(manifest, runner, request)
        if gpu is not None and manifest.resources.gpu != "none":   # and held against other runs here (GPU_LEASES)
            command = [*lease_command(str(self.lease_folder()), gpu, manifest.resources.gpu,
                                      holder(json.loads(request.read_text()))), *command]
        if prepare_secrets is not None and manifest.download_secrets and self.downloads(manifest, runner):
            # its prepare first, with them, then the node (with its lease: no card is held while weights download)
            # without them (prepare_first.py)
            from .. import prepare_first
            from ..plan import prepared_marks
            config = json.loads(request.read_text()).get("config", {})
            sync = self.sync_command(runner)
            # by its path, in isolated mode: the node's environment has none of this interpreter's import path
            command = [sys.executable, "-I", prepare_first.__file__, "--secrets", ",".join(manifest.download_secrets),
                       "--offline", json.dumps(OFFLINE), *(["--sync", json.dumps(sync)] if sync else []),
                       "--prepare", json.dumps(self.prepare_command(manifest, runner, config)),
                       "--marks", json.dumps([str(p) for p in prepared_marks(manifest, runner)]), "--", *command]
            env = {**env, **prepare_secrets}
        log = open(attempt_dir / "log.txt", "ab")
        proc = subprocess.Popen(command, cwd=manifest.directory, env=env,
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)   # its own process group: cancel stops the whole tree
        log.close()
        self._children[proc.pid] = proc
        handle = {"pid": proc.pid, "request": str(request)}
        tmp = attempt_dir / "handle.json.tmp"   # at once, before the coordinator records it: see recorded_handle
        tmp.write_text(json.dumps(handle))
        os.replace(tmp, attempt_dir / "handle.json")
        return handle

    def downloads(self, manifest: Manifest, runner: Runner) -> bool:
        """Whether a weight the runner loads is not in this machine's cache yet (plan.model_cached): its prepare fetches it."""
        from ..plan import fetched, model_cached
        return any(fetched(x) and not model_cached(x, bool(manifest.download_secrets)) for x in manifest.models_for(runner))

    def sync_command(self, runner: Runner) -> list[str] | None:
        """The node's environment synced from its lock, as `uv run` would before its first use (None: `self`)."""
        if self.kind == "self":
            return None
        return [shutil.which("uv") or "uv", "sync", "--project", str(REPO / runner.env), "--frozen", "--no-dev"]

    def prepare_command(self, manifest: Manifest, runner: Runner, config: dict) -> list[str]:
        """The node's prepare script with its settings, in its environment as sync_command left it, as `perception
        setup` runs it (setup.py)."""
        script = str(manifest.directory / manifest.prepare)
        if self.kind == "self":
            return [sys.executable, script, json.dumps(config)]
        return [shutil.which("uv") or "uv", "run", "--project", str(REPO / runner.env), "--frozen", "--no-dev", "--no-sync",
                "python", script, json.dumps(config)]

    def project_env(self, runner: Runner) -> dict[str, str]:
        """What the node's environment needs set: for the package, where uv makes it (in the store)."""
        return env_vars(runner.env) if self.kind == "native" and runner.env else {}

    def lease_folder(self) -> Path:
        """Where this machine's cards are leased (GPU_LEASES, under this user's home): the folder a coordinator elsewhere
        leases them in when it runs nodes here as a host, made the user's own and private (LEASE_FOLDER), or refused."""
        made = subprocess.run(["sh", "-c", LEASE_FOLDER], capture_output=True, text=True)
        if made.returncode:   # the same every time: refused for good (as a host's start refuses it, exit 3: Refused)
            raise LeaseRefused(f"cannot lease a GPU here: {made.stderr.strip()[-300:]}")
        return Path.home() / GPU_LEASES

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
    """Is `pid` still the node process for this request? (pids are reused, so check its command line too.)

    The rule a host's check uses (remote_driver.running_sh): a command line that names the request, or none for the
    moment. Linux shows no command line while a process execs, and a node on a known card is reached through several
    (the GPU lease's perl, then uv, then python). A coordinator that adopted such a node after a restart (not its child,
    so only this check sees it) and polled it mid-exec took it for gone, found no response and ran it again beside the
    one still running. ps names a process with no command line in brackets ("[perl]"), as it does a zombie and a kernel
    thread, which are not running nodes (_shown). A command line naming anything else is a reused pid."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    command = _shown(pid)
    return command is not None and (request in command or command == "")


def _shown(pid: int) -> str | None:
    """`pid`'s command line as ps shows it, "" while it has none for the moment (mid-exec), or None when there is no such
    process, it has ended (a zombie: its node has finished, whether or not anything reaps it), or it is a kernel thread
    (shown in brackets as a process with no command line is: a pid reused by one is never a node; their parent is
    kthreadd, pid 2, or none). -ww: unlimited width (procps truncates columns to 80 characters when not writing to a
    terminal, and a long request path then made a live node look dead)."""
    out = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "ppid=", "-o", "stat=", "-o", "command="],
                         capture_output=True, text=True)
    if out.returncode:   # no such process any more
        return None
    fields = out.stdout.strip().split(None, 2)
    ppid, state, command = (fields + ["", "", ""])[:3]
    if state.startswith("Z"):
        return None
    if command == "" or command.startswith("["):
        return None if ppid in ("0", "2") else ""
    return command
