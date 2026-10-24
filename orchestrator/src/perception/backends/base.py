"""What every execution backend provides. The graph never names a provider; placement picks a backend whose runner
fits the node, and the backend turns the file protocol (request.json in, out/ + response.json back) into a process,
a remote job or an API call.

    probe_capabilities()   what this backend can run here (platform, accelerators, memory, tools)
    runner_for(manifest)   the manifest runner this backend would use, or why none fits
    identity(manifest, r)  what makes this runner's results this runner's (env lock, code, SDK): cache key
    estimate(...)          cost and time, before anything is reserved
    stage_inputs(...)      make the inputs readable where the node runs; returns the request's input entries
    submit(...)            start it; returns a JSON-able handle (kept in the run state, so a restarted coordinator
                           can find the job again)
    poll(handle)           running | exited
    cancel(handle)
    collect(attempt_dir)   the node's response.json, or None if it never wrote one (a crash)
    cleanup(attempt_dir)
"""

from __future__ import annotations

import json
import os
import platform
import re
import stat
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..manifests import Manifest, Runner
from ..store import Artifact


@dataclass
class Capabilities:
    platform: str                      # darwin/arm64, linux/amd64, windows/amd64...
    cpu_cores: int
    memory_gb: float
    accelerators: set[str]             # cpu, mps, cuda
    vram_gb: float = 0                 # CUDA: total; Apple GPU: unified memory (memory_gb)
    tools: dict[str, str | None] = field(default_factory=dict)   # uv, ffmpeg, ffprobe -> path or None


@dataclass
class Estimate:
    cost_usd: float                    # 0 for local runs; a range's upper end for billable ones
    seconds: float | None = None       # None: not known before a first run


def this_platform() -> str:
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(machine, machine)
    system = {"Darwin": "darwin", "Linux": "linux", "Windows": "windows"}.get(platform.system(), platform.system().lower())
    return f"{system}/{arch}"


class Backend(ABC):
    kind: str

    @abstractmethod
    def probe_capabilities(self) -> Capabilities: ...

    def runner_for(self, manifest: Manifest) -> tuple[Runner | None, str]:
        """The first of the manifest's runners this backend can use here, preferring an accelerated one."""
        caps = self.probe_capabilities()
        reasons = []
        mine = [r for r in manifest.runners if r.kind == self.kind]
        if not mine:
            return None, f"no {self.kind} runner"
        for r in sorted(mine, key=lambda r: r.accelerator == "cpu"):
            if caps.platform not in r.platforms:
                reasons.append(f"{r.accelerator} runner is for {', '.join(r.platforms)}, not {caps.platform}")
            elif r.accelerator not in caps.accelerators:
                reasons.append(f"{r.accelerator} runner needs {r.accelerator}, not available here")
            elif r.minimum_vram_gb > caps.vram_gb:
                reasons.append(f"{r.accelerator} runner needs {r.minimum_vram_gb} GB VRAM, {caps.vram_gb:.0f} here")
            else:
                missing = self.missing_tools(r)
                if missing:
                    reasons.append(f"{', '.join(missing)} not installed")
                else:
                    return r, ""
        return None, "; ".join(reasons)

    def missing_tools(self, runner: Runner) -> list[str]:
        return []

    @abstractmethod
    def identity(self, manifest: Manifest, runner: Runner) -> str: ...

    def estimate(self, manifest: Manifest, runner: Runner, inputs: dict[str, Artifact]) -> Estimate:
        return Estimate(0.0)

    @abstractmethod
    def stage_inputs(self, attempt_dir: Path, inputs: dict[str, Artifact]) -> dict[str, dict]: ...

    @abstractmethod
    def submit(self, manifest: Manifest, runner: Runner, attempt_dir: Path, env: dict[str, str]) -> dict: ...

    @abstractmethod
    def poll(self, handle: dict) -> str: ...

    @abstractmethod
    def cancel(self, handle: dict) -> None: ...

    def work_dir(self, attempt_dir: Path) -> Path:
        """Where the node itself reads and writes (its request, out/, response.json)."""
        return attempt_dir

    def output_dir(self, attempt_dir: Path) -> Path:
        return self.work_dir(attempt_dir) / "out"

    def collect(self, attempt_dir: Path) -> dict | None:
        """The node's response, read only if it is a regular file: the node controls this directory, and a symlink
        here could point anywhere."""
        path = self.work_dir(attempt_dir) / "response.json"
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(st.st_mode):
            return None
        return json.loads(path.read_text())

    def cleanup(self, attempt_dir: Path) -> None:
        return None

    def recorded_handle(self, attempt_dir: Path) -> dict | None:
        """The handle `submit` recorded in the attempt's folder as soon as the node started, if it did: a coordinator that
        died before recording the task running leaves that node to be adopted, not started a second time."""
        return None


# --- the child's environment: no credentials it did not ask for -------------------------------------------------

SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)
# the coordinator's own interpreter settings: a node runs in its own environment, never the coordinator's
INTERPRETER_VARS = {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "CONDA_PREFIX"}


def child_env(requested: list[str], secrets: dict[str, str]) -> dict[str, str]:
    """The parent's environment without anything that looks like a credential, plus the secrets the node's manifest
    names. A node never sees a key it did not declare."""
    env = {k: v for k, v in os.environ.items() if not SECRET_NAME.search(k) and k not in INTERPRETER_VARS}
    for name in requested:
        if name in secrets:
            env[name] = secrets[name]
    return env
