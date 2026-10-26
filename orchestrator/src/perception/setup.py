"""`perception setup`: everything a node needs on this machine, ready before a run rather than fetched in the middle of
one. For each node that would run here: its uv environment synced from its lock (`uv sync --frozen`, what `uv run`
would do on first use), then its `prepare` script run in that environment (its weights downloaded into the cache it
reads). Nodes call no tools of the machine's (they read video with PyAV). Each step is timed and reported; a failure is said, with the
end of its output, and the rest still runs. Doing it again finds everything in place and takes seconds.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .backends.base import child_env
from .backends.native import REPO
from .manifests import Manifest, Runner


@dataclass
class Step:
    node: str          # the node's ref, or "" for the machine's tools
    what: str          # environment | models | tools
    ok: bool
    seconds: float
    message: str = ""


@dataclass
class Setup:
    steps: list[Step] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[bool, float, str]:
    """Run a step in the node's own environment; what it said last, or the end of what went wrong."""
    start = time.monotonic()
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=env)
    said = (out.stdout.strip().splitlines() or [""])[-1] if out.returncode == 0 else \
        " / ".join((out.stdout + out.stderr).strip().splitlines()[-3:])
    return out.returncode == 0, round(time.monotonic() - start, 1), said


def setup_node(m: Manifest, runner: Runner, uv: str | None = None, config: dict | None = None,
               secrets: dict[str, str] | None = None) -> list[Step]:
    """Sync the node's environment (native runners) and run its prepare script (its weights) with its settings: a graph's
    for the node when there is one (a setting can choose the model), else its defaults. With the environment a run gives
    the node (child_env): none of this interpreter's settings, and no credential but the secrets it declares (a gated
    model's token)."""
    uv = uv or shutil.which("uv") or "uv"
    env = child_env(m.secrets, secrets or {})
    steps: list[Step] = []
    if runner.kind == "native" and runner.env:
        ok, seconds, tail = _run([uv, "sync", "--project", str(REPO / runner.env), "--frozen", "--no-dev"], REPO, env)
        steps.append(Step(m.ref, "environment", ok, seconds, "" if ok else tail))
        if not ok:
            return steps   # its prepare would only fail the same way
    if m.prepare:
        python = ([uv, "run", "--project", str(REPO / runner.env), "--frozen", "--no-dev", "python"]
                  if runner.kind == "native" and runner.env else [sys.executable])
        ok, seconds, tail = _run([*python, str(m.directory / m.prepare), json.dumps(m.defaults() if config is None else config)], m.directory, env)
        steps.append(Step(m.ref, "models", ok, seconds, tail))
    return steps

