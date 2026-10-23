"""Container runs: the node's image (pinned by digest) under Docker on this machine, for linux/amd64 + CUDA nodes and
for nodes whose dependencies only install on Linux.

The container may be hostile, and it can write wherever it is mounted, so it sees only `<attempt>/container/`
(mounted at /work: its request, out/ and response.json) and each input read-only at /in/<name>. Everything the host
writes or trusts (request.json, log.txt) stays in the attempt directory outside that mount, and what the host reads
back from it is checked: the response only if it is a regular file, outputs only if they contain no symlinks.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ..manifests import Manifest, Runner
from ..store import Artifact
from .base import Backend, Capabilities
from .probe import probe


def append_log(path: Path, text: str) -> None:
    """Append to a host-side log without following a symlink (O_NOFOLLOW): never write through a link."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o644)
    with os.fdopen(fd, "a") as f:
        f.write(text)


class OciBackend(Backend):
    kind = "oci"

    def __init__(self):
        self._caps: Capabilities | None = None

    def probe_capabilities(self) -> Capabilities:
        if self._caps is None:
            caps = probe()
            # containers run Linux: a Mac's Docker cannot use the Apple GPU, and runs arm64 or emulated amd64
            arch = caps.platform.split("/")[1]
            caps.platform = f"linux/{arch}"
            caps.accelerators = caps.accelerators - {"mps"}
            if "cuda" not in caps.accelerators:
                caps.vram_gb = 0
            self._caps = caps
        return self._caps

    def missing_tools(self, runner: Runner) -> list[str]:
        return [] if self.probe_capabilities().tools.get("docker") else ["docker (with a running daemon)"]

    def identity(self, manifest: Manifest, runner: Runner) -> str:
        return f"oci:{runner.image.split('@sha256:')[1][:16]}"

    def work_dir(self, attempt_dir: Path) -> Path:
        return attempt_dir / "container"

    def stage_inputs(self, attempt_dir: Path, inputs: dict[str, Artifact]) -> dict[str, dict]:
        return {name: {"type": a.type, "sha256": a.hash, "path": f"/in/{name}", "host_path": str(a.path)}
                for name, a in inputs.items()}

    @staticmethod
    def _name(attempt_dir: Path) -> str:
        return f"perception-{attempt_dir.parent.parent.name}-{attempt_dir.parent.name}-{attempt_dir.name}"

    def command(self, runner: Runner, attempt_dir: Path, env: dict[str, str]) -> list[str]:
        """The `docker run` command, after writing the container's own request into its folder."""
        request = json.loads((attempt_dir / "request.json").read_text())
        mounts = []
        for name, entry in request["inputs"].items():
            mounts += ["-v", f"{entry.pop('host_path')}:/in/{name}:ro"]   # host paths never reach the container
        request["out_dir"] = "/work/out"
        work = self.work_dir(attempt_dir)
        work.mkdir()
        (work / "request.container.json").write_text(json.dumps(request, indent=1))
        secret_env = [x for k in request.get("secrets", []) if k in env for x in ("-e", k)]
        gpus = ["--gpus", "all"] if runner.accelerator == "cuda" else []
        return ["docker", "run", "-d", "--name", self._name(attempt_dir), "--network", "bridge", *gpus, *secret_env,
                "-v", f"{work}:/work", *mounts, runner.image, "--request", "/work/request.container.json"]

    def submit(self, manifest: Manifest, runner: Runner, attempt_dir: Path, env: dict[str, str]) -> dict:
        out = subprocess.run(self.command(runner, attempt_dir, env), env=env, capture_output=True, text=True)
        if out.returncode:
            raise RuntimeError(f"docker run failed: {out.stderr.strip()}")
        return {"container": out.stdout.strip(), "name": self._name(attempt_dir)}

    def poll(self, handle: dict) -> str:
        out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", handle["container"]], capture_output=True, text=True)
        return "running" if out.stdout.strip() == "true" else "exited"

    def cancel(self, handle: dict) -> None:
        subprocess.run(["docker", "stop", handle["container"]], capture_output=True)

    def cleanup(self, attempt_dir: Path) -> None:
        name = self._name(attempt_dir)
        logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        append_log(attempt_dir / "log.txt", logs.stdout + logs.stderr)   # outside the container's mount
        subprocess.run(["docker", "rm", name], capture_output=True)
