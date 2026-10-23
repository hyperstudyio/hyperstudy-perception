"""Container runs: the node's image (pinned by digest) under Docker on this machine, for linux/amd64 + CUDA nodes and
for nodes whose dependencies only install on Linux. The attempt directory is mounted at /work and every input
read-only at /in/<name>; the request the container sees uses those paths.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..manifests import Manifest, Runner
from ..store import Artifact
from .base import Backend, Capabilities
from .probe import probe


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

    def stage_inputs(self, attempt_dir: Path, inputs: dict[str, Artifact]) -> dict[str, dict]:
        return {name: {"type": a.type, "sha256": a.hash, "path": f"/in/{name}", "host_path": str(a.path)}
                for name, a in inputs.items()}

    def submit(self, manifest: Manifest, runner: Runner, attempt_dir: Path, env: dict[str, str]) -> dict:
        request = json.loads((attempt_dir / "request.json").read_text())
        mounts = []
        for name, entry in request["inputs"].items():
            mounts += ["-v", f"{entry.pop('host_path')}:/in/{name}:ro"]
        request["out_dir"] = "/work/out"
        (attempt_dir / "request.container.json").write_text(json.dumps(request, indent=1))
        name = f"perception-{attempt_dir.parent.parent.name}-{attempt_dir.parent.name}-{attempt_dir.name}"
        secret_env = [x for k in request.get("secrets", []) if k in env for x in ("-e", k)]
        gpus = ["--gpus", "all"] if runner.accelerator == "cuda" else []
        cmd = ["docker", "run", "-d", "--name", name, "--network", "bridge", *gpus, *secret_env,
               "-v", f"{attempt_dir}:/work", *mounts, runner.image, "--request", "/work/request.container.json"]
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if out.returncode:
            raise RuntimeError(f"docker run failed: {out.stderr.strip()}")
        return {"container": out.stdout.strip(), "name": name}

    def poll(self, handle: dict) -> str:
        out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", handle["container"]], capture_output=True, text=True)
        return "running" if out.stdout.strip() == "true" else "exited"

    def cancel(self, handle: dict) -> None:
        subprocess.run(["docker", "stop", handle["container"]], capture_output=True)

    def cleanup(self, attempt_dir: Path) -> None:
        name = f"perception-{attempt_dir.parent.parent.name}-{attempt_dir.parent.name}-{attempt_dir.name}"
        logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        with open(attempt_dir / "log.txt", "a") as f:
            f.write(logs.stdout + logs.stderr)
        subprocess.run(["docker", "rm", name], capture_output=True)
