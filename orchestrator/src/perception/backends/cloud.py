"""Cloud runs: a node on a cloud GPU machine started for it and deleted after it (`perception cloud on`).

The machine is an SSH host for one attempt: the same driver (remote_driver.py) runs the node there, after first
starting the machine through SkyPilot with this repository's project config (cloud/skypilot/sky.yaml: no laptop
credentials on the machine, and only the perception-node role, which can end perception-* machines and nothing else).
Before anything is sent the driver checks the machine (guard.sh: that role, no AWS credentials, and a shutdown
deadline), and whatever happens it deletes the machine (`sky down`). Backstops if the laptop never comes back: the
machine's shutdown deadline (an OS shutdown terminates it), and SkyPilot's autodown once idle (no jobs, no SSH).

A cloud place costs money: reserved before it starts at the hourly price for the whole deadline (the most it can run),
and charged for the time the machine existed. The deadline is 20 minutes (starting the machine, installing the node's
environment) plus the video's length, at least 30 minutes: a GPU node slower than real time on an L4 is not expected
(the face node ran at ~4x). Only CUDA runners are offered: the machine is a GPU machine.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

from ..hosts import Host
from ..manifests import Manifest, Runner
from ..store import Artifact
from .base import Capabilities, Estimate
from .native import REPO
from .remote import SshBackend

SKYPILOT = REPO / "cloud" / "skypilot"


class CloudBackend(SshBackend):
    """One machine type (an L4 with 8 vCPUs: the face node's fastest measured setup, perception #4) in one region."""
    task = SKYPILOT / "node-gpu.yaml"
    project_config = SKYPILOT / "sky.yaml"
    guard = SKYPILOT / "guard.sh"
    setup = SKYPILOT / "node-setup.sh"
    region = "us-east-1"
    machine_title = "AWS g6.2xlarge, NVIDIA L4"
    # on-demand, us-east-1, from SkyPilot's AWS catalogue on 2026-10-23 (spot had no L4 capacity there then)
    hourly_usd = 0.9776
    # what placement checks runners against; the driver probes the real machine (its tools come from node-gpu.yaml's setup)
    caps = Capabilities(platform="linux/amd64", cpu_cores=8, memory_gb=32, accelerators={"cpu", "cuda"}, vram_gb=22)

    def __init__(self, deadline_s: float | None = None, runner_kind: str = "native", python: str | None = None,
                 launch_s: float = 15 * 60, down_s: float = 15 * 60, aws_s: float = 120, **paths):
        """`deadline_s` fixes the deadline whatever the video (tests); by default it follows the video's length.
        `launch_s`: how long starting the machine (and installing ffmpeg and uv on it) may take: within the deadline's
        20 minutes for starting, which also cover installing the node's environment."""
        self.launch_s, self.down_s, self.aws_s = launch_s, down_s, aws_s   # how long sky down, and an EC2 call, may take
        host = Host("cloud", "(started for each node)", "", self.caps, 0)
        super().__init__(host, runner_kind=runner_kind, python=python)
        self.kind = "cloud"
        self.title = "Cloud GPU (AWS L4)"   # short: it names the place in menus (`cloud status` gives the machine)
        self.deadline_s = deadline_s
        for name, value in paths.items():   # tests: stand-in task, guard
            setattr(self, name, value)

    @property
    def machine(self) -> str | None:
        return None   # a machine of its own for every node

    def runners_for(self, manifest: Manifest) -> tuple[list[Runner], str]:
        runners, why = super().runners_for(manifest)
        gpu = [r for r in runners if r.accelerator == "cuda"]
        return gpu, "" if gpu else why or "it has no CUDA runner (a cloud machine is started only for GPU work)"

    def missing_tools(self, runner: Runner) -> list[str]:
        return []   # the machine's setup installs them

    def deadline(self, video_s: float | None) -> float:
        if self.deadline_s is not None:
            return self.deadline_s
        if video_s is None:
            return 60 * 60
        # the plan (ffprobe) and the run (decode's media.v1, also ffprobe's format duration) read the same length, so they
        # reserve the same; whole 5 minutes keep the price a round figure
        return max(30 * 60, math.ceil((20 * 60 + video_s) / 300) * 300)

    def estimate(self, manifest: Manifest, runner: Runner, inputs: dict[str, Artifact], video_s: float | None = None) -> Estimate:
        # the most it can cost: the machine runs at most until its deadline
        video_s = video_s or video_seconds((a.type, a.path) for a in inputs.values())
        return Estimate(round(self.hourly_usd * self.deadline(video_s) / 3600, 4))

    def driver_spec(self, attempt_dir: Path) -> dict:
        request = json.loads((attempt_dir / "request.json").read_text())
        return {"host": "the cloud machine", "cloud": {
            "cluster": cluster_name(request["run"], request["node"], request["attempt"]),
            "task": str(self.task), "project_config": str(self.project_config), "guard": str(self.guard), "setup": str(self.setup),
            "region": self.region, "hourly_usd": self.hourly_usd, "launch_s": self.launch_s, "down_s": self.down_s, "aws_s": self.aws_s,
            "deadline_s": self.deadline(video_seconds((e["type"], Path(e["path"])) for e in request["inputs"].values())),
            "sky": os.environ.get("PERCEPTION_SKY") or shutil.which("sky") or "sky",
            "aws": os.environ.get("PERCEPTION_AWS") or shutil.which("aws") or "aws"}}

    def submit(self, manifest: Manifest, runner: Runner, attempt_dir: Path, env: dict[str, str]) -> dict:
        # SkyPilot and the AWS CLI need the user's cloud credentials, which child_env keeps from nodes: the driver is
        # this package's code, and passes them to no node (the node runs on the machine, with sshd's environment)
        env = {**env, **{k: v for k, v in os.environ.items() if k.startswith(("AWS_", "SKYPILOT_"))}}
        return super().submit(manifest, runner, attempt_dir, env)


def video_seconds(inputs) -> float | None:
    """The video's length, from a node's inputs as (type, path): its media.v1 (decode's), else the video itself."""
    inputs = list(inputs)
    for type_id, path in inputs:
        if type_id == "media.v1":
            return float(json.loads(Path(path).read_text())["duration"])
    for type_id, path in inputs:
        if type_id == "video.v1":
            from ..estimates import probe_seconds
            return probe_seconds(Path(path))
    return None


def cluster_name(run_id: str, node: str, attempt: int) -> str:
    """perception-<run>-<node>-<hash>-<attempt>: the perception-node role can end only perception-* machines, and
    SkyPilot names are lower-case letters, digits and hyphens. The hash of the whole node id keeps two nodes whose
    shortened names agree (face_track_a, face-track-a) on machines of their own."""
    short = re.sub(r"[^a-z0-9-]", "-", node.lower()).strip("-")[:10] or "node"
    return f"perception-{run_id.rsplit('-', 1)[-1]}-{short}-{hashlib.sha256(node.encode()).hexdigest()[:6]}-{attempt}"


class Cloud:
    """cloud.json in the store's home: whether cloud places are offered (`perception cloud on`)."""

    def __init__(self, home: Path):
        self.path = home / "cloud.json"

    @property
    def enabled(self) -> bool:
        return self.path.exists() and json.loads(self.path.read_text()).get("enabled", False)

    def set(self, enabled: bool):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"enabled": enabled}))
