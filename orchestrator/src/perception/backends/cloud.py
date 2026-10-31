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
        # the plan (estimates.probe_seconds) and the run (decode's media.v1) read the same length, the container's
        # duration through PyAV, so they reserve the same; whole 5 minutes keep the price a round figure
        return max(30 * 60, math.ceil((20 * 60 + video_s) / 300) * 300)

    def estimate(self, manifest: Manifest, runner: Runner, inputs: dict[str, Artifact], video_s: float | None = None) -> Estimate:
        # the most it can cost: the machine runs at most until its deadline
        video_s = video_s or video_seconds((a.type, a.path) for a in inputs.values())
        return Estimate(round(self.hourly_usd * self.deadline(video_s) / 3600, 4))

    def download_refusal(self, manifest: Manifest) -> str:
        # a machine started for one attempt: whether, and to which kinds of machine, a download secret may go is phase
        # 10's (RUN_ANYWHERE_PLAN.md: Secure Cloud pods by default, a Community one only with its own yes)
        return f"its weights are downloaded with {', '.join(manifest.download_secrets)}, which is not sent to cloud machines yet"

    def driver_spec(self, attempt_dir: Path) -> dict:
        request = json.loads((attempt_dir / "request.json").read_text())
        return {"host": "the cloud machine", "cloud": {
            "cluster": cluster_name(request["run"], request["node"], request["attempt"]),
            "task": str(self.task), "project_config": str(self.project_config), "guard": str(self.guard), "setup": str(self.setup),
            "region": self.region, "hourly_usd": self.hourly_usd, "launch_s": self.launch_s, "down_s": self.down_s, "aws_s": self.aws_s,
            "deadline_s": self.deadline(video_seconds((e["type"], Path(e["path"])) for e in request["inputs"].values())),
            "sky": os.environ.get("PERCEPTION_SKY") or shutil.which("sky") or "sky",
            "aws": os.environ.get("PERCEPTION_AWS") or shutil.which("aws") or "aws"}}

    def submit(self, manifest: Manifest, runner: Runner, attempt_dir: Path, env: dict[str, str], gpu: int | None = None,
               prepare_secrets: dict[str, str] | None = None) -> dict:
        # a machine of its own (machine None): the scheduler gives it no card, and the node has the machine's one
        # SkyPilot and the AWS CLI need the user's cloud credentials, which child_env keeps from nodes: the driver is
        # this package's code, and passes them to no node (the node runs on the machine, with sshd's environment)
        env = {**env, **{k: v for k, v in os.environ.items() if k.startswith(("AWS_", "SKYPILOT_"))}}
        return super().submit(manifest, runner, attempt_dir, env, prepare_secrets=prepare_secrets)


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


class RunPodBackend(CloudBackend):
    """A RunPod pod (Secure Cloud, in the US, Canada or Europe) for each attempt: an account, credit and an API key are all it needs.

    Any of several GPUs with at least 20 GB, the cheapest available first: RunPod's capacity for any one type comes and
    goes. A pod is reached over SSH with a key made for its attempt, holds the node's inputs on its own disk only (no
    network volume: nothing stays after it is deleted), and is deleted through the API by the driver. Its backstops if
    this laptop never comes back: the pod removes itself at its deadline (its start command sets that before anything
    else runs; guard.sh checks it), and the next RunPod launch deletes any perception pod past its deadline (the
    deadline is in its name)."""
    guard = REPO / "cloud" / "runpod" / "guard.sh"
    # perception's own image (cloud/runpod/image): RunPod's cached PyTorch image with ffmpeg, rsync, uv and runpodctl 3
    # added; software only. Its tag is tag.py's, from what goes into it (a test checks they agree)
    image_tag = "efba77cf5632"
    # and the digest of the build that was measured: a rebuild under the same tag (apt's versions move) is never pulled
    image_digest = "sha256:472c7316c3bb200d88948b43acbb4afd054f9d56621f869d700a6680884e234a"
    # under the repository's owner, where .github/workflows/runpod-image.yml pushes it (a test checks)
    image = f"ghcr.io/hyperstudyio/perception-runpod:{image_tag}@{image_digest}"
    start = "/perception-start.sh"   # its own start (host keys, PUBLIC_KEY, sshd), run after the pod's removal is set
    # Secure Cloud prices on 2026-10-24 (RunPod's gpuTypes securePrice), the order RunPod tries them in: the cheap ones
    # first, the dearer ones when none of those is free (capacity for any one type comes and goes). Ampere, Ada and
    # Blackwell (the nodes' torch is built for CUDA 13, which has Blackwell's kernels); not MIG slices of one.
    gpus = {"NVIDIA RTX A4500": 0.25, "NVIDIA RTX A5000": 0.27, "NVIDIA RTX 4000 Ada Generation": 0.28,
            "NVIDIA GeForce RTX 3090": 0.50, "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition": 0.50,
            "NVIDIA RTX PRO 4000 Blackwell": 0.57, "NVIDIA A40": 0.59, "NVIDIA L4": 0.59, "NVIDIA RTX A6000": 0.59,
            "NVIDIA RTX PRO 4500 Blackwell": 0.72, "NVIDIA L40": 0.82,
            "NVIDIA GeForce RTX 4090": 0.89, "NVIDIA RTX PRO 5000 Blackwell": 0.96, "NVIDIA RTX 6000 Ada Generation": 0.99,
            "NVIDIA L40S": 1.09, "NVIDIA GeForce RTX 5090": 1.19}
    # each one's memory as nvidia-smi reports it (GB; the probe's memory_gb), against the steps' minimum_vram_gb: a "24 GB"
    # card reports 24 or 23.99, an L4 22.5, a "48 GB" A40, L40 or L40S 45
    gpu_vram_gb = {"NVIDIA RTX A4500": 20, "NVIDIA RTX A5000": 24, "NVIDIA RTX 4000 Ada Generation": 20,
                   "NVIDIA GeForce RTX 3090": 24, "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition": 95.6,
                   "NVIDIA RTX PRO 4000 Blackwell": 24, "NVIDIA A40": 45, "NVIDIA L4": 22.5, "NVIDIA RTX A6000": 48,
                   "NVIDIA RTX PRO 4500 Blackwell": 32, "NVIDIA L40": 45, "NVIDIA GeForce RTX 4090": 24,
                   "NVIDIA RTX PRO 5000 Blackwell": 48, "NVIDIA RTX 6000 Ada Generation": 48, "NVIDIA L40S": 45,
                   "NVIDIA GeForce RTX 5090": 31.8}
    # datacenters in the US, Canada and Europe, whichever has a GPU free: those RunPod's REST API accepts (its schema's
    # enum, 2026-10-25; gpus above are in its enum too)
    datacenters = ["US-CA-2", "US-DE-1", "US-GA-1", "US-GA-2", "US-IL-1", "US-KS-2", "US-KS-3", "US-MD-1", "US-MO-1", "US-MO-2",
                   "US-NC-1", "US-TX-1", "US-TX-3", "US-TX-4", "US-WA-1", "CA-MTL-1", "CA-MTL-2", "CA-MTL-3",
                   "EU-CZ-1", "EU-FR-1", "EU-NL-1", "EU-RO-1", "EU-SE-1", "EUR-IS-1", "EUR-IS-2", "EUR-IS-3", "EUR-NO-1"]
    disk_gb = 40   # the container's disk: the node's environment, weights and inputs
    # the nodes' torch is PyPI's CUDA 13 build (envs/faces/uv.lock): an NVIDIA driver of 580 or later (guard.sh)
    min_driver = 580
    region = "US, Canada, Europe (Secure Cloud)"
    machine_title = "RunPod, one of " + ", ".join(g.removeprefix("NVIDIA ").removeprefix("GeForce ") for g in gpus)
    hourly_usd = max(gpus.values())   # reserved at the dearest one it may get; charged at the price of the one it got
    caps = Capabilities(platform="linux/amd64", cpu_cores=8, memory_gb=30, accelerators={"cpu", "cuda"}, vram_gb=20)

    def __init__(self, api_key: str | None = None, api_s: float = 60, unknown_s: float = 180, **kw):
        """`api_s`: how long one API call may take. `unknown_s`: how long a pod whose create got no clear answer is
        looked for by its name before it is taken as never made (within down_s)."""
        super().__init__(**kw)
        self.api_s, self.unknown_s = api_s, unknown_s
        self.kind = "runpod"
        self.title = "Cloud GPU (RunPod)"
        self.api_key = api_key

    def driver_spec(self, attempt_dir: Path) -> dict:
        from ..runpod import api_base
        request = json.loads((attempt_dir / "request.json").read_text())
        return {"host": "the RunPod pod", "cloud": {
            "provider": "runpod", "name": cluster_name(request["run"], request["node"], request["attempt"]),
            "api": api_base(), "image": self.image, "start": self.start, "gpus": list(self.gpus), "datacenters": self.datacenters,
            "disk_gb": self.disk_gb, "min_driver": self.min_driver, "guard": str(self.guard), "setup": str(self.setup), "hourly_usd": self.hourly_usd,
            "launch_s": self.launch_s, "down_s": self.down_s, "api_s": self.api_s, "unknown_s": self.unknown_s,
            "deadline_s": self.deadline(video_seconds((e["type"], Path(e["path"])) for e in request["inputs"].values()))}}

    def submit(self, manifest: Manifest, runner: Runner, attempt_dir: Path, env: dict[str, str], gpu: int | None = None,
               prepare_secrets: dict[str, str] | None = None) -> dict:
        # the account key for the driver alone (this package's code), in its environment, never in remote.json: it
        # creates and deletes the pod from here, and sends the key nowhere
        return SshBackend.submit(self, manifest, runner, attempt_dir, {**env, "RUNPOD_API_KEY": self.api_key or ""},
                                 prepare_secrets=prepare_secrets)


class Cloud:
    """cloud.json in the store's home: which cloud places are offered (`perception cloud on [aws|runpod]`)."""
    providers = ("aws", "runpod")

    def __init__(self, home: Path):
        self.path = home / "cloud.json"

    def read(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    @property
    def enabled(self) -> bool:   # AWS (the first cloud place: its key is "enabled")
        return self.read().get("enabled", False)

    @property
    def runpod(self) -> bool:
        return self.read().get("runpod", False)

    @property
    def runpod_keys(self) -> bool:
        """Whether a whole run on a RunPod pod (pod_run.py) may be given the keys its steps declare: the user's explicit
        yes (`perception cloud keys runpod on`), off until then. Per-step pods are never given any."""
        return self.read().get("runpod_keys", False) is True

    def set_keys(self, allowed: bool):
        doc = self.read()
        doc["runpod_keys"] = allowed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(doc))

    def set(self, enabled: bool, provider: str = "aws"):
        doc = self.read()
        doc["enabled" if provider == "aws" else provider] = enabled
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(doc))
