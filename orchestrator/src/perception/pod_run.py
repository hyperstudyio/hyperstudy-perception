"""A whole run on a RunPod pod made for it (`perception remote run runpod GRAPH ...`): run-anywhere phase 10.

The run goes as it goes to a lab server (remote_run.py, phase 7: the pipeline's copy, the video, the run's spec and
keys on ssh's stdin, a coordinator there started detached), with the pod as the host. The pod is made for the run
(pod_session.PodSession: checked by guard.sh before anything is sent) and deleted after it: by `fetch` once the run is
imported, by `cancel`, or, when nothing comes back for it, by itself at the end of its linger window after the run
ends (the coordinator there writes that time: remote_coordinator.linger) or at its deadline, whichever is sooner.

What it may cost is fixed before the pod is made: the pod is reserved at the dearest GPU it may get for its whole
deadline, and the run's max_cost_usd less that reservation is what the coordinator there may spend on its API steps.
A run whose remainder would not cover their estimate is refused here, before anything is made.

The pod is remembered in the run's record (remote_run.record: `pod`), with its ssh key and known hosts in a private
folder beside it (remote/<run>.pod/), so status, attach, fetch and cancel reach it from any later command. The RunPod
account key stays on this machine (it makes and deletes the pod); the run's keys go to the pod only with
`perception cloud keys runpod on`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path

from .backends.cloud import RunPodBackend
from .hosts import Host
from .pod_session import PodSession, pod_ssh

NAME = "runpod"
# until measured on a pod (RUN_ANYWHERE_PLAN.md, phase 10): starting it and making the orchestrator's and nodes'
# environments and fetching the gated weights; then the steps, at most this many times the video's length (the full
# graph's steps took 7 to 23 times it, summed, on liquidswords2 with eight runs sharing its two GPUs, 2026-10-30)
SETUP_S = 30 * 60
PER_VIDEO_S = 12
LINGER_S = 30 * 60
MIN_DEADLINE_S = 60 * 60
# a whole run's pod: the RunPod backend's GPUs up to $0.60/h (20 GB or more, as its steps need), so the reservation of
# a short clip's run leaves room for its API steps under the full graph's $1.50
GPUS = {name: usd for name, usd in RunPodBackend.gpus.items() if usd <= 0.60}
DISK_GB = 100   # every step's environment and the gated weights, with the video and what the steps make (estimate)
# the run's keys a pod is never given, whatever its steps declare: the accounts' own (they make and delete machines)
NEVER_SENT = ("RUNPOD_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


def hourly_usd() -> float:
    return max(GPUS.values())


def deadline_s(video_s: float | None, linger_s: float = LINGER_S) -> int:
    """How long the pod may exist: setup, the steps (by the video's length; an hour when it is not known) and the linger
    window, in whole 5 minutes, at least an hour."""
    steps = PER_VIDEO_S * video_s if video_s else 60 * 60
    return max(MIN_DEADLINE_S, math.ceil((SETUP_S + steps + linger_s) / 300) * 300)


def reservation_usd(deadline: float) -> float:
    return round(hourly_usd() * deadline / 3600, 4)


def spec(deadline: float) -> dict:
    """The pod's spec for PodSession (as RunPodBackend.driver_spec's "cloud", for a whole run)."""
    from .runpod import api_base
    b = RunPodBackend()
    return {"provider": "runpod", "api": api_base(), "image": b.image, "start": b.start, "gpus": list(GPUS),
            "datacenters": b.datacenters, "disk_gb": DISK_GB, "min_driver": b.min_driver, "guard": str(b.guard), "setup": str(b.setup),
            "hourly_usd": hourly_usd(), "launch_s": b.launch_s, "down_s": b.down_s, "api_s": b.api_s,
            "unknown_s": b.unknown_s, "deadline_s": deadline}


def keys_folder(run: str) -> Path:
    from .remote_run import record
    return record(run).with_suffix(".pod")


@dataclasses.dataclass
class PodHost(Host):
    """A pod as a host for remote_run: reached with its own key and known hosts (pod_session.pod_ssh), nothing of the
    user's ssh config."""
    pod_keys: str = ""

    def ssh(self, fresh: bool = False) -> list[str]:
        return pod_ssh(Path(self.pod_keys), self.port or 22)

    @property
    def keys_command(self) -> str:
        return "perception cloud keys runpod on"


def planned_host(keys: bool) -> PodHost:
    """The pod as the checks before it is made see it: RunPod's machine (RunPodBackend.caps), with the image's tools."""
    caps = dataclasses.replace(RunPodBackend.caps, tools={"uv": "uv", "ffmpeg": "ffmpeg", "ffprobe": "ffprobe"})
    return PodHost(NAME, "(a pod made for the run)", "", caps, time.time(), keys=keys)


def host(pod: dict, keys: bool) -> PodHost:
    """The pod a run's record names, as a host."""
    from .backends.base import Capabilities
    return PodHost(NAME, pod["dest"], pod["root"], Capabilities.from_json(pod["caps"]), pod["made_at"], port=int(pod["port"]),
                   keys=keys, pod_keys=pod["keys"])


def launch(run: str, api, deadline: float) -> tuple[PodSession, PodHost, dict]:
    """Make the run's pod and check it (PodSession.launch), and remember it in the run's record before anything is sent
    to it. Returns the session (to delete the pod if what follows fails), the pod as a host, and what the record holds."""
    from .remote_run import record
    session = PodSession(api, spec(deadline), keys_folder(run))
    made_at = time.time()
    try:
        caps, root = session.launch(f"perception-{run.rsplit('-', 1)[-1]}-run")
        caps = dataclasses.replace(caps, tools={**caps.tools, "uv": set_up(session)})
    except BaseException:
        session.teardown()
        raise
    pod = {"id": session.pod, "name": session.name, "deadline_at": session.deadline_at, "deadline_s": deadline,
           "dest": session.dest, "port": int(session.ssh[session.ssh.index("-p") + 1]), "root": root,
           "caps": caps.to_json(),
           "keys": str(keys_folder(run)), "hourly_usd": session.usd, "made_at": made_at}
    mine = json.loads(record(run).read_text()) if record(run).exists() else {}
    record(run).parent.mkdir(parents=True, exist_ok=True)
    record(run).write_text(json.dumps({**mine, "pod": pod}))
    return session, host(pod, keys=False), pod


def set_up(session: PodSession) -> str:
    """What the run needs on the pod, after guard.sh and before anything is sent (cloud/skypilot/node-setup.sh, as a
    step's pod has it: ffmpeg, rsync, and uv, which the image keeps off PATH in ~/.perception-uv). The uv it says."""
    import subprocess
    import sys
    from .pod_session import LaunchFailed, say
    say("checking ffmpeg, rsync and uv on the pod")
    try:
        done = session.sh("bash -s", check=False, data=Path(session.c["setup"]).read_bytes(), timeout=900)
    except subprocess.TimeoutExpired:
        raise LaunchFailed(f"setting up pod {session.pod} took more than 15 min")
    sys.stdout.buffer.write(done.stderr)
    uv = done.stdout.decode().strip().splitlines()[-1] if done.stdout.strip() else ""
    if done.returncode or not uv:
        raise LaunchFailed(f"setting up pod {session.pod} failed (exit {done.returncode})" + ("" if uv else ": it did not say where uv is"))
    return uv


def delete(run: str, api) -> dict:
    """Delete the run's pod (PodSession.teardown) and record what it cost: its time at its price, from when it was made.
    Asked again after it is gone, it says so and changes nothing."""
    from .remote_run import record
    mine = json.loads(record(run).read_text())
    pod = mine.get("pod")
    if not pod:
        raise RuntimeError(f"run {run} has no pod of its own")
    if pod.get("deleted"):
        return pod
    session = PodSession(api, spec(pod["deadline_s"]), Path(pod["keys"]), started=time.monotonic() - (time.time() - pod["made_at"]))
    session.pod, session.name, session.usd, session.deadline_set = pod["id"], pod["name"], pod["hourly_usd"], True
    gone, usd, _ = session.teardown()
    pod = {**pod, "deleted": gone, "cost_usd": usd}
    record(run).write_text(json.dumps({**json.loads(record(run).read_text()), "pod": pod}))
    return pod
