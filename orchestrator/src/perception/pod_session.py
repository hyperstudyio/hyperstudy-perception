"""A RunPod pod made for one job, and deleted after it: a step's attempt (remote_driver, the RunPod backend) or a whole
run coordinated there (remote_run, `perception remote run runpod`: run-anywhere phase 10).

Created through RunPod's API with the account key, which stays in this process: any of several GPUs, in any of the US,
Canadian and European datacenters (whichever RunPod has free), with its own disk only (no network volume: nothing
outlives it). Its start command sets its removal at its deadline before anything else runs (removal_at_boot, with the
pod's own key, scoped to that pod), and its name carries the deadline, so a later launch deletes it if nothing else did
(sweep). It is reached over ssh with a key made for it, its host key pinned in a file of its own, and nothing of this
machine's forwarded; probed like a host, its clock checked, and cloud/runpod/guard.sh run there before anything is sent.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

POLL_S = 3.0
# how long past its deadline a perception pod is left before a RunPod launch deletes it: what made it deletes it before
# its deadline, and it removes itself at it (guard.sh)
SWEEP_GRACE_S = 10 * 60
# where a pod is told to go sooner than its deadline (a whole run's end plus its linger window): a Unix time, read by the
# removal its start command set (removal_at_boot)
REMOVE_AT = "/perception-remove-at"


class Unreachable(Exception):
    pass


class Refused(Exception):
    """The host answered, and the attempt cannot go on: the same would happen again (an input that changed since the run
    read it, a root that is not the user's). It fails as the node's own error, not a crash to retry."""
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class GuardRefused(Exception):
    pass


class LaunchFailed(Exception):
    """The cloud machine did not start (no capacity, a setup that failed or hung): worth another attempt."""


def say(msg: str, log=None):
    print(f"[remote] {msg}", file=log or sys.stdout, flush=True)


def removal_at_boot(at: int, start: str, remove_at: str = REMOVE_AT) -> str:
    """A pod's start command: its removal at `at` (a Unix time), or sooner at the time REMOVE_AT holds once something
    writes it there, set before anything else runs on it and asked again until RunPod does it (runpodctl, with the pod's
    own key, scoped to that pod), then the image's own start (sshd). The backstop for a pod whose driver never comes
    back, from its first moment; perception-deadline names it (guard.sh). What REMOVE_AT holds can only bring the
    removal forward: anything but a whole number there leaves the deadline as it is."""
    # asked every 30 s until it happens, whatever runpodctl's exit says: the pod's removal ends this loop with it
    f = shlex.quote(remove_at)
    removal = (f"until t=$(date +%s); [ $t -ge {int(at)} ] || {{ s=$(cat {f} 2>/dev/null); "
               f'case "$s" in ""|*[!0-9]*) false;; *) [ $t -ge "$s" ];; esac; }}; do sleep 15; done; '
               'while :; do RUNPOD_NO_UPDATE_CHECK=1 runpodctl pod delete "$RUNPOD_POD_ID"; sleep 30; done')
    return f"nohup sh -c {shlex.quote(removal)} perception-deadline > /dev/null 2>&1 & exec {start}"


class PodSession:
    """One pod: `launch` makes it and checks it, `teardown` deletes it and says what it cost. `c` is the backend's
    spec (RunPodBackend.driver_spec's "cloud", or remote_run's for a whole run): api, image, start, gpus, datacenters,
    disk_gb, min_driver, guard, hourly_usd, launch_s, down_s, api_s, unknown_s, deadline_s. `keys`: a private folder
    of this job's own, for the pod's ssh key and known hosts. `started`: when the job began (time.monotonic()), from
    which its deadline and its time limits count. `log`: where its progress goes (stdout, the attempt's log, for a
    step's driver; stderr for the CLI, whose stdout may be --json's)."""

    def __init__(self, api, c: dict, keys: Path, started: float | None = None, log=None):
        self.api, self.c, self.keys, self.log = api, c, keys, log or sys.stdout
        self.started = time.monotonic() if started is None else started
        # its id once created, the price RunPod gave it, and whether a create was sent without an answer
        self.pod: str | None = None
        self.name: str | None = None
        self.usd: float | None = None
        self.unknown = False
        self.deadline_set = False   # its removal at the deadline is running (guard.sh): it cannot outlive it
        self.deadline_at: int | None = None
        self.dest: str | None = None
        self.ssh: list[str] = []

    # --- making it --------------------------------------------------------------------------------------------------

    def launch(self, name: str):
        """Create the pod named `name` plus its deadline, wait for its SSH port, probe it like a host and run guard.sh
        there before anything is sent. Returns its capabilities and root (hosts.probe's)."""
        from .hosts import HostError, probe
        from .runpod import RunPodError
        c = self.c
        self.sweep()
        self.deadline_at = int(time.time() + c["deadline_s"] - (time.monotonic() - self.started))
        self.name = f"{name}-d{self.deadline_at}"
        say(f"starting a RunPod pod, {self.name} (at most {c['deadline_s'] / 60:.0f} min, at most ${c['hourly_usd']}/h)", self.log)
        self.keys.mkdir(mode=0o700, parents=True, exist_ok=True)   # a key made for this pod alone: the user's own are never offered to it
        key = self.keys / "id_ed25519"
        if not key.exists():
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "perception", "-f", str(key)],
                           check=True, capture_output=True, stdin=subprocess.DEVNULL)
        body = {"name": self.name, "imageName": c["image"], "gpuTypeIds": c["gpus"], "gpuTypePriority": "custom",
                "gpuCount": 1, "cloudType": "SECURE", "dataCenterIds": c["datacenters"], "dataCenterPriority": "availability",
                # its own disk only: no volume, so nothing (the inputs above all) outlives the pod
                "containerDiskInGb": c["disk_gb"], "volumeInGb": 0, "ports": ["22/tcp"],
                "env": {"PUBLIC_KEY": (self.keys / "id_ed25519.pub").read_text().strip()},
                # no allowedCudaVersions: its values stop at 13.0 and match a host's exactly, so it excludes the hosts that
                # report 13.2 (driver 595; checked 2026-10-25). guard.sh checks the driver instead
                "dockerStartCmd": ["bash", "-c", removal_at_boot(self.deadline_at - 60, c["start"])]}
        self.unknown = True   # until RunPod answers, a pod may exist that only its name can find
        try:
            pod = self.api.create_pod(body)
        except RunPodError as e:
            # only a 4xx, a create never sent (this laptop offline) and RunPod's answer when none of the GPUs is free (a
            # 500) prove no pod was made; any other failure may come after it was made (a gateway's 502): teardown looks
            # for it by its name
            if e.nothing_done or "no instances currently available" in str(e):
                self.unknown = False
            if e.nothing_done and e.sent:   # RunPod refused it: the same would happen again
                raise Refused("internal", f"RunPod refused to create the pod: {e}")
            raise LaunchFailed(f"RunPod did not start a pod: {e}")
        self.pod, self.unknown = pod["id"], False
        self.usd = float(pod.get("costPerHr") or c["hourly_usd"])
        say(f"pod {self.pod}: ${self.usd}/h", self.log)
        if self.usd > c["hourly_usd"]:   # its price rose past the list's: the reservation would not cover it
            raise Refused("internal", f"RunPod's price for pod {self.pod} is ${self.usd}/h, above the ${c['hourly_usd']}/h "
                                      "reserved for it: perception's RunPod price list needs updating")
        while True:
            try:
                p = self.api.pod(self.pod)
            except RunPodError as e:
                if e.status:
                    raise LaunchFailed(f"RunPod could not say how pod {self.pod} is: {e}")
                p = {}   # no answer this time: the launch's time limit still applies
            if p is None:
                raise LaunchFailed(f"pod {self.pod} went away while it was starting")
            if p.get("publicIp") and (p.get("portMappings") or {}).get("22"):
                break
            if time.monotonic() - self.started > c["launch_s"]:
                raise LaunchFailed(f"pod {self.pod} had no SSH address after {c['launch_s'] / 60:.0f} min")
            time.sleep(POLL_S)
        self.dest = f"root@{p['publicIp']}"
        self.ssh = pod_ssh(self.keys, int(p["portMappings"]["22"]))
        while True:   # its sshd starts a moment after the port is mapped
            try:
                if self.sh("true", check=False, timeout=60).returncode == 0:
                    break
            except (Unreachable, subprocess.TimeoutExpired):
                pass
            if time.monotonic() - self.started > c["launch_s"]:
                raise LaunchFailed(f"pod {self.pod}'s SSH did not answer within {c['launch_s'] / 60:.0f} min")
            time.sleep(POLL_S)
        try:
            caps, root = probe(self.dest, ssh=self.ssh)
        except HostError as e:
            raise LaunchFailed(f"pod {self.pod} started but could not be probed: {e}")
        if "cuda" not in caps.accelerators:
            raise Refused("resources", f"pod {self.pod} has no NVIDIA GPU that nvidia-smi can see")
        # it removes itself at its deadline, a moment by this laptop's clock: the clocks must agree, to a minute
        skew = int(self.sh("date +%s").stdout.decode().strip()) - time.time()
        if abs(skew) > 60:
            raise Refused("internal", f"pod {self.pod}'s clock and this machine's differ by {skew:.0f} s: set this machine's "
                                      "clock (its deadline is a moment by it)")
        # its removal at the deadline is running (its start command), and it holds no key to the account
        result = self.sh(f"bash -s {int(c['min_driver'])}", check=False, data=Path(c["guard"]).read_bytes())
        self.log.write((result.stdout + result.stderr).decode(errors="replace"))
        self.log.flush()
        if result.returncode == 75:   # not this pod (an old driver), or its checks could not be made: another may do
            why = (result.stderr.decode(errors="replace").strip().splitlines() or ["see the log"])[-1]
            raise LaunchFailed(f"pod {self.pod} was not used: {why.removeprefix('guard: ')}")
        if result.returncode:
            raise GuardRefused(f"guard.sh refused pod {self.pod} (exit {result.returncode}): nothing was sent; see the log")
        self.deadline_set = True
        return caps, root

    def sh(self, script: str, check: bool = True, data: bytes | None = None, timeout: float = 600) -> subprocess.CompletedProcess:
        out = subprocess.run([*self.ssh, self.dest, f"sh -c {shlex.quote(script)}"], input=data, capture_output=True, timeout=timeout)
        if out.returncode == 255:
            raise Unreachable(out.stderr.decode(errors="replace").strip()[-300:])
        if check and out.returncode:
            raise RuntimeError(f"on pod {self.pod}: {script[:120]}... exited {out.returncode}: "
                               f"{out.stderr.decode(errors='replace').strip()[-500:]}")
        return out

    def sweep(self):
        """Delete any perception pod past its deadline (by its name): one whose maker never came back (a laptop that
        died) and whose own removal at the deadline (guard.sh) did not happen either. A pod still within its deadline
        may be another job's, and is left alone."""
        from .runpod import POD_NAME, RunPodError
        try:
            pods = self.api.pods()
        except RunPodError as e:
            say(f"WARNING: could not list RunPod pods to delete any left past their deadline ({e})", self.log)
            return
        for p in pods:
            m = POD_NAME.match(p.get("name") or "")
            if m and int(m.group(1)) + SWEEP_GRACE_S < time.time():
                say(f"deleting {p['name']}: a perception pod past its deadline that was never deleted", self.log)
                try:
                    self.api.delete_pod(p["id"])
                except RunPodError as e:
                    say(f"WARNING: could not delete {p['name']} ({e}): delete it at https://console.runpod.io/pods", self.log)

    # --- deleting it ------------------------------------------------------------------------------------------------

    def teardown(self) -> tuple[bool, float, str | None]:
        """Delete the pod through the API and wait until RunPod no longer has it (404), within down_s. A create with no
        clear answer may have made one a moment later: looked for by its name for up to unknown_s, and taken as never
        made if it does not appear. Returns whether it is gone (or was never made), what it cost (its time at its price;
        all of its deadline if it could not be deleted), and, when it may still be running with nothing to end it (its
        removal at the deadline never confirmed), the message saying so."""
        from .runpod import RunPodError
        c = self.c
        if self.pod is None and not self.unknown:
            return True, 0.0, None   # none was made (RunPod refused, or no key): nothing to delete or charge
        say(f"deleting pod {self.pod or self.name}", self.log)
        start = time.monotonic()
        end = start + c["down_s"]
        gone, problem = False, ""
        while not gone and time.monotonic() < end:
            try:   # deleting is idempotent: asked again until RunPod no longer has the pod or the time is up
                if self.pod is None:
                    found = [p["id"] for p in self.api.pods() if p.get("name") == self.name]
                    if not found:
                        if time.monotonic() - start < min(c.get("unknown_s", 180), c["down_s"]):
                            time.sleep(2)
                            continue
                        say(f"no pod named {self.name} appeared in {c.get('unknown_s', 180):.0f} s: none was made. If one "
                            "appears at https://console.runpod.io/pods, delete it (the next RunPod run does, after its deadline)", self.log)
                        self.unknown = False
                        return True, 0.0, None
                    self.pod = found[0]   # names are the job's own: one at most
                self.api.delete_pod(self.pod)
                while time.monotonic() < end and self.api.pod(self.pod) is not None:
                    time.sleep(2)
                gone = self.api.pod(self.pod) is None
            except RunPodError as e:
                problem = str(e)
                say(f"RunPod: {e}", self.log)
                time.sleep(min(5.0, max(0.0, end - time.monotonic())))
        price = self.usd or c["hourly_usd"]
        seconds = time.monotonic() - self.started
        message = None
        if gone:   # its time; never more than its deadline once its own removal was set (it cannot outlive it then)
            usd = round(price * (min(seconds, c["deadline_s"]) if self.deadline_set else seconds) / 3600, 4)
        elif self.deadline_set:   # it removes itself at its deadline at most (guard.sh), and is charged that
            usd = round(price * c["deadline_s"] / 3600, 4)
            say(f"WARNING: could not delete pod {self.pod} ({problem or 'still there'}): it removes itself at its deadline; "
                "or delete it at https://console.runpod.io/pods", self.log)
        else:   # nothing ends it but someone
            usd = round(price * c["deadline_s"] / 3600, 4)
            message = (f"pod {self.pod or self.name} could not be deleted ({problem or 'still there'}), and its removal at "
                       "its deadline was not confirmed: it may still be running. Delete it at https://console.runpod.io/pods")
            say(f"WARNING: {message}", self.log)
        say(f"the pod cost about ${usd:.2f}" + ("" if gone else " (at least)"), self.log)
        return gone, usd, message


def pod_ssh(keys: Path, port: int) -> list[str]:
    """ssh to a pod: its key alone, its host key accepted the first time in a file of the job's own and held to after
    that (quoted: ssh reads an unquoted path with a space as two files). None of the user's ssh config applies (-F none:
    its IdentityFiles would be offered too), and nothing of this machine's is forwarded to it (an agent, ports, X11)."""
    from .hosts import ssh_command
    # its connection shared by its key too: a pod made later at the same address and port is never reached through
    # a connection to the one before
    return [*ssh_command(identity=str(keys / "id_ed25519")), "-F", "none", "-i", str(keys / "id_ed25519"), "-p", str(int(port)), "-o", "IdentitiesOnly=yes",
            "-o", f'UserKnownHostsFile="{keys / "known_hosts"}"', "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes", "-o", "ForwardX11=no"]
