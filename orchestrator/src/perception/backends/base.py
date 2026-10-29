"""What every execution backend provides. The graph never names a provider; placement picks a backend whose runner
fits the node, and the backend turns the file protocol (request.json in, out/ + response.json back) into a process,
a remote job or an API call.

    probe_capabilities()   what this backend can run here (platform, accelerators, memory, tools)
    runner_for(manifest)   the manifest runner this backend would use, or why none fits
    identity(manifest, r)  what makes a result this node's wherever it was made (the code that runs, the accelerator,
                           the platform; never the machine): the cache key's runner, one rule for every backend
    estimate(...)          cost and time, before anything is reserved
    stage_inputs(...)      make the inputs readable where the node runs; returns the request's input entries
    submit(...)            start it, on the card the scheduler chose if any; returns a JSON-able handle (kept in
                           the run state, so a restarted coordinator can find the job again)
    poll(handle)           running | exited
    cancel(handle)
    collect(attempt_dir)   the node's response.json, or None if it never wrote one (a crash)
    cleanup(attempt_dir)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..manifests import Manifest, Runner
from ..store import Artifact


@dataclass(frozen=True)
class Gpu:
    """One NVIDIA card, as nvidia-smi lists it. `index` is nvidia-smi's, which counts in PCI bus order: CUDA counts the
    fastest card first unless CUDA_DEVICE_ORDER=PCI_BUS_ID, so a node confined to a card gets both (gpu_env). None: the
    machine's GPU taken as one, its cards not known one by one (an Apple GPU, a host probed before cards were recorded)."""
    index: int | None
    name: str
    memory_gb: float


@dataclass
class Capabilities:
    platform: str                      # darwin/arm64, linux/amd64, windows/amd64...
    cpu_cores: int
    memory_gb: float
    accelerators: set[str]             # cpu, mps, cuda
    vram_gb: float = 0                 # CUDA: the largest card's; Apple GPU: unified memory (memory_gb)
    tools: dict[str, str | None] = field(default_factory=dict)   # uv, ffmpeg, ffprobe -> path or None
    gpus: list[Gpu] = field(default_factory=list)   # every NVIDIA card; empty on a Mac, and for a host probed before

    def devices(self) -> list[Gpu]:
        """What GPU nodes are placed on, one at a time for an exclusive node (scheduler.card_for): each card when the
        cards are known, else the machine's GPU as one (index None), which is how every machine was counted before the
        cards were: one exclusive node at a time on it, its memory not divided."""
        return list(self.gpus) or [Gpu(None, "", self.vram_gb)]

    @classmethod
    def from_json(cls, d: dict) -> "Capabilities":
        """As to_json (or a host saved before cards were recorded: no `gpus`) wrote it."""
        return cls(**{**d, "accelerators": set(d["accelerators"]), "gpus": [Gpu(**g) for g in d.get("gpus", [])]})

    def to_json(self) -> dict:
        return {**asdict(self), "accelerators": sorted(self.accelerators)}


# A card is leased on its machine for as long as a node runs on it, through a lock file every coordinator there honours
# (one on this Mac and one on the lab server, or two runs started side by side): the scheduler's own count covers only
# its run, so it only asks for a card, and the node's wrapper on the machine takes it (LEASE). In a folder of their own
# under the user's home on that machine, whatever root a host was added with: a coordinator elsewhere and one on the
# machine itself (NativeBackend.lease_folder) then always take the same leases. Not inside a root: ~/.cache/
# hyperstudy-perception also holds the nodes' weights, often with no root's marker, and a root must be empty or marked
# (hosts.PRIVATE_ROOT). The folder is private in itself (LEASE_FOLDER). Named for the machine (uname -n) as
# well as the card: lab servers often share one home over NFS, and a card is only another machine's card of the same
# index there. Not shared with other users of the machine (their roots are their own). flock is not fair: a node
# waiting for an exclusive lease waits until no shared node holds the card, however long a stream of them lasts.
GPU_LEASES = ".cache/hyperstudy-perception-gpus"   # under the home

# sh that makes the lease folder ($HOME/GPU_LEASES) the user's own and private, or fails (exit 3: refused, as a root
# that is someone else's is): only the user then writes the leases there.
LEASE_FOLDER = r"""d="$HOME/.cache/hyperstudy-perception-gpus"
(umask 077 && mkdir -p "$d") || exit 3
if [ -L "$d" ] || [ "$(ls -ldn "$d" | awk '{print $3}')" != "$(id -u)" ]; then echo "$d is not your own folder" >&2; exit 3; fi
chmod 700 "$d" || exit 3"""

# perl (on every Linux server and Mac, which flock(1) is not): flock the lease <dir>/<machine>-<index>, shared for a
# `gpu: shared` node and exclusive for a `gpu: exclusive` one, write who holds it (the run, node and attempt, and the
# pid), then exec the node with the lock's descriptor left open ($^F: not closed on exec), so the node's process holds
# the lease until it ends, however it ends: a lost SSH session does not end it. Opened for reading and writing: on NFS,
# flock is an fcntl lock, and a shared one needs a descriptor that can read. A lease another node holds is waited for,
# and the node's log says so and who holds it. The file names only the nodes holding the card now: an exclusive node
# empties it, and a shared one keeps only the lines of nodes still running (two shared nodes taking it at the same
# moment may each drop the other's line: the file is what is shown, the lock is what decides).
LEASE = r"""$^F = 1023;
my ($dir, $index, $use, $who) = splice(@ARGV, 0, 4);
my $path = "$dir/" . (POSIX::uname())[1] . "-$index";
open(my $lease, "+>>", $path) or die "perception: cannot open the GPU lease $path: $!\n";
my $how = $use eq "exclusive" ? LOCK_EX : LOCK_SH;
sub running { seek($lease, 0, 0); my @held = <$lease>; chomp(@held); return grep { /pid (\d+)$/ && kill(0, $1) } @held }
if (!flock($lease, $how | LOCK_NB)) {
  my @held = running();
  print STDERR "perception: waiting for $path ($use): another node on this machine has that GPU (held by " . ($held[-1] // "a node") . ")\n";
  flock($lease, $how) or die "perception: GPU lease $path: $!\n";
  print STDERR "perception: $path is free: starting\n";
}
my @keep = $how == LOCK_EX ? () : grep { !/^exclusive / } running();
truncate($lease, 0);
syswrite($lease, join("", map { "$_\n" } @keep) . "$use $who pid $$\n");
exec { $ARGV[0] } @ARGV or die "perception: cannot run $ARGV[0]: $!\n";
"""
LEASE_PERL = ["perl", "-MFcntl=:flock", "-MPOSIX", "-e", LEASE]


class LeaseRefused(PermissionError):
    """The machine's lease folder is not the user's own (LEASE_FOLDER): the same every time, so the node fails once."""


class SubmitRefused(Exception):
    """A submit that would be refused again, for the same reason: the node fails once, with this error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def lease_command(leases: str, index: int, use: str, who: str) -> list[str]:
    """What goes before a node's command to run it holding card `index`'s lease in the folder `leases` (LEASE), as
    `use` (shared, exclusive); `who` says which attempt holds it (run, node, attempt)."""
    return [*LEASE_PERL, leases, str(int(index)), use, who]


def holder(request: dict) -> str:
    """Who a node's lease says holds it (lease_command's `who`): its run, node and attempt."""
    return f"run {request['run']} node {request['node']} attempt {request['attempt']}"


def gpu_env(index: int | None) -> dict[str, str]:
    """The environment that confines a node to one card: nvidia-smi's index, in its order (Gpu). Nothing for the
    machine's GPU as one: the node sees every card, as before."""
    return {} if index is None else {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(int(index))}


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
    kind: str            # names the backend in a place (`kind:accelerator`): unique among a coordinator's backends
    runner_kind: str = ""   # the manifest runners it runs (default: its kind); a remote machine runs `native` ones
    remote = False       # runs elsewhere: a node's inputs leave this machine to reach it
    title = "This machine"   # where it is, for people: "gpu1 (SSH)", "Cloud GPU (AWS L4)"

    @property
    def machine(self) -> str | None:
        """Which machine's CPU, memory and GPU a node placed here shares with other nodes; None when every node gets a
        machine of its own (a cloud machine started for it)."""
        return "local"

    def sends(self, manifest: Manifest) -> list[str]:
        """The kinds of data that leave this machine when the node runs here: its inputs', for a remote machine (every
        kind, for an input that could hold anything)."""
        if not self.remote:
            return []
        from ..artifacts import get
        from ..ir import UPLOAD_KINDS
        kinds = [get(t).kinds for t in manifest.inputs.values()]
        return sorted(UPLOAD_KINDS) if None in kinds else sorted({k for ks in kinds for k in ks})

    def remote_refusal(self, manifest: Manifest) -> str:
        """Why a node may not run on this remote machine at all, or "" (local backends: never)."""
        return ""

    @abstractmethod
    def probe_capabilities(self) -> Capabilities: ...

    def runner_for(self, manifest: Manifest) -> tuple[Runner | None, str]:
        """The first of the manifest's runners this backend can use here, preferring an accelerated one."""
        runners, why = self.runners_for(manifest)
        return (runners[0], "") if runners else (None, why)

    def runners_for(self, manifest: Manifest) -> tuple[list[Runner], str]:
        """Every runner of the manifest this backend can use here, accelerated ones first; why none, when none."""
        caps = self.probe_capabilities()
        reasons, usable = [], []
        kind = self.runner_kind or self.kind
        mine = [r for r in manifest.runners if r.kind == kind]
        if not mine:
            return [], f"no {kind} runner"
        if refusal := self.remote_refusal(manifest):
            return [], refusal
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
                    usable.append(r)
        return usable, "" if usable else "; ".join(reasons)

    def missing_tools(self, runner: Runner) -> list[str]:
        return []

    def identity(self, manifest: Manifest, runner: Runner) -> str:
        """What makes a result this node's wherever it was made: the cache key's `runner`. One rule for every backend
        (decided 2026-10-28): the code that runs, the runner that runs it, its accelerator and the platform, and nothing
        that names the machine or the backend. A result made on one machine is reused on any other of the same kind (a
        lab server's on a RunPod pod, or on the same server added under a second name), and a store made by a
        coordinator on another machine can be imported, as long as that coordinator's copy of the pipeline lists its
        files (a checkout, or the package's RELEASE.json; the staged copy cloud/skypilot/launch.sh sends has neither,
        so its keys are unlisted ones and match no other machine's). Where a result was made is in its usage instead
        (`ran_on`, which the coordinator records).

        The code is the files remote.code_files lists, hashed by path and content: the files git tracks in the node's
        directory, a script it shares with another node, its uv project with its lock, and the node SDK (installed
        editable, outside the lock). What another machine runs is exactly what is sent there, so its results and this
        machine's agree when the code does. Here a node also runs the files beside it that git does not track yet (a
        module being written), so those the repository's .gitignore files do not ignore count too: a result made with them is never served to
        another machine, which runs without them. A node git cannot list (outside the repository, a checkout without
        git) is hashed from every file in its folders, and its results stay this machine's (no other machine is sent
        it).

        The platform is the OS and architecture (`linux/amd64`, `darwin/arm64`, as the machine was probed) and the
        accelerator is the runner's (`cpu`, `mps`, `cuda`). Nothing finer: not the GPU model, its driver or the CPU.
        The lock pins every library, torch and its CUDA build included (and each uv project Python's minor version), so
        two machines of one platform and accelerator give the same results up to float rounding, which is accepted (a face run on a RunPod RTX A5000
        was byte-identical to one on an AWS L4). A Mac and a Linux machine stay apart: their wheels and kernels differ
        (the face node on the Mac and on an L4 agreed on 185 of 186 tracks).

        Every identity starts with the platform and accelerator (`linux/amd64/cuda:...`), which no identity under the
        rule before this one did (those began with the backend's kind: `native:`, `ssh-gpu1:`, `runpod:`), so no
        cache key made then is ever read as one made now: results cached under the old rule are made again, once."""
        from .remote import code_files, code_hash   # remote imports this module
        here = not self.remote
        try:
            code = code_hash(code_files(manifest, runner, untracked=here))
        except (RuntimeError, OSError) as e:
            # here: hashed from the files themselves, under a name no listed code has; elsewhere nothing could be sent
            # (submit says why), and no result is shared
            from .native import unlisted_code_hash
            code = f"unlisted {unlisted_code_hash(manifest, runner)}" if here else f"unsendable {e}"
        plat = self.probe_capabilities().platform
        doc = {"code": code, "runner": [runner.kind, runner.entry, runner.env], "accelerator": runner.accelerator, "platform": plat}
        return f"{plat}/{runner.accelerator}:{hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]}"

    def estimate(self, manifest: Manifest, runner: Runner, inputs: dict[str, Artifact], video_s: float | None = None) -> Estimate:
        """`video_s`: the video's length when the caller knows it (a plan, before decode has run)."""
        return Estimate(0.0)

    @abstractmethod
    def stage_inputs(self, attempt_dir: Path, inputs: dict[str, Artifact]) -> dict[str, dict]: ...

    @abstractmethod
    def submit(self, manifest: Manifest, runner: Runner, attempt_dir: Path, env: dict[str, str], gpu: int | None = None,
               prepare_secrets: dict[str, str] | None = None) -> dict:
        """`gpu`: the card the scheduler gave the node on its machine (Gpu.index), which it is to see alone (gpu_env);
        None: no card of its own (no GPU, or the machine's GPU as one). `env` is the node's (node_env);
        `prepare_secrets` its download secrets, for its prepare alone (a backend that runs no prepare ignores them, and
        the node, offline, then finds only what is in the cache)."""

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


# a node whose downloads are its prepare's (download_secrets) reads only the cache: a missing weight is an error at
# once, not a download it has no token for. Nor does it load a token stored on the machine (`hf auth login`'s, or
# HF_TOKEN_PATH's), which huggingface_hub otherwise does even offline
OFFLINE = {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}
# a prepare on another machine: the token it was sent, or none, never the machine's own login (its HF_TOKEN in a
# profile, its stored token): access to a gated model is a person's. A prepare that prepare_first runs has it from OFFLINE
SENT_TOKEN_ONLY = {"HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}


def hf_cache_env(home: str) -> dict[str, str]:
    """A Hugging Face cache to read and write, whatever the machine sets: HF_HUB_CACHE (which huggingface_hub takes over
    HF_HOME's hub/, and a lab server may point at a cache every user shares) to the one in `home`."""
    return {"HF_HOME": home, "HF_HUB_CACHE": f"{home}/hub"}
# where such a node's gated weights are, when not in the machine's own Hugging Face cache: on a host, its account's
# (hosts.hf_home), which a coordinator there is told (remote_coordinator). Its HF_HOME, and only its: other nodes keep
# the machine's cache, so public weights are not kept once per person
GATED_HF_HOME = "PERCEPTION_GATED_HF_HOME"


def node_env(manifest: Manifest, secrets: dict[str, str]) -> dict[str, str]:
    """What a node runs with: child_env with its run-time secrets, offline when it has download secrets, which only its
    prepare is given (download_for)."""
    env = child_env(manifest.secrets, secrets)
    if not manifest.download_secrets:
        return env
    gated = hf_cache_env(env[GATED_HF_HOME]) if env.get(GATED_HF_HOME) else {}
    return {**env, **gated, **OFFLINE}


def download_for(manifest: Manifest, secrets: dict[str, str]) -> dict[str, str]:
    """The node's download secrets that are set: for its prepare, never for the node."""
    return {k: secrets[k] for k in manifest.download_secrets if secrets.get(k)}
