"""Other machines nodes can run on: lab servers reached over SSH (`perception hosts add NAME DEST`).

DEST is what you would give ssh (user@host, or a Host from ~/.ssh/config). --user, --port and --identity override what
~/.ssh/config says for it; every command sent to the host uses them (Host.ssh). `perception hosts probe DEST` reaches a
destination and says what it has without adding it or changing anything there (a connection test).

A host is probed when it is added (and by `hosts check`): its platform, cores, memory, GPUs and tools are recorded in
$PERCEPTION_HOME/hosts.json, so planning never waits on the network. SSH runs in batch mode with the user's own keys
and ~/.ssh/config: nothing here asks for or stores a password. Connections are shared (ControlMaster), so the many short
commands a remote node needs cost one handshake.

On the host everything lives under its root (default ~/.cache/hyperstudy-perception), which is private: created 0700, and
refused if someone else owns it, so the code, inputs, outputs and markers kept there can be trusted and are not readable
by others:
    code/<hash>/         the files git tracks for the node, its uv project and the node SDK, by content
    envs/<key>/          a uv environment (its project, its lock and its .venv), by the lock's content: shared by every
                         node and version of the code with that lock, so a change to a node's code makes no new one
    objects/<sha256>/data   inputs, by content: the video is sent once per host (each attempt links it under its name);
                         and the outputs of the nodes that ran there (face embeddings, transcripts: whatever a node
                         wrote), kept under the same hashes so a node after it there is not sent them again. Inputs
                         and outputs alike stay until no step has used them for 7 days (KEEP_DAYS; `--keep-days`), or,
                         while the host's disk has less than 20 GB free (or a tenth of it, if less; `--min-free-gb`),
                         until they are the ones unused longest; never while a step holds them. A removed input is sent
                         again when a node there reads it
    holds/<sha256>/      who holds an object (a file per attempt, naming its folder in work/: released when that
                         folder is removed) and when it was last used (.used). The cleanup removes only what no attempt
                         whose folder exists holds, under objects.lock (remote_driver.CLEANUP). It runs on each attempt
                         (at most every 10 minutes), every hour in a sweeper of the host's own that the attempts start
                         (so a host used once is cleaned too; it ends once nothing is kept there, and a host that
                         restarts has none until its next attempt), and with `perception hosts clean NAME --expired`
    work/<run>/<node>/<attempt>/   the request, out/, response.json, log.txt and pid of one attempt, made before its
                         inputs are sent: removed after it, or by the host's watchdog when no driver has asked about it
                         for a while (a finished node's results wait a day for its driver), or by the cleanup when
                         no watchdog will (no node running and no heartbeat for a day and an hour: the host restarted)
    pipelines/<hash>/    a listed copy of the pipeline, by content, for whole runs coordinated there (`perception remote
                         run`, remote_run.py), read-only; envs/orchestrator-<key>/ its orchestrator's environment
    coordinator/         the store of those runs (remote_coordinator.py): their state, results and node environments,
                         and per run (coordinator/remote/<run>/) its inputs' links, its coordinator's pid and log, and the
                         bundle `remote fetch` exports. Whether such runs may hold the user's keys, and a gated
                         model's token may go to a step's prepare there, is per host, in hosts.json (`perception hosts
                         keys NAME on|off`, off by default)
    hf/<account>/        the Hugging Face cache (HF_HOME) of one account's gated weights (hf_home): downloaded by the
                         prepare of a step that declares download_secrets, with that account's token, and read offline
                         by its steps; kept for later runs, removed by `perception hosts clean NAME`
Outside the root, ~/.cache/hyperstudy-perception-gpus/<machine>-<index> is a card's lease (backends/base.LEASE): a lock
a GPU node holds while it runs, which every coordinator's nodes there honour (one on the host itself too, whatever root
the host was added with), and who holds it.
An attempt's unfinished uploads (objects/.incoming-*, code/.incoming-*, envs/.incoming-*) are removed when it ends; an
output a host never finished keeping (objects/.keeping-*: the host restarted meanwhile) by the next keep or cleanup there
a day later. Code and environments stay until `perception hosts clean NAME`, which removes all the root holds: inputs, the
outputs kept there, code and environments, copies of the pipeline and the store of runs coordinated there, unless a
node is running there, a run is being coordinated there, or the root is not marked as perception's. A
host with a node running there is not removed either: that node could no longer be stopped.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .backends.base import Capabilities
from .backends.probe import GPU_QUERY, parse_gpus, visible_gpus

NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")
# an ssh destination: user@host, a host, or a ~/.ssh/config alias; never an option
DEST = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@%+:\[\]-]*$")
# a path on the host: plain characters, no `..`; relative ones are under the host's home
ROOT = re.compile(r"^[A-Za-z0-9_./~-]+$")
DEFAULT_ROOT = ".cache/hyperstudy-perception"
RESERVED = {"native", "self", "cloud", "runpod", "local"}
# What a host keeps (objects/: inputs sent there, outputs kept there) stays while unused for KEEP_DAYS, and its cleanup
# keeps MIN_FREE_GB free on the host's disk, or a tenth of the disk if that is less (remote_driver.CLEANUP). A week
# covers a project's runs on the same clips (the video, the largest object, is sent once); 20 GB is room for what one
# step writes on an hour of video, or a model's weights, on a shared server that is often nearly full. Per host:
# `hosts add/set --keep-days N --min-free-gb N`.
KEEP_DAYS = 7
MIN_FREE_GB = 20
USER = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._-]{0,63}$")


def hf_home(root: str, account: str) -> str:
    """HF_HOME on a host for one Hugging Face account (keycheck.hf_account): where its token's prepares there download
    gated weights and its steps there read them, in the host's private root. One cache per account, not per host: access
    to a gated model is a person's, so weights fetched under one person's approval never serve another's run, even when
    people share the host's Unix account. Kept for later runs (RUN_ANYWHERE_PLAN.md, gated weights)."""
    return f"{root}/hf/{account}"


# What the host has, as key=value lines. POSIX sh: Linux servers and Macs, no Python needed. Tools are looked for on the
# PATH a login shell would add too (~/.local/bin is where uv installs itself), since ssh commands get a minimal PATH.
# Each NVIDIA card is a line of its own (gpu=index, MiB, name), in nvidia-smi's order; then what the host's own
# environment says CUDA may see (visible_gpus).
PROBE = r"""
PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
echo "home=$HOME"
echo "system=$(uname -s)"
echo "machine=$(uname -m)"
echo "cores=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)"
if [ -r /proc/meminfo ]; then echo "memory_kb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)"
else echo "memory_kb=$(( $(sysctl -n hw.memsize) / 1024 ))"; fi
for t in uv ffmpeg ffprobe; do echo "tool_$t=$(command -v $t || true)"; done
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=%s --format=csv,noheader,nounits 2>/dev/null | sed 's/^/gpu=/'
fi
[ -n "${CUDA_VISIBLE_DEVICES+set}" ] && echo "cuda_visible=$CUDA_VISIBLE_DEVICES"
echo "cuda_order=${CUDA_DEVICE_ORDER-}"
""" % GPU_QUERY


class HostError(RuntimeError):
    pass


@dataclass
class Host:
    name: str
    dest: str
    root: str                  # absolute, on the host
    caps: Capabilities
    probed_at: float
    # overrides of what ~/.ssh/config says for dest (None: ssh's own)
    user: str | None = None
    port: int | None = None
    identity: str | None = None   # a private key on this machine, absolute
    # how long what a host keeps (objects/) stays there unused, and the free space its cleanup keeps (None: the defaults)
    keep_days: float | None = None
    min_free_gb: float | None = None
    # whether whole runs coordinated there (`perception remote run`) may be given the keys their steps declare, and a
    # step placed there the token its prepare downloads gated weights with (download_secrets): the user's explicit yes
    # for this host (`perception hosts keys NAME on`), off until then and again when it is added anew
    keys: bool = False

    @property
    def keep_s(self) -> float:
        return (KEEP_DAYS if self.keep_days is None else self.keep_days) * 86400

    def to_json(self) -> dict:
        """As hosts.json holds it. The cards beside `caps`, not in it: a reader from before cards were recorded (the
        app's own copy of the CLI, reading the same store) passes `caps` whole to Capabilities and would refuse a key it
        does not know, while it ignores one beside it."""
        caps = self.caps.to_json()
        return {**asdict(self), "caps": {k: v for k, v in caps.items() if k != "gpus"}, "gpus": caps["gpus"]}

    @classmethod
    def from_json(cls, d: dict) -> "Host":
        # a host saved before its cards were recorded has none: counted as one GPU
        caps = Capabilities.from_json({**d["caps"], "gpus": d.get("gpus", d["caps"].get("gpus", []))})
        return cls(d["name"], d["dest"], d["root"], caps, d["probed_at"], d.get("user"), d.get("port"), d.get("identity"),
                   d.get("keep_days"), d.get("min_free_gb"), d.get("keys") is True)

    def ssh(self, fresh: bool = False) -> list[str]:
        """ssh to this host: every command sent to it (probe, nodes, clean) goes through this, with its overrides.
        `fresh`: its own connection, not a shared one (a check: see ssh_command)."""
        return [*ssh_command(shared=not fresh, identity=self.identity), *ssh_options(self.user, self.port, self.identity)]


def ssh_options(user: str | None, port: int | None, identity: str | None) -> list[str]:
    return [*(["-l", user] if user else []), *(["-p", str(port)] if port else []),
            *(["-i", identity, "-o", "IdentitiesOnly=yes"] if identity else [])]


def check_overrides(dest: str, user: str | None, port: int | None, identity: str | None) -> str | None:
    """Refuse overrides ssh would misread; the identity file as an absolute path on this machine."""
    if user is not None:
        if not USER.match(user):
            raise HostError(f"not a user name: {user!r}")
        if "@" in dest:
            raise HostError(f"{dest} already names its user: give the user once (in the destination, or --user)")
    if port is not None and not 1 <= port <= 65535:
        raise HostError(f"not a port: {port}")
    if identity is None:
        return None
    path = Path(identity).expanduser()
    if not path.is_file():
        raise HostError(f"no key file at {identity}")
    path = path.resolve()
    if any(ord(c) < 32 for c in str(path)):
        raise HostError(f"not a plain path: {identity!r}")
    if "%" in str(path) or "${" in str(path):   # ssh expands %u, ${HOME}… in -i: it would read another file than this
        raise HostError(f"ssh would expand % or ${{ in a key file's path: rename {identity}")
    return str(path)


def ssh_command(shared: bool = True, identity: str | None = None) -> list[str]:
    """ssh with the options every remote command uses. PERCEPTION_SSH replaces the ssh program (the tests use a
    stand-in that runs commands on this machine). A connection is shared by host, port and user (%C), and by key file
    too (`identity`): ssh's own sharing ignores the key, so a connection opened with one key would carry commands meant
    for another. Not `shared`: a connection of its own, for a check (add, probe, check), whose yes must be this
    connection's, never one another key opened."""
    control = Path(f"/tmp/perception-ssh-{os.getuid()}")   # short: a socket path must fit in 104 bytes on macOS
    control.mkdir(mode=0o700, exist_ok=True)
    st = control.stat()
    if st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise HostError(f"{control} must be a directory only you can use")
    base = [os.environ.get("PERCEPTION_SSH", "ssh"), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
    if not shared:
        return [*base, "-o", "ControlMaster=no", "-o", "ControlPath=none"]
    import hashlib
    keyed = f"-{hashlib.sha256(identity.encode()).hexdigest()[:12]}" if identity else ""   # short: the socket path's limit
    return [*base, "-o", "ControlMaster=auto", "-o", f"ControlPath={control}/%C{keyed}", "-o", "ControlPersist=120"]


def check_retention(keep_days: float | None, min_free_gb: float | None):
    for flag, v in (("--keep-days", keep_days), ("--min-free-gb", min_free_gb)):
        if v is not None and not (v >= 0 and v < 1e6):
            raise HostError(f"{flag} is a number of days or GB, 0 or more: {v}")


def check_names(name: str, dest: str, root: str):
    if not NAME.match(name) or name in RESERVED:
        raise HostError(f"a host's name is lower-case letters, digits and -, and not {', '.join(sorted(RESERVED))}: {name!r}")
    if not DEST.match(dest):
        raise HostError(f"not an ssh destination: {dest!r}")
    parts = [x for x in root.removeprefix("~/").split("/") if x not in ("", ".")]
    if not ROOT.match(root) or ".." in parts or "~" in root.removeprefix("~/") or not parts:
        raise HostError(f"a host's root is a plain path below / or ~ (its own folder: `hosts clean` empties it): {root!r}")


# The root is perception's own and private. It is used only if it is missing, empty or already marked as perception's
# (MARKER): a folder of the user's own given by mistake is never changed or cleaned. It is created 0700, and refused if
# someone else owns it: only the user can then write the code, inputs and completion markers kept there, so they are
# trusted across runs, and others on a shared server cannot read the inputs.
MARKER = ".hyperstudy-perception"
PRIVATE_ROOT = r"""
umask 077
# the contents first, then the marker: perception writes there only after the marker exists, and never removes it, so
# a root that another attempt is setting up at this moment is never taken for someone else's
if [ -d "$root" ] && [ -n "$(ls -A "$root")" ] && [ ! -e "$root/.hyperstudy-perception" ]; then
  echo "$root already holds other files: give perception a folder of its own (--root)" >&2; exit 3
fi
mkdir -p "$root"
if [ "$(ls -ldn "$root" | awk '{print $3}')" != "$(id -u)" ]; then echo "$root belongs to someone else" >&2; exit 3; fi
chmod 700 "$root"
touch "$root/.hyperstudy-perception"
"""


def private_root_script(root: str) -> str:
    """sh that makes `root` (absolute, or under the home) the user's own private folder, or fails."""
    import shlex
    path = shlex.quote(root) if root.startswith("/") else '"$HOME"/' + shlex.quote(root.removeprefix("~/"))
    return f"root={path}\n" + PRIVATE_ROOT


def probe(dest: str, root: str = DEFAULT_ROOT, ssh: list[str] | None = None, make_root: bool = True) -> tuple[Capabilities, str]:
    """What the host has, and its root as an absolute path (made private there, or refused: PRIVATE_ROOT). `ssh`: the
    command to reach it, when it needs options of its own (overrides, a RunPod pod's key and port). Without `make_root`
    nothing on the host is changed: the root is only where it would be (a connection test)."""
    script = (private_root_script(root) if make_root else "") + PROBE
    out = subprocess.run([*(ssh or ssh_command()), dest, "sh -s"], input=script, capture_output=True, text=True, timeout=120)
    if out.returncode:
        raise HostError(f"could not reach {dest} (ssh exit {out.returncode}): {out.stderr.strip()[-500:]}")
    facts = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)
    found = parse_gpus("\n".join(line.removeprefix("gpu=") for line in out.stdout.splitlines() if line.startswith("gpu=")))
    # the cards a node there may be given, if the host's own environment narrows what CUDA sees (visible_gpus)
    gpus = visible_gpus(found, facts.get("cuda_visible"), facts.get("cuda_order") or None)
    try:
        system = {"Linux": "linux", "Darwin": "darwin"}[facts["system"]]
        arch = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}[facts["machine"]]
        memory = int(facts["memory_kb"]) / 2**20
        cores = int(facts["cores"])
    except (KeyError, ValueError) as e:
        raise HostError(f"{dest} answered, but not as expected ({e}): {out.stdout[-500:]}")
    accelerators, vram = {"cpu"}, 0.0
    if found:
        accelerators.add("cuda")
        vram = max(g.memory_gb for g in gpus or found)   # what a runner's minimum is checked against: the largest card
    if (system, arch) == ("darwin", "arm64"):
        accelerators.add("mps")
        vram = vram or memory   # unified memory
    caps = Capabilities(platform=f"{system}/{arch}", cpu_cores=cores, memory_gb=round(memory, 1), accelerators=accelerators,
                        vram_gb=round(vram, 1), tools={t: facts.get(f"tool_{t}") or None for t in ("uv", "ffmpeg", "ffprobe")},
                        gpus=gpus)
    home = facts["home"]
    absolute = (root if root.startswith("/") else f"{home}/{root.removeprefix('~/')}").rstrip("/")
    # plain characters in the whole path, the host's home too: rsync passes remote paths through the host's shell
    if not ROOT.match(absolute):
        raise HostError(f"{dest}'s folder for perception would be {absolute!r}, which is not a plain path (spaces?): give --root /a/plain/path")
    return caps, absolute


class Hosts:
    """hosts.json in the store's home."""

    def __init__(self, home: Path):
        self.path = home / "hosts.json"

    def load(self) -> dict[str, Host]:
        if not self.path.exists():
            return {}
        return {name: Host.from_json(d) for name, d in json.loads(self.path.read_text()).items()}

    def save(self, hosts: dict[str, Host]):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({n: h.to_json() for n, h in sorted(hosts.items())}, indent=1))
        os.replace(tmp, self.path)

    def add(self, name: str, dest: str, root: str = DEFAULT_ROOT, user: str | None = None, port: int | None = None,
            identity: str | None = None, keep_days: float | None = None, min_free_gb: float | None = None) -> Host:
        check_names(name, dest, root)
        identity = check_overrides(dest, user, port, identity)
        check_retention(keep_days, min_free_gb)
        caps, absolute = probe(dest, root, [*ssh_command(shared=False), *ssh_options(user, port, identity)])
        host = Host(name, dest, absolute, caps, time.time(), user, port, identity, keep_days, min_free_gb)
        hosts = self.load()
        hosts[name] = host
        self.save(hosts)
        return host

    def check(self, name: str) -> Host:
        hosts = self.load()
        if name not in hosts:
            raise HostError(f"no host {name!r} (perception hosts list)")
        old = hosts[name]
        caps, _ = probe(old.dest, old.root, old.ssh(fresh=True))
        hosts[name] = replace(old, caps=caps, probed_at=time.time())
        self.save(hosts)
        return hosts[name]

    def set(self, name: str, keep_days: float | None = None, min_free_gb: float | None = None) -> Host:
        """Change how long the host keeps what is unused there, or the free space its cleanup keeps (None: unchanged)."""
        hosts = self.load()
        if name not in hosts:
            raise HostError(f"no host {name!r} (perception hosts list)")
        check_retention(keep_days, min_free_gb)
        h = hosts[name]
        hosts[name] = replace(h, keep_days=h.keep_days if keep_days is None else keep_days,
                              min_free_gb=h.min_free_gb if min_free_gb is None else min_free_gb)
        self.save(hosts)
        return hosts[name]

    def allow_keys(self, name: str, on: bool) -> Host:
        """Whether runs coordinated on this host may hold the keys their steps declare (remote_run.py): in memory there,
        for the run's length, readable meanwhile by root and by processes of the same user on that machine."""
        hosts = self.load()
        if name not in hosts:
            raise HostError(f"no host {name!r} (perception hosts list)")
        hosts[name].keys = on
        self.save(hosts)
        return hosts[name]

    def remove(self, name: str):
        hosts = self.load()
        if hosts.pop(name, None) is None:
            raise HostError(f"no host {name!r}")
        self.save(hosts)


def probe_dest(dest: str, root: str = DEFAULT_ROOT, user: str | None = None, port: int | None = None,
               identity: str | None = None) -> tuple[Capabilities, str]:
    """Reach a destination as `hosts add` would and say what it has, changing nothing there and adding nothing here."""
    check_names("probe", dest, root)
    identity = check_overrides(dest, user, port, identity)
    return probe(dest, root, [*ssh_command(shared=False), *ssh_options(user, port, identity)], make_root=False)
