"""Other machines nodes can run on: lab servers reached over SSH (`perception hosts add NAME DEST`).

A host is probed when it is added (and by `hosts check`): its platform, cores, memory, GPU and tools are recorded in
$PERCEPTION_HOME/hosts.json, so planning never waits on the network. SSH runs in batch mode with the user's own keys
and ~/.ssh/config: nothing here asks for or stores a password. Connections are shared (ControlMaster), so the many short
commands a remote node needs cost one handshake.

On the host everything lives under its root (default ~/.cache/hyperstudy-perception), which is private: created 0700, and
refused if someone else owns it, so the code, inputs and markers kept there can be trusted and are not readable by others:
    code/<hash>/         the files git tracks for the node, its uv project and the node SDK, by content
    objects/<sha256>/data   inputs, by content: the video is sent once per host (each attempt links it under its name)
    work/<run>/<node>/<attempt>/   the request, out/, response.json, log.txt and pid of one attempt: removed after it,
                         or by the host's watchdog when no driver has asked about it for a while
`perception hosts clean NAME` removes the root's inputs and code (what stays on the host between runs), unless a node
is running there. A host with a node running there is not removed either: that node could no longer be stopped.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .backends.base import Capabilities

NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")
# an ssh destination: user@host, a host, or a ~/.ssh/config alias; never an option
DEST = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@%+:\[\]-]*$")
# a path on the host: plain characters, no `..`; relative ones are under the host's home
ROOT = re.compile(r"^[A-Za-z0-9_./~-]+$")
DEFAULT_ROOT = ".cache/hyperstudy-perception"
RESERVED = {"native", "self", "cloud", "local"}

# What the host has, as key=value lines. POSIX sh: Linux servers and Macs, no Python needed. Tools are looked for on the
# PATH a login shell would add too (~/.local/bin is where uv installs itself), since ssh commands get a minimal PATH.
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
  echo "vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)"
fi
"""


class HostError(RuntimeError):
    pass


@dataclass
class Host:
    name: str
    dest: str
    root: str                  # absolute, on the host
    caps: Capabilities
    probed_at: float

    def to_json(self) -> dict:
        d = asdict(self)
        d["caps"]["accelerators"] = sorted(self.caps.accelerators)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Host":
        caps = Capabilities(**{**d["caps"], "accelerators": set(d["caps"]["accelerators"])})
        return cls(d["name"], d["dest"], d["root"], caps, d["probed_at"])


def ssh_command() -> list[str]:
    """ssh with the options every remote command uses. PERCEPTION_SSH replaces the ssh program (the tests use a
    stand-in that runs commands on this machine)."""
    control = Path(f"/tmp/perception-ssh-{os.getuid()}")   # short: a socket path must fit in 104 bytes on macOS
    control.mkdir(mode=0o700, exist_ok=True)
    st = control.stat()
    if st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise HostError(f"{control} must be a directory only you can use")
    return [os.environ.get("PERCEPTION_SSH", "ssh"), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
            "-o", "ControlMaster=auto", "-o", f"ControlPath={control}/%C", "-o", "ControlPersist=120"]


def check_names(name: str, dest: str, root: str):
    if not NAME.match(name) or name in RESERVED:
        raise HostError(f"a host's name is lower-case letters, digits and -, and not {', '.join(sorted(RESERVED))}: {name!r}")
    if not DEST.match(dest):
        raise HostError(f"not an ssh destination: {dest!r}")
    parts = [x for x in root.removeprefix("~/").split("/") if x not in ("", ".")]
    if not ROOT.match(root) or ".." in parts or "~" in root.removeprefix("~/") or not parts:
        raise HostError(f"a host's root is a plain path below / or ~ (its own folder: `hosts clean` empties it): {root!r}")


# The root is private: created 0700, and refused if someone else owns it. Only the user can then write the code, inputs and
# completion markers kept there, so they are trusted across runs; and others on a shared server cannot read the inputs.
PRIVATE_ROOT = r"""
umask 077
mkdir -p "$root"
if [ "$(ls -ldn "$root" | awk '{print $3}')" != "$(id -u)" ]; then echo "$root belongs to someone else" >&2; exit 3; fi
chmod 700 "$root"
"""


def private_root_script(root: str) -> str:
    """sh that makes `root` (absolute, or under the home) the user's own private folder, or fails."""
    import shlex
    path = shlex.quote(root) if root.startswith("/") else '"$HOME"/' + shlex.quote(root.removeprefix("~/"))
    return f"root={path}\n" + PRIVATE_ROOT


def probe(dest: str, root: str = DEFAULT_ROOT) -> tuple[Capabilities, str]:
    """What the host has, and its root as an absolute path (made private there, or refused: PRIVATE_ROOT)."""
    out = subprocess.run([*ssh_command(), dest, "sh -s"], input=private_root_script(root) + PROBE, capture_output=True,
                         text=True, timeout=120)
    if out.returncode:
        raise HostError(f"could not reach {dest} (ssh exit {out.returncode}): {out.stderr.strip()[-500:]}")
    facts = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)
    try:
        system = {"Linux": "linux", "Darwin": "darwin"}[facts["system"]]
        arch = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}[facts["machine"]]
        memory = int(facts["memory_kb"]) / 2**20
        cores = int(facts["cores"])
    except (KeyError, ValueError) as e:
        raise HostError(f"{dest} answered, but not as expected ({e}): {out.stdout[-500:]}")
    accelerators, vram = {"cpu"}, 0.0
    if facts.get("vram_mib", "").strip().isdigit():
        accelerators.add("cuda")
        vram = int(facts["vram_mib"]) / 1024
    if (system, arch) == ("darwin", "arm64"):
        accelerators.add("mps")
        vram = vram or memory   # unified memory
    caps = Capabilities(platform=f"{system}/{arch}", cpu_cores=cores, memory_gb=round(memory, 1), accelerators=accelerators,
                        vram_gb=round(vram, 1), tools={t: facts.get(f"tool_{t}") or None for t in ("uv", "ffmpeg", "ffprobe")})
    home = facts["home"]
    absolute = root if root.startswith("/") else f"{home}/{root.removeprefix('~/')}"
    return caps, absolute.rstrip("/")


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

    def add(self, name: str, dest: str, root: str = DEFAULT_ROOT) -> Host:
        check_names(name, dest, root)
        caps, absolute = probe(dest, root)
        host = Host(name, dest, absolute, caps, time.time())
        hosts = self.load()
        hosts[name] = host
        self.save(hosts)
        return host

    def check(self, name: str) -> Host:
        hosts = self.load()
        if name not in hosts:
            raise HostError(f"no host {name!r} (perception hosts list)")
        old = hosts[name]
        caps, _ = probe(old.dest, old.root)
        hosts[name] = Host(name, old.dest, old.root, caps, time.time())
        self.save(hosts)
        return hosts[name]

    def remove(self, name: str):
        hosts = self.load()
        if hosts.pop(name, None) is None:
            raise HostError(f"no host {name!r}")
        self.save(hosts)
