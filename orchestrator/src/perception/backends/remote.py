"""Remote runs: a node on another machine, reached over SSH (a lab server, `perception hosts add`).

The node runs there exactly as a native node runs here (its script in its uv environment, the same file protocol), but
the coordinator never talks to the machine itself. `submit` starts a local driver process (`python -m
perception.remote`, remote_driver.py) that copies the node's code and inputs over, starts it, streams its log back,
and brings out/ and response.json back into the attempt's folder. To the coordinator the driver is a local process
like any native node: poll, cancel (the driver stops the remote node), adoption after a coordinator restart and
collect all work unchanged.

What reaches the other machine: the files git tracks for the node, its uv project and the node SDK, and the node's
inputs. Those inputs leave this machine, so a remote place is offered only when the graph's policy allows each kind of
data in them (policy.allow_upload), and runs only with the run's consent (--allow-uploads), as a cloud API's node does.
Nodes that send data out themselves (to an API) or need secrets are not run remotely: their data would go to two
places, and secrets stay here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from ..hosts import Host, ssh_command
from ..manifests import Manifest, Runner
from .base import Capabilities
from .native import REPO, NativeBackend


def code_files(manifest: Manifest, runner: Runner) -> list[str]:
    """What a node needs on the other machine, relative to the repository: the files git tracks in its directory, its
    uv project and the node SDK. Only tracked files: a working copy can hold images, crops and bundles that must not
    leave the machine (as launch.sh stages for whole-graph cloud runs)."""
    dirs = [manifest.directory.resolve().relative_to(REPO).as_posix(), "sdk"]
    if runner.env:
        dirs.append(runner.env)
    entry_dir = (manifest.directory / runner.entry).resolve().parent.relative_to(REPO).as_posix()
    if not (entry_dir + "/").startswith(dirs[0] + "/"):
        dirs.append(entry_dir)   # a script shared with another node (../work/node.py)
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "--", *dirs], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"{REPO} is not a git checkout: only files git tracks are sent to other machines ({out.stderr.strip()})")
    files = sorted(f for f in out.stdout.split("\0") if f)
    refuse_links(files)
    if not any(f.startswith(dirs[0] + "/") for f in files):
        raise RuntimeError(f"git tracks no files in {dirs[0]}: commit the node before running it on another machine")
    return files


def refuse_links(files: list[str], root: Path = REPO):
    """A link git tracks would be followed when the code is copied and hashed, and what it points at (anything on this
    machine) would go to the host: only content git tracks leaves the machine, so a node with one is not sent."""
    links = [f for f in files if (root / f).is_symlink()]
    if links:
        raise RuntimeError(f"git tracks links among the code to send ({', '.join(links[:5])}): links are not sent to other machines")


def code_hash(files: list[str], root: Path = REPO) -> str:
    """The code's content, as the name of its folder on the other machine: the same code is sent once."""
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode() + b"\0" + hashlib.sha256((root / f).read_bytes()).digest())
    return h.hexdigest()[:20]


class SshBackend(NativeBackend):
    runner_kind = "native"
    remote = True

    def __init__(self, host: Host, runner_kind: str = "native", python: str | None = None, lost_after_s: float = 600):
        """`runner_kind="self"` with `python` (an interpreter on the other machine that has the node SDK) is for tests."""
        super().__init__()
        self.host = host
        self.kind = f"ssh-{host.name}"
        self.title = f"{host.name} (SSH)"
        self.runner_kind = runner_kind
        self.python = python
        self.lost_after_s = lost_after_s

    @property
    def machine(self) -> str | None:
        return self.kind

    def probe_capabilities(self) -> Capabilities:
        return self.host.caps   # as recorded by `hosts add` / `hosts check`: planning never waits on the network

    def missing_tools(self, runner: Runner) -> list[str]:
        if self.runner_kind == "self":
            return []
        return [] if self.host.caps.tools.get("uv") else [f"uv (on {self.host.name})"]

    def remote_refusal(self, manifest: Manifest) -> str:
        if manifest.privacy.external_upload:
            return f"it sends {', '.join(manifest.privacy.external_upload)} to {manifest.service or 'a service'} itself, so it runs on this machine"
        if manifest.secrets:
            return f"it needs {', '.join(manifest.secrets)}, and secrets are not sent to other machines"
        return ""

    def identity(self, manifest: Manifest, runner: Runner) -> str:
        """Exactly the code that runs there (what code_files sends: the node's tracked files, a shared entry script, its
        uv project with its lock, the SDK), the accelerator and the host's platform."""
        try:
            code = code_hash(code_files(manifest, runner))
        except RuntimeError as e:   # not a git checkout: nothing could be sent, and submit says so; no cache entry is shared
            code = f"unsendable {e}"
        h = hashlib.sha256(f"{code} {runner.accelerator} {self.host.caps.platform}".encode())
        return f"{self.kind}:{h.hexdigest()[:16]}"

    def command(self, manifest: Manifest, runner: Runner, request: Path) -> list[str]:
        # the request path on the command line: how a restarted coordinator recognises the driver (native._alive)
        return [sys.executable, "-m", "perception.remote_driver", "--request", str(request)]

    def submit(self, manifest: Manifest, runner: Runner, attempt_dir: Path, env: dict[str, str]) -> dict:
        files = code_files(manifest, runner)
        caps = self.host.caps
        spec = {
            "dest": self.host.dest, "root": self.host.root, "host": self.host.name, "ssh": ssh_command(),
            "code": code_hash(files), "files": files,
            "node_dir": manifest.directory.resolve().relative_to(REPO).as_posix(), "entry": runner.entry, "env": runner.env,
            "runner_kind": self.runner_kind, "python": self.python, "uv": caps.tools.get("uv"),
            # the folders of the tools it found: ssh commands get a minimal PATH (ffmpeg in /opt/homebrew/bin on a Mac)
            "path": sorted({str(Path(p).parent) for p in caps.tools.values() if p}),
            "lost_after_s": self.lost_after_s,
        }
        (attempt_dir / "remote.json").write_text(json.dumps(spec, indent=1))
        return super().submit(manifest, runner, attempt_dir, env)
