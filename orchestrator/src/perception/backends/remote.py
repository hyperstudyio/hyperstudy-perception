"""Remote runs: a node on another machine, reached over SSH (a lab server, `perception hosts add`).

The node runs there exactly as a native node runs here (its script in its uv environment, the same file protocol), but
the coordinator never talks to the machine itself. `submit` starts a local driver process (`python -m
perception.remote`, remote_driver.py) that copies the node's code and inputs over, starts it, streams its log back,
and brings out/ and response.json back into the attempt's folder. To the coordinator the driver is a local process
like any native node: poll, cancel (the driver stops the remote node), adoption after a coordinator restart and
collect all work unchanged.

What reaches the other machine: the files git tracks for the node, its uv project and the node SDK, and the node's
inputs. The node's environment is made there from its uv project once per lock (envs/<env_key>), and shared by every
node and version of the code that has that lock: a change to a node's code reuses it. Those inputs leave this machine, so a remote place is offered only when the graph's policy allows each kind of
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

from ..hosts import Host
from ..manifests import Manifest, Runner
from ..release import files_in, release
from .base import Capabilities
from .native import REPO, NativeBackend


def code_files(manifest: Manifest, runner: Runner, untracked: bool = False) -> list[str]:
    """What a node needs on the other machine, relative to the repository: the files git tracks in its directory, its
    uv project and the node SDK. Only tracked files: a working copy can hold images, crops and bundles that must not
    leave the machine (as launch.sh stages for whole-graph cloud runs). `untracked`: also the files there git neither
    tracks nor the repository ignores, which a node running on this machine runs with (its identity: Backend.identity)."""
    if not manifest.directory.resolve().is_relative_to(REPO):
        raise RuntimeError(f"{manifest.ref} is not in the pipeline's repository ({manifest.directory}): only files it tracks are sent")
    dirs = [manifest.directory.resolve().relative_to(REPO).as_posix(), "sdk"]
    if runner.env:
        dirs.append(runner.env)
    entry_dir = (manifest.directory / runner.entry).resolve().parent.relative_to(REPO).as_posix()
    if not (entry_dir + "/").startswith(dirs[0] + "/"):
        dirs.append(entry_dir)   # a script shared with another node (../work/node.py)
    if release():
        files = files_in(dirs)   # a copy shipped with an app: every file in it was tracked where it was made
    else:
        # ignored by the repository's own .gitignore files only: not one checkout's .git/info/exclude or a user's global
        # excludes, which would leave a file that runs here out of what is code here
        others = ["--others", "--exclude-per-directory=.gitignore"] if untracked else []
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "--cached", *others, "--", *dirs], capture_output=True, text=True)
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


# The node SDK: a path dependency of every environment (envs/*/pyproject.toml), installed editable here. On another
# machine it is left out of the environment (uv sync --no-install-package) and imported from the code's own folder
# (PYTHONPATH): an environment shared by many versions of the code then never runs another version's SDK.
SDK_PACKAGE = "perception-node"
SDK_SOURCE = "sdk/src"
# what of the SDK an environment holds: its project file (its dependencies, which the environment's lock records too)
SDK_PROJECT = "sdk/pyproject.toml"
ENV_LAYOUT = "env-1"   # how an environment's folder is made (above): a change gives every environment a new key


def env_files(files: list[str], env: str | None) -> list[str]:
    """What a node's uv environment is made from, among the code sent (code_files): its uv project (pyproject.toml,
    uv.lock) and the SDK's project file. Not the node's own code, nor the SDK's: a change to either reuses the
    environment."""
    if not env:
        return []
    return [f for f in files if f.startswith(env.rstrip("/") + "/") or f == SDK_PROJECT]


def env_key(files: list[str], env: str | None, root: Path = REPO) -> str | None:
    """The environment's folder on the other machine (envs/<key>): its content (env_files), so every node and every
    version of the code with the same lock shares one environment there."""
    made_of = env_files(files, env)
    if not made_of:
        return None
    return hashlib.sha256(f"{ENV_LAYOUT} {env} {code_hash(made_of, root)}".encode()).hexdigest()[:20]


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
        if not manifest.directory.resolve().is_relative_to(REPO):
            return "it is not in the pipeline's repository, and only files the repository tracks are sent to other machines"
        return ""

    def driver_spec(self, attempt_dir: Path) -> dict:
        """What else the driver needs (a cloud machine to start first)."""
        return {}

    def command(self, manifest: Manifest, runner: Runner, request: Path) -> list[str]:
        # the request path on the command line: how a restarted coordinator recognises the driver (native._alive)
        return [sys.executable, "-m", "perception.remote_driver", "--request", str(request)]

    def submit(self, manifest: Manifest, runner: Runner, attempt_dir: Path, env: dict[str, str]) -> dict:
        files = code_files(manifest, runner)
        caps = self.host.caps
        spec = {
            "dest": self.host.dest, "root": self.host.root, "host": self.host.name, "ssh": self.host.ssh(),
            "code": code_hash(files), "files": files,
            # its environment, by its lock: shared with every node that has the same one
            "env_key": env_key(files, runner.env), "env_files": env_files(files, runner.env),
            "node_dir": manifest.directory.resolve().relative_to(REPO).as_posix(), "entry": runner.entry, "env": runner.env,
            "runner_kind": self.runner_kind, "python": self.python, "uv": caps.tools.get("uv"), "prepare": manifest.prepare,
            # the folders of the tools it found: ssh commands get a minimal PATH (ffmpeg in /opt/homebrew/bin on a Mac)
            "path": sorted({str(Path(p).parent) for p in caps.tools.values() if p}),
            "lost_after_s": self.lost_after_s,
            **self.driver_spec(attempt_dir),
        }
        (attempt_dir / "remote.json").write_text(json.dumps(spec, indent=1))
        return super().submit(manifest, runner, attempt_dir, env)
