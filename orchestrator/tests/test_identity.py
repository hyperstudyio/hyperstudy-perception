"""A result's identity (Backend.identity, the cache key's runner): one rule for every backend. The code that runs, the
runner, its accelerator and the platform, never the machine or the backend; against a small git repository of its
own, so the code, lock and SDK can change without touching this checkout."""

import subprocess
import time
from pathlib import Path

import pytest

import perception.backends.native as native
import perception.backends.remote as remote
from perception.backends.base import Capabilities
from perception.backends.cloud import CloudBackend, RunPodBackend
from perception.backends.native import NativeBackend, SelfBackend
from perception.backends.remote import SshBackend
from perception.hosts import Host
from perception.manifests import Registry

MANIFEST = """id: test.id
version: 1.0.0
inputs: {a: json.v1}
outputs: {out: json.v1}
runners:
  - {kind: native, entry: node.py, env: envs/e, accelerator: cuda, platforms: [linux/amd64, darwin/arm64]}
  - {kind: native, entry: node.py, env: envs/e, accelerator: cpu, platforms: [linux/amd64, darwin/arm64]}
"""


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository with one node (nodes/x), its environment (envs/e) and the SDK, all tracked, standing in for this one."""
    root = tmp_path.resolve() / "repo"   # as REPO is: a manifest's directory is resolved
    for path, text in {"nodes/x/manifest.yaml": MANIFEST, "nodes/x/node.py": "print('x')\n",
                       "envs/e/pyproject.toml": "[project]\nname = 'e'\n", "envs/e/uv.lock": "version = 1\n",
                       "sdk/src/perception_node/__init__.py": "PROTOCOL = 1\n", ".gitignore": "__pycache__/\n"}.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    monkeypatch.setattr(remote, "REPO", root)
    monkeypatch.setattr(remote.code_hash, "__defaults__", (root,))   # its root defaults to REPO as it was imported
    monkeypatch.setattr(native, "REPO", root)
    return root


def node(repo: Path):
    m = Registry.load(repo / "nodes").get("test.id@1.0.0")
    return m, {r.accelerator: r for r in m.runners}


def caps(platform="linux/amd64") -> Capabilities:
    return Capabilities(platform=platform, cpu_cores=16, memory_gb=64, accelerators={"cpu", "cuda"}, vram_gb=24,
                        tools={"uv": "/usr/bin/uv", "ffmpeg": None, "ffprobe": None})


def host(name: str, platform="linux/amd64") -> SshBackend:
    return SshBackend(Host(name, f"me@{name}", f"/home/me/.cache/{name}", caps(platform), time.time()))


def here(platform="linux/amd64") -> NativeBackend:
    b = NativeBackend()
    b._caps = caps(platform)   # this machine, as probed
    return b


def test_one_machine_under_two_names_and_a_pod_give_one_identity(repo):
    m, r = node(repo)
    ids = {b.identity(m, r["cuda"]) for b in (host("lab"), host("lab-again"), CloudBackend(), RunPodBackend(), here())}
    assert len(ids) == 1, ids
    (one,) = ids
    assert one.startswith("linux/amd64/cuda:") and "lab" not in one


def test_this_machine_and_another_agree_when_the_code_does(repo):
    m, r = node(repo)
    assert here("darwin/arm64").identity(m, r["cpu"]) == host("mac-mini", "darwin/arm64").identity(m, r["cpu"])


@pytest.mark.parametrize("change", [
    "nodes/x/node.py",               # the node's code
    "envs/e/uv.lock",                # its environment's lock
    "envs/e/pyproject.toml",         # its uv project
    "sdk/src/perception_node/__init__.py",   # the node SDK (installed editable, outside the lock)
])
@pytest.mark.parametrize("backend", [lambda: here(), lambda: host("lab")])
def test_changing_what_runs_changes_the_identity(repo, change, backend):
    m, r = node(repo)
    b = backend()
    before = b.identity(m, r["cuda"])
    with open(repo / change, "a") as f:
        f.write("# changed\n")
    assert b.identity(m, r["cuda"]) != before


def test_the_accelerator_and_the_platform_are_part_of_it(repo):
    m, r = node(repo)
    lab = host("lab")
    assert lab.identity(m, r["cuda"]) != lab.identity(m, r["cpu"])
    assert lab.identity(m, r["cpu"]) != host("mac-mini", "darwin/arm64").identity(m, r["cpu"])
    assert here("linux/amd64").identity(m, r["cpu"]) != here("darwin/arm64").identity(m, r["cpu"])


def test_a_file_git_does_not_track_counts_here_but_is_never_sent(repo):
    """A module being written beside the node runs here, not on another machine (only tracked files are sent): this
    machine's result is then not the other machine's, and editing it makes this machine's again."""
    m, r = node(repo)
    lab, mine = host("lab"), here()
    assert mine.identity(m, r["cuda"]) == lab.identity(m, r["cuda"])
    sent = lab.identity(m, r["cuda"])
    (repo / "nodes" / "x" / "helper.py").write_text("SCALE = 1\n")
    with_helper = mine.identity(m, r["cuda"])
    assert with_helper != sent and lab.identity(m, r["cuda"]) == sent
    (repo / "nodes" / "x" / "helper.py").write_text("SCALE = 2\n")
    assert mine.identity(m, r["cuda"]) != with_helper
    # what git ignores (bytecode, an environment) is neither run as code nor sent
    (repo / "nodes" / "x" / "__pycache__").mkdir()
    (repo / "nodes" / "x" / "__pycache__" / "node.cpython-312.pyc").write_bytes(b"\0")
    (repo / "nodes" / "x" / "helper.py").unlink()
    assert mine.identity(m, r["cuda"]) == sent


def test_only_the_repositorys_own_ignore_rules_leave_a_file_out(repo):
    """A file left out only by this checkout's .git/info/exclude (or the user's global excludes) still runs here, so it
    still counts: what is code here must not depend on one person's git settings."""
    m, r = node(repo)
    mine = here()
    (repo / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (repo / ".git" / "info" / "exclude").write_text("helper.py\n")
    (repo / "nodes" / "x" / "helper.py").write_text("SCALE = 1\n")
    before = mine.identity(m, r["cuda"])
    (repo / "nodes" / "x" / "helper.py").write_text("SCALE = 2\n")
    assert mine.identity(m, r["cuda"]) != before


def test_a_node_outside_the_repository_has_an_identity_here_from_its_files(repo, tmp_path):
    outside = tmp_path / "mine" / "x"
    outside.mkdir(parents=True)
    (outside / "manifest.yaml").write_text(MANIFEST)
    (outside / "node.py").write_text("print('mine')\n")
    m = Registry.load(tmp_path / "mine").get("test.id@1.0.0")
    cuda = next(r for r in m.runners if r.accelerator == "cuda")
    before = here().identity(m, cuda)
    assert before != host("lab").identity(m, cuda)   # another machine is never sent it: no result is shared
    (outside / "node.py").write_text("print('mine, changed')\n")
    assert here().identity(m, cuda) != before


def test_a_shared_script_outside_the_repository_counts_with_its_folder(repo, tmp_path):
    """As code_files takes the whole folder of a script a node shares (../shared/node.py): a module beside it, which it
    imports, is part of the code too."""
    root = tmp_path / "mine"
    (root / "x").mkdir(parents=True)
    (root / "shared").mkdir()
    (root / "x" / "manifest.yaml").write_text(MANIFEST.replace("entry: node.py", "entry: ../shared/node.py"))
    (root / "shared" / "node.py").write_text("from util import f\n")
    (root / "shared" / "util.py").write_text("def f(): return 1\n")
    m = Registry.load(root / "x").get("test.id@1.0.0")
    cuda = next(r for r in m.runners if r.accelerator == "cuda")
    before = here().identity(m, cuda)
    (root / "shared" / "util.py").write_text("def f(): return 2\n")
    assert here().identity(m, cuda) != before


def test_a_folder_a_runner_shares_counts_where_git_cannot_list_it_too(repo):
    """Runner.shares is the node's code with git (code_files) and without it (unlisted_code_hash: a launch.sh staged
    copy, a checkout with no .git): a change in the shared folder is a new identity either way."""
    import shutil
    (repo / "nodes" / "lib").mkdir()
    (repo / "nodes" / "lib" / "util.py").write_text("def f(): return 1\n")
    (repo / "nodes" / "x" / "manifest.yaml").write_text(MANIFEST.replace("env: envs/e,", "env: envs/e, shares: [nodes/lib],"))
    shutil.rmtree(repo / ".git")
    m, r = node(repo)
    with pytest.raises(RuntimeError):
        remote.code_files(m, r["cuda"])   # git cannot list it: the identity is the unlisted hash
    before = here().identity(m, r["cuda"])
    (repo / "nodes" / "lib" / "util.py").write_text("def f(): return 2\n")
    assert here().identity(m, r["cuda"]) != before

def test_no_identity_is_one_the_rule_before_gave(repo):
    """Those began with the backend's kind (native:, self:, ssh-NAME:, cloud:, runpod:); every one now begins with the
    platform, so no cache key made under the old rule is read as one made under this one."""
    m, r = node(repo)
    for b in (here(), SelfBackend(), host("lab"), CloudBackend(), RunPodBackend()):
        identity = b.identity(m, r["cuda"])
        assert not identity.startswith(("native:", "self:", "ssh-", "cloud:", "runpod:")) and "/" in identity.split(":")[0]


def test_the_same_code_in_a_copy_without_git_has_the_checkouts_identity(repo, monkeypatch):
    """The package (no .git) lists its files in RELEASE.json: a result made from a checkout and one made by the app's
    copy of the same commit agree."""
    import json
    import perception.release as release
    m, r = node(repo)
    checkout = here().identity(m, r["cuda"])
    files = subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    (repo / "RELEASE.json").write_text(json.dumps({"commit": "0" * 40, "files": files}))
    monkeypatch.setattr(release, "RELEASE", repo / "RELEASE.json")
    release.release.cache_clear()
    try:
        assert here().identity(m, r["cuda"]) == checkout
    finally:
        release.release.cache_clear()

