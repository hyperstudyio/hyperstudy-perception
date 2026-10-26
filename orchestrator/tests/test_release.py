"""The package PyPI publishes (scripts/build-dist.sh) carries the pipeline's files with no .git: RELEASE.json says which
commit they are and lists them, nothing asks git (on a Mac without the developer tools, git opens an install dialog),
and node environments are made in the store, not in the package's folder."""

import json
import subprocess
from pathlib import Path

import pytest

import perception.backends.remote as remote
import perception.export as export
import perception.release as release
from perception.manifests import Registry

NODES = Path(__file__).parent / "nodes"


@pytest.fixture
def copy(tmp_path, monkeypatch):
    """This checkout, seen as a copy: RELEASE.json lists `files`, and running git fails the test."""
    def made(files):
        (tmp_path / "RELEASE.json").write_text(json.dumps({"commit": "0123456789abcdef", "files": files}))
        monkeypatch.setattr(release, "RELEASE", tmp_path / "RELEASE.json")
        release.release.cache_clear()

        def no_git(cmd, *a, **k):
            raise AssertionError(f"a copy ran {cmd}")
        monkeypatch.setattr(subprocess, "run", no_git)
    yield made
    release.release.cache_clear()


def test_a_copy_sends_the_node_files_it_lists(copy):
    m = Registry.load(NODES).get("test.gpu@1.0.0")   # its entry is ../work/node.py
    runner = m.runners[0]
    tracked = remote.code_files(m, runner)            # git, in this checkout
    copy(tracked + ["orchestrator/tests/nodes/gpu-other/node.py", "docs/notes.md"])
    assert remote.code_files(m, runner) == tracked


def test_a_copys_bundles_name_its_commit(copy):
    copy([])
    assert export._git_commit() == "0123456"


def test_the_packages_node_environments_are_made_in_the_store_by_lock_and_version(copy, tmp_path, monkeypatch):
    monkeypatch.setenv("PERCEPTION_HOME", str(tmp_path / "home"))
    assert release.env_vars("envs/core") == {}   # a checkout: uv's own place, envs/core/.venv
    copy([])
    core, faces = release.env_dir("envs/core"), release.env_dir("envs/faces")
    assert core.parent == tmp_path / "home" / "envs" and core.name.startswith("core-") and faces.name.startswith("faces-")
    assert release.env_vars("envs/core") == {"UV_PROJECT_ENVIRONMENT": str(core)}
    monkeypatch.setattr(release, "version", lambda _: "9.9.9")   # another version of the package: another environment
    assert release.env_dir("envs/core") != core


def test_a_graph_is_a_file_or_one_of_the_pipelines_by_name(tmp_path, monkeypatch):
    from perception.cli import graph_path, graphs
    monkeypatch.chdir(tmp_path)
    assert graph_path("full.yaml") == release.REPO / "graphs" / "full.yaml"
    (tmp_path / "full.yaml").write_text("mine")   # a file of the user's comes first
    assert graph_path("full.yaml") == Path("full.yaml")
    assert graph_path("sub/missing.yaml") == Path("sub/missing.yaml")   # a path is never looked up by name
    full = next(g for g in graphs() if g["file"] == "full.yaml")
    assert full["pipeline"] and full["description"].startswith("Everything the pipeline can do")


def test_a_checkout_has_no_release_file():
    assert release.RELEASE.name in (Path(__file__).parents[2] / ".gitignore").read_text().split()
