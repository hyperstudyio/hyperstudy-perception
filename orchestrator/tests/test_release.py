"""A copy of the pipeline shipped with an app (scripts/export-release.sh): no .git, so RELEASE.json says which commit it
is and which files it has, and nothing asks git (on a Mac without the developer tools, git opens an install dialog)."""

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


def test_a_checkout_has_no_release_file():
    assert release.RELEASE.name in (Path(__file__).parents[2] / ".gitignore").read_text().split()
