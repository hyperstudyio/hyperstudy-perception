"""A copy of the pipeline made to ship with an app (scripts/export-release.sh): it has no .git, so what git says in a
checkout (which commit this is, which files are tracked) is in its RELEASE.json instead. Read only where the file is:
a checkout never has one (.gitignore), and a copy never asks git (on a Mac without the developer tools, running git
opens a dialog offering to install them)."""

import json
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RELEASE = REPO / "RELEASE.json"


@cache
def release() -> dict | None:
    """{"commit": the full hash, "files": every file of the copy, relative to it}; None in a checkout."""
    if not RELEASE.exists():
        return None
    data = json.loads(RELEASE.read_text())
    if not isinstance(data.get("commit"), str) or not isinstance(data.get("files"), list):
        raise RuntimeError(f"{RELEASE} is not a release file (commit and files)")
    return data


def files_in(dirs: list[str]) -> list[str]:
    """The copy's files under these directories, as `git ls-files -- dirs` lists them in a checkout."""
    return sorted(f for f in release()["files"] if any(f.startswith(d.rstrip("/") + "/") for d in dirs))
