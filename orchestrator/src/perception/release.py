"""Where the pipeline's files are: a checkout of this repository, or the package installed from PyPI, which carries
them in perception/_pipeline (scripts/build-dist.sh: nodes, environments, the node SDK, graphs, schemas and cloud
scripts). The package has no .git, so what git says in a checkout (which commit this is, which files are tracked) is in
its RELEASE.json instead. Read only where the file is: a checkout never has one (.gitignore), and the package never
asks git (on a Mac without the developer tools, running git opens a dialog offering to install them)."""

import hashlib
import json
import os
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGED = Path(__file__).resolve().parent / "_pipeline"
REPO = PACKAGED if PACKAGED.is_dir() else Path(__file__).resolve().parents[3]
RELEASE = REPO / "RELEASE.json"


def home() -> Path:
    """The store: artifacts, run state and (for the package) node environments."""
    return Path(os.environ.get("PERCEPTION_HOME", Path.home() / ".hyperstudy-perception"))


@cache
def release() -> dict | None:
    """{"commit": the full hash, "files": every file it carries, relative to REPO}; None in a checkout."""
    if not RELEASE.exists():
        return None
    data = json.loads(RELEASE.read_text())
    if not isinstance(data.get("commit"), str) or not isinstance(data.get("files"), list):
        raise RuntimeError(f"{RELEASE} is not a release file (commit and files)")
    return data


def files_in(dirs: list[str]) -> list[str]:
    """The package's files under these directories, as `git ls-files -- dirs` lists them in a checkout."""
    return sorted(f for f in release()["files"] if any(f.startswith(d.rstrip("/") + "/") for d in dirs))


def env_dir(env: str) -> Path | None:
    """Where a node environment (envs/NAME) is made: in a checkout, uv's own place (envs/NAME/.venv, None here); for
    the package, under the store, by the lock, the commit and the package version (the package's folder may not be
    writable, and an upgrade replaces it)."""
    if release() is None:
        return None
    lock = REPO / env / "uv.lock"
    h = hashlib.sha256(lock.read_bytes() if lock.exists() else b"no lock")
    # the commit too: the SDK is installed editable from this copy, so two builds of one version (PyPI's and a local
    # one) must not share an environment and point its SDK at each other's files
    h.update(release()["commit"].encode())
    try:
        h.update(version("hyperstudy-perception").encode())
    except PackageNotFoundError:   # the files without the package's metadata (a copy run in place)
        pass
    return home() / "envs" / f"{Path(env).name}-{h.hexdigest()[:12]}"


def env_vars(env: str) -> dict[str, str]:
    """What uv is told so a node environment is made where env_dir says (nothing in a checkout)."""
    d = env_dir(env)
    return {} if d is None else {"UV_PROJECT_ENVIRONMENT": str(d)}
