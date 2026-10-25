"""The RunPod image's tag, from everything that goes into it, and its build context:

    python3 cloud/runpod/image/tag.py                  prints the tag (12 hex digits)
    python3 cloud/runpod/image/tag.py --context DIR    also copies exactly those files into DIR (what is built)

The same files give the same tag: RunPodBackend.image_tag must equal it (a test checks), so a change to an environment's
lock or to the image needs a new build (.github/workflows/runpod-image.yml) and that tag. Standard library only."""

import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def envs() -> list[str]:
    return [line.strip() for line in (HERE / "envs.txt").read_text().splitlines() if line.strip()]


def files() -> dict[str, Path]:
    """Each file in the context, by its path there: the image's own files at its root, the repository's at theirs."""
    out = {name: HERE / name for name in ("Dockerfile", "start.sh", "envs.txt")}
    out["sdk/pyproject.toml"] = REPO / "sdk" / "pyproject.toml"
    for env in envs():
        for name in ("pyproject.toml", "uv.lock"):
            out[f"{env}/{name}"] = REPO / env / name
    return out


def tag() -> str:
    h = hashlib.sha256()
    for name, path in sorted(files().items()):
        h.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()[:12]


def stage(dest: Path):
    for name, path in files().items():
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest / name)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--context":
        stage(Path(sys.argv[2]))
    elif len(sys.argv) != 1:
        sys.exit(__doc__)
    print(tag())
