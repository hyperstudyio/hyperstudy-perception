"""Model weights a node fetches from a URL (not the Hugging Face hub): one shared cache, keyed by the file's sha256,
so a file is fetched once for every node and environment that uses it, and is checked before it is used.

    from perception_node.weights import fetch
    onnx = fetch("https://download.openmmlab.com/.../rtmw-....zip", sha256="bd03...", member=".onnx")

`plan` (the coordinator) looks for the same path to say whether a run would download it: cache_dir() is mirrored in
perception/plan.py (weights_cached), and a test keeps the two the same. Standard library only, like the rest of the SDK.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

CHUNK = 1 << 20


def cache_dir() -> Path:
    """PERCEPTION_WEIGHTS, else ~/.cache/hyperstudy-perception/weights."""
    return Path(os.environ.get("PERCEPTION_WEIGHTS") or Path.home() / ".cache" / "hyperstudy-perception" / "weights")


def cached(sha256: str) -> Path | None:
    """The cached file with this sha256, or None."""
    folder = cache_dir() / sha256
    files = [p for p in folder.iterdir() if p.is_file()] if folder.is_dir() else []
    return files[0] if len(files) == 1 else None


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, sha256: str, member: str | None = None) -> Path:
    """The file with this sha256, from the cache or else from `url`. A zip's file is the one member whose name ends in
    `member` (".onnx"). A file whose checksum differs is not kept: the error names both, and nothing is cached."""
    if hit := cached(sha256):
        return hit
    folder = cache_dir() / sha256
    cache_dir().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir(), prefix=".fetch-") as tmp:   # same disk: the move into place is a rename
        download = Path(tmp) / "download"
        with urllib.request.urlopen(url, timeout=60) as response, open(download, "wb") as out:
            shutil.copyfileobj(response, out, CHUNK)
        if member:
            with zipfile.ZipFile(download) as z:
                names = [n for n in z.namelist() if n.endswith(member) and not n.endswith("/")]
                if len(names) != 1:
                    raise ValueError(f"{url}: {len(names)} files end in {member!r} ({names[:5]}), not one")
                name = Path(names[0]).name
                file = Path(tmp) / name
                with z.open(names[0]) as src, open(file, "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
        else:
            name = Path(urllib.request.urlparse(url).path).name or "weights"
            file = download.rename(Path(tmp) / name)
        got = digest(file)
        if got != sha256:
            raise ValueError(f"{url}: {name} has sha256 {got}, not {sha256}")
        folder.mkdir(parents=True, exist_ok=True)
        return Path(shutil.move(str(file), folder / name))
