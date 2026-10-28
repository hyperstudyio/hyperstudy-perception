"""Publishing a run as a track bundle (schema/bundle-0.1): the view the annotator and HyperStudy read.

A bundle is not a second copy of the truth. Its layers are the run's committed `bundle.<layer>@0.1` artifacts, copied
byte for byte (with the side files their nodes wrote beside them, JSON or Parquet, such as layers/faces_detail.json or
layers/gaze_heatmaps.parquet), and its manifest records their checksums, the video (from the run's `media.v1` artifact) and each layer's licence mode: the strictest
of its node's models and of every node upstream of it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from . import artifacts
from .release import release
from .scheduler import Coordinator, LICENCE_ORDER

SCHEMA_VERSION = "0.1.0"


def _version() -> str:
    """The installed package's version: what made the bundle ("unknown" in a checkout not installed; its git_commit
    says which)."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("hyperstudy-perception")
    except PackageNotFoundError:
        return "unknown"


def _git_commit() -> str | None:
    if r := release():
        return r["commit"][:7]
    out = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or None


def bundle(coord: Coordinator, run_id: str, out: Path) -> Path:
    graph = coord.graph_of(run_id)
    tasks = coord._tasks(run_id)
    not_done = [i for i, t in tasks.items() if t["state"] not in ("succeeded", "cached")]
    if not_done:
        raise SystemExit(f"run {run_id} has unfinished nodes: {', '.join(not_done)}")
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} already holds a bundle")

    own = {i: max([x.licence_mode for x in n.manifest.models] or ["commercial-ok"], key=LICENCE_ORDER.index) for i, n in graph.nodes.items()}
    effective: dict[str, str] = {}

    def licence(i: str) -> str:
        if i not in effective:
            effective[i] = max([own[i], *(licence(u) for u in graph.nodes[i].upstream)], key=LICENCE_ORDER.index)
        return effective[i]

    media, layers, refined, asset_trees, writers, sides = None, {}, {}, [], {}, {}
    for node_id, t in tasks.items():
        for name, a in json.loads(t["outputs"]).items():
            kind = artifacts.get(a["type"])
            if a["type"] == "assets.v1":
                asset_trees.append(Path(a["path"]))
            elif a["type"] == "media.v1":
                media = json.loads(Path(a["path"]).read_text())
            elif kind.layer:
                writers.setdefault(kind.layer, []).append((node_id, Path(a["path"])))
            elif kind.side_of:
                sides.setdefault((node_id, kind.side_of), []).append((kind.bundle_file, Path(a["path"])))
    for layer_name, ws in writers.items():
        # a node that refines an upstream node's layer (captions on the shots layer) supersedes it; two unrelated
        # writers of one layer are a graph error
        last = [w for w in ws if not any(o[0] in graph.descendants(w[0]) for o in ws)]   # no other writer downstream
        if len(last) != 1 or any(o is not last[0] and last[0][0] not in graph.descendants(o[0]) for o in ws):
            raise SystemExit(f"the {layer_name} layer is written by unrelated nodes: {', '.join(w[0] for w in ws)}")
        layers[layer_name] = last[0]
        refined[layer_name] = [w[0] for w in ws if w is not last[0]]   # the writers it supersedes, all upstream of it
    if media is None:
        raise SystemExit("the run has no media.v1 artifact (a decode node): the bundle cannot name its video")

    (out / "layers").mkdir(parents=True)
    entries = {}
    for name, (node_id, path) in layers.items():
        dest = out / "layers" / f"{name}.json"
        # stored layers carry no run time or timings (they would defeat the cache); the bundle gets them from the run
        # that produced the layer, which for a cache hit is the original run
        usage = json.loads(tasks[node_id]["usage"] or "{}")
        doc = json.loads(path.read_text())
        doc["producer"]["run_at"] = usage.get("finished_at")
        doc["producer"]["timings"] = {**usage.get("timings", {}), "seconds": usage.get("seconds")}
        dest.write_text(json.dumps(doc, separators=(",", ":")))
        entries[name] = {"file": f"layers/{name}.json", "sha256": artifacts.sha256_file(dest), "bytes": dest.stat().st_size,
                         "licence_mode": licence(node_id), "stage": graph.nodes[node_id].manifest.stage or graph.nodes[node_id].manifest.id}
        # its side files, byte for byte: those the node that wrote this layer wrote with it (a layer's ids are its own),
        # and those it points at that an upstream writer of the layer wrote: a node that refines a layer and keeps its
        # pointers (diarize keeps asr's raw.tokens) keeps the ids they index. The nearest such writer's, if several
        copies = dict(sides.get((node_id, name), []))
        for field, referenced in artifacts.side_pointers(doc):   # detail.file, ..., raw.<role>.file
            if referenced in copies:
                continue
            upstream = [w for w in refined[name] if referenced in dict(sides.get((w, name), []))]
            if not upstream:
                by = (f"neither {node_id} nor the {name} writers upstream of it ({', '.join(refined[name])}) wrote"
                      if refined[name] else f"{node_id} did not write")
                raise SystemExit(f"the {name} layer from {node_id} points at {referenced} ({field}), which {by}")
            nearest = next(w for w in upstream if not any(o in graph.descendants(w) for o in upstream))
            copies[referenced] = dict(sides[(nearest, name)])[referenced]
        side_files = []
        for file, src in sorted(copies.items()):
            side = out / file
            shutil.copyfile(src, side)
            side.chmod(0o644)
            side_files.append({"file": file, "sha256": artifacts.sha256_file(side), "bytes": side.stat().st_size})
        if side_files:
            entries[name]["side_files"] = side_files
    assets = []
    for tree in asset_trees:   # each node's assets tree merges into assets/; layers refer to "assets/<path>"
        for src in sorted(p for p in tree.rglob("*") if p.is_file()):
            rel = src.relative_to(tree).as_posix()
            dest = out / "assets" / rel
            if dest.exists():
                raise SystemExit(f"two nodes write assets/{rel}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            dest.chmod(0o644)
            assets.append({"file": f"assets/{rel}", "sha256": artifacts.sha256_file(dest), "bytes": dest.stat().st_size})
    manifest = {
        "format": "hyperstudy-perception-bundle", "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "video": {k: media[k] for k in ("sha256", "duration", "fps", "width", "height")} | ({"filename": media["filename"]} if media.get("filename") else {}),
        "pipeline": {"name": "hyperstudy-perception", "version": _version(), "git_commit": _git_commit(), "source": f"perception run {run_id}"},
        "layers": entries,
        "assets": assets,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out
