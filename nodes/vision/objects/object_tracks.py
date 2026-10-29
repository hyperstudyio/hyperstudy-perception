"""The track step's output (object_tracks.v1, orchestrator/src/perception/artifacts.py): everything the link step needs
from SAM 3 and DINOv3, so it runs on the CPU with no model. A folder:

    tracks.json           {"fps", "device", "config", "models", "classes", "tracks": [...]}: the session and body
                          tracks in the order the track step made them (linking.py says what a track holds), each with
                          "embedding" (its row in embeddings.npy, or null when no crop was big enough to embed),
                          "views" ([{"crop": "views/<n>_<k>.jpg", "embedding": its row in view_embeddings.npy}], the
                          embedded crops) and "context" ("context/<n>.jpg", the crop check's image); "rows" are indices
                          into masks.parquet. The rest says how they were made, for the objects layer's producer:
                          SAM 3's device, the track step's settings and models, and the name classes it used (the
                          vocabulary's with garments.classes), which the link step's wearers and identities go by.
    embeddings.npy        float32, a row per embedded track: its mean DINOv3 embedding, normalised
    view_embeddings.npy   float32, a row per embedded crop, track after track: each crop's own, normalised
    masks.parquet         the objects_masks side file's rows (object_raw.py) as they stand if every track passes the
                          crop check: the link step marks the rejected ones' rows and writes the side file
    views/<n>_<k>.jpg     the displayed crop of track obj:<n>'s k-th embedded sample, as the objects layer's identity
                          crops are saved (PIL JPEG, quality 90): the link step copies the one it picks
    context/<n>.jpg       its three outlined moments side by side (geometry.context_strip) as JPEG (haiku.jpeg): the
                          bytes the crop check sends

Arrays and images go in their own files, not in tracks.json, in the form they are used in: the embeddings stay
float32 (the linking's cosines are float32 dot products, as in the unsplit node), and the crops are the JPEG bytes the
unsplit node saved or sent. So the link step makes the same layer, side file and crops from it as the unsplit node made
from the same tracks (test_split.py). Deterministic: the same tracks give the same bytes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

TRACKS, EMBEDDINGS, VIEW_EMBEDDINGS, MASKS = "tracks.json", "embeddings.npy", "view_embeddings.npy", "masks.parquet"
# a track's fields that are files here, not values in tracks.json
FILED = ("embedding", "view_embeddings", "views", "context")


def number(track_id: str) -> str:
    """A track's file stem: its number ("obj:12" -> "12"; a colon is no file name on every system)."""
    return track_id.split(":")[1]


def write(folder: Path, tracks: list[dict], info: dict) -> None:
    """tracks.json, the embeddings and the images, for tracks as the track step leaves them: embedded (embedding,
    view_embeddings: float32 arrays or None), with views (each embedded crop as JPEG bytes) and context (the crop
    check's JPEG). info: fps, device, config, models and classes. masks.parquet is object_raw.write's, at
    folder / MASKS."""
    (folder / "views").mkdir(parents=True, exist_ok=True)
    (folder / "context").mkdir(exist_ok=True)
    means, crops, entries = [], [], []
    for t in tracks:
        n = number(t["id"])
        entry = {k: v for k, v in t.items() if k not in FILED}
        entry["embedding"] = None
        if t["embedding"] is not None:
            entry["embedding"] = len(means)
            means.append(np.asarray(t["embedding"], np.float32))
        entry["views"] = []
        for k, crop in enumerate(t["views"]):
            name = f"views/{n}_{k}.jpg"
            (folder / name).write_bytes(crop)
            entry["views"].append({"crop": name, "embedding": len(crops)})
            crops.append(np.asarray(t["view_embeddings"][k], np.float32))
        entry["context"] = f"context/{n}.jpg"
        (folder / entry["context"]).write_bytes(t["context"])
        entries.append(entry)
    width = len(means[0]) if means else 0
    np.save(folder / EMBEDDINGS, np.stack(means) if means else np.zeros((0, width), np.float32), allow_pickle=False)
    np.save(folder / VIEW_EMBEDDINGS, np.stack(crops) if crops else np.zeros((0, width), np.float32), allow_pickle=False)
    (folder / TRACKS).write_text(json.dumps({**info, "tracks": entries}, separators=(",", ":")))


def read(folder: Path) -> tuple[list[dict], dict]:
    """(tracks, info) as write() was given them, except that views are the crops' paths and context the strip's JPEG
    bytes; embedding and view_embeddings are float32 again (each a row, or rows, of its file)."""
    doc = json.loads((folder / TRACKS).read_text())
    means = np.load(folder / EMBEDDINGS, allow_pickle=False)
    crops = np.load(folder / VIEW_EMBEDDINGS, allow_pickle=False)
    tracks = []
    for entry in doc.pop("tracks"):
        t = {k: v for k, v in entry.items() if k not in FILED}
        t["embedding"] = None if entry["embedding"] is None else means[entry["embedding"]]
        rows = [v["embedding"] for v in entry["views"]]
        t["view_embeddings"] = crops[rows] if t["embedding"] is not None else None
        t["views"] = [folder / v["crop"] for v in entry["views"]]
        t["context"] = (folder / entry["context"]).read_bytes()
        tracks.append(t)
    return tracks, doc


def copy_crop(view: Path, dest: Path) -> None:
    """An identity's crop: the view's JPEG as the track step saved it, byte for byte."""
    shutil.copyfile(view, dest)
