"""Artifact types: what flows between nodes, and how a node's output is checked before it is published.

The bundle's layer schemas are artifact types (`bundle.<layer>@0.1`): a layer a node writes is exactly the layer the
annotator and HyperStudy read, and so are the side files a layer points to (SIDE_FILES: `bundle.faces_detail@0.1` is
the faces layer's layers/faces_detail.json, `bundle.gaze_heatmaps@0.1` the gaze layer's layers/gaze_heatmaps.parquet).
A side file is JSON or Parquet; schema/README.md says how a step adds one. A few internal types carry what bundles do
not (media metadata, audio, keyframes).
A type is a file or a tree (a directory); its validator returns a list of problems (empty: valid).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Callable, Literal

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .release import REPO

SCHEMA_DIR = REPO / "schema"
BUNDLE_SCHEMA = SCHEMA_DIR / "bundle-0.1.schema.json"
BUNDLE_LAYERS = ("shots", "transcript", "speakers", "faces", "bodies", "hands", "gestures", "facing", "gaze", "objects",
                 "sounds", "text", "prosody", "waveform")


@dataclass(frozen=True)
class ArtifactType:
    id: str
    shape: Literal["file", "tree", "source"]   # source: an input that stays where it is (the video), never copied
    validate: Callable[[Path], list[str]]
    layer: str | None = None                    # the bundle layer this type is, if any
    # a side file of a bundle layer (heavy per-sample arrays the layer points to): the layer, and the file's path in a
    # bundle. Export publishes it with that layer when the node that wrote the layer wrote it too
    side_of: str | None = None
    bundle_file: str | None = None
    # the kinds of data it holds, in policy.allow_upload's terms: what leaves the machine when a node that reads it runs
    # elsewhere (a lab server, a cloud machine). None: anything (a JSON or text file could hold any of them)
    kinds: tuple[str, ...] | None = None


@cache
def _bundle_schema():
    schema = json.loads(BUNDLE_SCHEMA.read_text())
    return schema, Registry().with_resource(schema["$id"], Resource.from_contents(schema))


def _json_schema(schema: dict) -> Callable[[Path], list[str]]:
    validator = Draft202012Validator(schema)

    def check(path: Path) -> list[str]:
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            return [f"not JSON: {e}"]
        return [f"{e.json_path}: {e.message[:200]}" for e in list(validator.iter_errors(doc))[:20]]
    return check


def _layer(name: str) -> Callable[[Path], list[str]]:
    def check(path: Path) -> list[str]:
        schema, registry = _bundle_schema()
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            return [f"not JSON: {e}"]
        validator = Draft202012Validator({"$ref": f"{schema['$id']}#/$defs/layer/{name}"}, registry=registry)
        problems = [f"{e.json_path}: {e.message[:200]}" for e in list(validator.iter_errors(doc))[:20]]
        if doc.get("layer") != name:
            problems.append(f"layer is {doc.get('layer')!r}, not {name!r}")
        return problems or _raw_pointer_problems(name, doc)
    return check


def _side(name: str) -> Callable[[Path], list[str]]:
    def check(path: Path) -> list[str]:
        schema, registry = _bundle_schema()
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            return [f"not JSON: {e}"]
        validator = Draft202012Validator({"$ref": f"{schema['$id']}#/$defs/side/{name}"}, registry=registry)
        problems = [f"{e.json_path}: {e.message[:200]}" for e in list(validator.iter_errors(doc))[:20]]
        return problems or _named_lengths(doc, SIDE_NAMED.get(name, {}))
    return check


PARQUET_MAGIC = b"PAR1"


def _parquet(path: Path) -> list[str]:
    """A Parquet file, checked at the format level only: the PAR1 magic at the start and at the end, and a footer
    length that fits in the file (its sha256 and size are recorded like any file's). The orchestrator does not read
    Parquet: no pyarrow here (the app installs this package with uv, and it stays light). The node that writes a
    Parquet side file owns its columns and tests them, and pipeline/bundle/validate.py reads them with pyarrow.
    A file with an encrypted footer ends in PARE: no reader without its key can open it."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(4)
            f.seek(max(size - 8, 0))
            tail = f.read(8)
    except OSError as e:
        return [f"not readable: {e}"]
    if size < 12:
        return [f"not Parquet: {size} bytes, fewer than its magic, a footer length and the closing magic"]
    problems = [] if head == PARQUET_MAGIC else [f"not Parquet: starts with {head!r}, not {PARQUET_MAGIC!r}"]
    if tail[4:] != PARQUET_MAGIC:
        problems.append(f"not Parquet: ends with {tail[4:]!r}, not {PARQUET_MAGIC!r}")
    footer = int.from_bytes(tail[:4], "little")
    if not 0 < footer <= size - 12:
        problems.append(f"not Parquet: a footer of {footer} bytes does not fit in a file of {size}")
    return problems


def _raw_pointer_problems(layer: str, doc: dict) -> list[str]:
    """A layer's raw pointers ({role: {file, format, description}}) must each name a side file of this layer
    (SIDE_FILES), as layers/<name>.<format> in the format SIDE_FILES gives it. `doc` has passed its schema."""
    problems, raw = [], doc.get("raw") or {}
    for role, pointer in raw.items():
        name, _, extension = pointer["file"].removeprefix("layers/").rpartition(".")
        side = SIDE_FILES.get(name)
        if extension != pointer["format"]:
            problems.append(f"raw.{role}: {pointer['file']} is not named as a {pointer['format']} file "
                            f"(layers/<name>.{pointer['format']})")
        elif side is None or side.layer != layer:
            problems.append(f"raw.{role}: {pointer['file']} is not a side file of the {layer} layer (artifacts.SIDE_FILES)")
        elif side.format != pointer["format"]:
            problems.append(f"raw.{role}: {name} is a {side.format} side file, not {pointer['format']}")
    files = [p["file"] for p in raw.values()]
    if len(files) != len(set(files)):
        problems.append("raw: two pointers name the same file")
    return problems


def side_pointers(doc: dict) -> list[tuple[str, str]]:
    """(field, file) for every side file a layer points at: the faces layer's named pointers (detail.file, mesh.file,
    embeddings.file) and any layer's raw.<role>.file."""
    named = [(f"{p}.file", doc[p]["file"]) for p in SIDE_POINTERS if isinstance(doc.get(p), dict) and doc[p].get("file")]
    raw = [(f"raw.{role}.file", p["file"]) for role, p in (doc.get("raw") or {}).items()
           if isinstance(p, dict) and p.get("file")]
    return named + raw


def _named_lengths(doc: dict, named: dict[str, str]) -> list[str]:
    """Per-sample arrays read by a list of names (a faces_detail sample's au by au_names) must have one value per name,
    which the schema cannot say. `doc` has passed its schema."""
    problems = []
    for track, samples in doc["tracks"].items():
        for i, s in enumerate(samples):
            problems += [f"$.tracks['{track}'][{i}]: {field} has {len(s[field])} values for {len(doc[names])} {names}"
                         for field, names in named.items() if len(s[field]) != len(doc[names])]
    return problems[:20]


def _wav(path: Path) -> list[str]:
    with open(path, "rb") as f:
        head = f.read(12)
    return [] if head[:4] == b"RIFF" and head[8:12] == b"WAVE" else ["not a RIFF/WAVE file"]


def _tree_of(suffixes: tuple[str, ...]) -> Callable[[Path], list[str]]:
    def check(path: Path) -> list[str]:
        if not path.is_dir():
            return ["not a directory"]
        files = [p for p in path.rglob("*") if p.is_file()]
        bad = [str(p.relative_to(path)) for p in files if p.suffix.lower() not in suffixes and p.name != "index.json"]
        return [f"unexpected files: {bad[:5]}"] if bad else []
    return check


MEDIA_SCHEMA = {
    "type": "object",
    "required": ["sha256", "duration", "width", "height", "fps", "time_base", "variable_frame_rate"],
    "properties": {
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "duration": {"type": "number", "exclusiveMinimum": 0},
        "width": {"type": "integer"}, "height": {"type": "integer"},
        "fps": {"type": "number", "exclusiveMinimum": 0, "description": "average frame rate"},
        "r_frame_rate": {"type": "string"}, "avg_frame_rate": {"type": "string"},
        "time_base": {"type": "string", "description": "the video stream's PTS time base, e.g. 1/30000"},
        "start_pts": {"type": ["integer", "null"]},
        "variable_frame_rate": {"type": "boolean"},
        "audio": {"type": ["object", "null"]},
        "filename": {"type": "string"},
    },
}

# What each layer can hold. Shots carry captions, which describe and name the people on screen; prosody measures the
# audio and carries each utterance's words; the waveform is a measure of the audio. Every layer is listed: a new one must say what it holds.
LAYER_KINDS: dict[str, tuple[str, ...]] = {
    "shots": ("transcript", "faces"), "transcript": ("transcript",), "speakers": ("transcript",), "text": ("transcript",),
    "faces": ("faces", "embeddings"), "bodies": (), "hands": (), "gestures": (), "facing": (), "gaze": (), "objects": (),
    "sounds": (), "prosody": ("audio", "transcript"), "waveform": ("audio",),
}
assert set(LAYER_KINDS) == set(BUNDLE_LAYERS)

@dataclass(frozen=True)
class Side:
    """A side file of a bundle layer: model output the layer does not draw, kept beside it."""
    layer: str
    kinds: tuple[str, ...]                      # what it holds, in LAYER_KINDS' terms
    format: Literal["json", "parquet"] = "json"


# Side files: name -> Side. Its file is layers/<name>.<format> and its type bundle.<name>@0.1. A JSON one validates
# against #/$defs/side/<name>; a Parquet one is checked here at the format level (_parquet), and its columns are
# #/$defs/parquet/<name>, which pipeline/bundle/validate.py reads with pyarrow. JSON when small, Parquet for large arrays.
# The faces detail is py-feat's per-sample landmarks, action units, emotions and pose, and the faces mesh its per-sample
# 478-point face mesh: face data, no embeddings. The faces embeddings are its per-sample ArcFace identity embeddings:
# identity data, so a step that reads them elsewhere needs the policy and the run's consent to name embeddings.
# The gaze heatmaps are Gaze-LLE's 64x64 map per head of where it looks, computed from the whole frame: no pixels, but
# together a coarse picture of where the people and things in each frame are, beyond the gaze layer's one peak per
# head, so they count as frames. Not faces: no face image, landmarks or box, only the face track ids and times the gaze
# layer (which holds no kind) has too.
SIDE_FILES: dict[str, Side] = {
    "faces_detail": Side("faces", ("faces",)),
    "faces_mesh": Side("faces", ("faces",)),
    "faces_embeddings": Side("faces", ("faces", "embeddings")),
    "gaze_heatmaps": Side("gaze", ("frames",), "parquet"),
}
# the faces layer's fields that point at one of its side files, as {"file": "layers/<name>.json", ...} or null. They
# predate raw, the general pointer: any layer points at its side files with raw: {<role>: {file, format, description}}
SIDE_POINTERS = ("detail", "mesh", "embeddings")
# a side file's per-sample arrays that hold one value per name in a list of the file's: field -> that list
SIDE_NAMED: dict[str, dict[str, str]] = {"faces_detail": {"au": "au_names", "emo": "emotion_names"}}

TYPES: dict[str, ArtifactType] = {
    "video.v1": ArtifactType("video.v1", "source", lambda p: [] if p.is_file() else ["missing"], kinds=("video",)),
    "media.v1": ArtifactType("media.v1", "file", _json_schema(MEDIA_SCHEMA), kinds=()),   # the video's format and timing
    "audio.wav.v1": ArtifactType("audio.wav.v1", "file", _wav, kinds=("audio",)),
    # small images (keyframes, crops) a layer refers to as "assets/<path>"; export merges every assets.v1 tree into the
    # bundle's assets/ directory. Either kind of image: both kinds are counted
    "assets.v1": ArtifactType("assets.v1", "tree", _tree_of((".jpg", ".jpeg", ".png", ".webp")), kinds=("frames", "faces")),
    "json.v1": ArtifactType("json.v1", "file", _json_schema({})),
    "text.v1": ArtifactType("text.v1", "file", lambda p: []),
    **{f"bundle.{name}@0.1": ArtifactType(f"bundle.{name}@0.1", "file", _layer(name), layer=name, kinds=LAYER_KINDS[name])
       for name in BUNDLE_LAYERS},
    **{f"bundle.{name}@0.1": ArtifactType(f"bundle.{name}@0.1", "file", _side(name) if side.format == "json" else _parquet,
                                          side_of=side.layer, bundle_file=f"layers/{name}.{side.format}", kinds=side.kinds)
       for name, side in SIDE_FILES.items()},
}


def get(type_id: str) -> ArtifactType:
    if type_id not in TYPES:
        raise KeyError(f"unknown artifact type {type_id}")
    return TYPES[type_id]


# --- content hashes ---------------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(path: Path) -> str:
    """A directory's hash: over its sorted relative paths and their file hashes (so renames and edits both count)."""
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(path).as_posix()).encode() + b"\0" + sha256_file(p).encode() + b"\n")
    return h.hexdigest()


def special_files(path: Path) -> list[str]:
    """Entries that are not plain files or directories (symlinks, sockets, devices), found without following links.
    Outputs must have none: publishing a symlink would copy whatever it points at on the host."""
    import os
    import stat
    bad = []
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        return [f"{path.name} (symlink)"]
    if stat.S_ISDIR(st.st_mode):
        for root, dirs, files in os.walk(path, followlinks=False):
            for name in dirs + files:
                mode = os.lstat(os.path.join(root, name)).st_mode
                if stat.S_ISLNK(mode):
                    bad.append(f"{os.path.relpath(os.path.join(root, name), path)} (symlink)")
                elif not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    bad.append(f"{os.path.relpath(os.path.join(root, name), path)} (not a regular file)")
    elif not stat.S_ISREG(st.st_mode):
        bad.append(f"{path.name} (not a regular file)")
    return bad


def content_hash(path: Path) -> str:
    return sha256_tree(path) if path.is_dir() else sha256_file(path)


def canonical(value) -> str:
    """JSON with sorted keys and no whitespace: equal values hash equal."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
