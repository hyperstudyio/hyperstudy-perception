"""Artifact types: what flows between nodes, and how a node's output is checked before it is published.

The bundle's layer schemas are artifact types (`bundle.<layer>@0.1`): a layer a node writes is exactly the layer the
annotator and HyperStudy read, and so are the side files a layer points to (`bundle.faces_detail@0.1` and
`bundle.faces_mesh@0.1`, the faces layer's layers/faces_detail.json and layers/faces_mesh.json). A few internal types
carry what bundles do not (media metadata, audio, keyframes).
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
        return problems
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

# Side files: name (its schema, #/$defs/side/<name>, and its file, layers/<name>.json) -> (layer, what it holds).
# The faces detail is py-feat's per-sample landmarks, action units, emotions and pose, and the faces mesh its per-sample
# 478-point face mesh: face data, no embeddings.
SIDE_FILES: dict[str, tuple[str, tuple[str, ...]]] = {"faces_detail": ("faces", ("faces",)),
                                                      "faces_mesh": ("faces", ("faces",))}
# the fields of a layer that point at one of its side files, as {"file": "layers/<name>.json", ...} or null
SIDE_POINTERS = ("detail", "mesh")
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
    **{f"bundle.{name}@0.1": ArtifactType(f"bundle.{name}@0.1", "file", _side(name), side_of=layer,
                                          bundle_file=f"layers/{name}.json", kinds=kinds)
       for name, (layer, kinds) in SIDE_FILES.items()},
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
