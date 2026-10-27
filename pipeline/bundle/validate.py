"""Validate a track bundle directory: schema, checksums, and cross-layer references.

1. manifest.json against the root of schema/bundle-<major.minor>.schema.json, and each layer file against
   #/$defs/layer/<name>.
2. Every layer and side file exists, and its sha256 and size match the manifest. A side file with a schema
   (#/$defs/side/<name>) validates against it, and a layer's detail.file is one of its side files.
3. References: every id a layer points at (a shot, utterance, voice, face track, person, body, hand, object track
   or object identity) is defined by the layer that owns that kind of id. JSON Schema checks an id's shape, not
   whether it exists.

Exit status 1 with every problem listed, 0 if the bundle is valid.

Usage (from pipeline/bundle/): uv run python validate.py BUNDLE_DIR
"""

import hashlib
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

SCHEMAS = Path(__file__).resolve().parent.parent.parent / "schema"

# which layer defines each id kind, and where (list key, id key)
DEFINES = {"shot": ("shots", [("shots", "id")]), "utt": ("transcript", [("utterances", "id")]),
           "voice": ("transcript", [("voices", "id")]), "face": ("faces", [("tracks", "id")]),
           "person": ("faces", [("people", "id")]), "body": ("bodies", [("tracks", "id")]),
           "hand": ("hands", [("hands", "id")]), "obj": ("objects", [("tracks", "id"), ("rejected_tracks", "id")]),
           "objid": ("objects", [("identities", "id")])}
# keys whose string values (or lists of strings) are references to ids defined elsewhere
REF_KEYS = {"shot", "shots", "utt", "voice", "face", "person", "body", "hand", "from", "to", "also", "object", "target",
            "identity", "wearer", "worn_by", "tracks", "a", "b"}


def load_schema(version):
    major_minor = ".".join(version.split(".")[:2])
    path = SCHEMAS / f"bundle-{major_minor}.schema.json"
    if not path.exists():
        raise SystemExit(f"no schema for bundle version {version} ({path.name})")
    schema = json.loads(path.read_text())
    registry = Registry().with_resource(schema["$id"], Resource.from_contents(schema))
    return schema, registry


def references(node, key=None):
    """Yield (key, value) for every string under a reference key, at any depth."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "id":
                continue
            yield from references(v, k)
    elif isinstance(node, list):
        for v in node:
            yield from references(v, key)
    elif isinstance(node, str) and key in REF_KEYS and ":" in node:
        yield key, node


def crop_paths(node, key=None):
    """Yield (key, path) for every non-null `crop` at any depth."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from crop_paths(v, k)
    elif isinstance(node, list):
        for v in node:
            yield from crop_paths(v, key)
    elif key == "crop" and isinstance(node, str):
        yield key, node


def side_file_problems(bundle, name, entry, doc, schema, registry):
    """A layer's detail.file must be one of its side files; a side file with a schema (#/$defs/side/<name>) must
    validate, and one keyed by face track ids (faces_detail) must hold the layer's tracks, a sample for each sample."""
    problems, listed = [], {f["file"] for f in entry.get("side_files", [])}
    referenced = (doc.get("detail") or {}).get("file") if isinstance(doc.get("detail"), dict) else None
    if referenced and referenced not in listed:
        problems.append(f"{name}: detail.file {referenced} is not one of the layer's side files")
    for file in sorted(listed):
        side_name, path = Path(file).stem, bundle / file
        if side_name not in schema["$defs"].get("side", {}) or not path.exists():
            continue
        try:
            side = json.loads(path.read_text())
        except ValueError as e:
            problems.append(f"{name}: {file}: not JSON: {e}")
            continue
        validator = Draft202012Validator({"$ref": f"{schema['$id']}#/$defs/side/{side_name}"}, registry=registry)
        errors = sorted(validator.iter_errors(side), key=lambda e: list(e.absolute_path))
        problems += [f"{name}: {file}: {e.json_path}: {e.message[:200]}" for e in errors[:20]]
        if errors or side_name != "faces_detail" or not isinstance(doc.get("tracks"), list):
            continue   # the cross-checks below read the shape the schema checks
        for track, samples in side["tracks"].items():
            for i, s in enumerate(samples):
                problems += [f"{name}: {file}: $.tracks['{track}'][{i}]: {field} has {len(s[field])} values for "
                             f"{len(side[names])} {names}" for field, names in (("au", "au_names"), ("emo", "emotion_names"))
                             if len(s[field]) != len(side[names])]
        layer_samples = {t.get("id"): [s.get("t") for s in t.get("samples", []) if isinstance(s, dict)]
                         for t in doc["tracks"] if isinstance(t, dict)}
        detail = {k: [s["t"] for s in v] for k, v in side["tracks"].items()}
        if layer_samples != detail:
            differ = sorted(k for k in layer_samples.keys() | detail.keys() if layer_samples.get(k) != detail.get(k))
            problems.append(f"{name}: {file} does not match the layer's track samples: {len(differ)} tracks, e.g. {differ[:5]}")
    return problems


def main():
    bundle = Path(sys.argv[1])
    problems = []
    manifest = json.loads((bundle / "manifest.json").read_text())
    schema, registry = load_schema(manifest.get("schema_version", "0.0.0"))
    for e in Draft202012Validator(schema, registry=registry).iter_errors(manifest):
        problems.append(f"manifest: {e.json_path}: {e.message}")

    docs = {}
    for name, entry in manifest.get("layers", {}).items():
        for f in [entry] + entry.get("side_files", []):
            path = bundle / f["file"]
            if not path.exists():
                problems.append(f"{name}: missing {f['file']}")
                continue
            data = path.read_bytes()
            if len(data) != f["bytes"] or hashlib.sha256(data).hexdigest() != f["sha256"]:
                problems.append(f"{name}: {f['file']} does not match its manifest checksum or size")
        path = bundle / entry["file"]
        if not path.exists():
            continue
        docs[name] = json.loads(path.read_text())
        validator = Draft202012Validator({"$ref": f"{schema['$id']}#/$defs/layer/{name}"}, registry=registry)
        for e in sorted(validator.iter_errors(docs[name]), key=lambda e: list(e.absolute_path))[:20]:
            problems.append(f"{name}: {e.json_path}: {e.message[:200]}")
        if docs[name].get("layer") != name:
            problems.append(f"{name}: file says layer {docs[name].get('layer')!r}")
        problems += side_file_problems(bundle, name, entry, docs[name], schema, registry)

    for f in manifest.get("assets", []):
        path = bundle / f["file"]
        if not path.exists():
            problems.append(f"assets: missing {f['file']}")
        elif path.stat().st_size != f["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != f["sha256"]:
            problems.append(f"assets: {f['file']} does not match its manifest checksum or size")
    listed = {f["file"] for f in manifest.get("assets", [])}
    for name, doc in docs.items():
        for key, value in crop_paths(doc):
            if value not in listed:
                problems.append(f"{name}: {key} points at {value}, which the manifest's assets do not list")

    defined = {}
    for kind, (layer_name, places) in DEFINES.items():
        if layer_name in docs:
            defined[kind] = {item[id_key] for list_key, id_key in places for item in docs[layer_name].get(list_key, [])}
    for name, doc in docs.items():
        missing = {}
        for key, value in references({k: v for k, v in doc.items() if k != "producer"}):
            kind = value.split(":", 1)[0]
            if kind not in DEFINES:
                continue
            if kind not in defined:
                missing.setdefault(f"refers to {kind} ids but layer {DEFINES[kind][0]} is absent", []).append(value)
            elif value not in defined[kind]:
                missing.setdefault(f"unknown {kind} ids", []).append(value)
        for what, values in missing.items():
            problems.append(f"{name}: {what}: {len(values)}, e.g. {sorted(set(values))[:5]}")

    if problems:
        print(f"{bundle}: INVALID ({len(problems)} problems)")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"{bundle}: valid ({len(docs)} layers, schema {manifest['schema_version']})")


if __name__ == "__main__":
    main()
