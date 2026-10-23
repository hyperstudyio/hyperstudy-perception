"""Validate a track bundle directory: schema, checksums, and cross-layer references.

1. manifest.json against the root of schema/bundle-<major.minor>.schema.json, and each layer file against
   #/$defs/layer/<name>.
2. Every layer and side file exists, and its sha256 and size match the manifest.
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
