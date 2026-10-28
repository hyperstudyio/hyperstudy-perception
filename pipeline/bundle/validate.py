"""Validate a track bundle directory: schema, checksums, and cross-layer references.

1. manifest.json against the root of schema/bundle-<major.minor>.schema.json, and each layer file against
   #/$defs/layer/<name>.
2. Every layer and side file exists, and its sha256 and size match the manifest. Every side file a layer points at
   (the faces layer's detail.file, mesh.file and embeddings.file; any layer's raw.<role>.file) is among its side
   files. A JSON side file with a schema (#/$defs/side/<name>) validates against it. A Parquet side file must have
   Parquet's magic at both ends; with pyarrow (this project's dev group, which `uv run` installs) its columns must be
   #/$defs/parquet/<name>'s, none null or not finite, and a per-sample one must hold a row for each of the layer's
   samples. Without pyarrow that content check is skipped, and the output says so.
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

try:
    import pyarrow.parquet as pq
except ImportError:   # the Parquet content check is skipped, and says so (main)
    pq = None

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


# the faces layer's fields that point at one of its side files ({"file": ...} or null); any layer's raw.<role> does too
SIDE_POINTERS = ("detail", "mesh", "embeddings")
# side files keyed by face track ids, a sample for each of the faces layer's samples
PER_SAMPLE_SIDES = ("faces_detail", "faces_mesh", "faces_embeddings")
FORMATS = {".json": "json", ".parquet": "parquet"}


def side_pointers(doc):
    """(field, file, format or None) for every side file a layer points at."""
    named = [(f"{p}.file", doc[p].get("file"), None) for p in SIDE_POINTERS if isinstance(doc.get(p), dict)]
    raw = doc.get("raw") if isinstance(doc.get("raw"), dict) else {}
    return [x for x in named + [(f"raw.{role}.file", p.get("file"), p.get("format")) for role, p in raw.items()
                                if isinstance(p, dict)] if x[1]]


def side_file_problems(bundle, name, entry, doc, schema, registry, notes):
    """Every side file a layer points at must be among its side files, in the format its pointer says; a JSON side
    file with a schema (#/$defs/side/<name>) must validate, and one keyed by face track ids (faces_detail, faces_mesh,
    faces_embeddings) must hold the layer's tracks, a sample for each sample. Parquet: parquet_problems."""
    problems, listed = [], {f["file"] for f in entry.get("side_files", [])}
    for field, referenced, fmt in side_pointers(doc):
        if referenced not in listed:
            problems.append(f"{name}: {field} {referenced} is not one of the layer's side files")
        if fmt is not None and FORMATS.get(Path(referenced).suffix) != fmt:
            problems.append(f"{name}: {field} {referenced} is not a {fmt} file")
    for file in sorted(listed):
        side_name, path = Path(file).stem, bundle / file
        if not path.exists():
            continue
        if Path(file).suffix == ".parquet":
            problems += parquet_problems(name, file, path, schema["$defs"].get("parquet", {}).get(side_name), doc, notes)
            continue
        if side_name not in schema["$defs"].get("side", {}):
            continue
        try:
            side = json.loads(path.read_text())
        except ValueError as e:
            problems.append(f"{name}: {file}: not JSON: {e}")
            continue
        validator = Draft202012Validator({"$ref": f"{schema['$id']}#/$defs/side/{side_name}"}, registry=registry)
        errors = sorted(validator.iter_errors(side), key=lambda e: list(e.absolute_path))
        problems += [f"{name}: {file}: {e.json_path}: {e.message[:200]}" for e in errors[:20]]
        if errors or side_name not in PER_SAMPLE_SIDES or not isinstance(doc.get("tracks"), list):
            continue   # the cross-checks below read the shape the schema checks
        if side_name == "faces_detail":
            for track, samples in side["tracks"].items():
                for i, s in enumerate(samples):
                    problems += [f"{name}: {file}: $.tracks['{track}'][{i}]: {field} has {len(s[field])} values for "
                                 f"{len(side[names])} {names}"
                                 for field, names in (("au", "au_names"), ("emo", "emotion_names"))
                                 if len(s[field]) != len(side[names])]
        layer_samples = {t.get("id"): [s.get("t") for s in t.get("samples", []) if isinstance(s, dict)]
                         for t in doc["tracks"] if isinstance(t, dict)}
        side_samples = {k: [s["t"] for s in v] for k, v in side["tracks"].items()}
        if layer_samples != side_samples:
            differ = sorted(k for k in layer_samples.keys() | side_samples.keys()
                            if layer_samples.get(k) != side_samples.get(k))
            problems.append(f"{name}: {file} does not match the layer's track samples: {len(differ)} tracks, e.g. {differ[:5]}")
    return problems


PARQUET_MAGIC = b"PAR1"


def parquet_problems(name, file, path, spec, doc, notes):
    """A Parquet side file: the PAR1 magic at both ends (what the orchestrator checks too). With pyarrow, its columns
    in order with #/$defs/parquet/<name>'s Arrow types, no nulls, no value that is not finite (in a float column or a
    fixed-size list of floats), and for a per-sample one (per_sample: true) a row for each sample of the layer's tracks
    (face, sample, t), in time order with the heads of one time in the layer's track order. Without pyarrow, the content check is skipped and a note says so."""
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != PARQUET_MAGIC or data[-4:] != PARQUET_MAGIC:
        return [f"{name}: {file}: not Parquet (no PAR1 magic at both ends)"]
    if spec is None:
        return [f"{name}: {file}: the schema has no #/$defs/parquet/{Path(file).stem} for its columns"]
    if pq is None:
        notes.append(f"{name}: {file}: Parquet content not checked (pyarrow is not installed; `uv run` in "
                     "pipeline/bundle installs it from the dev group)")
        return []
    try:
        table = pq.read_table(path)
    except Exception as e:   # pyarrow raises ArrowInvalid, OSError and others for a damaged file
        return [f"{name}: {file}: not readable as Parquet: {type(e).__name__}: {str(e)[:200]}"]
    want = [(c["name"], c["arrow"]) for c in spec["columns"]]
    got = [(f.name, str(f.type)) for f in table.schema]
    if got != want:
        return [f"{name}: {file}: columns are {got}, not {want}"]
    problems = [f"{name}: {file}: column {c} has {table[c].null_count} nulls" for c, _ in want if table[c].null_count]
    import pyarrow as pa
    import pyarrow.compute as pc
    for field in table.schema:   # a float column, or a fixed-size list of floats (each value of each list)
        column = table[field.name]
        if pa.types.is_fixed_size_list(field.type) and pa.types.is_floating(field.type.value_type):
            column = pc.list_flatten(column)
        elif not pa.types.is_floating(field.type):
            continue
        if not pc.all(pc.is_finite(column)).as_py():
            problems.append(f"{name}: {file}: column {field.name} holds a value that is not finite")
    if problems or not spec.get("per_sample"):
        return problems
    if not isinstance(doc.get("tracks"), list):
        return [f"{name}: {file} is per sample, but the layer has no tracks"]
    order = []
    for k, track in enumerate(t for t in doc["tracks"] if isinstance(t, dict)):
        samples = track.get("samples")
        if not isinstance(samples, list) or not all(
                isinstance(s, dict) and isinstance(s.get("t"), (int, float)) and not isinstance(s.get("t"), bool)
                for s in samples):
            return problems   # the rows check reads the shape the layer's schema checks, and its errors report this
        face = track.get("face", track.get("id"))
        order += [(s["t"], k, face, j) for j, s in enumerate(samples)]
    expected = [(face, j, t) for t, _, face, j in sorted(order, key=lambda x: (x[0], x[1]))]
    rows = list(zip(table["face"].to_pylist(), table["sample"].to_pylist(), table["t"].to_pylist()))
    if rows != expected:
        first = next((i for i, (a, b) in enumerate(zip(rows, expected)) if a != b), min(len(rows), len(expected)))
        problems.append(f"{name}: {file} does not match the layer's samples: {len(rows)} rows for {len(expected)} "
                        f"samples, first difference at row {first}")
    return problems


def main():
    bundle = Path(sys.argv[1])
    problems = []
    manifest = json.loads((bundle / "manifest.json").read_text())
    schema, registry = load_schema(manifest.get("schema_version", "0.0.0"))
    for e in Draft202012Validator(schema, registry=registry).iter_errors(manifest):
        problems.append(f"manifest: {e.json_path}: {e.message}")

    docs, notes = {}, []
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
        problems += side_file_problems(bundle, name, entry, docs[name], schema, registry, notes)

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

    for note in notes:
        print(f"note: {note}")
    if problems:
        print(f"{bundle}: INVALID ({len(problems)} problems)")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    unchecked = f"; {len(notes)} check{'s' if len(notes) > 1 else ''} skipped, see the notes" if notes else ""
    print(f"{bundle}: valid ({len(docs)} layers, schema {manifest['schema_version']}{unchecked})")


if __name__ == "__main__":
    main()
