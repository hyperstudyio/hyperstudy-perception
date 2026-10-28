from pathlib import Path

from perception_node import Node

# faces_pyfeat.parquet and faces_untracked.parquet: written by face-track's own writers (make_fixture.py), for face:0's
# samples below
HERE = Path(__file__).parent

with Node() as node:
    node.model("test-faces", licence_mode="academic-only")
    samples = [0.0, 0.125]
    node.write_json("detail", "faces_detail.json", {
        "au_names": ["AU01"], "emotion_names": ["Neutral"], "angles": "radians",
        "tracks": {"face:0": [{"t": t, "lmk": [1.5] * 136, "au": [0.1], "emo": [0.9], "va": [0.1, 0.2],
                               "head": [0.0, 0.1, 0.2], "gaze": [0.0, 0.0]} for t in samples]}})
    node.write_json("mesh", "faces_mesh.json", {
        "points": 478, "layout": "all 478 x, then all 478 y",
        "tracks": {"face:0": [{"t": t, "xy": [1] * 956} for t in samples]}})
    node.write_json("embeddings", "faces_embeddings.json", {
        "model": "test-arcface", "dims": 512, "normalised": False,
        "tracks": {"face:0": [{"t": t, "v": [0.25] * 512} for t in samples]}})
    for role in ("pyfeat", "untracked"):
        node.path(role, f"faces_{role}.parquet").write_bytes((HERE / f"faces_{role}.parquet").read_bytes())
    node.write_json("faces", "faces.json", {
        "layer": "faces", "schema_version": "0.1.0", "producer": node.producer(), "sample_fps": 8, "angles": "radians",
        "people": [], "person_pairs": [],
        "tracks": [{"id": "face:0", "person": None, "shot": "shot:0", "start": 0.0, "end": 0.125, "linked_body": None, "crop": None,
                    "samples": [{"t": t, "box": [0, 0, 2, 2], "mouth": 0.0, "gaze": [0.0, 0.0]} for t in samples]}],
        "detail": {"file": node.config["detail_file"], "fields": ["lmk", "au", "emo", "va", "head", "gaze"]},
        "mesh": {"file": node.config["mesh_file"]}, "embeddings": {"file": node.config["embeddings_file"]},
        "raw": {role: {"file": f"layers/faces_{role}.parquet", "format": "parquet", "description": f"py-feat's {role} output"}
                for role in ("pyfeat", "untracked")}})
