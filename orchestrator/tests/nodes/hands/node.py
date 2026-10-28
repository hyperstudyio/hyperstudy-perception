from pathlib import Path

from perception_node import Node

# hands_detections.parquet: written by the hands node's own writer (make_fixture.py), for hand:1's samples below
HERE = Path(__file__).parent

with Node() as node:
    node.model("test-hands", licence_mode="academic-only")
    node.path("detections", "hands_detections.parquet").write_bytes((HERE / "hands_detections.parquet").read_bytes())
    kpts = [[1.0, 1.0, 0.5]] * 21

    def hand(i, source, times):
        return {"id": f"hand:{i}", "shot": "shot:0", "side": None, "body": None, "face": None, "box_source": source,
                "start": times[0], "end": times[-1], "samples": [{"t": t, "box": [1, 1, 2, 2], "kpts": kpts} for t in times]}

    node.write_json("hands", "hands.json", {
        "layer": "hands", "schema_version": "0.1.0", "producer": node.producer(), "keypoint_names": ["k"] * 21,
        "score_threshold": None, "fps": 8, "hands": [hand(0, "wrist", [0.0, 0.125]), hand(1, "detector", [0.0, 0.125])],
        "raw": {"detections": {"file": "layers/hands_detections.parquet", "format": "parquet", "description": "test detections"}}})
