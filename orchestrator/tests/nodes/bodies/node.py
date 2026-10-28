from pathlib import Path

from perception_node import Node

# bodies_scores.parquet and bodies_duplicates.parquet: written by the body-pose node's own writers (make_fixture.py),
# for the samples below
HERE = Path(__file__).parent

with Node() as node:
    node.model("test-pose", licence_mode="academic-only")
    for name in ("scores", "duplicates"):
        node.path(name, f"bodies_{name}.parquet").write_bytes((HERE / f"bodies_{name}.parquet").read_bytes())
    kpts = [[1.0, 1.0, 5.0]] * 133

    def track(i, times):
        return {"id": f"body:{i}", "shot": "shot:0", "face": None, "link": None, "start": times[0], "end": times[-1],
                "samples": [{"t": t, "box": [0, 0, 2, 4], "kpts": kpts} for t in times]}

    node.write_json("bodies", "bodies.json", {
        "layer": "bodies", "schema_version": "0.1.0", "producer": node.producer(), "keypoint_names": ["k"] * 133,
        "score_threshold": 3.0, "fps": 8, "tracks": [track(0, [0.0, 0.125]), track(1, [0.0])],
        "raw": {name: {"file": f"layers/bodies_{name}.parquet", "format": "parquet", "description": f"test {name}"}
                for name in ("scores", "duplicates")}})
