from pathlib import Path

from perception_node import Node

# gaze_heatmaps.parquet: written by the gaze node's own writer (make_fixture.py), for face:0's samples in test.faces
FIXTURE = Path(__file__).parent / "gaze_heatmaps.parquet"

with Node() as node:
    c = node.config
    node.model("test-gaze", licence_mode="needs-review")
    raw = node.path("heatmaps", "gaze_heatmaps.parquet")
    raw.write_bytes(FIXTURE.read_bytes() if c["heatmaps_bytes"] == "fixture" else c["heatmaps_bytes"].encode())
    node.write_json("gaze", "gaze.json", {
        "layer": "gaze", "schema_version": "0.1.0", "producer": node.producer(), "fps": 8,
        "tracks": [{"face": "face:0", "samples": [{"t": t, "peak": [2.0, 1.5], "in_frame": 0.9, "target": None,
                                                    "target_score": 0.0} for t in (0.0, 0.125)]}],
        "events": [],
        "raw": {"heatmaps": {"file": c["heatmaps_file"], "format": c["heatmaps_format"],
                             "description": "Gaze-LLE's 64x64 heatmap per head"}}})
