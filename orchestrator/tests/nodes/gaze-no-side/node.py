from perception_node import Node

with Node() as node:
    node.model("test-gaze", licence_mode="needs-review")
    node.write_json("gaze", "gaze.json", {
        "layer": "gaze", "schema_version": "0.1.0", "producer": node.producer(), "fps": 8,
        "tracks": [{"face": "face:0", "samples": [{"t": 0.0, "peak": [2.0, 1.5], "in_frame": 0.9, "target": None}]}],
        "events": [],
        "raw": {"heatmaps": {"file": "layers/gaze_heatmaps.parquet", "format": "parquet",
                             "description": "Gaze-LLE's 64x64 heatmap per head"}}})
