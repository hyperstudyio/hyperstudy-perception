from perception_node import Node

with Node() as node:
    sha = node.request["inputs"]["video"]["sha256"]
    node.write_json("media", "media.json", {"sha256": sha, "duration": 2.0, "width": 4, "height": 4, "fps": 25.0,
                                            "time_base": "1/12800", "variable_frame_rate": False, "filename": "clip.mp4"})
    node.model("test-shots", licence_mode="academic-only")
    node.write_json("shots", "shots.json", {"layer": "shots", "schema_version": "0.1.0", "producer": node.producer(),
                                            "shots": [{"id": "shot:0", "index": 0, "start": 0.0, "end": 2.0, "keyframes": [], "caption": None}]})
