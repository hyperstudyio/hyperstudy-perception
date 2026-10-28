from pathlib import Path

from perception_node import Node

# facing_poses.parquet and facing_angles.parquet: written by the facing node's own writers (make_fixture.py)
HERE = Path(__file__).parent

with Node() as node:
    node.model("test-mediapipe", licence_mode="commercial-ok")
    for name in ("poses", "angles"):
        node.path(name, f"facing_{name}.parquet").write_bytes((HERE / f"facing_{name}.parquet").read_bytes())
    # no event: the faces stand-in has one face track, and a facing event names two
    node.write_json("facing", "facing.json", {
        "layer": "facing", "schema_version": "0.1.0", "producer": node.producer(), "events": [],
        "raw": {name: {"file": f"layers/facing_{name}.parquet", "format": "parquet", "description": f"test {name}"}
                for name in ("poses", "angles")}})
