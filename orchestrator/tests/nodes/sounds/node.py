from pathlib import Path

from perception_node import Node

# sounds_tags.parquet and sounds_queries*.parquet: written by the sounds node's own writer (make_fixture.py)
HERE = Path(__file__).parent

with Node() as node:
    node.model("test-sounds", licence_mode="needs-review")
    node.path("tags", "sounds_tags.parquet").write_bytes((HERE / "sounds_tags.parquet").read_bytes())
    node.path("queries", "sounds_queries.parquet").write_bytes((HERE / node.config["queries_fixture"]).read_bytes())
    node.write_json("sounds", "sounds.json", {
        "layer": "sounds", "schema_version": "0.1.0", "producer": node.producer(),
        "events": [{"id": "sound:0", "label": "Laughter", "start": 0.0, "end": 1.0, "sources": ["panns"],
                    "agreement": None, "score": 0.5},
                   {"id": "sound:1", "label": "laughter", "start": 0.32, "end": 0.96, "sources": ["pe-a-frame"],
                    "agreement": True, "score": None}],
        "raw": {"tags": {"file": "layers/sounds_tags.parquet", "format": "parquet", "description": "the taggers' scores"},
                "queries": {"file": "layers/sounds_queries.parquet", "format": "parquet",
                            "description": "PE-A-Frame's probabilities"}}})
