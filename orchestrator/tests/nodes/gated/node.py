"""A test node with gated weights (test.gated): writes its input's value plus 1, and records what it was given
(record.py) and whether its weights were in the cache."""

import json

from perception_node import Node
from record import record, weights

with Node() as node:
    record(node.config["record_dir"], f"node-{node.request['attempt']}", weights=weights().is_file())
    value = json.loads(node.input("a").read_text())["value"]
    node.write_json("out", "out.json", {"value": value + 1})
    node.model("test-gated-weights", licence_mode="commercial-ok")
