"""A test node that declares a key (TEST_API_KEY): writes its input's value plus 1, and records which key-like variables
its environment had and the sha256 of the one it declares (never the key itself), for test_remote_run."""

import hashlib
import json
import os
from pathlib import Path

from perception_node import Node

with Node() as node:
    record = Path(node.config["record_dir"])
    record.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("TEST_API_KEY")
    (record / f"{node.request['node']}-{node.request['attempt']}-{os.getpid()}").write_text(json.dumps(
        {"keys": sorted(k for k in os.environ if k.endswith(("_KEY", "_TOKEN"))),
         "sha256": hashlib.sha256(key.encode()).hexdigest() if key else None}))
    value = json.loads(node.input("a").read_text())["value"]
    node.write_json("out", "out.json", {"value": value + 1})
    node.model("test-model", licence_mode="commercial-ok")
