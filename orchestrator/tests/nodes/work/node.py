"""A test node: sums its inputs' `value`s plus config `add`, after `sleep` seconds. It records each execution (a file
in `record_dir`) and can fail on purpose: `fail` = transient | crash | schema | internal | input, for attempts up to
`fail_until` (so a retried transient failure can then succeed). `oom` raises an exception named like PyTorch's
(torch.OutOfMemoryError) without needing torch."""

import json
import os
import time
from pathlib import Path

from perception_node import Node, NodeError

with Node() as node:
    c = node.config
    record = Path(c["record_dir"])
    record.mkdir(parents=True, exist_ok=True)
    (record / f"{node.request['node']}-{node.request['attempt']}-{os.getpid()}").write_text(json.dumps({"start": time.time()}))
    total = c["add"]
    for name in sorted(node.request["inputs"]):
        p = node.input(name)
        total += json.loads(p.read_text())["value"] if p.suffix == ".json" else 0
    time.sleep(c["sleep"])
    failing = node.request["attempt"] <= c["fail_until"]
    if failing and c["fail"] == "crash":
        os._exit(3)                                  # dies without a response, as an OOM kill would
    if failing and c["fail"] == "oom":
        class OutOfMemoryError(RuntimeError):
            pass
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 7.91 GiB.")
    if failing and c["fail"] in ("transient", "input", "internal"):
        if c["fail"] == "internal":
            raise ZeroDivisionError("a bug")
        raise NodeError(c["fail"], f"failing on purpose ({c['fail']})")
    if failing and c["fail"] == "schema":
        node.path("out", "out.json").write_text("{not json")   # violates json.v1: the coordinator must not publish it
    else:
        node.write_json("out", "out.json", {"value": total})
    node.model("test-model", licence_mode="commercial-ok")
    (record / f"{node.request['node']}-{node.request['attempt']}-{os.getpid()}.end").write_text(json.dumps({"end": time.time()}))
