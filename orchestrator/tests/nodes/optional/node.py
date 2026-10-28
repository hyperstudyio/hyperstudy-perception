"""A test node with an optional input: a's value plus `add`, plus b's when the request carries b (has_input)."""

from perception_node import Node

with Node() as node:
    total = node.input_json("a")["value"] + node.config["add"]
    has_b = node.has_input("b")
    if has_b:
        total += node.input_json("b")["value"]
    node.write_json("out", "out.json", {"value": total, "had_b": has_b})
    node.model("test-model", licence_mode="commercial-ok")
