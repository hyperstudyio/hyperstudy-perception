from perception_node import Node, NodeError

with Node() as node:
    doc = node.input_json("shots")
    if node.request["attempt"] <= node.config["fail_until"]:
        node.usage(usd=node.config["usd"])               # money spent before the failure
        raise NodeError("transient", "failing on purpose after spending")
    node.model("test-captioner", licence_mode="api-terms")
    shots = [{**s, "caption": {"description": f"shot {s['index']}"}} for s in doc["shots"]]
    node.write_json("shots", "shots.json", {**doc, "producer": node.producer(), "shots": shots})
    node.usage(usd=node.config["usd"])
