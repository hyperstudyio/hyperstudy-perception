from perception_node import Node

with Node() as node:
    doc = node.input_json("gaze")
    node.model("test-gaze", licence_mode="needs-review")
    node.write_json("gaze", "gaze.json", {**doc, "producer": node.producer()})   # raw kept, as written upstream
