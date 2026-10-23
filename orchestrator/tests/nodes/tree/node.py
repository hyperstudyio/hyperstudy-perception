import os

from perception_node import Node

with Node() as node:
    tree = node.path("assets", "assets")
    (tree / "crops").mkdir(parents=True, exist_ok=True)
    (tree / "crops" / "a.jpg").write_bytes(b"jpeg")
    if node.config["link_to"]:
        os.symlink(node.config["link_to"], tree / "crops" / "b.jpg")   # what a hostile or buggy node could do
