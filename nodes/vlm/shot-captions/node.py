"""vlm.shot-captions: one description per shot from its keyframes and speech, with the face roster.

Ported from pipeline/bakeoff/caption.py (Claude Haiku 5.5 with the face roster: 126 of 135 boxed people named on
OOO, about $0.04 a clip). Each shot sees its three keyframes, with tracked faces boxed and labelled, the speech that
starts inside it, the people seen so far and the previous shot's description; never anything from later shots.
That chaining is why the calls run in order rather than as a Message Batch.

Writes the shots layer with captions (it refines the decode node's shots layer) and the on-screen text layer.
"""

from perception_node import Node
from captioning import run

with Node() as node:
    run(node)
