"""vision.gestures: hand over mouth, self-touch of the face, pointing (and at what) and raised hands, as the bundle's
gestures layer, by the bake-off's rules over the hands layer's keypoints (gesture_rules.py), with every hand sample's
measurements in layers/gestures_evidence.parquet (gesture_evidence.py).

It reads the hands layer (RTMPose-hand's 21 keypoints, each hand's body and face), the bodies layer (shoulders and nose,
for raised hands), the faces layer and its faces_detail (py-feat's 68 landmarks: the mouth and the face region), and,
when the run has it, the objects layer (pointing targets besides faces). objects is an optional input: the user can
leave the objects step out (it uploads frames and costs money), and gestures then runs without it, pointing at faces
only. The video is read for one frame, to know the upright frame size the keypoints are in.

No model: geometry only, on the CPU. Licence: the rules are this repository's (commercial-ok), but they read the hands,
bodies and py-feat face layers, all academic-only, so the layer is academic-only through its inputs.
"""

import time

from perception_node import Node, NodeError
import gesture_evidence
from gesture_inputs import Context, InputError, check_hands_model, upright_size
from gesture_rules import KINDS, hand_events, layer_events, sample_evidence

MODEL_ID = "rules over hand keypoints and face landmarks (bake-off hand_events.py)"

with Node() as node:
    hands_layer = node.input_json("hands")
    hands = hands_layer["hands"]
    has_objects = "objects" in node.request["inputs"]
    t0 = time.time()
    try:
        check_hands_model(hands_layer)
        width, height = upright_size(node.input("video"))
        ctx = Context(hands, node.input_json("bodies"), node.input_json("faces"), node.input_json("detail"),
                      node.input_json("objects") if has_objects else None, width, height)
    except InputError as e:
        raise NodeError("input", str(e)) from None
    load_s = round(time.time() - t0, 1)
    node.model(MODEL_ID, licence_mode="commercial-ok", licence="this repository")

    t1 = time.time()
    events, statuses = [], {}
    with gesture_evidence.Writer(node.path("evidence", "gestures_evidence.parquet")) as writer:
        for hand in hands:
            try:
                rows = [sample_evidence(hand, j, ctx) for j in range(len(hand["samples"]))]
            except InputError as e:
                raise NodeError("input", str(e)) from None
            for row in rows:
                writer.add(row)
                for key in ("face_status", "point_status", "raise_status"):
                    statuses.setdefault(key, {}).setdefault(row[key], 0)
                    statuses[key][row[key]] += 1
            events += hand_events(hand, rows)
    events = layer_events(events)
    rules_s = round(time.time() - t1, 1)

    node.write_json("gestures", "gestures.json", {
        "layer": "gestures", "schema_version": "0.1.0",
        "producer": node.producer(timings={"load_s": load_s, "rules_s": rules_s}, device="cpu"),
        "events": events,
        "raw": {"evidence": {"file": gesture_evidence.FILE, "format": "parquet", "description": gesture_evidence.DESCRIPTION}}})
    node.usage(hands=len(hands), samples=writer.rows, objects_input=has_objects,
               events={kind: sum(e["kind"] == kind for e in events) for kind in KINDS},
               **{key: dict(sorted(counts.items())) for key, counts in statuses.items()})
