"""vlm.shot-captions-objects: the shot captions (vlm.shot-captions), with the things on screen named in the same call,
as the vocabulary the objects' track step reads (run-anywhere 1b, B; A is vision.objects-names, a call of its own).

Each shot's captions call (captioning.run) also returns `objects`: up to 8 names, under the names step's rules (haiku.py's
VOCAB_INSTRUCTIONS, told the names used in earlier shots), cleaned and split as that step does (linking.clean_names,
garments.session_names). The captions' keyframes have the faces boxed and labelled; the names are of the things in
them, never the boxes. Then one call (Haiku 5.5, as the names step) classes the clip's names (haiku.name_classes), and
the vocabulary is written as the names step writes it (vocabulary.document), with the captions' model as its namer.
"""

import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path[:0] = [str(here.parent / "shot-captions"), str(here.parents[1] / "vision" / "objects")]   # its runner shares them

from perception_node import Node, NodeError   # noqa: E402
import captioning   # noqa: E402
import vocabulary   # noqa: E402
from garments import session_names   # noqa: E402
from haiku import MAX_OBJECTS, MODEL as HAIKU, VOCAB_INSTRUCTIONS, Haiku, spent as haiku_spent   # noqa: E402
from linking import check_classes, clean_names   # noqa: E402

# the names step's rules (haiku.VOCAB_INSTRUCTIONS' list), for the reply's objects part
RULES = VOCAB_INSTRUCTIONS.split("Rules:\n", 1)[1]

with Node() as node:
    prompts, body_named, earlier = {}, {}, set()

    def more(shot):
        return (f"\n\nobjects: up to {MAX_OBJECTS} concrete physical objects clearly visible in the keyframes that could be "
                f"outlined and tracked with a segmentation mask. The face boxes and labels are not objects.\nRules for "
                f"objects:\n{RULES}\nNames used in earlier shots: {', '.join(sorted(earlier)) or '(none yet)'}")

    def each(shot, reply):
        named = clean_names(reply["objects"], MAX_OBJECTS)
        earlier.update(named)
        prompts[shot["index"]], body_named[shot["index"]] = session_names(named)

    captions = captioning.run(node, more=more, each=each,
                                  schema_more={"objects": {"type": "array", "items": {"type": "string"}}})
    shots = node.input_json("shots")["shots"]
    totals = {"in": 0, "out": 0, "calls": 0}

    def spend(n_in, n_out):
        totals.update({"in": totals["in"] + n_in, "out": totals["out"] + n_out, "calls": totals["calls"] + 1})
        node.usage(usd=round(captions["usd"] + haiku_spent(totals["in"], totals["out"]), 4), class_calls=totals["calls"],
                   tokens_in=captions["tokens_in"] + totals["in"], tokens_out=captions["tokens_out"] + totals["out"])

    names = sorted({n for shot_names in prompts.values() for n in shot_names})
    if names:   # the class call is Haiku's, whichever model captions
        node.model(HAIKU, licence_mode="api-terms", licence="Anthropic API terms")
    try:
        classes = check_classes(names, Haiku(node.secret("ANTHROPIC_API_KEY"), spend).name_classes(names)) if names else {}
    except ValueError as e:
        raise NodeError("internal", str(e)) from None
    node.write_json("vocabulary", "vocabulary.json",
                    vocabulary.document(shots, prompts, body_named, classes, node.config["model"]))
    node.usage(vocabulary={str(k): v for k, v in prompts.items()}, classes=classes,
               body_named={str(k): v for k, v in body_named.items() if v})
