"""vision.objects-names: the names of the things on screen, shot by shot, and what kind of thing each is, as a
vocabulary artifact (vocabulary.py) for the track step. The first of the objects steps (names, track, link: the objects node
1.1.0 split so its GPU step neither waits on the API nor holds a key, RUN_ANYWHERE_PLAN.md item 1).

1. Vocabulary (haiku.py): Claude Haiku 5.5 sees each shot's keyframes (decode's) and names up to 8 concrete, trackable
   objects in generic detector nouns, told the names it used in earlier shots so one bike stays "bicycle". Clothing,
   shoes, hats, glasses and necklaces are not its to name (the track step finds them on every person); a name it gives
   anyway that the body pass asks for (a garment noun, or "person") is not a session prompt (garments.session_names),
   and the artifact keeps it apart (body_named).
2. Name classes (haiku.py): one call classifies the clip's prompts into families, lookalike or individual, and worn or
   not (the body nouns' classes are fixed, garments.classes, and the track step adds them).

It needs decode's keyframes and shots only, so it starts as soon as decode is done, beside the faces and bodies the
track step also waits for. The keyframes are raw frames: they show the people on screen (frames and faces, as the
captions' keyframes do). What 1.1.0 checked first, before anything was uploaded or paid for (the gated models' access,
the longest shot's memory), is the track step's now: a run whose track step fails there has paid for the names
(about $0.003 a minute of video), which are cached for the run that follows.
"""

import time

from perception_node import Node, NodeError
import vocabulary
from garments import session_names
from haiku import MAX_OBJECTS, MODEL as HAIKU, Haiku, spent
from linking import check_classes, clean_names

with Node() as node:
    shots = node.input_json("shots")["shots"]
    keyframes_dir = node.input("keyframes").resolve()
    totals = {"in": 0, "out": 0, "calls": 0}

    def spend(n_in, n_out):
        """The running total after every billed call: if a later call fails, the response still reports it."""
        totals.update({"in": totals["in"] + n_in, "out": totals["out"] + n_out, "calls": totals["calls"] + 1})
        node.usage(usd=spent(totals["in"], totals["out"]), tokens_in=totals["in"], tokens_out=totals["out"], calls=totals["calls"])

    haiku = Haiku(node.secret("ANTHROPIC_API_KEY"), spend)

    # 1. one call per shot, in order, each told the names used before it (never after)
    t0 = time.time()
    prompts, body_named, earlier = {}, {}, set()
    for shot in shots:
        images = []
        for kf in shot.get("keyframes") or []:
            path = (keyframes_dir / kf["file"].removeprefix("assets/")).resolve()   # follows links: one out is caught too
            if not path.is_relative_to(keyframes_dir):
                raise NodeError("input", f"keyframe {kf['file']} is outside the keyframes input; only those images are sent")
            images.append(path.read_bytes())
        if not images:
            raise NodeError("input", f"shot {shot['index']} has no keyframes: the object names come from them")
        named = clean_names(haiku.vocabulary(images, earlier), MAX_OBJECTS)
        earlier.update(named)
        # the body pass's own nouns (garments, "person") are not session prompts
        prompts[shot["index"]], body_named[shot["index"]] = session_names(named)

    # 2. the names' families, kinds and whether they are worn: from the vocabulary alone
    names = sorted({n for shot_names in prompts.values() for n in shot_names})
    try:
        classes = check_classes(names, haiku.name_classes(names)) if names else {}
    except ValueError as e:
        raise NodeError("internal", str(e)) from None

    node.model(HAIKU, licence_mode="api-terms", licence="Anthropic API terms")
    node.write_json("vocabulary", "vocabulary.json", vocabulary.document(shots, prompts, body_named, classes, HAIKU))
    node.usage(timings={"vocabulary_s": round(time.time() - t0, 1)},   # with the name classification, as 1.1.0 timed it
               vocabulary={str(k): v for k, v in prompts.items()}, classes=classes,
               body_named={str(k): v for k, v in body_named.items() if v})
