"""The vocabulary artifact (vocabulary.v1, orchestrator/src/perception/artifacts.py): what the track step asks SAM 3 for
in each shot, and what kind of thing each name is. The names step (names.py) writes it from Haiku's answers; the track
step (track.py) reads it. Where the names come from is one edge of the graph, so another step (the captions call,
RUN_ANYWHERE_PLAN.md 1b) can write the same artifact.

    {"model": "claude-haiku-5-5",                   who named them
     "shots": [{"shot": "shot:0",                   every shot of the shots layer, in its order
                "names": ["bicycle", "ramp"],       the session's text prompts: up to 8, most prominent first
                "body_named": ["jacket"]}],         names given that the body pass asks for itself (garments.PROMPTS),
                                                    so not prompts: kept to say what the namer saw
     "classes": {"bicycle": {"family": "bicycle", "kind": "individual", "worn": false}, ...}}
                                                    each session name's class, exactly those (haiku.py's name classes)

The body nouns' classes are not in it: they are fixed (garments.classes) and the track step adds them.
"""

from __future__ import annotations



def document(shots: list[dict], prompts: dict[int, list[str]], body_named: dict[int, list[str]], classes: dict[str, dict],
             model: str) -> dict:
    """The artifact for these shots (the shots layer's): prompts and body_named, {shot index: names}, as
    garments.session_names split each shot's cleaned names (linking.clean_names); classes, the prompts' classes."""
    return {"model": model, "shots": [{"shot": s["id"], "names": prompts[s["index"]], "body_named": body_named[s["index"]]}
                                      for s in shots], "classes": classes}


def session_prompts(doc: dict, shots: list[dict]) -> tuple[dict[int, list[str]], dict[int, list[str]], dict[str, dict]]:
    """({shot index: its prompts}, {shot index: its body-named names}, classes) from an artifact, which must name every
    shot of `shots` (the shots layer's) once and no other, and class exactly the prompts it gives: ValueError otherwise
    (vocabulary.v1's own check covers its shape, not which shots layer it was made for)."""
    index = {s["id"]: s["index"] for s in shots}
    given = [s["shot"] for s in doc["shots"]]
    if sorted(given) != sorted(index) or len(set(given)) != len(given):
        missing, extra = sorted(set(index) - set(given)), sorted(set(given) - set(index))
        raise ValueError(f"the vocabulary's shots are not the shots layer's: missing {missing}, unexpected {extra}"
                         + (", some named twice" if len(set(given)) != len(given) else ""))
    prompts = {index[s["shot"]]: list(s["names"]) for s in doc["shots"]}
    body_named = {index[s["shot"]]: list(s.get("body_named", [])) for s in doc["shots"]}
    names = {n for shot_names in prompts.values() for n in shot_names}
    if names != set(doc["classes"]):
        raise ValueError(f"the vocabulary's classes are not its names': unclassed {sorted(names - set(doc['classes']))}, "
                         f"unused {sorted(set(doc['classes']) - names)}")
    return prompts, body_named, {n: dict(c) for n, c in doc["classes"].items()}
