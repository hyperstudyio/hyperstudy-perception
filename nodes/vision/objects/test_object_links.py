"""Wearers, cross-shot identities and the objects layer (linking.py), without the models or the API.
Run: uv run --no-project --with pytest --with numpy --with av==17.1.0 --with pillow --with-editable ./sdk pytest -q nodes/vision/objects"""

import numpy as np
import pytest

from linking import (assign_wearers, box_iou, by_shot, check_classes, check_views, clean_names, crop_files,
                     different_people, duplicates, layer_content, link_identities, pair_score, representative_view, slug)
from test_garments import box_of, figure

CFG = {"link_threshold": 0.62, "cross_name_margin": 0.05, "same_wearer_bonus": 0.05, "min_link_px": 48, "duplicate_iou": 0.6}
CLASSES = {
    "necklace": {"family": "necklace", "kind": "individual", "worn": True},
    "pearl necklace": {"family": "necklace", "kind": "individual", "worn": True},
    "chair": {"family": "chair", "kind": "individual", "worn": False},
    "tree": {"family": "tree", "kind": "lookalike", "worn": False},
}


def unit(*v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


def track(id, label, shot, times, box=(100, 100, 80, 80), embedding=None, wearer=None):
    return {"id": id, "label": label, "shot": shot, "start": times[0], "end": times[-1], "score": 0.9, "wearer": wearer,
            "samples": [{"t": t, "box": list(box), "poly": [[0, 0], [1, 0], [1, 1]], "score": 0.8} for t in times],
            "embedding": embedding, "view_embeddings": None if embedding is None else [embedding], "views": [f"view of {id}"]}


def test_names_are_kept_as_the_bake_off_kept_them():
    assert clean_names([" Bicycle", "bicycle", "", "  ", "Wooden Ramp", "car"], 2) == ["bicycle", "wooden ramp"]


def test_the_vocabulary_leaves_clothing_to_the_body_pass_and_names_things_generically():
    from haiku import VOCAB_INSTRUCTIONS
    assert "ordinary shirts" not in VOCAB_INSTRUCTIONS and "notable" not in VOCAB_INSTRUCTIONS
    for rule in ("found on every person", "hair, wigs", "patterns, prints", "body parts", '"helmet", not "full-face helmet"'):
        assert rule in VOCAB_INSTRUCTIONS


def test_classes_must_cover_exactly_the_vocabulary():
    answer = [{"name": "Necklace ", "family": " Necklace", "kind": "individual", "worn": True}]
    assert check_classes(["necklace"], answer) == {"necklace": {"family": "necklace", "kind": "individual", "worn": True}}
    with pytest.raises(ValueError, match="missing \\['chair'\\]"):
        check_classes(["necklace", "chair"], answer)
    with pytest.raises(ValueError, match="unexpected \\['necklace'\\]"):
        check_classes([], answer)


def test_crop_check_must_answer_every_view_once():
    assert check_views([{"view": 2, "verdict": "absent"}, {"view": 1, "verdict": "object"}], 2) == {1: "object", 2: "absent"}
    with pytest.raises(ValueError):
        check_views([{"view": 1, "verdict": "object"}], 2)
    with pytest.raises(ValueError):
        check_views([{"view": 1, "verdict": "object"}, {"view": 1, "verdict": "absent"}, {"view": 2, "verdict": "object"}], 2)


def test_box_iou_and_duplicates_need_shared_frames():
    assert box_iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(50 / 150)
    a = track("obj:0", "necklace", 1, [1.0, 1.167])
    b = track("obj:1", "pearl necklace", 1, [1.0, 1.167], box=(104, 100, 80, 80))
    elsewhere = track("obj:2", "pearl necklace", 2, [1.0, 1.167])
    assert duplicates(a, b, 0.6) and not duplicates(a, elsewhere, 0.6)
    assert not duplicates(a, track("obj:3", "necklace", 1, [1.0], box=(160, 100, 80, 80)), 0.6)


FACES = {"tracks": [{"id": "face:0", "person": "person:Person A"}, {"id": "face:1", "person": "person:Person B"},
                    {"id": "face:2", "person": None}],
         "person_pairs": [{"a": "person:Person A", "b": "person:Person B", "distance": 0.95, "co_visible": True},
                          {"a": "person:Person A", "b": "person:Person C", "distance": 0.4, "co_visible": False}]}


def test_tracks_by_shot():
    layer = {"tracks": [{"id": "body:0", "shot": "shot:3", "samples": []}, {"id": "body:1", "shot": "shot:0", "samples": []},
                        {"id": "body:2", "shot": "shot:3", "samples": []}]}
    out = by_shot(layer, {"shot:0": 0, "shot:3": 3})
    assert {k: [s.track["id"] for s in v] for k, v in out.items()} == {3: ["body:0", "body:2"], 0: ["body:1"]}
    with pytest.raises(KeyError):
        by_shot(layer, {"shot:3": 3})


def test_different_people_are_pairs_far_apart():
    assert different_people(FACES, 0.8) == {frozenset(("person:Person A", "person:Person B"))}


# two seated people side by side (test_garments.figure), A at x 400 and B at x 600, and their faces round their noses
PERSONS = {"face:0": "person:Person A", "face:1": "person:Person B"}


def seated(times, wide_a=False):
    """{shot 3: [Series]} bodies and faces of A and B at these times. wide_a: A's body box takes in all of B, as a body
    box of a person leaning across does (the bake-off took the smallest box holding the object's centre)."""
    a_box = [250, 100, 600, 830] if wide_a else box_of(400, 100)
    bodies = {"tracks": [
        {"id": "body:0", "shot": "shot:3", "face": "face:0",
         "samples": [{"t": t, "kpts": figure(400, 100), "box": a_box} for t in times]},
        {"id": "body:1", "shot": "shot:3", "face": "face:1",
         "samples": [{"t": t, "kpts": figure(600, 100), "box": box_of(600, 100)} for t in times]}]}
    faces = {"tracks": [{"id": "face:0", "shot": "shot:3", "samples": [{"t": t, "box": [350, 110, 100, 100]} for t in times]},
                        {"id": "face:1", "shot": "shot:3", "samples": [{"t": t, "box": [550, 110, 100, 100]} for t in times]}]}
    shots = {"shot:3": 3}
    return by_shot(bodies, shots), by_shot(faces, shots)


def test_a_worn_track_is_worn_by_the_body_whose_keypoint_region_holds_it():
    """B's necklace (on B's neck, x 600) is inside A's wide body box: the bake-off's rule gave it to... the smaller box,
    B, here; but A's region does not hold it, so A never could take it, nor take B's on a narrower box."""
    bodies, faces = seated([1.0, 1.167], wide_a=True)
    necklace = track("obj:0", "necklace", 3, [1.0, 1.167], box=(580, 230, 40, 30))
    chair = track("obj:1", "chair", 3, [1.0, 1.167], box=(580, 230, 40, 30))
    rare = track("obj:2", "necklace", 3, [1.0, 1.5, 2.0], box=(580, 230, 40, 30))   # a body near one of three samples
    ring = track("obj:3", "necklace", 3, [1.0, 1.167], box=(290, 470, 20, 20))      # at A's right hand (x 300, y 480)
    garment = {**track("obj:4", "pants", 3, [1.0, 1.167]), "found": "body", "body": "body:1", "wearer": "person:Person B"}
    assign_wearers([necklace, chair, rare, ring, garment], bodies, faces, PERSONS, {**CLASSES, "pants": CLASSES["chair"]},
                   0.5, 3.0)
    assert necklace["wearer"] == "person:Person B" and ring["wearer"] == "person:Person A"
    assert chair["wearer"] is None and rare["wearer"] is None
    assert garment["wearer"] == "person:Person B"   # a body garment keeps its body's person


def test_a_neighbour_in_front_does_not_take_the_necklace():
    """A small person in front (A) whose body box holds B's necklace: the bake-off gave it to the smallest body box
    holding the object's centre, A (OOO: the seated P3's items went to her neighbour's cluster in every run). A's
    keypoint regions do not hold it; B's neck does. Failed with the body-box rule."""
    times = [1.0, 1.167]
    shots = {"shot:3": 3}
    bodies = by_shot({"tracks": [
        {"id": "body:0", "shot": "shot:3", "face": "face:0",
         "samples": [{"t": t, "kpts": figure(540, 200, scale=0.6), "box": box_of(540, 200, scale=0.6)} for t in times]},
        {"id": "body:1", "shot": "shot:3", "face": "face:1",
         "samples": [{"t": t, "kpts": figure(600, 100), "box": box_of(600, 100)} for t in times]}]}, shots)
    faces = by_shot({"tracks": [
        {"id": "face:0", "shot": "shot:3", "samples": [{"t": t, "box": [510, 210, 60, 60]} for t in times]},
        {"id": "face:1", "shot": "shot:3", "samples": [{"t": t, "box": [550, 110, 100, 100]} for t in times]}]}, shots)
    necklace = track("obj:0", "necklace", 3, times, box=(590, 230, 20, 20))
    assign_wearers([necklace], bodies, faces, PERSONS, CLASSES, 0.5, 3.0)
    assert necklace["wearer"] == "person:Person B"


def test_a_body_is_one_person_over_its_whole_track():
    """A's body is linked (bodies layer) to B's face. Its nose is seen in A's own face at 1.0 and 1.333, not at 1.167
    (a turned head). A ring on A's hand at 1.167 only: the body's person is decided over all its samples (A), not over
    that one frame, where "head never seen" brought back the layer's link and gave the ring to B. Failed while the
    person was decided per object sample."""
    times = [1.0, 1.167, 1.333]
    shots = {"shot:3": 3}
    turned = np.array(figure(400, 100))
    turned[:5, 2] = 0                                                   # no nose, eyes or ears at 1.167
    bodies = by_shot({"tracks": [{"id": "body:0", "shot": "shot:3", "face": "face:1", "samples": [
        {"t": t, "kpts": turned.tolist() if t == 1.167 else figure(400, 100), "box": box_of(400, 100)} for t in times]}]}, shots)
    faces = by_shot({"tracks": [{"id": "face:0", "shot": "shot:3", "samples": [{"t": t, "box": [350, 110, 100, 100]} for t in times]},
                                {"id": "face:1", "shot": "shot:3", "samples": [{"t": t, "box": [550, 110, 100, 100]} for t in times]}]},
                    shots)
    ring = track("obj:0", "necklace", 3, [1.167], box=(290, 470, 20, 20))   # at A's right hand
    assign_wearers([ring], bodies, faces, PERSONS, CLASSES, 0.5, 3.0)
    assert ring["wearer"] == "person:Person A"


def test_a_worn_track_finds_bodies_sampled_at_their_own_rate():
    """Body-pose samples at ~7.49 a second: an object sample 67 ms from the nearest body sample was outside the bake-off's
    window (half of the layer's 1/8 s) and got no wearer. Failed before the window was the body's own step."""
    times = [round(1 + k * 4 / 29.97, 3) for k in range(8)]            # 1.0, 1.133, 1.267, ...
    bodies, faces = seated(times)
    necklace = track("obj:0", "necklace", 3, [1.2, 1.333, 1.5], box=(580, 230, 40, 30))   # 1.2: 67 ms from 1.133 and 1.267
    assign_wearers([necklace], bodies, faces, PERSONS, CLASSES, 1.0, 3.0)
    assert necklace["wearer"] == "person:Person B"


def test_pairs_that_may_never_be_one_object():
    e = unit(1, 0)
    a = track("obj:0", "necklace", 1, [1.0, 1.5], embedding=e, wearer="person:Person A")
    same_time = track("obj:1", "necklace", 1, [1.2, 2.0], embedding=e)
    other_person = track("obj:2", "necklace", 2, [1.0], embedding=e, wearer="person:Person B")
    maybe_same = track("obj:3", "necklace", 2, [1.0], embedding=e, wearer="person:Person C")   # 0.4 apart: not certain
    tiny = track("obj:4", "necklace", 2, [1.0], box=(0, 0, 30, 20), embedding=e)
    tiny_same_wearer = track("obj:5", "necklace", 3, [1.0], box=(0, 0, 30, 20), embedding=e, wearer="person:Person A")
    unembedded = track("obj:6", "necklace", 2, [1.0])
    different = {frozenset(("person:Person A", "person:Person B"))}
    assert pair_score(a, same_time, different, CFG) is None
    assert pair_score(a, other_person, different, CFG) is None
    assert pair_score(a, maybe_same, different, CFG) == pytest.approx(1.0)
    assert pair_score(a, tiny, different, CFG) is None
    assert pair_score(a, tiny_same_wearer, different, CFG) == pytest.approx(1.05)
    assert pair_score(a, unembedded, different, CFG) is None
    pearl = track("obj:7", "pearl necklace", 2, [1.0], embedding=e)
    assert pair_score(a, pearl, different, CFG) == pytest.approx(0.95)
    # two body garments of two persons never link, however close the persons' faces; with one wearer unknown they may
    jeans_c = {**track("obj:8", "pants", 4, [1.0], embedding=e, wearer="person:Person C"), "found": "body", "body": "body:1"}
    jeans_a = {**track("obj:9", "pants", 5, [1.0], embedding=e, wearer="person:Person A"), "found": "body", "body": "body:2"}
    jeans_x = {**track("obj:10", "pants", 6, [1.0], embedding=e, wearer=None), "found": "body", "body": "body:3"}
    assert frozenset(("person:Person A", "person:Person C")) not in different
    assert pair_score(jeans_c, jeans_a, different, CFG) is None
    # nor with a wearer unknown (a face missed, a body seen from behind): it could be anyone's, and jeans look alike
    assert pair_score(jeans_c, jeans_x, different, CFG) is None
    assert pair_score(jeans_x, {**jeans_x, "id": "obj:11", "shot": 7, "body": "body:4"}, different, CFG) is None
    assert pair_score(jeans_c, {**jeans_c, "id": "obj:12", "shot": 7, "body": "body:5"}, different, CFG) == pytest.approx(1.05)
    assert pair_score({**jeans_a, "found": "vocabulary"}, jeans_c, different, CFG) == pytest.approx(1.0)


def test_identities_link_one_family_by_average_linkage():
    tracks = [
        track("obj:0", "necklace", 1, [1.0, 1.5], embedding=unit(1, 0)),
        track("obj:1", "pearl necklace", 2, [3.0, 3.5], embedding=unit(1, 0.1)),   # cosine 0.995 - 0.05 margin: links
        track("obj:2", "necklace", 3, [5.0, 5.5], embedding=unit(0, 1)),           # unlike both
        track("obj:3", "chair", 1, [1.0, 1.5], embedding=unit(1, 0)),              # another family, never with necklaces
        track("obj:4", "chair", 2, [3.0, 3.5], embedding=unit(1, 0)),
        track("obj:5", "tree", 1, [1.0, 1.5], embedding=unit(1, 0)),               # lookalikes never link
        track("obj:6", "tree", 2, [3.0, 3.5], embedding=unit(1, 0)),
    ]
    ids = link_identities(tracks, CLASSES, set(), CFG)
    assert [(i["id"], i["family"], i["tracks"]) for i in ids] == [
        ("chair 1", "chair", ["obj:3", "obj:4"]),
        ("necklace 1", "necklace", ["obj:0", "obj:1"]),
        ("necklace 2", "necklace", ["obj:2"]),
        ("tree 1", "tree", ["obj:5"]), ("tree 2", "tree", ["obj:6"])]


def test_a_shots_lookalikes_are_one_identity():
    """A row of ceiling lights is one identity per shot, never linked across shots: 86 of OOO's 108 identities were
    single light tubes."""
    lights = [track(f"obj:{k}", "tree", 1 + k // 3, [1.0 + k, 1.5 + k], box=(100 * k, 0, 50, 50), embedding=unit(1, 0))
              for k in range(6)]
    ids = link_identities(lights, CLASSES, set(), CFG)
    assert [(i["id"], i["tracks"]) for i in ids] == [("tree 1", ["obj:0", "obj:1", "obj:2"]), ("tree 2", ["obj:3", "obj:4", "obj:5"])]


def test_duplicate_tracks_are_one_object_from_the_start():
    """One necklace tracked under two names at once: never linkable as a pair (same time), but one object."""
    a = track("obj:0", "necklace", 1, [1.0, 1.167], embedding=unit(1, 0))
    b = track("obj:1", "pearl necklace", 1, [1.0, 1.167], box=(102, 100, 80, 80), embedding=unit(0, 1))
    ids = link_identities([a, b], CLASSES, set(), CFG)
    assert [i["tracks"] for i in ids] == [["obj:0", "obj:1"]]
    assert ids[0]["id"] == "necklace 1"   # the most common label; ties go to the first in sorted order


def test_two_garments_of_one_body_are_never_one_object():
    """A cardigan and the top under it share a box, not a mask: the body pass made them two garments. A vocabulary
    name for the cardigan in the same frames is one object with it."""
    classes = {"cardigan": {"family": "upper-body clothing", "kind": "individual", "worn": True},
               "t-shirt": {"family": "upper-body clothing", "kind": "individual", "worn": True},
               "knit cardigan": {"family": "upper-body clothing", "kind": "individual", "worn": True}}
    outer = {**track("obj:0", "cardigan", 1, [1.0, 1.333], embedding=unit(1, 0)), "found": "body", "body": "body:3"}
    inner = {**track("obj:1", "t-shirt", 1, [1.0, 1.333], box=(110, 105, 70, 70), embedding=unit(0, 1)), "found": "body",
             "body": "body:3"}
    named = {**track("obj:2", "knit cardigan", 1, [1.0, 1.333], box=(102, 100, 80, 80), embedding=unit(1, 0)),
             "found": "vocabulary"}
    assert not duplicates(outer, inner, 0.6) and duplicates(outer, named, 0.6)
    ids = link_identities([outer, inner, named], classes, set(), CFG)
    assert sorted(i["tracks"] for i in ids) == [["obj:0", "obj:2"], ["obj:1"]]


def test_a_merge_needs_every_cross_pair_linkable():
    """A and B link, B and C link, but A and C are on screen together: C never joins A's identity."""
    a = track("obj:0", "chair", 1, [1.0, 2.0], embedding=unit(1, 0))
    c = track("obj:2", "chair", 1, [1.5, 2.5], box=(900, 100, 80, 80), embedding=unit(1, 0.05))
    b = track("obj:1", "chair", 2, [5.0, 6.0], embedding=unit(1, 0.02))
    ids = link_identities([a, b, c], CLASSES, set(), CFG)
    assert sorted(len(i["tracks"]) for i in ids) == [1, 2]


def test_worn_by_lists_the_identitys_wearers():
    a = track("obj:0", "necklace", 1, [1.0], embedding=unit(1, 0), wearer="person:Person A")
    b = track("obj:1", "necklace", 2, [1.0], embedding=unit(1, 0), wearer=None)
    assert link_identities([a, b], CLASSES, set(), CFG)[0]["worn_by"] == ["person:Person A"]


def test_the_representative_view_is_closest_to_the_identitys_mean():
    a = track("obj:0", "chair", 1, [1.0], embedding=unit(1, 0))
    b = track("obj:1", "chair", 2, [1.0], embedding=unit(1, 0.2))
    a["view_embeddings"], a["views"] = [unit(0, 1), unit(1, 0.1)], ["drifted", "good"]
    b["view_embeddings"], b["views"] = [unit(1, 0.3)], ["fair"]
    by_id = {"obj:0": a, "obj:1": b}
    assert representative_view({"tracks": ["obj:0", "obj:1"]}, by_id) == "good"
    assert representative_view({"tracks": ["obj:9"]}, {"obj:9": track("obj:9", "chair", 1, [1.0])}) is None


def test_crop_files_are_named_as_convert_bakeoff_named_them_and_never_collide():
    """In their own folder: face-track writes crops/face_<i>.jpg and crops/person_<name>.jpg, and an object Haiku
    called "face" would otherwise be crops/face_1.jpg, which stops the bundle export (two nodes write one asset)."""
    assert slug("camo t-shirt 1") == "camo_t_shirt_1"
    files = crop_files([{"id": "t-shirt 1"}, {"id": "t shirt 1"}, {"id": "necklace 12"}, {"id": "face 1"}])
    assert files == {"t-shirt 1": "crops/objects/t_shirt_1.jpg", "t shirt 1": "crops/objects/t_shirt_1_2.jpg",
                     "necklace 12": "crops/objects/necklace_12.jpg", "face 1": "crops/objects/face_1.jpg"}


def test_the_layer_is_what_convert_bakeoff_wrote():
    kept = track("obj:0", "necklace", 4, [1.0, 1.167], wearer="person:Person A")
    gone = track("obj:1", "chair", 5, [2.0, 2.167])
    ids = [{"id": "necklace 1", "label": "necklace", "family": "necklace", "worn_by": ["person:Person A"], "tracks": ["obj:0"]}]
    doc = layer_content(6, [kept], [gone], ids, {"obj:0": "object", "obj:1": "graphic"}, {"necklace 1": "assets/crops/necklace_1.jpg"})
    s = {"t": 1.0, "box": [100, 100, 80, 80], "poly": [[0, 0], [1, 0], [1, 1]], "score": 0.8}
    assert doc["fps"] == 6
    assert doc["tracks"] == [{"id": "obj:0", "label": "necklace", "identity": "objid:necklace 1", "shot": "shot:4", "start": 1.0,
                              "end": 1.167, "wearer": "person:Person A", "verdict": "object", "found": "vocabulary",
                              "samples": [s, {**s, "t": 1.167}]}]
    assert doc["identities"] == [{"id": "objid:necklace 1", "label": "necklace", "family": "necklace", "worn_by": ["person:Person A"],
                                  "tracks": ["obj:0"], "crop": "assets/crops/necklace_1.jpg"}]
    assert doc["rejected_tracks"] == [{"id": "obj:1", "label": "chair", "verdict": "graphic", "shot": "shot:5", "start": 2.0,
                                       "end": 2.167, "found": "vocabulary", "samples": [{**s, "t": 2.0}, {**s, "t": 2.167}]}]
    shirt = {**track("obj:2", "t-shirt", 4, [1.0]), "found": "body", "body": "body:7", "wearer": None}
    body_doc = layer_content(6, [shirt], [], [{"id": "t-shirt 1", "label": "t-shirt", "family": "upper-body clothing",
                                               "worn_by": [], "tracks": ["obj:2"]}], {"obj:2": "object"}, {})
    assert {k: body_doc["tracks"][0][k] for k in ("found", "body", "wearer")} == {"found": "body", "body": "body:7", "wearer": None}
    assert layer_content(6, [], [], [{"id": "chair 1", "label": "chair", "family": "chair", "worn_by": [], "tracks": []}],
                         {}, {})["identities"][0]["crop"] is None


# --- linking at a film's scale ----------------------------------------------------------------------------------------

def bake_off_link(members, different, c):
    """The bake-off's merge loop (objects.py link_identities), for one non-lookalike family: groups of member indices."""
    n = len(members)
    score = [[pair_score(members[i], members[j], different, c) if i != j else None for j in range(n)] for i in range(n)]
    groups = [[i] for i in range(n)]
    while True:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                pairs = [score[a][b] for a in groups[i] for b in groups[j]]
                if any(p is None for p in pairs):
                    continue
                avg = float(np.mean(pairs))
                if avg >= c["link_threshold"] and (best is None or avg > best[0]):
                    best = (avg, i, j)
        if best is None:
            return groups
        _, i, j = best
        groups[i] += groups.pop(j)


def chairs(n, shots, seed):
    """n chair tracks over `shots` shots, a few looks each with noise, some on screen together."""
    rng = np.random.default_rng(seed)
    looks = rng.normal(size=(6, 16))
    out = []
    for k in range(n):
        shot = int(rng.integers(shots))
        start = float(rng.integers(0, 4))
        e = looks[k % 6] + rng.normal(scale=0.6, size=16)
        out.append(track(f"obj:{k}", "chair", shot, [start, start + 0.5], box=(100 * k, 100, 80, 80), embedding=e / np.linalg.norm(e)))
    return out


@pytest.mark.parametrize("seed", range(5))
def test_linking_merges_as_the_bake_offs_loop_did(seed):
    tracks = chairs(40, 12, seed)
    want = sorted(sorted(f"obj:{i}" for i in g) for g in bake_off_link(tracks, set(), CFG))
    got = sorted(sorted(i["tracks"]) for i in link_identities(tracks, CLASSES, set(), CFG))
    assert got == want and len(got) < 40   # and some tracks did link


def test_linking_a_films_tracks_takes_seconds_not_hours(monkeypatch):
    """A family of a film's size, and a large lookalike family, which never links. Counted, not timed: each pair of
    tracks is scored once (the bake-off's loop also re-read every pair on every merge, cubic in the family's size: over
    a minute for these 400 tracks), and a lookalike family scores none. The time limit is only a backstop, far above
    what the counted work takes (well under a second here)."""
    import time
    import linking
    calls = []
    scored = linking.pair_score
    monkeypatch.setattr(linking, "pair_score", lambda *a, **k: calls.append(1) or scored(*a, **k))
    t0 = time.time()
    link_identities(chairs(400, 120, 0), CLASSES, set(), CFG)
    assert len(calls) <= 400 * 399 // 2
    calls.clear()
    lights = [track(f"obj:{k}", "tree", k // 10, [1.0, 1.5], box=(30 * (k % 10), 0, 20, 20), embedding=unit(1, 0)) for k in range(3000)]
    assert len(link_identities(lights, CLASSES, set(), CFG)) == 300 and not calls   # one per shot
    assert time.time() - t0 < 30
