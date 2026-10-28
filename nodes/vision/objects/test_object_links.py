"""Wearers, cross-shot identities and the objects layer (linking.py), without the models or the API.
Run: uv run --no-project --with pytest --with numpy --with av==16.1.0 --with pillow --with-editable ./sdk pytest -q nodes/vision/objects"""

import numpy as np
import pytest

from linking import (assign_wearers, body_samples, box_iou, check_classes, check_views, clean_names, crop_files,
                     different_people, duplicates, layer_content, link_identities, pair_score, representative_view, slug)

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


def test_body_samples_are_those_of_bodies_linked_to_a_person():
    bodies = {"tracks": [
        {"shot": "shot:3", "face": "face:0", "samples": [{"t": 1.0, "box": [0, 0, 500, 900]}]},
        {"shot": "shot:3", "face": "face:2", "samples": [{"t": 1.0, "box": [600, 0, 500, 900]}]},   # a face with no person
        {"shot": "shot:3", "face": None, "samples": [{"t": 1.0, "box": [0, 0, 1900, 1000]}]}]}
    assert body_samples(bodies, FACES, {"shot:3": 3}) == {3: [(1.0, [0, 0, 500, 900], "person:Person A")]}


def test_different_people_are_pairs_far_apart():
    assert different_people(FACES, 0.8) == {frozenset(("person:Person A", "person:Person B"))}


def test_a_worn_track_is_worn_by_the_smallest_body_holding_it_in_enough_samples():
    bodies = {3: [(1.0, [0, 0, 1000, 1000], "person:Person A"), (1.0, [50, 50, 300, 300], "person:Person B"),
                  (1.125, [0, 0, 1000, 1000], "person:Person A")]}
    worn = track("obj:0", "necklace", 3, [1.0, 1.167])           # centre (140, 140): both bodies at 1.0, only A near 1.167
    chair = track("obj:1", "chair", 3, [1.0, 1.167])
    rare = track("obj:2", "necklace", 3, [1.0, 1.5, 2.0])         # held by a body in one of three samples
    assign_wearers([worn, chair, rare], bodies, 8, CLASSES, 0.5)
    # one vote each (B at 1.0 as the smaller box, A at 1.125, within half a body sample of 1.167): the first counted wins
    assert worn["wearer"] == "person:Person B"
    assert chair["wearer"] is None and rare["wearer"] is None


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


def test_duplicate_tracks_are_one_object_from_the_start():
    """One necklace tracked under two names at once: never linkable as a pair (same time), but one object."""
    a = track("obj:0", "necklace", 1, [1.0, 1.167], embedding=unit(1, 0))
    b = track("obj:1", "pearl necklace", 1, [1.0, 1.167], box=(102, 100, 80, 80), embedding=unit(0, 1))
    ids = link_identities([a, b], CLASSES, set(), CFG)
    assert [i["tracks"] for i in ids] == [["obj:0", "obj:1"]]
    assert ids[0]["id"] == "necklace 1"   # the most common label; ties go to the first in sorted order


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
                              "end": 1.167, "wearer": "person:Person A", "verdict": "object",
                              "samples": [s, {**s, "t": 1.167}]}]
    assert doc["identities"] == [{"id": "objid:necklace 1", "label": "necklace", "family": "necklace", "worn_by": ["person:Person A"],
                                  "tracks": ["obj:0"], "crop": "assets/crops/necklace_1.jpg"}]
    assert doc["rejected_tracks"] == [{"id": "obj:1", "label": "chair", "verdict": "graphic", "shot": "shot:5", "start": 2.0,
                                       "end": 2.167, "samples": [{**s, "t": 2.0}, {**s, "t": 2.167}]}]
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


def test_linking_a_films_tracks_takes_seconds_not_hours():
    """A family of a film's size (the bake-off's loop grew with the cube of it), and a large lookalike family, which
    never links and needs no pair scores at all."""
    import time
    t0 = time.time()
    link_identities(chairs(400, 120, 0), CLASSES, set(), CFG)
    lights = [track(f"obj:{k}", "tree", k // 10, [1.0, 1.5], box=(30 * (k % 10), 0, 20, 20), embedding=unit(1, 0)) for k in range(3000)]
    assert len(link_identities(lights, CLASSES, set(), CFG)) == 3000
    assert time.time() - t0 < 5
