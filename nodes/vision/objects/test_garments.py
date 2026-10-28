"""Clothing from bodies (garments.py), without the models: crops, keypoint regions, whose a detection is, which become
items, items into tracks, and whose body it is.
Run: uv run --no-project --with pytest --with numpy --with av==16.1.0 --with pillow --with pyarrow==26.0.0 --with-editable ./sdk pytest -q nodes/vision/objects"""

import numpy as np
import pytest

import garments
from garments import (KEEP_SCORE, NOUNS, PROMPTS, Series, body_person, crop_box, garment_tracks, head_point, owner,
                      person_owner, pick, regions, session_names)

SEEN = 9.0   # a keypoint score above the bodies layer's threshold (3.0)


def figure(cx, top, scale=1.0, face=True, legs=True):
    """133 COCO-WholeBody keypoints of a person standing at x = cx with the top of the head at y = top: what the bodies
    layer gives. Face points (23-90) only with face; knees, ankles and feet only with legs."""
    k = np.zeros((133, 3))
    s = scale

    def put(i, x, y):
        k[i] = [cx + x * s, top + y * s, SEEN]
    put(0, 0, 60)                                       # nose
    put(1, 15, 45), put(2, -15, 45), put(3, 30, 50), put(4, -30, 50)
    put(5, 80, 160), put(6, -80, 160)                   # shoulders
    put(7, 100, 280), put(8, -100, 280), put(9, 100, 380), put(10, -100, 380)
    put(11, 50, 420), put(12, -50, 420)                 # hips
    if legs:
        put(13, 50, 600), put(14, -50, 600), put(15, 50, 780), put(16, -50, 780)
        for i, x in zip(range(17, 23), (60, 40, 50, -60, -40, -50)):
            put(i, x, 800)
    if face:
        for i in range(23, 91):                         # a face from y 20 to 110, x -40..40; the chin (face-8) lowest
            a = (i - 23) / 68 * 2 * np.pi
            put(i, 40 * np.cos(a), 65 + 45 * np.sin(a))
        put(23 + 8, 0, 110)
    return k.tolist()


def box_of(cx, top, scale=1.0):
    return [cx - 130 * scale, top, 260 * scale, 830 * scale]


def test_the_prompts_are_the_nouns_then_person():
    assert PROMPTS == (*NOUNS, "person")
    assert set(NOUNS.values()) == set(garments.FAMILIES)
    assert garments.classes()["jacket"] == {"family": "upper-body clothing", "kind": "individual", "worn": True}


def test_the_body_pass_takes_its_own_nouns_out_of_the_vocabulary():
    """A vocabulary name the body pass asks for is not a session prompt: a garment would be found twice, and "person"
    (NES: Haiku named it in 5 shots despite the rule against people) made people into objects."""
    assert session_names(["helmet", "person", "jacket", "bicycle", "necklace"]) == (["helmet", "bicycle"],
                                                                                    ["person", "jacket", "necklace"])


def test_a_crop_is_a_square_about_the_body_inside_the_frame():
    assert crop_box([100, 100, 200, 400], 1920, 1080) == (0, 80, 420, 520)   # 440 wide about x 200, cut at 0
    assert crop_box([1800, 900, 100, 100], 1920, 1080) == (1795, 895, 1905, 1005)


def test_regions_follow_the_keypoints_and_need_them():
    r = regions(figure(500, 100), box_of(500, 100), 3.0)
    assert set(r) == {"upper", "lower", "feet", "hands", "head", "eyes", "neck"}
    x0, y0, x1, y1 = r["upper"]
    assert x0 < 420 and x1 > 580 and y0 < 260 < 520 < y1          # shoulders (y 260) to hips (y 520)
    assert r["lower"][1] < 520 and r["lower"][3] > 880             # hips to ankles
    assert r["head"][3] < r["upper"][3]
    close_up = regions(figure(500, 100, legs=False), [370, 100, 260, 500], 3.0)
    assert "feet" not in close_up and close_up["lower"][3] == pytest.approx(600 + 0.1 * 260)   # the box's bottom
    assert regions(np.zeros((133, 3)).tolist(), [0, 0, 100, 100], 3.0) == {}   # nothing seen: no region


def test_a_detection_is_the_body_whose_region_holds_it():
    bodies = {"body:0": regions(figure(400, 100), box_of(400, 100), 3.0),
              "body:1": regions(figure(600, 100), box_of(600, 100), 3.0)}
    assert owner("upper", (420, 400), bodies) == "body:0"
    assert owner("upper", (520, 400), bodies) == "body:1"            # in both regions: the nearer centre
    assert owner("upper", (1500, 400), bodies) is None
    assert owner("feet", (400, 100), bodies) is None                 # a slot's own region only


def test_a_person_mask_is_the_body_with_most_keypoints_inside():
    kpts = {"body:0": np.array(figure(400, 100)), "body:1": np.array(figure(700, 100))}
    mask = np.zeros((1000, 600), bool)
    mask[:, :240] = True                    # crop at x 200: frame x 200-439 holds body:0's middle points
    assert person_owner(mask, (200, 0), kpts, 3.0)[0] == "body:0"
    assert person_owner(np.zeros((1000, 600), bool), (200, 0), kpts, 3.0) == (None, 0)


def det(noun, score, mask):
    return {"noun": noun, "score": score, "mask": mask}


def blob(x0, x1, y0=0, y1=10):
    m = np.zeros((10, 20), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_pick_takes_each_slots_best_and_an_inner_layer_only_if_it_is_another_region():
    detections = [
        det("cardigan", 0.9, blob(0, 10)),        # 0: outer
        det("shirt", 0.95, blob(0, 10)),          # 1: same region as the outer: not an inner layer
        det("t-shirt", 0.7, blob(10, 14)),        # 2: another region: the inner layer
        det("pants", 0.6, blob(0, 20, 5)),        # 3
        det("skirt", 0.8, blob(0, 20, 6)),        # 4: the better lower item
        det("shoe", 0.7, blob(0, 2)), det("shoe", 0.6, blob(18, 20)),   # 5, 6: both shoes, one item
        det("hat", 0.4, blob(0, 5)),              # 7: below KEEP_SCORE
        det("necklace", 0.9, blob(4, 6)),         # 8: someone else's
        det("person", 0.9, blob(0, 20)),          # 9
    ]
    owners = ["body:0"] * 8 + ["body:1", "body:0"]
    assert pick("body:0", detections, owners) == {"person": [9], "upper/outer": [0], "upper/inner": [2], "lower": [4],
                                                   "feet": [5, 6]}
    assert pick("body:1", detections, owners) == {"neck": [8]}
    assert 0.4 < KEEP_SCORE


def test_garment_tracks_are_a_body_tracks_items_of_one_kind_labelled_by_their_commonest_noun():
    def item(body, key, t, noun, score=0.8):
        return {"body": body, "key": key, "t": t, "noun": noun, "score": score, "sample": {"t": t}}
    items = [item("body:2", "upper/inner", 1.0, "shirt"), item("body:2", "upper/inner", 1.333, "t-shirt"),
             item("body:2", "upper/inner", 0.667, "t-shirt", 0.6), item("body:2", "lower", 1.0, "pants"),
             item("body:10", "upper/outer", 1.0, "jacket"), item("body:10", "upper/outer", 1.333, "jacket"),
             item("body:2", "person", 1.0, "person"), item("body:2", "person", 1.333, "person")]
    out = garment_tracks(items, 2)
    assert [(g["body"], g["key"], g["label"], g["family"], [it["t"] for it in g["items"]]) for g in out] == [
        ("body:2", "upper/inner", "t-shirt", "upper-body clothing", [0.667, 1.0, 1.333]),
        ("body:10", "upper/outer", "jacket", "upper-body clothing", [1.0, 1.333])]   # one lower sample: too short; no person
    assert out[0]["score"] == pytest.approx((0.8 + 0.8 + 0.6) / 3)
    tie = garment_tracks([item("body:0", "lower", 1.0, "skirt"), item("body:0", "lower", 2.0, "pants")], 2)
    assert tie[0]["label"] == "pants"    # a tie goes to the first in NOUNS


def test_a_series_finds_the_sample_within_half_its_own_step():
    """Body-pose samples at ~7.49 a second on 29.97 fps video. Half of its layer's 1/8 s missed object samples between
    two body samples (the bake-off's wearer window, ~6% of OOO's object samples); half its own step does not."""
    times = [round(k * 4 / 29.97, 3) for k in range(5)]          # 0, 0.133, 0.267, ...
    s = Series({"samples": [{"t": t} for t in times]})
    assert s.at(0.2)["t"] == 0.133 and s.at(0.21)["t"] == 0.267   # 67 ms from both: the first
    assert abs(0.2 - 0.133) > 1 / 16                              # outside the old window
    assert s.at(0.7) is None and s.at(-0.1) is None
    assert Series({"samples": [{"t": 1.0}]}).at(1.0) == {"t": 1.0} and Series({"samples": [{"t": 1.0}]}).at(1.01) is None
    assert Series({"samples": []}).at(0.0) is None


def face_track(id, box, times):
    return Series({"id": id, "samples": [{"t": t, "box": box} for t in times]})


def test_a_body_is_the_person_whose_face_holds_its_nose():
    """On OOO a panelist's body at the frame's edge was linked (bodies layer) to the face of the panelist beside her.
    Its nose was in no face's box: the body is nobody's, and its clothes have no wearer, not the neighbour."""
    times = [0.0, 0.125, 0.25]
    body = Series({"face": "face:9", "samples": [{"t": t, "kpts": figure(400, 100), "box": box_of(400, 100)} for t in times]})
    persons = {"face:1": "person:Person A", "face:9": "person:Person B"}
    own_face = face_track("face:1", [350, 120, 100, 100], times)          # holds the nose at (400, 160)
    neighbour = face_track("face:9", [700, 120, 100, 100], times)
    assert head_point(figure(400, 100), 3.0) == pytest.approx([400, 160])
    assert body_person(body, times, [own_face, neighbour], persons, 3.0) == "person:Person A"
    assert body_person(body, times, [neighbour], persons, 3.0) is None    # its head is seen, in no face: nobody's
    back = np.array(figure(400, 100))
    back[:5, 2] = 0                                                        # seen from behind: no nose, no eyes
    away = Series({"face": "face:9", "samples": [{"t": t, "kpts": back.tolist(), "box": box_of(400, 100)} for t in times]})
    assert body_person(away, times, [neighbour], persons, 3.0) == "person:Person B"   # the layer's link, then
