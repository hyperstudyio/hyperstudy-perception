"""Body linking and tracking (bodies.py), without the models.
Run: uv run --project envs/vision --with pytest pytest nodes/vision/body-pose"""

import json
from pathlib import Path

import numpy as np
import pytest

from bodies import (drop_duplicate_bodies, face_boxes_at, face_in_body, keypoints_in_face, layer_tracks, link_faces,
                    one_body_per_face, track_bodies)

KEYPOINTS = json.loads((Path(__file__).parent / "keypoints.json").read_text())
HEAD = [KEYPOINTS["names"].index(n) for n in ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder")]
THRESHOLD = 3.0


def body(box, head_at=None, score=0.9, visible=5.0):
    """A body with 133 keypoints, all invisible but the head and shoulders, which sit around head_at (x, y)."""
    kpts = np.zeros((133, 3))
    if head_at is not None:
        hx, hy = head_at
        for i, (dx, dy) in zip(HEAD, [(0, 0), (-5, -5), (5, -5), (-10, 0), (10, 0), (-30, 40), (30, 40)]):
            kpts[i] = [hx + dx, hy + dy, visible]
    return {"box": box, "score": score, "kpts": kpts}


def stats():
    return {"box_candidates": 0, "keypoint_check_failed": 0, "linked_samples": 0, "unlinked_samples": 0}


def test_keypoints_are_coco_wholebody_with_nose_and_eyes_first():
    assert len(KEYPOINTS["names"]) == 133 and KEYPOINTS["names"][:3] == ["nose", "left_eye", "right_eye"]
    assert all(0 <= a < 133 and 0 <= b < 133 for a, b in KEYPOINTS["skeleton"])
    assert sorted(i for g in KEYPOINTS["groups"].values() for i in g) == list(range(133))


def test_face_boxes_are_the_shots_faces_interpolated_never_extrapolated():
    tracks = [{"id": "face:0", "shot": "shot:0", "start": 1.0, "end": 1.25,
               "samples": [{"t": 1.0, "box": [0, 0, 10, 10]}, {"t": 1.25, "box": [10, 0, 10, 10]}]},
              {"id": "face:1", "shot": "shot:1", "start": 1.0, "end": 1.25, "samples": [{"t": 1.0, "box": [0, 0, 1, 1]}]}]
    assert face_boxes_at(1.0, "shot:0", tracks) == {"face:0": [0, 0, 10, 10]}
    assert face_boxes_at(1.125, "shot:0", tracks)["face:0"] == pytest.approx([5, 0, 10, 10])
    assert face_boxes_at(1.5, "shot:0", tracks) == {}                                    # after the track ends
    gap = [{**tracks[0], "end": 3.0, "samples": [{"t": 1.0, "box": [0, 0, 10, 10]}, {"t": 3.0, "box": [0, 0, 10, 10]}]}]
    assert face_boxes_at(2.0, "shot:0", gap) == {}                                       # too long a gap to fill


def test_a_face_links_only_inside_the_top_of_a_body_and_where_its_eyes_are():
    face = [90, 10, 20, 20]
    assert face_in_body(face, [50, 0, 100, 200])
    assert not face_in_body(face, [50, -150, 100, 200])                                 # in the body's lower half
    assert keypoints_in_face(body([50, 0, 100, 200], head_at=(100, 20))["kpts"], face, THRESHOLD) == 3
    assert keypoints_in_face(body([50, 0, 100, 200], head_at=(300, 20))["kpts"], face, THRESHOLD) == 0


def test_two_bodies_on_one_face_link_the_one_whose_eyes_are_in_it():
    face = {"face:0": [90, 10, 20, 20]}
    here, elsewhere = body([50, 0, 100, 200], head_at=(100, 20)), body([40, 0, 120, 220], head_at=(300, 20), score=0.99)
    s = stats()
    link_faces([elsewhere, here], face, THRESHOLD, s)
    assert here.get("face") == "face:0" and "face" not in elsewhere
    assert s["keypoint_check_failed"] == 1 and s["linked_samples"] == 1 and s["unlinked_samples"] == 1


def test_a_second_detection_of_one_person_is_dropped_but_a_person_beside_them_is_not():
    one = body([0, 0, 100, 200], head_at=(50, 20), score=0.74)
    twin = body([5, 10, 90, 150], head_at=(51, 21), score=0.50)          # same head and shoulders: the same person
    beside = body([60, 0, 100, 200], head_at=(110, 20), score=0.6)       # overlapping box, another head
    kept = drop_duplicate_bodies([one, twin, beside], HEAD, THRESHOLD)
    assert kept == [one, beside]


def test_tracks_follow_faces_and_iou_end_at_cuts_and_never_share_a_face():
    linked = lambda t: {**body([0, 0, 100, 200]), "face": "face:0"}
    frames = [(0.0, [linked(0)]), (0.125, [linked(0), body([300, 0, 100, 200])]), (0.25, [body([302, 0, 100, 200])]),
              (0.375, [linked(0)])]
    tracks = track_bodies(frames, ["shot:0", "shot:0", "shot:0", "shot:1"])
    assert [(tr["shot"], tr["face"], [s["t"] for s in tr["samples"]]) for tr in tracks] == [
        ("shot:0", "face:0", [0.0, 0.125]), ("shot:0", None, [0.125, 0.25]), ("shot:1", "face:0", [0.375])]
    with pytest.raises(AssertionError):
        one_body_per_face([{"face": "face:0", "samples": [{"t": 0}, {"t": 2}]}, {"face": "face:0", "samples": [{"t": 1}, {"t": 3}]}])


def test_a_face_moving_to_an_overlapping_unlinked_track_starts_a_new_track_rather_than_two_at_once():
    a = lambda: {**body([0, 0, 100, 200]), "face": "face:0"}
    c = lambda: body([10, 0, 100, 200])                              # overlaps a's box; unlinked
    c_linked = lambda: {**c(), "face": "face:0"}
    frames = [(0.0, [a(), c()]), (0.125, [c()]), (0.6, [c_linked()])]   # a unseen for longer than the gap by 0.6
    tracks = track_bodies(frames, ["shot:0"] * 3)                   # no AssertionError: one face, one track at a time
    linked = sorted(([s["t"] for s in tr["samples"]], tr["face"]) for tr in tracks)
    assert linked == [([0.0], "face:0"), ([0.0, 0.125], None), ([0.6], "face:0")]


def test_layer_tracks_are_body_ids_in_shot_order_rounded():
    b = {**body([0.04, 1.26, 10, 20]), "face": None}
    tracks = layer_tracks([{"shot": "shot:10", "face": "face:3", "samples": [{"t": 2.0, **b}]},
                           {"shot": "shot:2", "face": None, "samples": [{"t": 5.0, **b}]}])
    assert [(t["id"], t["shot"], t["face"], t["link"]) for t in tracks] == [
        ("body:0", "shot:2", None, None), ("body:1", "shot:10", "face:3", "face-in-box+keypoints")]
    assert tracks[0]["samples"][0]["box"] == [0.0, 1.3, 10.0, 20.0] and len(tracks[0]["samples"][0]["kpts"]) == 133
