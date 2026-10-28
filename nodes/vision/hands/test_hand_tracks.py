"""Hand proposals, matching and tracking (hand_tracks.py), without the models.
Run: uv run --no-project --with pytest --with numpy --with-editable ./sdk pytest nodes/vision/hands"""

import json
from pathlib import Path

import numpy as np
import pytest

from hand_tracks import (HAND_BOX_PAD, KEYPOINTS, MIN_HAND_PX, SKELETON, body_indices, bodies_by_time, frame_hands,
                         layer_hands, match_detections, propose, square_box, track_hands)

NAMES = json.loads((Path(__file__).parents[1] / "body-pose" / "keypoints.json").read_text())["names"]
IDX = body_indices(NAMES)
THRESHOLD = 3.0
W, H = 1920, 1080


def body_kpts(hands=None, wrists=None, score=5.0):
    """133 whole-body keypoints, all invisible but the given wrists {side: (x, y)} and hands {side: [(x, y), ...]}."""
    kpts = np.zeros((133, 3))
    for side, xy in (wrists or {}).items():
        kpts[IDX[side]["wrist"]] = [*xy, score]
    for side, points in (hands or {}).items():
        for i, xy in zip(IDX[side]["hand"], points):
            kpts[i] = [*xy, score]
    return kpts


def a_body(name="body:0", face="face:0", **kw):
    return {"person": {"body": name, "face": face}, "kpts": body_kpts(**kw)}


def hand_at(x, y, r=40):
    """Five hand keypoints spread around (x, y): a 2r x 2r hand."""
    return [(x, y), (x - r, y - r), (x + r, y - r), (x - r, y + r), (x + r, y + r)]


class Fake:
    """Detector and keypoint model: fixed detections; keypoints are the box's corners and centre, repeated."""

    def __init__(self, detections=()):
        self.detections, self.posed = list(detections), []

    def detect(self, frame):
        return list(self.detections)

    def keypoints(self, frame, box):
        self.posed.append(box)
        return np.tile([box[0], box[1], 0.5], (21, 1))


def stats():
    return dict.fromkeys(("wrist_below_gate", "detector_below_gate", "detector_too_large", "detector_duplicate",
                          "detector_linked", "detector_unlinked"), 0)


FRAME = np.zeros((H, W, 3), np.uint8)


def test_keypoints_are_the_21_hand_keypoints_and_the_body_layer_keeps_hands_where_coco_wholebody_does():
    assert len(KEYPOINTS) == 21 and all(0 <= a < 21 and 0 <= b < 21 for a, b in SKELETON)
    assert IDX == {"left": {"wrist": 9, "hand": list(range(91, 112))}, "right": {"wrist": 10, "hand": list(range(112, 133))}}
    with pytest.raises(ValueError, match="no hand keypoints"):
        body_indices(NAMES[:91])


def test_a_proposal_is_a_square_around_the_visible_hand_and_its_wrist():
    body = propose(a_body(hands={"left": hand_at(500, 500)}, wrists={"left": (500, 600)}), IDX, THRESHOLD, W, H)
    points = np.array(hand_at(500, 500) + [(500, 600)], dtype=float)
    assert body["proposals"] == {"left": square_box(points, HAND_BOX_PAD)}
    assert body["proposals"]["left"] == pytest.approx([500 - 84, 530 - 84, 168, 168])   # 460-600: 140 px tall x 1.2
    assert list(body["wrists"]["left"]) == [500, 600] and body["wrists"]["right"] is None


def test_a_hand_needs_four_visible_keypoints_inside_the_frame():
    few = propose(a_body(hands={"left": hand_at(500, 500)[:3]}), IDX, THRESHOLD, W, H)
    assert few["proposals"] == {}
    faint = propose(a_body(hands={"left": hand_at(500, 500)}, score=2.9), IDX, THRESHOLD, W, H)
    assert faint["proposals"] == {}
    # an arm leaving the bottom of the frame: RTMW puts the hand just below it, with a high score
    off = propose(a_body(hands={"right": hand_at(500, H + 10, r=5)}, wrists={"right": (500, H + 5)}), IDX, THRESHOLD, W, H)
    assert off["proposals"] == {} and off["wrists"]["right"] is None


def test_detector_boxes_match_the_nearest_wrist_within_one_box_side():
    bodies = [propose(a_body(wrists={"left": (100, 100), "right": (400, 100)}), IDX, THRESHOLD, W, H)]
    detections = [([380, 80, 50, 50], 0.9), ([90, 150, 50, 50], 0.8), ([1000, 1000, 50, 50], 0.7)]
    # box 0 holds the right wrist; box 1 is 50 px (one side) below the left wrist: at the gate; box 2 is far from both
    assert match_detections(detections, bodies) == {0: (0, "right"), 1: (0, "left")}
    assert match_detections([([90, 151, 50, 50], 0.8)], bodies) == {}                    # just past the gate
    assert match_detections(detections, []) == {} and match_detections([], bodies) == {}


def test_two_wrists_in_one_box_go_to_the_first_as_scipy_assigns_them():
    bodies = [propose(a_body(name="body:0", wrists={"left": (100, 100)}), IDX, THRESHOLD, W, H),
              propose(a_body(name="body:1", wrists={"right": (110, 110)}), IDX, THRESHOLD, W, H)]
    assert match_detections([([50, 50, 100, 100], 0.9)], bodies) == {0: (0, "left")}


def test_wrist_hands_then_detector_hands_linked_dropped_or_unlinked():
    body = propose(a_body(hands={"left": hand_at(500, 500)}, wrists={"left": (500, 600), "right": (900, 600)}), IDX, THRESHOLD, W, H)
    left = body["proposals"]["left"]
    model = Fake([
        ([left[0] + 5, left[1] + 5, left[2], left[3]], 0.9),   # the left hand again (IoU >= 0.3): dropped
        ([880, 580, 60, 60], 0.8),                             # around the right wrist, which has no hand: linked
        ([1500, 200, 60, 60], 0.7),                            # nobody's wrist near: kept, unlinked
        ([1500, 600, 20, 60], 0.6),                            # narrower than MIN_HAND_PX
        ([100, 100, 600, 600], 0.5),                           # larger than half the frame height
    ])
    s = stats()
    hands = frame_hands(FRAME, [body], model, s)
    assert [(h["box_source"], h["side"], h["person"]) for h in hands] == [
        ("wrist", "left", {"body": "body:0", "face": "face:0"}), ("detector", "right", {"body": "body:0", "face": "face:0"}),
        ("detector", None, None)]
    assert s == {"wrist_below_gate": 0, "detector_below_gate": 1, "detector_too_large": 1, "detector_duplicate": 1,
                 "detector_linked": 1, "detector_unlinked": 1}
    assert model.posed == [left, [880, 580, 60, 60], [1500, 200, 60, 60]]   # keypoints only for the hands kept


def test_a_detector_box_on_a_wrist_that_already_has_its_hand_is_that_hand_again():
    body = propose(a_body(hands={"left": hand_at(500, 500)}, wrists={"left": (500, 600)}), IDX, THRESHOLD, W, H)
    s = stats()
    hands = frame_hands(FRAME, [body], Fake([([470, 580, 60, 60], 0.9)]), s)   # on the wrist, barely overlapping the hand box
    assert len(hands) == 1 and s["detector_duplicate"] == 1


@pytest.mark.parametrize("dropped", [[0, 0, 1000, 1000],      # frame-sized (black frames, bokeh)
                                     [490, 590, 20, 20],      # smaller than MIN_HAND_PX
                                     [495, 595, 0.0, 10]])    # zero width: no division by it
def test_a_box_the_gates_drop_takes_no_wrist_from_a_real_hand_box(dropped):
    """Both boxes hold the wrist (cost 0) and SciPy's tie rule gives it to the first, higher-scoring box: matched
    before the gates, the dropped box took the wrist and the real hand box was kept unlinked."""
    body = propose(a_body(wrists={"left": (500, 600)}), IDX, THRESHOLD, W, H)          # a wrist, no hand keypoints
    s = stats()
    hands = frame_hands(FRAME, [body], Fake([(dropped, 0.6), ([470, 570, 60, 60], 0.5)]), s)
    assert [(h["box"], h["side"], h["person"]) for h in hands] == [([470, 570, 60, 60], "left", {"body": "body:0", "face": "face:0"})]
    assert s["detector_linked"] == 1 and s["detector_unlinked"] == 0


def test_a_box_on_a_wrist_hand_takes_no_free_wrist_from_a_real_hand_box():
    """A detector box that is a wrist hand again (IoU >= DEDUP_IOU) is dropped, so it does not take part in matching
    either: here it would take body:1's wrist (inside it, cost 0), the only wrist in the frame, from the real box
    beside it (cost 0.5)."""
    a = propose(a_body(name="body:0", hands={"left": hand_at(500, 500)}), IDX, THRESHOLD, W, H)   # its wrist not seen
    b = propose(a_body(name="body:1", face=None, wrists={"right": (560, 520)}), IDX, THRESHOLD, W, H)
    left = a["proposals"]["left"]
    s = stats()
    hands = frame_hands(FRAME, [a, b], Fake([(left, 0.9), ([590, 490, 60, 60], 0.5)]), s)
    assert [(h["box_source"], h["side"], h["person"]["body"]) for h in hands] == [("wrist", "left", "body:0"), ("detector", "right", "body:1")]
    assert s["detector_duplicate"] == 1 and s["detector_linked"] == 1


def test_a_small_wrist_hand_is_not_posed():
    body = propose(a_body(hands={"left": hand_at(500, 500, r=10)}), IDX, THRESHOLD, W, H)
    assert body["proposals"]["left"][2] < MIN_HAND_PX
    s = stats()
    assert frame_hands(FRAME, [body], Fake(), s) == [] and s["wrist_below_gate"] == 1


def hand(box, side="left", source="wrist", body="body:0", face="face:0"):
    return {"box": box, "side": side, "box_source": source, "person": {"body": body, "face": face} if body else None,
            "kpts": np.zeros((21, 3))}


def test_linked_hands_continue_by_body_side_and_source_until_a_gap_or_a_cut():
    frames = [(0.0, "shot:0", [hand([0, 0, 50, 50]), hand([500, 0, 50, 50], source="detector")]),
              (0.125, "shot:0", [hand([300, 300, 50, 50])]),                  # moved far: still the same hand
              (0.75, "shot:0", [hand([300, 300, 50, 50])]),                   # after a 0.625 s gap: a new track
              (0.875, "shot:1", [hand([300, 300, 50, 50])])]                  # a cut: a new track
    tracks = track_hands(frames)
    assert [(tr["shot"], tr["box_source"], [s["t"] for s in tr["samples"]]) for tr in tracks] == [
        ("shot:0", "wrist", [0.0, 0.125]), ("shot:0", "detector", [0.0]), ("shot:0", "wrist", [0.75]), ("shot:1", "wrist", [0.875])]


def test_unlinked_hands_continue_by_iou_with_a_track_of_the_same_side():
    frames = [(0.0, "shot:0", [hand([0, 0, 50, 50], side=None, source="detector", body=None)]),
              (0.125, "shot:0", [hand([5, 0, 50, 50], side=None, source="detector", body=None),
                                 hand([200, 0, 50, 50], side=None, source="detector", body=None)])]
    tracks = track_hands(frames)
    assert [[s["box"] for s in tr["samples"]] for tr in tracks] == [[[0, 0, 50, 50], [5, 0, 50, 50]], [[200, 0, 50, 50]]]


def test_layer_hands_are_hand_ids_in_shot_order_rounded():
    k = np.full((21, 3), [10.04, 20.06, 0.123])
    tracks = [{"shot": "shot:10", "side": "left", "box_source": "wrist", "person": {"body": "body:3", "face": None},
               "samples": [{"t": 2.0, "box": [0.04, 1.26, 10, 20], "kpts": k}]},
              {"shot": "shot:2", "side": None, "box_source": "detector", "person": None,
               "samples": [{"t": 5.0, "box": [0, 0, 10, 10], "kpts": k}, {"t": 5.125, "box": [0, 0, 10, 10], "kpts": k}]}]
    out = layer_hands(tracks)
    assert [(h["id"], h["shot"], h["side"], h["body"], h["face"], h["box_source"], h["start"], h["end"]) for h in out] == [
        ("hand:0", "shot:2", None, None, None, "detector", 5.0, 5.125), ("hand:1", "shot:10", "left", "body:3", None, "wrist", 2.0, 2.0)]
    assert out[1]["samples"][0]["box"] == [0.0, 1.3, 10.0, 20.0] and out[1]["samples"][0]["kpts"][0] == [10.0, 20.1, 0.12]
    assert list(out[0]) == ["id", "shot", "side", "body", "face", "box_source", "start", "end", "samples"]   # convert_bakeoff's order


def test_bodies_by_time_keeps_track_order_within_a_frame():
    tracks = [{"id": "body:0", "face": "face:2", "samples": [{"t": 0.125, "kpts": [[0, 0, 0]] * 133}]},
              {"id": "body:1", "face": None, "samples": [{"t": 0.1250001, "kpts": [[0, 0, 0]] * 133}, {"t": 0.25, "kpts": [[0, 0, 0]] * 133}]}]
    frames = bodies_by_time(tracks)
    assert list(frames) == [0.125, 0.25]
    assert [b["person"] for b in frames[0.125]] == [{"body": "body:0", "face": "face:2"}, {"body": "body:1", "face": None}]
