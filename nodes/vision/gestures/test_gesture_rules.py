"""The gesture rules (gesture_rules.py) and what they read of the other layers (gesture_inputs.py), on synthetic hands,
faces and bodies. Run: uv run --no-project --with pytest --with numpy --with av==16.1.0 --with-editable ./sdk pytest nodes/vision/gestures"""

import json
from pathlib import Path

import numpy as np
import pytest

from gesture_inputs import Context, InputError, check_hands_model, upright_size
from gesture_rules import (EMPTY, MOUTH_RADIUS, Series, convex_hull, events_from_evidence, face_geometry, hand_events,
                           inside, layer_events, ray_target, sample_evidence)

NAMES = json.loads((Path(__file__).parents[1] / "body-pose" / "keypoints.json").read_text())["names"]
TIMES = [round(k * 0.125, 3) for k in range(9)]
W, H = 1920, 1080

# A pointing hand in its own frame, the index along +x from the wrist at the origin: index straight and long, the
# other three fingers folded back toward the palm (tips nearer the wrist than their PIP joints)
POINTING = np.array([[0, 0], [10, -10], [20, -20], [28, -25], [35, -28],          # wrist, thumb
                     [40, -10], [60, -10], [75, -10], [90, -10],                   # index MCP, PIP, DIP, tip
                     [40, 0], [55, 0], [50, 5], [35, 5],                           # middle
                     [38, 8], [52, 10], [47, 14], [33, 13],                        # ring
                     [35, 15], [46, 18], [42, 21], [30, 20]], dtype=float)         # pinky
OPEN = POINTING.copy()
for pip, tip in ((10, 12), (14, 16), (18, 20)):
    OPEN[tip] = OPEN[pip] + 2 * (OPEN[pip] - OPEN[pip - 1])                       # every finger straight


def kpts(points, score=0.9):
    return [[float(x), float(y), score] for x, y in points]


def landmarks(cx=500.0, cy=400.0, r=100.0):
    """68 py-feat landmarks [x0..x67, y0..y67] of a face of box side 2r centred at (cx, cy): jaw along the lower half
    circle, brows at cy - 0.7r, nose down the middle, eyes, and the mouth around (cx, cy + 0.5r)."""
    pts = [(cx - r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(0, np.pi, 17)]           # jaw 0-16
    pts += [(cx - 0.7 * r + 0.14 * r * i + (0.14 * r if i > 4 else 0), cy - 0.7 * r) for i in range(10)]  # brows 17-26
    pts += [(cx, cy - 0.5 * r + 0.1 * r * i) for i in range(4)] + [(cx - 0.2 * r + 0.1 * r * i, cy + 0.2 * r) for i in range(5)]
    for ex in (cx - 0.4 * r, cx + 0.4 * r):                                                          # eyes 36-47
        pts += [(ex + 0.15 * r * np.cos(a), cy - 0.4 * r + 0.05 * r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    pts += [(cx + 0.3 * r * np.cos(a), cy + 0.5 * r + 0.1 * r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 20, endpoint=False)]
    pts = np.array(pts)
    assert len(pts) == 68
    return list(pts[:, 0]) + list(pts[:, 1])


FACE_BOX = [400.0, 300.0, 200.0, 200.0]   # the face of landmarks(): mouth centre (500, 450), MOUTH_RADIUS x 200 = 40 px


def body(nose=(500, 450), left=(600, 600), right=(400, 600), score=5.0):
    """133 keypoints, all invisible but the nose and shoulders (a person facing the camera: their left on image right)."""
    k = np.zeros((133, 3))
    for name, xy in (("nose", nose), ("left_shoulder", left), ("right_shoulder", right)):
        k[NAMES.index(name)] = [*xy, score]
    return k.tolist()


def scene(hand_points, side="right", face="face:0", body_kpts=None, other_hand=None, objects=None, face_times=TIMES):
    """(hands, Context): one hand of body:0 with the same points at every time, face:0 with landmarks(), and a second
    face (face:1) in the shot at (1400, 300)."""
    samples = [{"t": t, "box": [0, 0, 10, 10], "kpts": kpts(hand_points)} for t in TIMES]
    hands = [{"id": "hand:0", "shot": "shot:0", "side": side, "body": "body:0", "face": face, "samples": samples}]
    if other_hand is not None:
        hands.append({"id": "hand:1", "shot": "shot:0", "side": "left" if side == "right" else "right", "body": "body:0",
                      "face": face, "samples": [{"t": t, "box": other_hand, "kpts": kpts(hand_points)} for t in TIMES]})
    faces = {"tracks": [{"id": "face:0", "shot": "shot:0", "samples": [{"t": t, "box": FACE_BOX} for t in face_times]},
                        {"id": "face:1", "shot": "shot:0", "samples": [{"t": t, "box": [1300, 200, 200, 200]} for t in TIMES]}]}
    detail = {"tracks": {"face:0": [{"t": t, "lmk": landmarks()} for t in face_times],
                         "face:1": [{"t": t, "lmk": landmarks(1400, 300)} for t in TIMES]}}
    bodies = {"keypoint_names": NAMES, "score_threshold": 3.0,
              "tracks": [{"id": "body:0", "samples": [{"t": t, "kpts": body_kpts or body()} for t in TIMES]}]}
    return hands, Context(hands, bodies, faces, detail, objects, W, H)


def evidence(hand_points, j=0, **kw):
    hands, ctx = scene(hand_points, **kw)
    return sample_evidence(hands[0], j, ctx)


# ---------------------------------------------------------------- geometry

def test_the_hull_keeps_the_corners_and_leaves_out_inner_and_collinear_points():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hull = convex_hull(square + [(5, 5), (5, 0), (0, 5)])
    assert sorted(map(tuple, hull)) == sorted(square)
    assert len(convex_hull([(0, 0), (1, 1)])) == 2


def test_inside_counts_the_boundary_and_a_degenerate_hull_holds_only_its_own_points():
    square = convex_hull([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert inside(square, (5, 5)) and inside(square, (10, 5)) and inside(square, (0, 0))
    assert not inside(square, (10.01, 5)) and not inside(square, (-1, -1))
    segment = np.array([(0, 0), (10, 10)], dtype=float)
    assert inside(segment, (5, 5)) and not inside(segment, (5, 6))


def test_series_interpolates_across_short_gaps_only():
    s = Series([{"t": 0.0, "v": [0.0]}, {"t": 0.25, "v": [10.0]}, {"t": 1.0, "v": [20.0]}], "v")
    assert s.at(0.25)[0] == 10.0 and s.at(0.125)[0] == pytest.approx(5.0)
    assert s.at(0.6) is None              # 0.75 s gap: longer than MAX_FACE_GAP_S
    assert s.at(-0.1) is None and s.at(1.1) is None


def test_face_geometry_is_the_mouth_centre_the_box_side_and_a_region_above_the_brows():
    mouth, size, region = face_geometry(np.array(landmarks()), FACE_BOX)
    assert mouth == pytest.approx([500, 450]) and size == 200
    assert inside(region, (500, 300)) and inside(region, (500, 285))      # forehead, above the brows at y = 330
    assert not inside(region, (500, 250)) and not inside(region, (700, 400))


# ---------------------------------------------------------------- face rules

def test_a_hand_over_the_mouth_scores_the_share_of_its_keypoints_near_the_mouth():
    points = np.vstack([np.tile([500, 450], (6, 1)) + [[i, 0] for i in range(6)], np.tile([900, 900], (15, 1))])
    row = evidence(points)
    assert row["face_status"] == "ok" and row["valid"] == 21 and row["mouth_near"] == 6
    assert row["hand_over_mouth"] and not row["self_touch_face"]
    assert row["mouth_dist"][0] == pytest.approx(0) and row["mouth_dist"][20] == pytest.approx(np.hypot(400, 450) / 200)
    [event] = events_from_evidence([{"id": "hand:0", "face": "face:0"}],
                                   [evidence(points, j) for j in range(len(TIMES))])
    assert (event["kind"], event["score"], event["start"], event["end"]) == ("hand_over_mouth", round(6 / 21, 2), 0.0, 1.0)


def test_the_mouth_inside_the_hand_is_over_the_mouth_with_no_keypoint_near_it():
    ring = [(500 + 60 * np.cos(a), 450 + 60 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 21, endpoint=False)]
    row = evidence(np.array(ring))
    assert row["mouth_near"] == 0 and row["mouth_in_hull"] and row["hand_over_mouth"]
    assert all(d > MOUTH_RADIUS for d in row["mouth_dist"])


def test_a_hand_on_the_forehead_touches_the_face_and_one_beside_it_does_not():
    forehead = np.vstack([[[470 + 10 * i, 300] for i in range(7)], np.tile([900, 900], (14, 1))])
    row = evidence(forehead)
    assert row["self_touch_face"] and not row["hand_over_mouth"] and row["face_inside"] == 7
    beside = evidence(forehead + np.array([[300, 0]] * 7 + [[0, 0]] * 14))
    assert beside["face_status"] == "ok" and not beside["self_touch_face"] and beside["face_inside"] == 0


def test_face_rules_need_the_face_its_landmarks_and_enough_keypoints():
    near_mouth = np.tile([500, 450], (21, 1))
    assert evidence(near_mouth, face=None)["face_status"] == "no_face"
    # landmarks at 0 and 1 s only: no landmarks at 0.5 s (a 1 s gap), interpolated at none
    row = evidence(near_mouth, j=4, face_times=[0.0, 1.0])
    assert row["face_status"] == "no_landmarks" and not row["hand_over_mouth"]
    few = [[500, 450, 0.9]] * 5 + [[500, 450, 0.29]] * 16      # 5 keypoints above 0.3: fewer than MIN_VALID_KPTS
    hands, ctx = scene(near_mouth)
    hands[0]["samples"][0]["kpts"] = few
    row = sample_evidence(hands[0], 0, ctx)
    assert row["valid"] == 5 and (row["face_status"], row["point_status"], row["raise_status"]) == ("few_keypoints",) * 3


def test_keypoints_outside_the_frame_do_not_count():
    hands, ctx = scene(np.tile([500, 450], (21, 1)))
    hands[0]["samples"][0]["kpts"] = kpts([[500, 450]] * 15 + [[W, 450]] * 3 + [[500, -1]] * 3)
    row = sample_evidence(hands[0], 0, ctx)
    assert row["valid"] == 15 and row["valid_mask"] == (1 << 15) - 1


# ---------------------------------------------------------------- pointing

def test_an_extended_index_with_the_other_fingers_curled_points_along_it_at_the_face_it_aims_at():
    # a hand at (1000, 300) pointing right, at face:1 (centre 1400, 300); its own face (face:0) is never a target
    row = evidence(POINTING * 2 + [1000, 310])
    assert row["point_status"] == "ok" and row["pointing"] and row["direction"] == pytest.approx([1, 0])
    assert row["target"] == "face:1" and row["index_straightness"] == pytest.approx(1.0)
    assert row["index_extension"] >= 1.2 and all(c < 1.0 for c in row["curl"])
    left = evidence(-POINTING * 2 + [1000, 310])   # pointing left: face:0 is its own, nothing else there
    assert left["pointing"] and left["target"] is None


def test_an_open_hand_does_not_point_and_neither_do_clasped_hands():
    assert not evidence(OPEN * 2 + [1000, 310])["pointing"]
    clasped = evidence(POINTING * 2 + [1000, 310], other_hand=[2, 2, 10, 10])     # the other hand's box overlaps
    assert clasped["point_status"] == "clasped" and clasped["clasp_iou"] >= 0.2 and not clasped["pointing"]


def test_pointing_needs_every_joint_it_measures():
    points = kpts(POINTING * 2 + [1000, 310])
    points[12][2] = 0.1   # the middle fingertip is not seen
    hands, ctx = scene(POINTING)
    hands[0]["samples"][0]["kpts"] = points
    assert sample_evidence(hands[0], 0, ctx)["point_status"] == "missing_keypoints"


def test_objects_are_targets_only_when_the_run_has_them():
    objects = {"tracks": [{"id": "obj:3", "shot": "shot:0", "samples": [{"t": t, "box": [1250, 270, 40, 40]} for t in TIMES]},
                          {"id": "obj:4", "shot": "shot:1", "samples": [{"t": t, "box": [1220, 280, 20, 20]} for t in TIMES]}]}
    points = POINTING * 2 + [1000, 310]
    assert evidence(points)["target"] == "face:1"
    assert evidence(points, objects=objects)["target"] == "obj:3"   # on the ray; obj:4, also on it, is in another shot


def test_ray_target_takes_the_box_centre_nearest_the_ray_within_half_its_side():
    origin, right = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    boxes = [("a", np.array([90, -10, 20, 20])), ("b", np.array([190, 5, 20, 20])), ("behind", np.array([-110, -10, 20, 20]))]
    assert ray_target(origin, right, boxes) == "a"                       # a is on the ray, b 15 px off it
    assert ray_target(origin, right, [("far", np.array([90, 30, 20, 20]))]) is None


# ---------------------------------------------------------------- raised hand

def hand_with_wrist(x, y):
    return OPEN + [x, y]    # the wrist (keypoint 0) at (x, y)


def test_a_wrist_well_above_its_shoulder_is_raised_to_the_shoulder_or_the_head():
    row = evidence(hand_with_wrist(400, 520))          # right shoulder (400, 600), width 200: 80 px > 50
    assert (row["raise_status"], row["raised_hand"], row["level"]) == ("ok", True, "shoulder")
    assert row["rise"] == pytest.approx(80) and row["shoulder_width"] == pytest.approx(200)
    head = evidence(hand_with_wrist(400, 400))          # above the nose at y = 450
    assert head["raised_hand"] and head["level"] == "head" and head["above_nose"]
    resting = evidence(hand_with_wrist(400, 570))       # 30 px: on the collarbone
    assert resting["raise_status"] == "ok" and not resting["raised_hand"] and resting["level"] == ""


def test_the_wrist_is_compared_with_its_own_sides_shoulder():
    # the left shoulder is at (600, 520) in this body; a right wrist at y = 540 is 60 px above the right one only
    tilted = body(left=(600, 520))
    assert evidence(hand_with_wrist(400, 540), side="right", body_kpts=tilted)["raised_hand"]
    assert not evidence(hand_with_wrist(600, 540), side="left", body_kpts=tilted)["raised_hand"]


def test_upside_down_a_hand_above_the_shoulders_in_the_image_is_not_raised():
    flipped = body(nose=(500, 750))                     # the body's up points down the image (a rider in a flip)
    row = evidence(hand_with_wrist(400, 500), body_kpts=flipped)
    assert row["raise_status"] == "ok" and not row["raised_hand"] and row["rise"] == pytest.approx(-100)


def test_a_hand_on_the_face_is_raised_only_above_the_nose():
    on_mouth = np.tile([500, 450], (21, 1)) + np.array([[0, 60]] + [[0, 0]] * 20)   # wrist at (500, 510), fingers on the mouth
    row = evidence(on_mouth, side="right", body_kpts=body(right=(500, 600)))
    assert row["hand_over_mouth"] and row["rise"] > 50 and not row["raised_hand"]


def test_raising_needs_the_shoulders_and_the_nose():
    row = evidence(hand_with_wrist(400, 400), body_kpts=body(score=2.0))
    assert row["raise_status"] == "no_shoulders_or_nose" and not row["raised_hand"]
    low = evidence(hand_with_wrist(400, 400), body_kpts=body(left=(600, H), right=(400, H)))   # shoulders below the frame
    assert low["raise_status"] == "no_shoulders_or_nose"


# ---------------------------------------------------------------- events

def row(t, hand="hand:0", **kw):
    return {**EMPTY, "hand": hand, "t": t, "valid": 10, **kw}


def test_positives_merge_across_one_missed_sample_and_short_runs_are_dropped():
    hand = {"id": "hand:0", "face": "face:2"}
    over = [row(t, hand_over_mouth=True, mouth_near=n) for t, n in ((0.0, 4), (0.125, 6), (0.375, 5), (2.0, 9), (2.125, 9))]
    [event] = hand_events(hand, over + [row(0.25)])
    assert (event["start"], event["end"], event["detail"], event["score"]) == (0.0, 0.375, {"samples": 3}, 0.5)
    assert event["face"] == "face:2" and event["hand"] == "hand:0"


def test_a_pointing_event_has_the_mean_direction_and_the_target_most_samples_name_the_first_on_a_tie():
    hand = {"id": "hand:0", "face": None}
    rows = [row(0.0, pointing=True, direction=[1.0, 0.0], target="obj:2"),
            row(0.125, pointing=True, direction=[0.0, 1.0], target="face:1"),
            row(0.25, pointing=True, direction=[1.0, 0.0], target=None)]
    [event] = hand_events(hand, rows)
    assert event["detail"] == {"direction": [0.89, 0.45], "target": "obj:2", "samples": 3}
    [event] = hand_events(hand, rows[1:2] + rows[:1] + rows[2:])   # the same samples given in another order
    assert event["detail"]["target"] == "obj:2"                    # first named in time, not in the list


def test_a_raised_hand_event_is_at_head_level_when_most_samples_are():
    rows = [row(t, raised_hand=True, level=lv) for t, lv in ((0.0, "head"), (0.125, "shoulder"), (0.25, "head"))]
    [event] = hand_events({"id": "hand:0", "face": None}, rows)
    assert event["detail"] == {"level": "head", "samples": 3}
    rows[2]["level"] = "shoulder"
    assert hand_events({"id": "hand:0", "face": None}, rows)[0]["detail"]["level"] == "shoulder"


def test_layer_events_sort_by_start_then_hand_id_as_text_and_number_from_zero():
    def ev(start, hand, kind="pointing"):
        return {"kind": kind, "start": start, "end": start + 1, "face": None, "hand": hand, "score": 1.0, "detail": {}}
    out = layer_events([ev(2.0, "hand:2"), ev(1.0, "hand:13"), ev(1.0, "hand:125"), ev(1.0, "hand:125", "raised_hand")])
    assert [(e["id"], e["hand"], e["kind"]) for e in out] == [
        ("gesture:0", "hand:125", "pointing"), ("gesture:1", "hand:125", "raised_hand"), ("gesture:2", "hand:13", "pointing"),
        ("gesture:3", "hand:2", "pointing")]
    assert list(out[0]) == ["id", "kind", "start", "end", "face", "hand", "score", "detail"]   # the bake-off's key order


def test_evidence_for_a_hand_the_layer_does_not_have_is_an_error():
    with pytest.raises(ValueError, match="hand:9"):
        events_from_evidence([{"id": "hand:0", "face": None}], [row(0.0, hand="hand:9")])


# ---------------------------------------------------------------- inputs

def test_the_layers_must_belong_together():
    hands, ctx = scene(OPEN)
    bodies = {"keypoint_names": NAMES, "score_threshold": 3.0, "tracks": [{"id": "body:0", "samples": []}]}
    faces = {"tracks": [{"id": "face:0", "shot": "shot:0", "samples": [{"t": 0.0, "box": FACE_BOX}]}]}
    with pytest.raises(InputError, match="faces_detail has no landmarks"):
        Context(hands, bodies, faces, {"tracks": {}}, None, W, H)
    detail = {"tracks": {"face:0": [{"t": 0.0, "lmk": landmarks()}]}}
    with pytest.raises(InputError, match="body:7 is not in the bodies layer"):
        Context([{**hands[0], "body": "body:7"}], bodies, faces, detail, None, W, H)
    with pytest.raises(InputError, match="face:5 is not in the faces layer"):
        Context([{**hands[0], "face": "face:5"}], bodies, faces, detail, None, W, H)
    with pytest.raises(InputError, match="no nose keypoint"):
        Context(hands, {**bodies, "keypoint_names": NAMES[1:]}, faces, detail, None, W, H)
    lonely = Context(hands, bodies, faces, detail, None, W, H)    # body:0 has no samples
    with pytest.raises(InputError, match="no sample at 0.0"):
        sample_evidence(hands[0], 0, lonely)


def test_the_hands_must_be_rtmpose_hands():
    check_hands_model({"producer": {"models": [{"id": "RTMPose-m Hand5 256x256 (rtmpose-m_simcc-hand5_pt-aic-coco_210e)"}]}})
    with pytest.raises(InputError, match="not RTMPose-m Hand5"):
        check_hands_model({"producer": {"models": [{"id": "MediaPipe Hands"}]}})


def test_the_frame_size_is_the_upright_one(rotated, clip):
    assert upright_size(rotated) == (32, 64)          # stored 64x32 with a 90 degree rotation
    assert upright_size(clip.path) == (clip.w, clip.h)
