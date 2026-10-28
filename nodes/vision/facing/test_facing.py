"""Torso direction and facing events (facing.py), without the model.
Run: uv run --project envs/vision --with pytest pytest nodes/vision/facing"""

import numpy as np
import pytest

from facing import (NAMES, TORSO, body_facing, body_poses, crop, crop_window, events_from_angles, face_number,
                    facing_events, facing_vector, in_frame, linked_samples, merge, nearest_index, nearest_pose,
                    pair_angles, stored, torso_centre, torso_facing)

LSH, RSH, LHIP, RHIP = TORSO


def world(yaw_deg=0.0, hips=True):
    """33 world landmarks (camera frame: x right, y down, z away) of a subject turned yaw_deg to their right: shoulders
    0.4 m wide, 0.5 m above the hips."""
    a = np.radians(yaw_deg)
    left = np.array([np.cos(a), 0.0, np.sin(a)])   # the subject's left-right axis: +x when facing the camera
    w = np.zeros((33, 3))
    w[LSH], w[RSH] = 0.2 * left + [0, -0.5, 0], -0.2 * left + [0, -0.5, 0]
    if hips:
        w[LHIP], w[RHIP] = 0.15 * left, -0.15 * left
    return w


def visible(*joints):
    v = np.zeros(33, bool)
    v[list(joints)] = True
    return v


def test_the_landmarks_are_mediapipes_33_with_shoulders_and_hips_where_the_bake_off_read_them():
    assert len(NAMES) == 33 and TORSO == (11, 12, 23, 24)


def test_a_subject_facing_the_camera_has_yaw_and_pitch_zero():
    yaw, pitch, source = torso_facing(world(0), visible(*TORSO))
    assert (yaw, pitch, source) == (pytest.approx(0, abs=1e-9), pytest.approx(0, abs=1e-9), "torso")
    assert facing_vector(yaw, pitch) == pytest.approx([0, 0, -1])   # towards the camera


def test_yaw_is_positive_turned_to_the_subjects_right_and_180_from_behind():
    # turned 30 degrees to their right: the left shoulder comes towards the camera (z < 0)... yaw +30
    assert torso_facing(world(-30), visible(*TORSO))[0] == pytest.approx(30)
    assert torso_facing(world(30), visible(*TORSO))[0] == pytest.approx(-30)
    assert abs(torso_facing(world(180), visible(*TORSO))[0]) == pytest.approx(180)
    # facing vector of a subject turned to their right points to image left (x < 0)
    assert facing_vector(30, 0)[0] < 0


def test_without_both_hips_the_shoulders_give_yaw_only_and_without_both_shoulders_nothing():
    yaw, pitch, source = torso_facing(world(-40, hips=False), visible(LSH, RSH))
    assert source == "shoulders" and pitch is None and yaw == pytest.approx(40)
    assert torso_facing(world(-40), visible(LSH, RSH, LHIP))[1:] == (None, "shoulders")   # one hip is not enough
    assert torso_facing(world(0), visible(LSH, LHIP, RHIP)) == (None, None, None)


def test_pitch_is_positive_tilted_back():
    w = world(0)
    w[[LSH, RSH], 2] += 0.2   # shoulders farther from the camera than the hips: leaning back, the chest faces up
    yaw, pitch, _ = torso_facing(w, visible(*TORSO))
    assert yaw == pytest.approx(0, abs=1e-9) and pitch > 0


def test_a_body_is_stored_as_the_bake_offs_pose_file_rounded_it():
    kpts, facing = stored(np.array([[10.04, 20.06, 0.4951]]), 12.345, None)
    assert kpts == [[10.0, 20.1, 0.5]] and facing == [12.3, None]
    assert stored(np.zeros((1, 3)), None, None)[1] is None


def test_visibility_decides_which_joints_count():
    kpts = np.zeros((33, 3))
    kpts[list(TORSO), 2] = [0.9, 0.9, 0.49, 0.9]   # one hip under 0.5
    k, facing, source = body_facing((kpts, world(-20)))
    assert source == "shoulders" and facing == [20.0, None] and len(k) == 33
    assert body_facing(None) is None


def test_the_crop_is_the_grown_box_clipped_to_the_frame_in_a_black_square():
    (x0, y0, side), (bx0, by0, bx1, by1) = crop_window([100, 50, 40, 100], 640, 360)
    # grown 2 px and 5 px a side; 1.1 - 1 is a hair over 0.1 in floating point and int() truncates, so 50 - 5 is 44,
    # as in the bake-off (the same expression)
    assert (bx0, by0, bx1, by1) == (98, 44, 142, 155) and side == 111 and (x0, y0) == (65, 44)
    # past the frame's edge: only the inside is copied, the square is still the grown box's longer side
    (x0, y0, side), (bx0, by0, bx1, by1) = crop_window([-10, 300, 50, 100], 640, 360)
    assert (bx0, by1) == (0, 360) and side == max(bx1 - bx0, by1 - by0)

    frame = np.full((360, 640, 3), 200, np.uint8)
    square, window = crop(frame, [100, 50, 40, 100])
    assert square.shape == (111, 111, 3) and window == (65, 44, 111)
    assert square[:, 33:77].min() == 200 and square[:, :33].max() == 0 and square[:, 77:].max() == 0   # neighbours masked


def landmarks(x, y, visibility, z=-0.3, presence=0.99):
    """33 landmarks as the landmarker returns them: x, y in [0, 1] of the crop, z, visibility, presence."""
    return np.tile([x, y, z, visibility, presence], (33, 1)).astype(float)


def test_of_two_poses_the_one_centred_nearest_the_box_is_kept_in_frame_pixels():
    near = landmarks(0.5, 0.5, 0.5)
    far = landmarks(0.05, 0.05, 1.0)
    poses = in_frame([(far, "far world"), (near, "near world")], (65, 45, 110))
    assert poses[1][0][0].tolist() == [65 + 55, 45 + 55, 0.5]   # centred on the box [100, 50, 40, 100]; visibility
    assert nearest_pose(poses, [100, 50, 40, 100])[1] == "near world"
    assert nearest_index(poses, [100, 50, 40, 100]) == 1 and nearest_index([], [0, 0, 1, 1]) is None
    assert nearest_pose([], [0, 0, 1, 1]) is None


def test_every_pose_of_a_crop_is_kept_for_the_side_file_and_the_nearest_is_what_the_events_use():
    window, box = (65, 45, 110), [100, 50, 40, 100]
    near = landmarks(0.5, 0.5, 0.9)
    near[list(TORSO), :2] = [[0.6, 0.4], [0.4, 0.4], [0.58, 0.7], [0.42, 0.7]]   # torso about the box centre
    far = landmarks(0.05, 0.05, 0.9)
    raw = [(far, world(30)), (near, world(-20))]
    poses, kept = body_poses(raw, window, box)
    assert [(p["pose"], p["kept"]) for p in poses] == [(0, False), (1, True)]
    assert poses[1]["landmarks"] is near and poses[1]["world"] is raw[1][1]   # as the landmarker returned them
    # the kept pose's facing and centre are what the layer's path computed from it (body_facing, torso_centre)
    k, facing, source = body_facing(nearest_pose(in_frame(raw, window), box))
    assert kept == (pytest.approx(torso_centre(k, box)), facing, source) and facing == [20.0, 0.0] and source == "torso"
    assert poses[1]["centre"].tolist() == torso_centre(k, box).tolist()
    assert poses[1]["yaw"] == pytest.approx(20) and poses[1]["pitch"] == pytest.approx(0, abs=1e-9)   # unrounded
    assert poses[0]["yaw"] == pytest.approx(-30) and poses[0]["source"] == "torso"   # the other pose has its own
    assert body_poses([], window, box) == ([], None)


def test_a_pose_without_visible_shoulders_is_kept_with_no_facing():
    hidden = landmarks(0.5, 0.5, 0.2)
    poses, kept = body_poses([(hidden, world(0))], (0, 0, 100), [10, 10, 80, 80])
    assert (poses[0]["source"], poses[0]["yaw"], poses[0]["pitch"]) == (None, None, None)
    assert kept[1:] == (None, None) and kept[0].tolist() == [50, 50]   # no facing; the box centre


def test_the_torso_centre_is_the_visible_shoulders_and_hips_else_the_box_centre():
    kpts = np.zeros((33, 3))
    kpts[LSH], kpts[RSH] = [10, 20, 0.9], [30, 20, 0.9]
    assert torso_centre(kpts, [0, 0, 100, 100]).tolist() == [20, 20]
    assert torso_centre(np.zeros((33, 3)), [0, 0, 100, 50]).tolist() == [50, 25]


def frames_of(times, bodies):
    """{(shot, t): bodies} for the same bodies at every time."""
    return {("shot:0", t): bodies(t) for t in times}


TIMES = [round(i / 8, 3) for i in range(12)]   # 1.375 s at 8 fps


def test_two_people_turned_to_each_other_face_each_other_mutually():
    # face 0 on the left turned 50 degrees to its left (towards image right), face 1 on the right turned to its right
    events = facing_events(frames_of(TIMES, lambda t: [(0, np.array([100.0, 100]), [-50.0, 0.0]),
                                                       (1, np.array([300.0, 100]), [50.0, 0.0])]))
    assert [(e["from"], e["to"], e["mutual"]) for e in events] == [("face:0", "face:1", True), ("face:1", "face:0", True)]
    assert events[0]["id"] == "facing:0" and events[0]["start"] == 0.0 and events[0]["end"] == 1.375
    assert events[0]["detail"] == {"mean_angle_deg": 40.0, "samples": 12, "mutual_s": 1.375}
    assert events[0]["score"] == round(np.cos(np.radians(40)), 3)


def test_someone_facing_the_camera_faces_no_one_and_a_one_sided_facing_is_not_mutual():
    events = facing_events(frames_of(TIMES, lambda t: [(0, np.array([100.0, 100]), [-50.0, None]),
                                                       (1, np.array([300.0, 100]), [0.0, 0.0])]))
    assert [(e["from"], e["to"], e["mutual"]) for e in events] == [("face:0", "face:1", False)]
    assert events[0]["detail"]["mutual_s"] == 0


def test_of_several_people_in_the_cone_the_nearest_in_the_image_is_faced():
    events = facing_events(frames_of(TIMES, lambda t: [(0, np.array([100.0, 100]), [-70.0, 0.0]),
                                                       (1, np.array([400.0, 100]), None),
                                                       (2, np.array([250.0, 100]), None)]))
    assert [(e["from"], e["to"]) for e in events] == [("face:0", "face:2")]


def test_short_facings_are_dropped_and_gaps_of_one_sample_bridged():
    def bodies(t):
        turned = [-50.0, 0.0] if t != 0.5 else [0.0, 0.0]   # one sample missed in the middle
        return [(0, np.array([100.0, 100]), turned), (1, np.array([300.0, 100]), None)]
    events = facing_events(frames_of(TIMES, bodies))
    assert len(events) == 1 and events[0]["detail"]["samples"] == 11 and events[0]["end"] == 1.375
    assert facing_events(frames_of(TIMES[:8], bodies)) == []   # 0.875 s < 1 s
    assert merge([0.0, 0.125, 0.5], 0.3) == [[0.0, 0.125], [0.5]]


def test_events_are_ordered_by_start_then_face_number_not_by_id_text():
    def bodies(t):
        return [(10, np.array([100.0, 100]), [-50.0, 0.0]), (9, np.array([300.0, 100]), [50.0, 0.0])]
    events = facing_events(frames_of(TIMES, bodies))
    assert [e["from"] for e in events] == ["face:9", "face:10"]   # "face:10" < "face:9" as text
    assert face_number("face:12") == 12


def test_only_bodies_linked_to_a_face_are_compared():
    layer = {"tracks": [
        {"id": "body:0", "shot": "shot:0", "face": "face:3", "samples": [{"t": 0.0, "box": [1, 2, 3, 4]}, {"t": 0.125, "box": [2, 2, 3, 4]}]},
        {"id": "body:1", "shot": "shot:0", "face": None, "samples": [{"t": 0.0, "box": [9, 9, 9, 9]}]}]}
    assert linked_samples(layer) == [("shot:0", 0.0, 3, [1, 2, 3, 4], "body:0", 0),
                                     ("shot:0", 0.125, 3, [2, 2, 3, 4], "body:0", 1)]


# ---------------------------------------------------------------- the angles side file

def facing_events_before(frames, cone_deg=60.0, min_facing_s=1.0, max_gap_s=0.3):
    """facing.facing_events as vision.facing 1.0.0 computed it, in one pass (before the angles were stored apart from
    the thresholding), verbatim but for the name: the reference the split must reproduce."""
    from collections import defaultdict
    positives = defaultdict(list)
    for (_, t), bodies in sorted(frames.items(), key=lambda kv: kv[0][1]):
        for a, centre_a, facing in bodies:
            if facing is None:
                continue
            f = facing_vector(*facing)
            inside = []
            for b, centre_b, _ in bodies:
                if b == a:
                    continue
                d = centre_b - centre_a
                dist = float(np.hypot(*d))
                if dist == 0:
                    continue
                angle = float(np.degrees(np.arccos(np.clip(f[0] * d[0] / dist + f[1] * d[1] / dist, -1, 1))))
                if angle <= cone_deg:
                    inside.append((dist, b, angle))
            if inside:
                _, b, angle = min(inside)
                positives[(a, b)].append((t, angle))
    runs = {}
    for pair, samples in positives.items():
        angle_at = dict(samples)
        for run in merge(sorted(angle_at), max_gap_s):
            if run[-1] - run[0] >= min_facing_s - 1e-6:
                runs.setdefault(pair, []).append((run, [angle_at[t] for t in run]))
    events = []
    for (a, b), pair_runs in runs.items():
        reverse = {t for t, _ in positives.get((b, a), [])}
        for run, angles in pair_runs:
            both = [t for t in run if t in reverse]
            mutual_s = sum(r[-1] - r[0] for r in merge(both, max_gap_s))
            events.append({"from": a, "to": b, "start": run[0], "end": run[-1],
                           "score": round(float(np.mean(np.cos(np.radians(angles)))), 3),
                           "mutual": mutual_s >= min_facing_s - 1e-6,
                           "detail": {"mean_angle_deg": round(float(np.mean(angles)), 1), "samples": len(run),
                                      "mutual_s": round(mutual_s, 3)}})
    events.sort(key=lambda e: (e["start"], e["from"]))
    return [{"id": f"facing:{i}", "from": f"face:{e['from']}", "to": f"face:{e['to']}", "start": e["start"], "end": e["end"],
             "mutual": e["mutual"], "score": e["score"], "detail": e["detail"]} for i, e in enumerate(events)]


def random_frames(seed, times=160, faces=5):
    """A random scene at 8 fps: up to `faces` people a time, wandering and turning, some without a facing or a pitch,
    sometimes two at one point (no direction between them), and a second shot later on."""
    rng = np.random.default_rng(seed)
    place = rng.uniform(0, 640, (faces, 2))
    yaw = rng.uniform(-180, 180, faces)
    frames = {}
    for i in range(times):
        place += rng.normal(0, 3, place.shape)
        yaw += rng.normal(0, 4, faces)
        bodies = []
        for face in rng.permutation(faces):
            r = rng.random()
            if r > 0.9:   # not posed at this time
                continue
            facing = None if r < 0.1 else [round(float(yaw[face]), 1), None if r < 0.3 else round(float(rng.normal(0, 10)), 1)]
            bodies.append((int(face) + 3, place[face].round(1), facing))
        if len(bodies) > 1 and rng.random() < 0.05:
            bodies[1] = (bodies[1][0], bodies[0][1].copy(), bodies[1][2])
        frames[(f"shot:{0 if i < times // 2 else 1}", round(i / 8, 3))] = bodies
    return frames


@pytest.mark.parametrize("seed", range(12))
def test_thresholding_the_stored_angles_gives_the_one_pass_events_at_any_thresholds(seed):
    frames = random_frames(seed)
    rows = pair_angles(frames)
    assert events_from_angles(rows) == facing_events(frames) == facing_events_before(frames)
    for cone, min_s, gap in ((30.0, 0.5, 0.3), (90.0, 1.0, 0.125), (60.0, 0.0, 1.0), (45.0, 2.0, 0.5), (60.0, 1.0, 0.0)):
        assert events_from_angles(rows, cone, min_s, gap) == facing_events_before(frames, cone, min_s, gap)


def test_the_random_scenes_have_events_to_compare():
    events = [e for seed in range(12) for e in facing_events_before(random_frames(seed))]
    assert len(events) > 20 and any(e["mutual"] for e in events) and not all(e["mutual"] for e in events)


def test_every_pair_is_stored_whatever_its_angle_and_coinciding_centres_are_not():
    frames = {("shot:0", 0.0): [(0, np.array([100.0, 100]), [-90.0, None]), (1, np.array([300.0, 100]), None),
                                (2, np.array([100.0, 300]), [0.0, 0.0]), (3, np.array([100.0, 300]), [180.0, 0.0])]}
    rows = pair_angles(frames)
    # face 0 (facing image right) to 1, 2 and 3; 2 and 3 (towards and away from the camera) to the others but each other
    assert [(a, b) for _, a, b, _, _ in rows] == [(0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (3, 0), (3, 1)]
    assert rows[0] == (0.0, 0, 1, pytest.approx(0.0, abs=1e-9), 200.0) and rows[1][3] == pytest.approx(90)
    assert rows[3][3] == pytest.approx(90) and rows[3][4] == 200.0   # facing the camera: 90 degrees from anyone
