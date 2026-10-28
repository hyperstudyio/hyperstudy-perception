"""Who faces whom: each body's torso direction in 3D (MediaPipe Pose Landmarker's world landmarks), then facing events
between face tracks, as the bake-off computed them (pose/pose3d.py --model mediapipe: MediapipePose and torso_facing;
pose/facing.py). numpy only, so it is tested without the model.

Per body sample (models.py poses it):
  crop_window   the body box grown by MARGIN, centred in a square whose rest is black: the landmarker is single-person
                and poses the most prominent person in its input, so neighbours outside the box are masked.
  nearest_pose  of the (up to two) poses found in the crop, the one whose landmarks centre nearest the box centre.
  torso_facing  the torso normal in the camera frame (x right, y down, z away). With a = unit(L shoulder - R shoulder)
                + unit(L hip - R hip) and u = mid-shoulder - mid-hip, f = a x u, normalised. A subject facing the
                camera has a = +x, u = -y and f = -z. yaw = atan2(-f_x, -f_z): 0 = facing the camera, + = turned to the
                subject's right (image left), +-180 = back to the camera; pitch = atan2(-f_y, hypot(f_x, f_z)), + =
                tilted back. Without both hips, a = L shoulder - R shoulder, u = -y, "shoulders", pitch None. Without
                both shoulders, no facing. A joint counts when its visibility >= SCORE_THRESHOLD.

Per sample time (facing_events), among the bodies linked to a face track: A faces B when the angle between A's facing
vector f = (-sin(yaw) cos(pitch), -sin(pitch), -cos(yaw) cos(pitch)) (pitch 0 when None) and the image direction from
A's torso centre to B's, at zero depth, is at most cone_deg. The 3D poses are root-relative, so they give no depth
between people: the bodies of a shot are taken to stand at equal depth. Of several bodies inside the cone, the nearest
in the image is the one faced. Positive samples of one (A, B) pair are merged when at most max_gap_s apart; runs
shorter than min_facing_s are dropped; score = mean cos(angle); mutual when B faces A for at least min_facing_s of it.

The bake-off wrote each body's facing and landmarks to a file (pose_mediapipe.json) and facing.py read them back:
stored() rounds them as that file did, so the events are the bake-off's to the last digit.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

MARGIN = 1.1           # crop = body box grown by this (bake-off: 1.1 matched RTMW's skeleton best on 457 OOO boxes)
POSES = 2              # poses asked of the landmarker per crop; nearest_pose keeps one
# MediaPipe's per-landmark visibility. Against RTMW >= 3.0 on the same OOO boxes, Youden's J is 0.74-0.75 for 0.3-0.5;
# 0.5 is also MediaPipe's default detection and presence confidence (bake-off)
SCORE_THRESHOLD = 0.5
CONE_DEG = 60.0        # a target sideways at the same depth is 90 - |yaw| from the facing vector: turned >= 30 degrees
MIN_FACING_S = 1.0     # under a second is mostly yaw jitter near the cone edge or a passing turn (>= 9 samples at 8 fps)
MAX_GAP_S = 0.3        # bridges one missed sample at 8 fps

NAMES = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
         "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
         "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
         "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
         "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"]
TORSO = tuple(NAMES.index(n) for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))   # 11, 12, 23, 24


# ---------------------------------------------------------------- per body

def crop_window(box, width: int, height: int):
    """The square crop for an [x, y, w, h] box in a width x height frame: (x0, y0, side) of the square in frame pixels
    (it may extend past the frame), and (bx0, by0, bx1, by1), the grown box clipped to the frame, the only part copied."""
    x1, y1, x2, y2 = box[0], box[1], box[0] + box[2], box[1] + box[3]
    mx, my = (x2 - x1) * (MARGIN - 1) / 2, (y2 - y1) * (MARGIN - 1) / 2
    bx0, by0 = int(max(x1 - mx, 0)), int(max(y1 - my, 0))
    bx1, by1 = int(min(x2 + mx, width)), int(min(y2 + my, height))
    side = max(bx1 - bx0, by1 - by0)
    x0, y0 = (bx0 + bx1) // 2 - side // 2, (by0 + by1) // 2 - side // 2
    return (x0, y0, side), (bx0, by0, bx1, by1)


def crop(frame, box):
    """The landmarker's input for one body: the grown box on black, in a square (frame's channel order), and where the
    square sits in the frame (x0, y0, side)."""
    height, width = frame.shape[:2]
    (x0, y0, side), (bx0, by0, bx1, by1) = crop_window(box, width, height)
    square = np.zeros((side, side, frame.shape[2]), np.uint8)
    square[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = frame[by0:by1, bx0:bx1]
    return square, (x0, y0, side)


def in_frame(poses, window):
    """The landmarker's poses for a crop at window (x0, y0, side), with their 2D landmarks in frame pixels."""
    x0, y0, side = window
    return [(np.c_[lms[:, 0] * side + x0, lms[:, 1] * side + y0, lms[:, 2]], world) for lms, world in poses]


def nearest_pose(poses, box):
    """The pose whose 2D landmarks centre nearest the box centre, or None. poses: [(kpts (33, 3) x, y, visibility in
    frame pixels, world (33, 3) metres)]."""
    centre = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])
    return min(poses, key=lambda p: np.linalg.norm(p[0][:, :2].mean(0) - centre)) if poses else None


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def torso_facing(world, visible):
    """(yaw_deg, pitch_deg or None, "torso" | "shoulders") from camera-frame world landmarks, or (None, None, None)."""
    left, right, left_hip, right_hip = TORSO
    if not (visible[left] and visible[right]):
        return None, None, None
    shoulders = world[left] - world[right]
    if visible[left_hip] and visible[right_hip]:
        normal = np.cross(unit(shoulders) + unit(world[left_hip] - world[right_hip]),
                          (world[left] + world[right]) / 2 - (world[left_hip] + world[right_hip]) / 2)
        source = "torso"
    else:
        normal = np.cross(shoulders, np.array([0.0, -1.0, 0.0]))
        source = "shoulders"
    if np.linalg.norm(normal) == 0:
        return None, None, None
    f = unit(normal)
    yaw = float(np.degrees(np.arctan2(-f[0], -f[2])))
    pitch = float(np.degrees(np.arctan2(-f[1], np.hypot(f[0], f[2])))) if source == "torso" else None
    return yaw, pitch, source


def stored(kpts, yaw, pitch):
    """A body's landmarks and facing as the bake-off's pose file stored them (x, y to 0.1 px, visibility to 0.01;
    degrees to 0.1), which its facing step then read: (kpts as lists, [yaw, pitch] or None)."""
    k = [[round(float(x), 1), round(float(y), 1), round(float(c), 2)] for x, y, c in kpts]
    return k, None if yaw is None else [round(yaw, 1), None if pitch is None else round(pitch, 1)]


def body_facing(pose):
    """(stored landmarks, stored facing, source) for the chosen pose, or None when the crop held no pose."""
    if pose is None:
        return None
    kpts, world = pose
    yaw, pitch, source = torso_facing(world, kpts[:, 2] >= SCORE_THRESHOLD)
    k, facing = stored(kpts, yaw, pitch)
    return k, facing, source


# ---------------------------------------------------------------- between bodies

def facing_vector(yaw_deg, pitch_deg):
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg or 0.0)
    return np.array([-np.sin(yaw) * np.cos(pitch), -np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])


def torso_centre(kpts, box):
    """2D mean of the visible shoulders and hips, else the box centre."""
    k = np.array(kpts)[list(TORSO)]
    visible = k[:, 2] >= SCORE_THRESHOLD
    if visible.any():
        return k[visible, :2].mean(0)
    x, y, w, h = box
    return np.array([x + w / 2, y + h / 2])


def merge(times, gap):
    runs = []
    for t in times:
        if runs and t - runs[-1][-1] <= gap + 1e-6:
            runs[-1].append(t)
        else:
            runs.append([t])
    return runs


def face_number(face_id: str) -> int:
    """'face:12' -> 12: faces are ordered by number (ties between equally near targets, event order), as the bake-off's
    integer face tracks were."""
    return int(face_id.split(":", 1)[1])


def facing_events(frames, cone_deg=CONE_DEG, min_facing_s=MIN_FACING_S, max_gap_s=MAX_GAP_S):
    """The facing layer's events. frames: {(shot, t): [(face number, torso centre (2,), [yaw, pitch] or None)]}, the
    face-linked bodies posed at each sample time."""
    positives = defaultdict(list)   # (A, B) -> [(t, angle)]
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


def linked_samples(bodies_layer):
    """[(shot, t, face number, box)] for every sample of a body track linked to a face, in track order."""
    return [(tr["shot"], s["t"], face_number(tr["face"]), s["box"])
            for tr in bodies_layer["tracks"] if tr["face"] is not None for s in tr["samples"]]
