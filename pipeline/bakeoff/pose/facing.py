"""Who faces whom, from the per-sample torso facing direction in a 3D pose file (pose3d.py).

Per sample time and shot, only bodies linked to a face track take part. For body A with a facing direction:
  1. A's facing vector in the camera frame (x right, y down, z away) from its [yaw, pitch]:
     f = (-sin(yaw) cos(pitch), -sin(pitch), -cos(yaw) cos(pitch)); pitch 0 when null (shoulders only).
     This keeps the depth component of A's own facing (a body turned 30 degrees toward a neighbour points mostly
     at the camera, and is not facing the neighbour).
  2. The direction to body B is from A's torso centre to B's (2D mean of the visible shoulders and hips, else the
     box centre), with zero depth: the 3D models are root-relative, so they give no depth BETWEEN people. Depth
     from body scale would need the camera's focal length, which is unknown, and the per-body scale is noisy, so
     the bodies of one shot are taken to stand at equal depth (true for OOO's line-ups and its seated pairs only
     approximately).
  3. A faces B when the angle between f and the direction to B is at most CONE_DEG. Of several bodies inside the
     cone, the nearest in the image is the one faced (in a line-up, the next person blocks the ones beyond).
Positive samples of one (A, B) pair are merged when at most MAX_GAP_S apart; runs shorter than MIN_FACING_S
(end - start) are dropped. score = mean cos(angle) over the event's samples. "mutual" is true when B faces A
for at least MIN_FACING_S of the event (detail.mutual_s).

Usage: uv run python facing.py ../runs/<name> --pose pose_<model>.json
Writes runs/<name>/facing_<model>.json.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# A target lying sideways at the same depth is at 90 - |yaw| degrees from the facing vector, so the cone sets the
# torso turn needed: 60 degrees = turned at least 30 degrees toward the person. See the module docstring for why
# the depth between people is not used.
CONE_DEG = 60.0
# Bodies reorient slowly; a facing that lasts under a second is mostly yaw jitter of a body near the cone edge or a
# passing turn. At 8 fps, 1.0 s is at least 9 samples.
MIN_FACING_S = 1.0
MAX_GAP_S = 0.3   # merge positives up to this apart: bridges one missed sample at 8 fps (as hand_events.py)


def facing_vector(yaw_deg, pitch_deg):
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg or 0.0)
    return np.array([-np.sin(yaw) * np.cos(pitch), -np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])


def torso_centre(sample, joints, threshold):
    k = np.array(sample["kpts"])[list(joints)]
    visible = k[:, 2] >= threshold
    if visible.any():
        return k[visible, :2].mean(0)
    x, y, w, h = sample["box"]
    return np.array([x + w / 2, y + h / 2])


def merge(times, gap):
    runs = []
    for t in times:
        if runs and t - runs[-1][-1] <= gap + 1e-6:
            runs[-1].append(t)
        else:
            runs.append([t])
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pose", required=True)
    args = parser.parse_args()
    pose = json.loads((args.run_dir / args.pose).read_text())
    if not any("facing" in s for tr in pose["tracks"] for s in tr["samples"]):
        raise SystemExit(f"{args.pose} has no per-sample 'facing' (write it with pose3d.py)")
    names = pose["keypoints"]
    joints = [names.index(n) for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    threshold = pose["score_threshold"]

    by_time = defaultdict(list)  # (shot, t) -> [(face track, centre, facing or None)]
    for track in pose["tracks"]:
        if track["face_track"] is None:
            continue
        for s in track["samples"]:
            by_time[(track["shot"], s["t"])].append((track["face_track"], torso_centre(s, joints, threshold), s["facing"]))

    positives = defaultdict(list)   # (A, B) -> [(t, angle)]
    per_sample = defaultdict(list)  # A -> [[t, target, angle]]
    for (shot, t), bodies in sorted(by_time.items(), key=lambda kv: kv[0][1]):
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
                if angle <= CONE_DEG:
                    inside.append((dist, b, angle))
            target = min(inside)[1:] if inside else (None, None)
            per_sample[a].append([t, target[0], None if target[1] is None else round(target[1], 1)])
            if target[0] is not None:
                positives[(a, target[0])].append((t, target[1]))

    runs = {}
    for pair, samples in positives.items():
        angle_at = dict(samples)
        for run in merge(sorted(angle_at), MAX_GAP_S):
            if run[-1] - run[0] >= MIN_FACING_S - 1e-6:
                runs.setdefault(pair, []).append((run, [angle_at[t] for t in run]))
    events = []
    for (a, b), pair_runs in runs.items():
        reverse = {t for t, _ in positives.get((b, a), [])}
        for run, angles in pair_runs:
            both = [t for t in run if t in reverse]
            mutual_s = sum(r[-1] - r[0] for r in merge(both, MAX_GAP_S))
            events.append({"kind": "facing", "from_face_track": a, "to_face_track": b, "start": run[0], "end": run[-1],
                           "score": round(float(np.mean(np.cos(np.radians(angles)))), 3),
                           "mutual": mutual_s >= MIN_FACING_S - 1e-6,
                           "detail": {"mean_angle_deg": round(float(np.mean(angles)), 1), "samples": len(run),
                                      "mutual_s": round(mutual_s, 3)}})
    events.sort(key=lambda e: (e["start"], e["from_face_track"]))
    out = {"source": args.pose, "cone_deg": CONE_DEG, "min_facing_s": MIN_FACING_S, "max_gap_s": MAX_GAP_S,
           "events": events, "per_sample": {str(k): v for k, v in sorted(per_sample.items())}}
    name = Path(args.pose).stem.replace("pose_", "facing_")
    (args.run_dir / f"{name}.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"{name}.json: {len(events)} facing events ({sum(e['mutual'] for e in events)} mutual) between "
          f"{len({e['from_face_track'] for e in events} | {e['to_face_track'] for e in events})} face tracks; "
          f"{sum(len(v) for v in per_sample.values())} samples with a facing direction, "
          f"{sum(1 for v in per_sample.values() for s in v if s[1] is not None)} facing someone")


if __name__ == "__main__":
    main()
