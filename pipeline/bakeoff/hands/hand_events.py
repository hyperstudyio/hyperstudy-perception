"""Hand events derived from hands_<model>.json: hand over mouth, self-touch of the face, pointing, raised hand.

Per hand sample (only keypoints above the model's KPT_SCORE_THRESHOLD count, and a sample needs MIN_VALID_KPTS of them):
  hand_over_mouth  the hand's owner (person.face_track) has py-feat landmarks at that time (face_detail.json,
                   interpolated over gaps up to MAX_FACE_GAP_S) and at least MIN_REGION_KPTS hand keypoints lie
                   within MOUTH_RADIUS x face size of the mouth centre (mean of landmarks 48-67), or the mouth
                   centre lies inside the hull of the hand keypoints. score = fraction of the hand's valid keypoints
                   near the mouth.
  self_touch_face  not over the mouth, and at least MIN_REGION_KPTS hand keypoints inside the owner's face region:
                   the hull of the 68 landmarks plus a forehead band (brow points raised by FOREHEAD x face size),
                   grown by FACE_MARGIN x face size. score = fraction of valid keypoints inside. In 2D a hand in front
                   of the face cannot be told from one touching it; both count.
  pointing         index finger extended (wrist-to-tip > EXTENDED_RATIO x wrist-to-PIP, index straight, index not
                   foreshortened) and middle, ring and pinky curled (wrist-to-tip < CURLED_RATIO x wrist-to-PIP).
                   detail.direction = mean unit vector index MCP -> tip; detail.target = the other face track or
                   objects.json object whose box centre is nearest the ray from the index tip (perpendicular distance
                   within half the box's larger side), else null. Works for unlinked hands too (face_track null).
                   Not evaluated while the same person's two hand boxes overlap by CLASPED_IOU (interlocked fingers).
  raised_hand      the wrist (hand keypoint 0) is above the owner's same-side shoulder by more than RAISE_MARGIN x
                   shoulder width BOTH in the image (y) and along the body's own up axis (shoulder midpoint -> nose).
                   Either alone fails on nes-biking: image y alone raises the hands of a rider upside down in a flip
                   (67.5 s); the body axis alone raises the hands of riders bent over the handlebars with the head
                   bowed toward the camera (7 of 8 events). Needs both shoulders and the nose visible in the body
                   keypoints of the pose file the hands came from.
                   A hand on the face also has its wrist above the shoulders in a close-up, so a hand that is on the
                   face counts only when its wrist is above the nose (hands on the head). detail.level = "head" when
                   the wrist is also above the nose (both ways), else "shoulder".
Positive samples of one hand and kind are merged when at most MAX_EVENT_GAP_S apart; events shorter than
MIN_EVENT_S are dropped. Face events need face_detail.json; without it they are skipped and the reason printed.

Usage: uv run python hand_events.py ../runs/<name> --hands hands_<model>.json [--pose pose_rtmw.json]
       (--pose must be the body file hands.py read, e.g. hands_body_rtmw.json for runs posed by body_rtmw.py)
Writes runs/<name>/hand_events.json.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Per-model keypoint score above which a keypoint is used. Scores are not comparable across models, so each threshold
# was set where the model's keypoints agree with an independent model equally well: on 1198 OOO hands found by both
# RTMPose-hand and MediaPipe, the median keypoint distance to MediaPipe's (as a fraction of the hand box) is 0.11 for
# RTMPose-hand scores 0.2-0.3, 0.066 for 0.3-0.4 and 0.04 above; for RTMW it is 0.18 at 3-4, 0.11 at 4-5, 0.063 at 5-6.
# So RTMPose-hand 0.3 and RTMW 5.0 (~0.065 of the box, about a finger width). RTMW's body-stage threshold of 3.0 is
# too loose for fingers. MediaPipe has no per-landmark score (hands.py writes 1.0), so all its landmarks pass.
KPT_SCORE_THRESHOLD = {"rtmw": 5.0, "rtmpose_hand": 0.3, "mediapipe": 0.5}
BODY_SCORE_THRESHOLD = 3.0  # RTMW body keypoints (shoulders, nose): pose.py's calibrated visibility threshold
MIN_VALID_KPTS = 6          # fewer valid keypoints than this and the sample says too little about the hand's shape
MAX_FACE_GAP_S = 0.5        # face landmarks are interpolated across gaps up to this (a hand over the mouth often hides the face from RetinaFace)
MIN_REGION_KPTS = 3         # keypoints in a region needed to call a contact; 1-2 points are often a stray fingertip or keypoint noise
MOUTH_RADIUS = 0.2          # x face size; the mouth spans ~0.35 face widths, so this is its half-width plus a small margin
FOREHEAD = 0.25             # x face size; py-feat's 68 landmarks stop at the brows, the forehead is about this tall above them
FACE_MARGIN = 0.05          # x face size added around the face region (hairline, cheek edge)
EXTENDED_RATIO = 1.2        # straight index: wrist-to-tip ~1.35 x wrist-to-PIP (palm 1 + finger 0.9 vs palm 1 + 0.45)
CURLED_RATIO = 1.0          # a curled finger's tip folds back to the palm, nearer the wrist than its own PIP joint
STRAIGHT_COS = 0.7          # index MCP->PIP and PIP->tip within ~45 degrees
MIN_INDEX_TO_PALM = 0.5     # index MCP->tip at least this x wrist->middle MCP; shorter is foreshortened (pointing at the camera) and has no 2D direction
# Both OOO pointing events before this rule were one man's interlocked hands read as an extended index (verified on frames).
CLASPED_IOU = 0.2
RAISE_MARGIN = 0.25         # x shoulder width above the shoulder line, so a hand resting on the shoulder or collarbone is not raised
MAX_EVENT_GAP_S = 0.3       # merge positives up to this apart: bridges one missed sample at 7.5-8 fps
MIN_EVENT_S = 0.25          # end - start; at 7.5-8 fps that is at least 3 consecutive samples. Single-sample positives are
                            # mostly keypoint glitches (one frame of a finger snapping across the face)

MOUTH = slice(48, 68)
WRIST, INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP = 0, 5, 6, 8, 9
CURLED_FINGERS = [(10, 12), (14, 16), (18, 20)]  # (PIP, tip) of middle, ring, pinky
BODY_NOSE, BODY_SHOULDER = 0, {"left": 5, "right": 6}


# ---------------------------------------------------------------- lookups

def series_at(series, t, value):
    """Linear interpolation of value(sample) at t over samples with "t", across gaps up to MAX_FACE_GAP_S; None outside."""
    times = [s["t"] for s in series]
    j = int(np.searchsorted(times, t))
    if j < len(times) and abs(times[j] - t) < 1e-3:
        return np.asarray(value(series[j]), dtype=float)
    if 0 < j < len(times) and times[j] - times[j - 1] <= MAX_FACE_GAP_S:
        w = (t - times[j - 1]) / (times[j] - times[j - 1])
        return (1 - w) * np.asarray(value(series[j - 1]), dtype=float) + w * np.asarray(value(series[j]), dtype=float)
    return None


def face_geometry(lmk, box):
    """(mouth centre, face size, face region polygon) from 68 py-feat landmarks [x0..x67, y0..y67] and the face box side."""
    pts = np.stack([lmk[:68], lmk[68:]], axis=1)
    size = float(box[2])
    brows_up = pts[17:27] - [0.0, FOREHEAD * size]
    hull = cv2.convexHull(np.vstack([pts, brows_up]).astype(np.float32))[:, 0]
    centre = hull.mean(0)
    grown = centre + (hull - centre) * (1 + FACE_MARGIN * size / max(np.linalg.norm(hull - centre, axis=1).mean(), 1.0))
    return pts[MOUTH].mean(0), size, grown.astype(np.float32)


def inside(polygon, point):
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


# ---------------------------------------------------------------- per-sample tests

def mouth_and_face(kpts, valid, geometry):
    """(over_mouth score or None, touch score or None)."""
    mouth, size, region = geometry
    pts = kpts[valid, :2]
    near = np.hypot(*(pts - mouth).T) <= MOUTH_RADIUS * size
    hull = cv2.convexHull(pts.astype(np.float32))
    if near.sum() >= MIN_REGION_KPTS or (len(pts) >= 3 and inside(hull, mouth)):
        return float(near.mean()), None
    on_face = np.array([inside(region, p) for p in pts])
    if on_face.sum() >= MIN_REGION_KPTS:
        return None, float(on_face.mean())
    return None, None


def pointing(kpts, valid):
    """Unit direction of a pointing index finger, or None."""
    needed = [WRIST, INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP] + [i for pair in CURLED_FINGERS for i in pair]
    if not valid[needed].all():
        return None
    p = kpts[:, :2]
    d = lambda a, b: float(np.hypot(*(p[a] - p[b])))
    if d(WRIST, INDEX_TIP) < EXTENDED_RATIO * d(WRIST, INDEX_PIP):
        return None
    if d(INDEX_MCP, INDEX_TIP) < MIN_INDEX_TO_PALM * d(WRIST, MIDDLE_MCP):
        return None
    a, b = p[INDEX_PIP] - p[INDEX_MCP], p[INDEX_TIP] - p[INDEX_PIP]
    if np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6) < STRAIGHT_COS:
        return None
    if any(d(WRIST, tip) >= CURLED_RATIO * d(WRIST, pip) for pip, tip in CURLED_FINGERS):
        return None
    v = p[INDEX_TIP] - p[INDEX_MCP]
    return v / np.linalg.norm(v)


def ray_target(origin, direction, candidates):
    """Id of the candidate box whose centre is nearest the ray (perpendicular distance <= half its larger side), or None."""
    best, best_ratio = None, 1.0
    for target_id, (x, y, w, h) in candidates:
        rel = np.array([x + w / 2, y + h / 2]) - origin
        along = float(rel @ direction)
        if along <= 0:
            continue
        ratio = float(np.linalg.norm(rel - along * direction)) / (max(w, h) / 2)
        if ratio <= best_ratio:
            best, best_ratio = target_id, ratio
    return best


def iou(a, b):
    """IoU of two [x, y, w, h] boxes."""
    w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def merge(times, gap):
    """Runs of sorted times whose neighbours are at most gap apart."""
    runs = []
    for t in times:
        if runs and t - runs[-1][-1] <= gap + 1e-6:
            runs[-1].append(t)
        else:
            runs.append([t])
    return runs


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--hands", required=True)
    parser.add_argument("--pose", default="pose_rtmw.json", help="the body file hands.py used (its pose track ids)")
    args = parser.parse_args()

    hands = json.loads((args.run_dir / args.hands).read_text())
    threshold = KPT_SCORE_THRESHOLD[hands["model"]]
    extract = json.loads((args.run_dir / "extract.json").read_text())
    width, height = extract["video"]["width"], extract["video"]["height"]
    faces = json.loads((args.run_dir / "faces.json").read_text())
    pose = json.loads((args.run_dir / args.pose).read_text())
    body = {tr["id"]: {round(s["t"], 3): np.array(s["kpts"]) for s in tr["samples"]} for tr in pose["tracks"]}
    detail_path = args.run_dir / "face_detail.json"
    detail = json.loads(detail_path.read_text())["tracks"] if detail_path.exists() else None
    if detail is None:
        print(f"hand_over_mouth and self_touch_face skipped: {detail_path} does not exist (run faces/face_detail.py)")
    objects_path = args.run_dir / "objects.json"
    objects = json.loads(objects_path.read_text())["tracks"] if objects_path.exists() else []

    def targets(t, shot, own_face):
        out = []
        for i, tr in enumerate(faces["tracks"]):
            if i != own_face and tr["shot"] == shot:
                box = series_at(tr["series"], t, lambda s: s["box"])
                if box is not None:
                    out.append((i, box))
        for ob in objects:
            if ob["shot"] == shot:
                box = series_at(ob["samples"], t, lambda s: s["box"])
                if box is not None:
                    out.append((ob["id"], box))
        return out

    boxes_at = {}  # (pose track, side, t) -> hand box, for the clasped-hands check
    for hand in hands["hands"]:
        if hand["person"] is not None:
            for s in hand["samples"]:
                boxes_at[(hand["person"]["pose_track"], hand["side"], s["t"])] = s["box"]

    events, stats = [], {"samples": 0, "too_few_kpts": 0, "no_face_landmarks": 0, "no_shoulders_or_nose": 0, "clasped": 0}
    for hand in hands["hands"]:
        owner = hand["person"]
        face_id = owner["face_track"] if owner else None
        positives = {"hand_over_mouth": {}, "self_touch_face": {}, "pointing": {}, "raised_hand": {}}
        for s in hand["samples"]:
            stats["samples"] += 1
            t, kpts = s["t"], np.array(s["kpts"])
            valid = (kpts[:, 2] >= threshold) & (kpts[:, 0] >= 0) & (kpts[:, 0] < width) & (kpts[:, 1] >= 0) & (kpts[:, 1] < height)
            if valid.sum() < MIN_VALID_KPTS:
                stats["too_few_kpts"] += 1
                continue
            on_face = False
            if detail is not None and face_id is not None:
                lmk = series_at(detail[face_id], t, lambda x: x["lmk"])
                box = series_at(faces["tracks"][face_id]["series"], t, lambda x: x["box"])
                if lmk is None or box is None:
                    stats["no_face_landmarks"] += 1
                else:
                    over, touch = mouth_and_face(kpts, valid, face_geometry(lmk, box))
                    if over is not None:
                        positives["hand_over_mouth"][t] = (over, {})
                    elif touch is not None:
                        positives["self_touch_face"][t] = (touch, {})
                    on_face = over is not None or touch is not None
            other = boxes_at.get((owner["pose_track"], {"left": "right", "right": "left"}[hand["side"]], t)) if owner else None
            clasped = other is not None and iou(other, s["box"]) >= CLASPED_IOU
            stats["clasped"] += clasped
            direction = None if clasped else pointing(kpts, valid)
            if direction is not None:
                positives["pointing"][t] = (1.0, {"direction": direction,
                                                  "target": ray_target(kpts[INDEX_TIP, :2], direction, targets(t, hand["shot"], face_id))})
            if owner is not None and valid[WRIST]:
                b = body[owner["pose_track"]].get(round(t, 3))
                if b is None:
                    raise SystemExit(f"{hand['id']} t={t}: {owner['pose_track']} has no sample in {args.pose}; was --pose the file hands.py used?")
                ls, rs, nose = b[BODY_SHOULDER["left"]], b[BODY_SHOULDER["right"]], b[BODY_NOSE]
                if min(ls[2], rs[2], nose[2]) < BODY_SCORE_THRESHOLD or max(ls[1], rs[1]) >= height:
                    stats["no_shoulders_or_nose"] += 1
                else:
                    up = nose[:2] - (ls[:2] + rs[:2]) / 2
                    up = up / max(np.linalg.norm(up), 1e-6)
                    wrist, shoulder = kpts[WRIST, :2], b[BODY_SHOULDER[hand["side"]]][:2]
                    image_up = np.array([0.0, -1.0])
                    rise = min(float((wrist - shoulder) @ up), float((wrist - shoulder) @ image_up))
                    head = float((wrist - nose[:2]) @ up) > 0 and float((wrist - nose[:2]) @ image_up) > 0
                    if rise > RAISE_MARGIN * float(np.hypot(*(ls[:2] - rs[:2]))) and (head or not on_face):
                        positives["raised_hand"][t] = (1.0, {"level": "head" if head else "shoulder"})

        for kind, samples in positives.items():
            for run in merge(sorted(samples), MAX_EVENT_GAP_S):
                if run[-1] - run[0] < MIN_EVENT_S:
                    continue
                event = {"kind": kind, "hand": hand["id"], "face_track": face_id, "start": run[0], "end": run[-1],
                         "score": round(float(np.mean([samples[t][0] for t in run])), 2), "detail": {}}
                if kind == "pointing":
                    mean = np.mean([samples[t][1]["direction"] for t in run], axis=0)
                    found = [samples[t][1]["target"] for t in run if samples[t][1]["target"] is not None]
                    event["detail"] = {"direction": [round(float(v), 2) for v in mean / np.linalg.norm(mean)],
                                       "target": max(set(found), key=found.count) if found else None,
                                       "samples": len(run)}
                elif kind == "raised_hand":
                    levels = [samples[t][1]["level"] for t in run]
                    event["detail"] = {"level": "head" if levels.count("head") * 2 > len(levels) else "shoulder", "samples": len(run)}
                else:
                    event["detail"] = {"samples": len(run)}
                events.append(event)

    events.sort(key=lambda e: (e["start"], e["hand"]))
    path = args.run_dir / "hand_events.json"
    path.write_text(json.dumps({"source": args.hands, "events": events}, indent=1))
    counts = {k: sum(e["kind"] == k for e in events) for k in ("hand_over_mouth", "self_touch_face", "pointing", "raised_hand")}
    print(f"{path.name} from {args.hands}: {len(events)} events {counts}; sample stats {stats}")


if __name__ == "__main__":
    main()
