"""Hand events from the hands layer's keypoints: hand over mouth, self-touch of the face, pointing (and at what), raised
hand. The bake-off's rules (tools/perception-bakeoff/hands/hand_events.py), reading the pipeline's layers. numpy only
(the convex hull and point-in-polygon test the bake-off took from OpenCV are here), so it is tested without them.

Per hand sample (only keypoints above KPT_SCORE_THRESHOLD and inside the frame count, and a sample needs
MIN_VALID_KPTS of them):
  hand_over_mouth  the hand's face track (the hands layer's face, from its body) has py-feat landmarks at that time
                   (faces_detail, interpolated over gaps up to MAX_FACE_GAP_S) and at least MIN_REGION_KPTS hand
                   keypoints lie within MOUTH_RADIUS x face size of the mouth centre (mean of landmarks 48-67), or the
                   mouth centre lies inside the hull of the hand keypoints. score = fraction of the hand's valid
                   keypoints near the mouth.
  self_touch_face  not over the mouth, and at least MIN_REGION_KPTS hand keypoints inside the face region: the hull of
                   the 68 landmarks plus a forehead band (brow points raised by FOREHEAD x face size), grown by
                   FACE_MARGIN x face size. score = fraction of valid keypoints inside. In 2D a hand in front of the
                   face cannot be told from one touching it; both count.
  pointing         index finger extended (wrist-to-tip > EXTENDED_RATIO x wrist-to-PIP, index straight, index not
                   foreshortened) and middle, ring and pinky curled (wrist-to-tip < CURLED_RATIO x wrist-to-PIP).
                   detail.direction = mean unit vector index MCP -> tip; detail.target = the other face track (face:N)
                   or object track (obj:N) of the shot whose box centre is nearest the ray from the index tip
                   (perpendicular distance within half the box's larger side), else null. Objects are targets only
                   when the run has the objects layer (it is an optional input: without it, faces only). Works for
                   hands with no face too. Not evaluated while the same body's two hand boxes overlap by CLASPED_IOU
                   (interlocked fingers).
  raised_hand      the wrist (hand keypoint 0) is above the body's same-side shoulder by more than RAISE_MARGIN x
                   shoulder width BOTH in the image (y) and along the body's own up axis (shoulder midpoint -> nose).
                   Either alone fails on nes-biking: image y alone raises the hands of a rider upside down in a flip
                   (67.5 s); the body axis alone raises the hands of riders bent over the handlebars with the head
                   bowed toward the camera. Needs both shoulders and the nose above the bodies layer's score threshold.
                   A hand on the face also has its wrist above the shoulders in a close-up, so a hand on the face
                   counts only when its wrist is above the nose (hands on the head). level "head" when the wrist is
                   also above the nose (both ways), else "shoulder".
Positive samples of one hand and kind are merged when at most MAX_EVENT_GAP_S apart; events shorter than MIN_EVENT_S
are dropped.

Each sample's measurements for every rule are kept (sample_evidence), and the events are made from them alone
(events_from_evidence): the side file of them (gesture_evidence.py) gives the layer back, and is where the thresholds
can be revisited.

Two deliberate differences from the bake-off: a pointing target is a bundle id (face:N, obj:N), where the bake-off's
layer copied its file's face track number (an int, which the app's GestureEvent.detail.target, a string, does not
read); and when two targets are named by equally many samples of a pointing event, the first named wins, where the
bake-off took whichever its set iterated first (Python's string hashing is seeded per process, so it could differ
between two runs). Neither occurs on the bake-off's clips (no pointing events).
"""

from __future__ import annotations

import numpy as np

# RTMPose-hand's keypoint score above which a keypoint is used (the bake-off's KPT_SCORE_THRESHOLD["rtmpose_hand"]):
# on 1198 OOO hands found by both RTMPose-hand and MediaPipe, the median keypoint distance to MediaPipe's (as a
# fraction of the hand box) is 0.11 for scores 0.2-0.3, 0.066 for 0.3-0.4 and 0.04 above: 0.3 is about a finger width.
# Scores are not comparable across models, so the node checks the hands layer is RTMPose-hand's (HAND_MODEL).
KPT_SCORE_THRESHOLD = 0.3
HAND_MODEL = "RTMPose-m Hand5"
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
POINTING_KPTS = [WRIST, INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP] + [i for pair in CURLED_FINGERS for i in pair]
KINDS = ("hand_over_mouth", "self_touch_face", "pointing", "raised_hand")
OTHER_SIDE = {"left": "right", "right": "left"}
IMAGE_UP = np.array([0.0, -1.0])


# ---------------------------------------------------------------- geometry (numpy, in place of OpenCV)

def convex_hull(points) -> np.ndarray:
    """The convex hull's vertices (Andrew's monotone chain), counter-clockwise in image coordinates, collinear points
    left out (as cv2.convexHull)."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)   # sorted by x, then y
    if len(pts) < 3:
        return pts

    def turn(o, a, b):   # z of (a - o) x (b - o): > 0 for a left turn
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and turn(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out
    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def inside(polygon, point) -> bool:
    """Whether point lies inside the polygon or on its boundary (cv2.pointPolygonTest(..., False) >= 0). A polygon of
    one or two vertices (a degenerate hull) holds only the points on it."""
    poly = np.asarray(polygon, dtype=float)
    x, y = float(point[0]), float(point[1])
    n, crossings = len(poly), 0
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        # on this edge: collinear with it and within its extent
        if (x1 - x0) * (y - y0) == (y1 - y0) * (x - x0) and min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1):
            return True
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            crossings += 1
    return n >= 3 and crossings % 2 == 1


def iou(a, b) -> float:
    """IoU of two [x, y, w, h] boxes."""
    w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


# ---------------------------------------------------------------- lookups

class Series:
    """A track's per-sample values by time (face landmarks, face or object boxes)."""

    def __init__(self, samples: list[dict], key: str):
        self.times = [s["t"] for s in samples]
        self.values = [np.asarray(s[key], dtype=float) for s in samples]

    def at(self, t: float):
        """Linear interpolation at t across gaps up to MAX_FACE_GAP_S; None outside (the bake-off's series_at)."""
        times, j = self.times, int(np.searchsorted(self.times, t))
        if j < len(times) and abs(times[j] - t) < 1e-3:
            return self.values[j]
        if 0 < j < len(times) and times[j] - times[j - 1] <= MAX_FACE_GAP_S:
            w = (t - times[j - 1]) / (times[j] - times[j - 1])
            return (1 - w) * self.values[j - 1] + w * self.values[j]
        return None


def face_geometry(lmk, box):
    """(mouth centre, face size, face region polygon) from 68 py-feat landmarks [x0..x67, y0..y67] and the face box."""
    pts = np.stack([lmk[:68], lmk[68:]], axis=1)
    size = float(box[2])
    brows_up = pts[17:27] - [0.0, FOREHEAD * size]
    hull = convex_hull(np.vstack([pts, brows_up]))
    centre = hull.mean(0)
    grown = centre + (hull - centre) * (1 + FACE_MARGIN * size / max(np.linalg.norm(hull - centre, axis=1).mean(), 1.0))
    return pts[MOUTH].mean(0), size, grown


def valid_keypoints(kpts, width: int, height: int):
    """Keypoints above KPT_SCORE_THRESHOLD and inside the width x height frame."""
    return ((kpts[:, 2] >= KPT_SCORE_THRESHOLD) & (kpts[:, 0] >= 0) & (kpts[:, 0] < width)
            & (kpts[:, 1] >= 0) & (kpts[:, 1] < height))


# ---------------------------------------------------------------- per-sample measurements

def face_measures(kpts, valid, geometry) -> dict:
    """Hand over mouth and on face: every keypoint's distance to the mouth centre (in face sizes), how many valid ones
    are near it, whether the mouth is inside the valid keypoints' hull, how many valid ones are inside the face region,
    and the two rules' outcomes (over the mouth wins)."""
    mouth, size, region = geometry
    dist = np.hypot(*(kpts[:, :2] - mouth).T)
    pts = kpts[valid, :2]
    near = int((dist[valid] <= MOUTH_RADIUS * size).sum())
    in_hull = len(pts) >= 3 and inside(convex_hull(pts), mouth)
    on_face = int(sum(inside(region, p) for p in pts))
    over = near >= MIN_REGION_KPTS or in_hull
    return {"face_size": size, "mouth_dist": dist / size, "mouth_near": near, "mouth_in_hull": bool(in_hull),
            "face_inside": on_face, "hand_over_mouth": bool(over), "self_touch_face": bool(not over and on_face >= MIN_REGION_KPTS)}


def finger_measures(kpts) -> dict:
    """Pointing: the index's extension (wrist-to-tip / wrist-to-PIP), reach (MCP-to-tip / wrist-to-middle MCP) and
    straightness (cosine of MCP->PIP and PIP->tip), each other finger's wrist-to-tip / wrist-to-PIP, the outcome, and
    the unit direction MCP -> tip when it points. The outcome compares as the bake-off did (products, not the ratios)."""
    p = kpts[:, :2]
    d = lambda a, b: float(np.hypot(*(p[a] - p[b])))
    ratio = lambda a, b: a / max(b, 1e-6)
    a, b = p[INDEX_PIP] - p[INDEX_MCP], p[INDEX_TIP] - p[INDEX_PIP]
    cos = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6))
    points = (d(WRIST, INDEX_TIP) >= EXTENDED_RATIO * d(WRIST, INDEX_PIP)
              and d(INDEX_MCP, INDEX_TIP) >= MIN_INDEX_TO_PALM * d(WRIST, MIDDLE_MCP)
              and cos >= STRAIGHT_COS
              and not any(d(WRIST, tip) >= CURLED_RATIO * d(WRIST, pip) for pip, tip in CURLED_FINGERS))
    v = p[INDEX_TIP] - p[INDEX_MCP]
    return {"index_extension": ratio(d(WRIST, INDEX_TIP), d(WRIST, INDEX_PIP)),
            "index_reach": ratio(d(INDEX_MCP, INDEX_TIP), d(WRIST, MIDDLE_MCP)), "index_straightness": cos,
            "curl": [ratio(d(WRIST, tip), d(WRIST, pip)) for pip, tip in CURLED_FINGERS], "pointing": bool(points),
            "direction": v / np.linalg.norm(v) if points else None}


def ray_target(origin, direction, candidates):
    """Id of the candidate box whose centre is nearest the ray (perpendicular distance <= half its larger side), or None;
    on a tie the later candidate (as the bake-off)."""
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


def raise_measures(wrist, side: str, body_kpts, idx: dict, threshold: float, height: int, on_face: bool) -> dict:
    """Raised hand: the wrist's rise above its same-side shoulder (px, the lesser of along the image's up and the
    body's), the shoulder width (px), whether the wrist is above the nose both ways, and the outcome. raise_status
    no_shoulders_or_nose when a shoulder or the nose is below the bodies layer's score threshold, or a shoulder is
    below the frame."""
    ls, rs, nose = body_kpts[idx["left_shoulder"]], body_kpts[idx["right_shoulder"]], body_kpts[idx["nose"]]
    if min(ls[2], rs[2], nose[2]) < threshold or max(ls[1], rs[1]) >= height:
        return {"raise_status": "no_shoulders_or_nose"}
    up = nose[:2] - (ls[:2] + rs[:2]) / 2
    up = up / max(np.linalg.norm(up), 1e-6)
    shoulder = (ls if side == "left" else rs)[:2]
    rise = min(float((wrist - shoulder) @ up), float((wrist - shoulder) @ IMAGE_UP))
    head = float((wrist - nose[:2]) @ up) > 0 and float((wrist - nose[:2]) @ IMAGE_UP) > 0
    width = float(np.hypot(*(ls[:2] - rs[:2])))
    raised = rise > RAISE_MARGIN * width and (head or not on_face)
    return {"raise_status": "ok", "rise": rise, "shoulder_width": width, "above_nose": bool(head), "raised_hand": bool(raised),
            "level": ("head" if head else "shoulder") if raised else ""}


EMPTY = {"face_status": "few_keypoints", "face_size": 0.0, "mouth_dist": np.zeros(21), "mouth_near": 0, "mouth_in_hull": False,
         "face_inside": 0, "hand_over_mouth": False, "self_touch_face": False,
         "point_status": "few_keypoints", "index_extension": 0.0, "index_reach": 0.0, "index_straightness": 0.0,
         "curl": [0.0, 0.0, 0.0], "pointing": False, "direction": None, "target": None,
         "raise_status": "few_keypoints", "rise": 0.0, "shoulder_width": 0.0, "above_nose": False, "raised_hand": False, "level": ""}


def sample_evidence(hand: dict, j: int, ctx) -> dict:
    """Sample j of a hands-layer hand: every rule's measurements and outcome (EMPTY's keys, plus hand, sample, t, valid,
    valid_mask and clasp_iou). `ctx` (gesture_inputs.Context) holds the frame size and the other layers' lookups."""
    s = hand["samples"][j]
    t, kpts = s["t"], np.asarray(s["kpts"], dtype=float)
    valid = valid_keypoints(kpts, ctx.width, ctx.height)
    other = ctx.hand_box(hand["body"], OTHER_SIDE.get(hand["side"]), t) if hand["body"] is not None else None
    row = {**EMPTY, "hand": hand["id"], "sample": j, "t": t, "valid": int(valid.sum()),
           "valid_mask": int(sum(1 << i for i in np.flatnonzero(valid))),
           "clasp_iou": iou(other, s["box"]) if other is not None else 0.0}
    if row["valid"] < MIN_VALID_KPTS:
        return row

    face = hand["face"]
    geometry = ctx.face_geometry(face, t) if face is not None else None
    if face is None:
        row["face_status"] = "no_face"
    elif geometry is None:
        row["face_status"] = "no_landmarks"
    else:
        row.update(face_status="ok", **face_measures(kpts, valid, geometry))

    if row["clasp_iou"] >= CLASPED_IOU:
        row["point_status"] = "clasped"
    elif not valid[POINTING_KPTS].all():
        row["point_status"] = "missing_keypoints"
    else:
        row.update(point_status="ok", **finger_measures(kpts))
        if row["pointing"]:
            row["target"] = ray_target(kpts[INDEX_TIP, :2], row["direction"], ctx.targets(t, hand["shot"], face))

    if hand["body"] is None:
        row["raise_status"] = "no_body"
    elif not valid[WRIST]:
        row["raise_status"] = "no_wrist"
    else:
        on_face = row["hand_over_mouth"] or row["self_touch_face"]
        row.update(raise_measures(kpts[WRIST, :2], hand["side"], ctx.body_at(hand["body"], t), ctx.body_idx,
                                  ctx.body_threshold, ctx.height, on_face))
    return row


# ---------------------------------------------------------------- events

def merge(times, gap: float) -> list[list[float]]:
    """Runs of sorted times whose neighbours are at most gap apart."""
    runs = []
    for t in times:
        if runs and t - runs[-1][-1] <= gap + 1e-6:
            runs[-1].append(t)
        else:
            runs.append([t])
    return runs


def sample_positive(row: dict, kind: str):
    """(score, detail) when the sample is positive for kind, else None. Scores are recomputed from the counts."""
    if not row[kind]:
        return None
    if kind == "hand_over_mouth":
        return row["mouth_near"] / row["valid"], {}
    if kind == "self_touch_face":
        return row["face_inside"] / row["valid"], {}
    if kind == "pointing":
        return 1.0, {"direction": np.asarray(row["direction"], dtype=float), "target": row["target"]}
    return 1.0, {"level": row["level"]}


def hand_events(hand: dict, rows: list[dict]) -> list[dict]:
    """One hand's events from its samples' evidence (rows, in any order): positives of each kind merged across
    MAX_EVENT_GAP_S, runs shorter than MIN_EVENT_S dropped. Kinds in KINDS order, each in time order."""
    positives = {kind: {} for kind in KINDS}
    for row in sorted(rows, key=lambda r: r["t"]):
        if row["hand"] != hand["id"]:
            raise ValueError(f"a row of {row['hand']} among {hand['id']}'s")
        for kind in KINDS:
            found = sample_positive(row, kind)
            if found is not None:
                positives[kind][row["t"]] = found
    events = []
    for kind, samples in positives.items():
        for run in merge(sorted(samples), MAX_EVENT_GAP_S):
            if run[-1] - run[0] < MIN_EVENT_S:
                continue
            event = {"kind": kind, "start": run[0], "end": run[-1], "face": hand["face"], "hand": hand["id"],
                     "score": round(float(np.mean([samples[t][0] for t in run])), 2)}
            if kind == "pointing":
                mean = np.mean([samples[t][1]["direction"] for t in run], axis=0)
                found = [samples[t][1]["target"] for t in run if samples[t][1]["target"] is not None]
                # the most often named target; on a tie the first named (the bake-off's set order was not fixed)
                event["detail"] = {"direction": [round(float(v), 2) for v in mean / np.linalg.norm(mean)],
                                   "target": max(dict.fromkeys(found), key=found.count) if found else None,
                                   "samples": len(run)}
            elif kind == "raised_hand":
                levels = [samples[t][1]["level"] for t in run]
                event["detail"] = {"level": "head" if levels.count("head") * 2 > len(levels) else "shoulder", "samples": len(run)}
            else:
                event["detail"] = {"samples": len(run)}
            events.append(event)
    return events


def layer_events(events: list[dict]) -> list[dict]:
    """Every hand's events as the layer's: sorted by start, then hand id as a string (as the bake-off; ties keep the
    order they came in), numbered gesture:N."""
    ordered = sorted(events, key=lambda e: (e["start"], e["hand"]))
    return [{"id": f"gesture:{i}", **e} for i, e in enumerate(ordered)]


def events_from_evidence(hands: list[dict], rows: list[dict]) -> list[dict]:
    """The layer's events from the evidence of all the hands' samples (the side file's rows, read back)."""
    by_hand: dict[str, list[dict]] = {hand["id"]: [] for hand in hands}
    for row in rows:
        if row["hand"] not in by_hand:
            raise ValueError(f"evidence for {row['hand']}, which is not in the hands layer")
        by_hand[row["hand"]].append(row)
    return layer_events([e for hand in hands for e in hand_events(hand, by_hand[hand["id"]])])
