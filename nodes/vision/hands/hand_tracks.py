"""Hands, from the bodies layer and one frame's detections to tracks linked to body and face tracks: the bake-off's
hands stage (hands.py --model rtmpose_hand), reading the pipeline's bodies layer. numpy only, so it is tested without
the models (the frame's models are passed in: anything with `detect(frame)` and `keypoints(frame, box)`).

Hands are a gated second stage on top of the bodies, after HaMeR's design:
  1. Hand proposals come from each body's whole-body hand keypoints: a hand with at least MIN_HAND_KPTS keypoints
     above the bodies layer's score threshold and inside the frame gives a square box around those keypoints and the
     body's wrist. These hands inherit the body track, its face track and the side (COCO-WholeBody's left_hand is the
     subject's left hand). box_source "wrist".
  2. RTMDet-nano hand on the full frame finds hands whose body was not posed. Boxes smaller than MIN_HAND_PX, larger
     than MAX_DETECTOR_BOX or on a wrist hand are dropped first; the rest are matched to the wrists of every body in
     the frame by Hungarian assignment (cost = the wrist's distance outside the box / box side, gated by WRIST_GATE).
     A box matched to a wrist that already gave a hand is that hand again and is dropped; one matched to a wrist with
     no hand inherits that body and side; an unmatched one is kept with no body and no side. box_source "detector".
     (The bake-off matched before dropping, so a dropped box could take a real box's wrist: see frame_hands.)
  3. RTMPose-hand runs only on hands whose box side is at least MIN_HAND_PX.
  4. Hands are tracked within the shot: linked hands by (body track, side, box source), unlinked ones by box IoU; a gap
     longer than MAX_TRACK_GAP_S ends a track, and so does a cut.

Boxes are [x, y, w, h] in pixels; keypoints [x, y, score], the score RTMPose-hand's own (a SimCC peak, 0-1 here, not a
probability), so the layer has no score threshold.
"""

from __future__ import annotations

import numpy as np

from scipy_lsap.rectangular_lsap import linear_sum_assignment

MIN_HAND_KPTS = 4        # HaMeR's demo builds a hand box only from more than 3 valid hand keypoints
HAND_BOX_PAD = 1.2       # square box side = this x the larger extent of the valid hand keypoints + wrist (tips sit on the extent edge)
# Second-stage gate on the hand box side. Contact sheets of 24 random OOO RTMW hand boxes per size bin: at 24-32 px
# the hands are blurred blobs (wide shots, hands in pockets or at the sides) with no fingers separable by eye; at
# 32-48 px finger separation is visible on some (clasped hands at a mouth, hands on a chest). 32 px is also the top
# of the bake-off research note's 24-32 px "no model resolves fingers below this" range.
MIN_HAND_PX = 32
WRIST_GATE = 1.0         # a detector box matches a wrist at most one box side outside it ("about one hand-length")
DEDUP_IOU = 0.3          # a detector box overlapping an accepted hand box this much is that hand again
TRACK_IOU = 0.3          # unlinked hands: box overlap with a track's last sample to continue it (as faces and bodies)
MAX_TRACK_GAP_S = 0.5    # a hand unseen for longer than this (4 samples at 8 fps) starts a new track, as bodies
# RTMDet-nano hand also returns frame-sized boxes (1900 px wide) on black frames and bokeh shots of nes-biking; a
# detector box whose longer side exceeds this fraction of the frame height is not a hand.
MAX_DETECTOR_BOX = 0.5

# COCO-WholeBody's 21 hand keypoints and RTMPose Hand5's share this order (wrist, then thumb, index, middle, ring,
# pinky from base to tip)
KEYPOINTS = ["wrist", "thumb1", "thumb2", "thumb3", "thumb4", "forefinger1", "forefinger2", "forefinger3", "forefinger4",
             "middle_finger1", "middle_finger2", "middle_finger3", "middle_finger4", "ring_finger1", "ring_finger2",
             "ring_finger3", "ring_finger4", "pinky_finger1", "pinky_finger2", "pinky_finger3", "pinky_finger4"]
SKELETON = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
SIDES = ("left", "right")


# ---------------------------------------------------------------- geometry

def iou(a, b) -> float:
    """IoU of two [x, y, w, h] boxes."""
    w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def square_box(points, pad: float) -> list[float]:
    """Square [x, y, w, h] around points, side = pad x the larger extent."""
    lo, hi = points.min(0), points.max(0)
    centre, side = (lo + hi) / 2, max(float((hi - lo).max()) * pad, 1.0)
    return [float(centre[0] - side / 2), float(centre[1] - side / 2), side, side]


def outside_distance(point, box) -> float:
    """Distance from a point to a [x, y, w, h] box (0 inside)."""
    dx = max(box[0] - point[0], 0.0, point[0] - box[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[1] - box[3])
    return float(np.hypot(dx, dy))


# ---------------------------------------------------------------- body proposals

def body_indices(names: list[str]) -> dict:
    """Where the bodies layer keeps each side's wrist and 21 hand keypoints (COCO-WholeBody: wrists 9 and 10, hands
    91-111 and 112-132), found by name."""
    try:
        return {side: {"wrist": names.index(f"{side}_wrist"),
                       "hand": list(range(names.index(f"{side}_hand_root"), names.index(f"{side}_hand_root") + 21))}
                for side in SIDES}
    except ValueError as e:
        raise ValueError(f"the bodies layer has no hand keypoints: {e}") from None


def visible(kpts, threshold: float, width: int, height: int):
    """Keypoints above the score threshold AND inside the frame. RTMW's SimCC heads put off-frame hands (arms leaving
    the bottom of a medium shot) just outside the image with scores well above the threshold."""
    return (kpts[:, 2] >= threshold) & (kpts[:, 0] >= 0) & (kpts[:, 0] < width) & (kpts[:, 1] >= 0) & (kpts[:, 1] < height)


def bodies_by_time(tracks: list[dict]) -> dict[float, list[dict]]:
    """{t: [body]} from the bodies layer's tracks, in track order: body = {person {body, face}, kpts (Nx3)}."""
    frames: dict[float, list[dict]] = {}
    for track in tracks:
        for s in track["samples"]:
            frames.setdefault(round(s["t"], 3), []).append(
                {"person": {"body": track["id"], "face": track.get("face")}, "kpts": np.array(s["kpts"], dtype=float)})
    return frames


def propose(body: dict, idx: dict, threshold: float, width: int, height: int) -> dict:
    """The body with its visible wrists {side: xy or None} and hand proposals {side: box}, in a width x height frame."""
    wrists, proposals = {}, {}
    kpts = body["kpts"]
    for side in SIDES:
        wrist = kpts[idx[side]["wrist"]]
        wrists[side] = wrist[:2] if visible(wrist[None], threshold, width, height)[0] else None
        hand = kpts[idx[side]["hand"]]
        valid = visible(hand, threshold, width, height)
        if valid.sum() < MIN_HAND_KPTS:
            continue
        points = hand[valid, :2]
        if wrists[side] is not None:
            points = np.vstack([points, wrists[side]])
        proposals[side] = square_box(points, HAND_BOX_PAD)
    return {**body, "wrists": wrists, "proposals": proposals}


# ---------------------------------------------------------------- per-frame assembly

def match_detections(detections, bodies) -> dict[int, tuple[int, str]]:
    """Hungarian match of detector boxes to body wrists: {detection index: (body index, side)}."""
    wrists = [(b, side, xy) for b, body in enumerate(bodies) for side, xy in body["wrists"].items() if xy is not None]
    if not detections or not wrists:
        return {}
    cost = np.array([[outside_distance(xy, box) / box[2] for _, _, xy in wrists] for box, _ in detections])
    rows, cols = linear_sum_assignment(cost)
    return {int(r): wrists[c][:2] for r, c in zip(rows, cols) if cost[r, c] <= WRIST_GATE}


def frame_hands(frame, bodies: list[dict], model, stats: dict) -> list[dict]:
    """One frame's hands, [{box, side, box_source, person, kpts}]: `bodies` from propose(), `model` the detector and
    the keypoint model (models.Hands). Wrist hands first, in body order, then detector hands, highest score first."""
    hands, done = [], set()   # done: (body index, side) already holding a hand
    for b, body in enumerate(bodies):
        for side, box in body["proposals"].items():
            if box[2] < MIN_HAND_PX:
                stats["wrist_below_gate"] += 1
                continue
            hands.append({"box": box, "side": side, "box_source": "wrist", "person": body["person"],
                          "kpts": model.keypoints(frame, box)})
            done.add((b, side))

    # Boxes that cannot be hands are dropped before matching: matched first (as the bake-off did), a dropped box could
    # take a wrist from a real hand box, which then lost its body, side and face (Hungarian assignment gives a wrist to
    # one box only, and of two boxes holding it, both at cost 0, to the first: the higher score)
    candidates = []
    for box, score in model.detect(frame):
        if min(box[2], box[3]) < MIN_HAND_PX:
            stats["detector_below_gate"] += 1
        elif max(box[2], box[3]) > MAX_DETECTOR_BOX * frame.shape[0]:
            stats["detector_too_large"] += 1
        # RTMDet's baked-in NMS leaves overlapping boxes on one hand (IoU ~0.5 seen on OOO), so a box overlapping a
        # wrist hand is that hand again
        elif any(iou(box, h["box"]) >= DEDUP_IOU for h in hands):
            stats["detector_duplicate"] += 1
        else:
            candidates.append((box, score))
    matches = match_detections(candidates, bodies)
    for d, (box, _) in enumerate(candidates):   # highest score first
        # ... and so is one overlapping a detector hand accepted before it
        if any(iou(box, h["box"]) >= DEDUP_IOU for h in hands):
            stats["detector_duplicate"] += 1
            continue
        if d in matches:
            b, side = matches[d]
            if (b, side) in done:
                stats["detector_duplicate"] += 1
                continue
            person = bodies[b]["person"]
            done.add((b, side))
            stats["detector_linked"] += 1
        else:
            side, person = None, None
            stats["detector_unlinked"] += 1
        hands.append({"box": box, "side": side, "box_source": "detector", "person": person, "kpts": model.keypoints(frame, box)})
    return hands


# ---------------------------------------------------------------- tracks

def track_hands(frames: list[tuple[float, str, list[dict]]]) -> list[dict]:
    """frames: [(t, shot, [hand])], in time order. Linked hands continue by (body track, side, source); unlinked ones by
    IoU with an open unlinked track of the same side. Tracks in the order they began."""
    tracks, open_keyed, open_free = [], {}, {}
    for t, shot, hands in frames:
        free_candidates = [tr for tr in open_free.get(shot, []) if t - tr["samples"][-1]["t"] <= MAX_TRACK_GAP_S + 1e-6]
        used = set()
        for hand in sorted(hands, key=lambda h: -h["box"][2]):
            if hand["person"] is not None:
                key = (shot, hand["person"]["body"], hand["side"], hand["box_source"])
                track = open_keyed.get(key)
                if track is None or t - track["samples"][-1]["t"] > MAX_TRACK_GAP_S + 1e-6:
                    track = {"shot": shot, "side": hand["side"], "box_source": hand["box_source"], "person": hand["person"], "samples": []}
                    open_keyed[key] = track
                    tracks.append(track)
            else:
                pool = [tr for tr in free_candidates if id(tr) not in used and tr["side"] == hand["side"]]
                track = max(pool, key=lambda tr: iou(tr["samples"][-1]["box"], hand["box"]), default=None)
                if track is None or iou(track["samples"][-1]["box"], hand["box"]) < TRACK_IOU:
                    track = {"shot": shot, "side": hand["side"], "box_source": "detector", "person": None, "samples": []}
                    open_free.setdefault(shot, []).append(track)
                    tracks.append(track)
                used.add(id(track))
            track["samples"].append({"t": t, "box": hand["box"], "kpts": hand["kpts"]})
    return tracks


def layer_hands(tracks: list[dict]) -> list[dict]:
    """The hands layer's hands: ids hand:N in shot then start order (ties in the order the tracks began), coordinates
    rounded as the bake-off's."""
    ordered = sorted(tracks, key=lambda tr: (int(tr["shot"].split(":")[1]), tr["samples"][0]["t"]))
    return [{
        "id": f"hand:{i}", "shot": tr["shot"], "side": tr["side"],
        "body": tr["person"]["body"] if tr["person"] else None, "face": tr["person"]["face"] if tr["person"] else None,
        "box_source": tr["box_source"], "start": tr["samples"][0]["t"], "end": tr["samples"][-1]["t"],
        "samples": [{"t": s["t"], "box": [round(float(v), 1) for v in s["box"]],
                     "kpts": [[round(float(x), 1), round(float(y), 1), round(float(c), 2)] for x, y, c in s["kpts"]]}
                    for s in tr["samples"]],
    } for i, tr in enumerate(ordered)]
