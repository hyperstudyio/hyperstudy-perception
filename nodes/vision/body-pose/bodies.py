"""Bodies, from one frame's detections to tracks linked to face tracks: the bake-off's pose stage (pose.py, rtmw),
reading the pipeline's faces layer. numpy only, so it is tested without the models.

Per frame: a second detection of the same person (same head and shoulder joints) is dropped (drop_duplicate_bodies);
each body is linked to a face track on screen then: the face box (interpolated to the frame time) must sit inside the
top part of the body box AND the body's own nose/eye keypoints must fall inside that face box (link_faces, one-to-one
per frame). Bodies are then tracked within shots (track_bodies): face-linked bodies first, onto their face's track,
the rest by box IoU; a cut always ends a track, and a face track is never linked to two body tracks at once.

Boxes are [x, y, w, h] in pixels; keypoints [x, y, score], the score RTMW's own (SimCC peaks, not probabilities).
"""

from __future__ import annotations

import numpy as np

DUPLICATE_HEAD_DIST = 0.1   # two bodies are one person if their head/shoulder joints lie this close (x box scale)
HEAD_JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder")
TRACK_IOU = 0.3             # box overlap with a track's last sample to continue it (as the faces node)
MAX_TRACK_GAP_S = 0.5       # a body unseen for longer than this (4 samples at 8 fps) starts a new track
MAX_FACE_GAP_S = 0.5        # face boxes are interpolated only between face samples at most this far apart
FACE_INSIDE_BODY = 0.7      # fraction of the face box's area that must lie inside the body box
FACE_CENTRE_TOP = 0.5       # the face centre must lie in this top fraction of the body box
LINK_MIN_KPTS = 2           # of nose, left eye, right eye that must fall inside the face box
FACE_BOX_MARGIN = 0.15      # RetinaFace boxes are tight; eyes on profile faces sit on the edge: grown by this per side
NOSE_EYES = (0, 1, 2)       # nose, left_eye, right_eye in COCO-WholeBody


def iou(a, b) -> float:
    """IoU of two [x, y, w, h] boxes."""
    w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def face_boxes_at(t: float, shot: str, face_tracks: list[dict]) -> dict[str, list[float]]:
    """{face track id: [x, y, w, h]} for the faces layer's tracks of this shot on screen at t (linear interpolation
    between samples at most MAX_FACE_GAP_S apart, no extrapolation)."""
    boxes = {}
    for track in face_tracks:
        if track["shot"] != shot or not track["start"] - 1e-6 <= t <= track["end"] + 1e-6:
            continue
        samples = track["samples"]
        times = [s["t"] for s in samples]
        j = int(np.searchsorted(times, t))
        if j < len(times) and abs(times[j] - t) < 1e-3:
            boxes[track["id"]] = samples[j]["box"]
        elif 0 < j < len(times) and times[j] - times[j - 1] <= MAX_FACE_GAP_S:
            w = (t - times[j - 1]) / (times[j] - times[j - 1])
            boxes[track["id"]] = [(1 - w) * a + w * b for a, b in zip(samples[j - 1]["box"], samples[j]["box"])]
    return boxes


def face_in_body(face, body) -> bool:
    """The face box lies inside the top part of the body box."""
    fx, fy, fw, fh = face
    bx, by, bw, bh = body
    inside = max(0.0, min(fx + fw, bx + bw) - max(fx, bx)) * max(0.0, min(fy + fh, by + bh) - max(fy, by))
    return inside >= FACE_INSIDE_BODY * fw * fh and by <= fy + fh / 2 <= by + FACE_CENTRE_TOP * bh


def keypoints_in_face(kpts, face, score_threshold: float) -> int:
    fx, fy, fw, fh = face
    mx, my = FACE_BOX_MARGIN * fw, FACE_BOX_MARGIN * fh
    return sum(1 for i in NOSE_EYES if kpts[i][2] >= score_threshold
               and fx - mx <= kpts[i][0] <= fx + fw + mx and fy - my <= kpts[i][1] <= fy + fh + my)


def drop_duplicate_bodies(bodies: list[dict], head_idx: list[int], score_threshold: float) -> list[dict]:
    """Remove second detections of the same person: of two bodies whose head and shoulder joints (visible in both)
    lie within DUPLICATE_HEAD_DIST x the smaller box's sqrt(area), keep the higher detector score.

    YOLOX's ONNX ends in its own NMS, so it can return two boxes on one person (on OOO at 69.0 s, one fully inside the
    other, IoU 0.63). Box IoU cannot tell duplicates from real overlapping people, but the joints can: same-frame pairs
    with IoU >= 0.2 had a median head/shoulder distance below 0.05 or above 0.16 of box size (bake-off, 1004 pairs)."""
    order = sorted(range(len(bodies)), key=lambda i: -bodies[i]["score"])
    kept = []
    for i in order:
        a = bodies[i]["kpts"][head_idx]
        duplicate = False
        for j in kept:
            b = bodies[j]["kpts"][head_idx]
            both = (a[:, 2] >= score_threshold) & (b[:, 2] >= score_threshold)
            if both.sum() < 2:
                continue
            scale = min(np.sqrt(bodies[i]["box"][2] * bodies[i]["box"][3]), np.sqrt(bodies[j]["box"][2] * bodies[j]["box"][3]))
            if np.median(np.linalg.norm(a[both, :2] - b[both, :2], axis=1)) < DUPLICATE_HEAD_DIST * scale:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    return [bodies[i] for i in sorted(kept)]


def link_faces(bodies: list[dict], faces: dict[str, list[float]], score_threshold: float, stats: dict) -> None:
    """One-to-one body -> face track links for one frame (sets body["face"]). A face passing the box test whose
    keypoints disagree is not linked. When two bodies pass for one face, the one with more nose/eye keypoints in the
    face wins, then the higher detector score; the other stays unlinked."""
    candidates = []
    for b, body in enumerate(bodies):
        for face_id, face in faces.items():
            if not face_in_body(face, body["box"]):
                continue
            stats["box_candidates"] += 1
            hits = keypoints_in_face(body["kpts"], face, score_threshold)
            if hits >= LINK_MIN_KPTS:
                candidates.append((hits, body["score"], b, face_id))
            else:
                stats["keypoint_check_failed"] += 1
    used_bodies, used_faces = set(), set()
    for hits, _, b, face_id in sorted(candidates, reverse=True):
        if b in used_bodies or face_id in used_faces:
            continue
        bodies[b]["face"] = face_id
        used_bodies.add(b)
        used_faces.add(face_id)
    stats["linked_samples"] += len(used_bodies)
    stats["unlinked_samples"] += len(bodies) - len(used_bodies)


def track_bodies(frames: list[tuple[float, list[dict]]], shots_of_frames: list[str]) -> list[dict]:
    """Greedy within-shot tracking. A body joins the open track holding its face link; else the best-IoU track whose
    face link does not conflict. Face-linked bodies are placed first, then the rest largest first: largest-first alone
    let an unlinked duplicate take the track of the face its twin was linked to."""
    tracks, open_tracks = [], {}
    face_ends: dict[str, float] = {}   # face track -> the last time a body track linked to it was seen
    for (t, bodies), shot in zip(frames, shots_of_frames):
        candidates = [tr for tr in open_tracks.setdefault(shot, []) if t - tr["samples"][-1]["t"] <= MAX_TRACK_GAP_S + 1e-6]
        used = set()
        for body in sorted(bodies, key=lambda b: (b.get("face") is None, -b["box"][2] * b["box"][3])):
            face = body.get("face")
            free = [tr for tr in candidates if id(tr) not in used]
            same_face = [tr for tr in free if face is not None and tr["face"] == face]
            # an unlinked track takes a face only if it began after every other track of that face ended: linking it
            # then would make two tracks of one face at once (the face moved to an overlapping body while its own
            # track went unseen for longer than MAX_TRACK_GAP_S)
            compatible = [tr for tr in free if face is None or tr["face"] == face
                          or (tr["face"] is None and tr["samples"][0]["t"] > face_ends.get(face, -1.0))]
            best = same_face[0] if same_face else max(compatible, key=lambda tr: iou(tr["samples"][-1]["box"], body["box"]), default=None)
            if best is None or (not same_face and iou(best["samples"][-1]["box"], body["box"]) < TRACK_IOU):
                best = {"shot": shot, "face": None, "samples": []}
                open_tracks[shot].append(best)
                tracks.append(best)
            if face is not None:
                best["face"] = face
            best["samples"].append({"t": t, **body})
            if best["face"] is not None:
                face_ends[best["face"]] = t
            used.add(id(best))
    one_body_per_face(tracks)
    return tracks


def one_body_per_face(tracks: list[dict]) -> None:
    """Invariant: tracks linked to the same face track never overlap in time."""
    spans = sorted((tr["face"], tr["samples"][0]["t"], tr["samples"][-1]["t"]) for tr in tracks if tr["face"] is not None)
    for (face_a, _, end_a), (face_b, start_b, _) in zip(spans, spans[1:]):
        if face_a == face_b and start_b <= end_a:
            raise AssertionError(f"{face_a} linked to two bodies at once ({start_b} <= {end_a})")


def layer_tracks(tracks: list[dict]) -> list[dict]:
    """The bodies layer's tracks: ids body:N in shot then time order, coordinates rounded as the bake-off's."""
    ordered = sorted(tracks, key=lambda tr: (int(tr["shot"].split(":")[1]), tr["samples"][0]["t"]))
    return [{
        "id": f"body:{i}", "shot": tr["shot"], "face": tr["face"],
        "link": "face-in-box+keypoints" if tr["face"] is not None else None,
        "start": tr["samples"][0]["t"], "end": tr["samples"][-1]["t"],
        "samples": [{"t": s["t"], "box": [round(float(v), 1) for v in s["box"]],
                     "kpts": [[round(float(x), 1), round(float(y), 1), round(float(c), 2)] for x, y, c in s["kpts"]]}
                    for s in tr["samples"]],
    } for i, tr in enumerate(ordered)]
