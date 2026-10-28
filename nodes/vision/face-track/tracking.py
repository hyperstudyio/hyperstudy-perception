"""Face tracks and identity clusters from per-sample detections. Pure functions (numpy, scikit-learn); the node feeds
them py-feat's rows. Ported unchanged from pipeline/bakeoff/faces/faces.py, whose comments record why each threshold
is what it is; detection_rows' size and score filter is `dropped`, which `untracked` also reads to list the faces no
track holds.
"""

from itertools import combinations

import numpy as np

SAME_TRACK_IOU = 0.3          # box overlap between consecutive samples to continue a track
SAME_PERSON_COSINE = 0.65     # anchor clustering; on OOO 0.75 merged panelists more fully but also merged junk (King Kong + blur + logo). Splits are fixable in review, wrong merges mislead agents
FRONTAL_YAW = 0.5             # radians (~30 deg); py-feat head pose and gaze are in radians
ANCHOR_FACE_PX = 60           # a track defines a person only if it has a near-frontal face at least this big
ATTACH_COSINE = 0.6           # profile/small/blurry tracks join a person only within this distance; else Unknown
MIN_FACE_PX = 40              # ignore faces smaller than this (too small to identify or read)
MIN_FACE_SCORE = 0.9          # RetinaFace confidence. On NES - Biking this removed the title lettering and one of two skull shirt prints (30 -> 14 people)
MOUTH_AUS = ("AU25", "AU26")  # lips part, jaw drop: mouth movement for linking voices to faces


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    w, h = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = w * h
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def shot_of(t, shots):
    for shot in shots:
        if shot["start"] <= t < shot["end"]:
            return shot["index"]
    return shots[-1]["index"]


def build_tracks(rows, shots):
    """Greedy IoU linking of detections in consecutive samples, never across a cut."""
    tracks = []
    open_tracks = {}  # shot index -> tracks still extendable
    for t in sorted({r["t"] for r in rows}):
        frame_rows = [r for r in rows if r["t"] == t]
        shot = shot_of(t, shots)
        candidates = open_tracks.setdefault(shot, [])
        used = set()
        for r in sorted(frame_rows, key=lambda r: -r["box"][2] * r["box"][3]):
            best = max((c for c in candidates if id(c) not in used),
                       key=lambda c: iou(c["rows"][-1]["box"], r["box"]), default=None)
            if best is not None and iou(best["rows"][-1]["box"], r["box"]) >= SAME_TRACK_IOU:
                best["rows"].append(r)
                used.add(id(best))
            else:
                track = {"shot": shot, "rows": [r]}
                candidates.append(track)
                tracks.append(track)
                used.add(id(track))
    return tracks


def is_anchor(track):
    """A track good enough to define a person: some near-frontal samples of a reasonably large face."""
    return any(abs(r["yaw"]) <= FRONTAL_YAW and min(r["box"][2], r["box"][3]) >= ANCHOR_FACE_PX for r in track["rows"])


def track_embedding(track):
    frontal = [r["embedding"] for r in track["rows"] if abs(r["yaw"]) <= FRONTAL_YAW]
    e = np.mean(frontal or [r["embedding"] for r in track["rows"]], axis=0)
    return e / np.linalg.norm(e)


def cluster_tracks(tracks, threshold=SAME_PERSON_COSINE):
    """Two passes. People are defined only by anchor tracks (average-linkage clustering of frontal ArcFace embeddings,
    never merging tracks on screen at the same time). Every other track (profile-only, small, blurry) joins its nearest
    person if it is within ATTACH_COSINE and does not co-occur with that person; otherwise it is unknown (-1)."""
    from sklearn.cluster import AgglomerativeClustering

    emb = np.stack([track_embedding(t) for t in tracks])
    times = [{r["t"] for r in t["rows"]} for t in tracks]
    anchors = [i for i, t in enumerate(tracks) if is_anchor(t)]
    labels = [-1] * len(tracks)
    if not anchors:
        return labels

    dist = np.clip(1 - emb[anchors] @ emb[anchors].T, 0, 2)
    for a in range(len(anchors)):
        for b in range(a + 1, len(anchors)):
            if times[anchors[a]] & times[anchors[b]]:
                dist[a, b] = dist[b, a] = 2.0  # on screen together: different people
    anchor_labels = ([0] if len(anchors) == 1 else
                     AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                             distance_threshold=threshold).fit_predict(dist))
    for i, label in zip(anchors, anchor_labels):
        labels[i] = int(label)

    people = sorted(set(anchor_labels))
    centroid = {p: np.mean([emb[i] for i in anchors if labels[i] == p], axis=0) for p in people}
    occupied = {p: set().union(*(times[i] for i in anchors if labels[i] == p)) for p in people}
    for i, track in enumerate(tracks):
        if labels[i] != -1:
            continue
        options = sorted((1 - float(emb[i] @ (centroid[p] / np.linalg.norm(centroid[p]))), p) for p in people
                         if not (times[i] & occupied[p]))
        if options and options[0][0] <= ATTACH_COSINE:
            labels[i] = options[0][1]
    return labels


def name_people(tracks, labels):
    """Person A, B, ... in order of first appearance; -1 stays unnamed (Unknown)."""
    first_seen = {}
    for track, label in zip(tracks, labels):
        if label != -1:
            first_seen[label] = min(first_seen.get(label, 1e9), track["rows"][0]["t"])
    order = sorted(first_seen, key=first_seen.get)
    return order, first_seen, {label: f"Person {chr(65 + i)}" if i < 26 else f"Person {i + 1}" for i, label in enumerate(order)}


def person_pairs(tracks, labels, names):
    """Cosine distance between people's ArcFace centroids, and whether they are ever on screen together (then they
    cannot be one person): the annotator's merge suggestions (pipeline/bakeoff/faces/identity_similarity.py)."""
    members = {}
    for i, label in enumerate(labels):
        if label != -1:
            members.setdefault(names[label], []).append(i)
    centroid = {}
    for person, idx in members.items():
        c = np.mean([track_embedding(tracks[i]) for i in idx], axis=0)
        centroid[person] = c / np.linalg.norm(c)
    times = {p: {round(r["t"], 3) for i in idx for r in tracks[i]["rows"]} for p, idx in members.items()}
    pairs = [{"a": a, "b": b, "distance": round(float(1 - centroid[a] @ centroid[b]), 3), "co_visible": bool(times[a] & times[b])}
             for a, b in combinations(sorted(members), 2)]
    return sorted(pairs, key=lambda p: p["distance"])


def dropped(r):
    """Why tracking leaves out a py-feat row: None (a face it tracks), "none" (no face: py-feat's NaN placeholder row for
    a frame without one), or "size", "score" or "size+score" (a face smaller than MIN_FACE_PX, scored below
    MIN_FACE_SCORE, or both)."""
    if np.isnan(r["FaceRectX"]):
        return "none"
    why = [name for name, out in (("size", min(r["FaceRectWidth"], r["FaceRectHeight"]) < MIN_FACE_PX),
                                  ("score", r["FaceScore"] < MIN_FACE_SCORE)) if out]
    return "+".join(why) or None


def untracked(fex, fps):
    """The faces py-feat detected that tracking leaves out (see dropped): (row label, frame, t, why), in py-feat's
    order. build_tracks puts every row detection_rows keeps in a track, so these are all the faces in no track."""
    out = []
    for index, r in fex.iterrows():
        why = dropped(r)
        if why not in (None, "none"):
            out.append((index, int(r["frame"]), round(float(r["frame"]) / fps, 3), why))
    return out


def detection_rows(fex, fps):
    """py-feat rows -> detections kept for tracking (size and score filtered: dropped), with ArcFace embeddings."""
    emotion_cols = list(fex.emotion_columns)
    rows = []
    for index, r in fex.iterrows():
        if dropped(r) is not None:
            continue
        rows.append({
            "index": index, "t": round(float(r["frame"]) / fps, 3), "frame": int(r["frame"]),
            "box": [float(r["FaceRectX"]), float(r["FaceRectY"]), float(r["FaceRectWidth"]), float(r["FaceRectHeight"])],
            "score": float(r["FaceScore"]),
            "embedding": np.array([r[f"Identity_{i + 1}"] for i in range(512)], dtype=np.float32),
            "yaw": float(r["Yaw"]), "pitch": float(r["Pitch"]),
            "gaze_yaw": float(r["gaze_yaw"]), "gaze_pitch": float(r["gaze_pitch"]),
            "emotions": {c: float(r[c]) for c in emotion_cols},
            "mouth": float(np.mean([r[au] for au in MOUTH_AUS])),
        })
    return rows
