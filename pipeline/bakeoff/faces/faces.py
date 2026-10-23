"""Face stage of the bake-off: who is on screen, across the whole film.

py-feat Detectorv2 (RetinaFace + multitask model + ArcFace) runs on frames
sampled from the video. Faces are then
  1. linked into tracks within each shot (a cut always ends a track),
  2. clustered across shots into people by ArcFace embedding, with the rule
     that two faces on screen at the same moment are never the same person,
  3. summarised per track (gaze, head pose, dominant emotion) and given one
     representative crop per person.

Licence note: py-feat's code is MIT, but the Detectorv2 multitask weights are
non-commercial and InsightFace-derived ArcFace weights are research-only.
The output records this so a track built here is marked academic-only.

Usage: uv run python faces.py ../runs/<name> [--device mps|cpu] [--fps 4] [--batch-size 32]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

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
    return any(abs(r["yaw"]) <= FRONTAL_YAW and min(r["box"][2], r["box"][3]) >= ANCHOR_FACE_PX
               for r in track["rows"])


def track_embedding(track):
    frontal = [r["embedding"] for r in track["rows"] if abs(r["yaw"]) <= FRONTAL_YAW]
    e = np.mean(frontal or [r["embedding"] for r in track["rows"]], axis=0)
    return e / np.linalg.norm(e)


def cluster_tracks(tracks, threshold=SAME_PERSON_COSINE):
    """Two passes. People are defined only by anchor tracks (average-linkage clustering of
    frontal ArcFace embeddings, never merging tracks on screen at the same time). Every other
    track (profile-only, small, blurry) joins its nearest person if it is within ATTACH_COSINE
    and does not co-occur with that person; otherwise it is labelled unknown (-1)."""
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


def detection_rows(fex, fps):
    """py-feat rows -> detections kept for tracking (size and score filtered), with ArcFace embeddings."""
    emotion_cols = list(fex.emotion_columns)
    rows = []
    for index, r in fex.iterrows():
        if (np.isnan(r["FaceRectX"]) or min(r["FaceRectWidth"], r["FaceRectHeight"]) < MIN_FACE_PX
                or r["FaceScore"] < MIN_FACE_SCORE):
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


def load_tracks(run_dir):
    """Rebuild faces.json's tracks from the cached detections, so track i here is track i in faces.json.

    Returns (fex, faces, tracks); each track's rows carry "index", the row of the py-feat Fex frame they came from.
    Stops if the rebuild does not match faces.json (the cache or tracking changed since faces.py ran).
    """
    import pickle

    faces = json.loads((run_dir / "faces.json").read_text())
    extract = json.loads((run_dir / "extract.json").read_text())
    cache = run_dir / f"faces_detections_{faces['sample_fps']:g}fps.pkl"
    fex, _ = pickle.loads(cache.read_bytes())  # our own cache, written by main() in this run dir
    tracks = build_tracks(detection_rows(fex, extract["video"]["fps"]), extract["shots"])
    if len(tracks) != len(faces["tracks"]) or any(
            built["rows"][0]["t"] != saved["start"] or built["shot"] != saved["shot"]
            for built, saved in zip(tracks, faces["tracks"])):
        raise SystemExit("rebuilt tracks do not match faces.json; re-run faces.py")
    return fex, faces, tracks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=SAME_PERSON_COSINE)
    parser.add_argument("--redetect", action="store_true", help="ignore cached detections")
    parser.add_argument("--batch-size", type=int, default=32)  # 8-32 tie on M5 Max MPS (decode-bound), 64 was 2x slower; re-measure on CUDA
    args = parser.parse_args()

    from feat.detector_v2 import Detectorv2

    extract = json.loads((args.run_dir / "extract.json").read_text())
    video, shots, fps = Path(extract["video"]["source"]), extract["shots"], extract["video"]["fps"]
    video = video if video.is_absolute() else (args.run_dir.parent.parent / video)
    skip = max(1, round(fps / args.fps))

    import pickle

    cache = args.run_dir / f"faces_detections_{args.fps:g}fps.pkl"
    if cache.exists() and not args.redetect:
        fex, info = pickle.loads(cache.read_bytes())
    else:
        detector = Detectorv2(device=args.device)
        fex = detector.detect(str(video), data_type="video", skip_frames=skip, batch_size=args.batch_size)
        info = dict(detector.info)
        cache.write_bytes(pickle.dumps((fex, info)))

    emotion_cols = list(fex.emotion_columns)
    rows = detection_rows(fex, fps)

    tracks = build_tracks(rows, shots)
    labels = cluster_tracks(tracks, args.threshold) if tracks else []

    # Label people in order of first appearance: Person A, B, ... Tracks no person claimed stay "Unknown".
    first_seen = {}
    for track, label in zip(tracks, labels):
        if label != -1:
            first_seen[label] = min(first_seen.get(label, 1e9), track["rows"][0]["t"])
    order = sorted(first_seen, key=first_seen.get)
    names = {label: f"Person {chr(65 + i)}" if i < 26 else f"Person {i + 1}" for i, label in enumerate(order)}
    names[-1] = "Unknown"

    crops_dir = args.run_dir / "faces"
    crops_dir.mkdir(exist_ok=True)
    people = []
    for label in order:
        members = [t for t, l in zip(tracks, labels) if l == label]
        best = max((r for t in members for r in t["rows"]),
                   key=lambda r: r["box"][2] * r["box"][3] * np.cos(min(abs(r["yaw"]), 1.4)))
        x, y, w, h = best["box"]
        pad = 0.25
        frame_path = crops_dir / "_frame.jpg"
        import subprocess
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{best['t']:.3f}", "-i", str(video),
                        "-frames:v", "1", str(frame_path)], check=True)
        image = Image.open(frame_path)
        crop = image.crop((max(0, x - pad * w), max(0, y - pad * h), x + (1 + pad) * w, y + (1 + pad) * h))
        crop.thumbnail((256, 256))
        crop_name = f"{names[label].replace(' ', '_')}.jpg"
        crop.save(crops_dir / crop_name, quality=85)
        frame_path.unlink()
        people.append({"label": names[label], "crop": f"faces/{crop_name}", "first_seen": round(first_seen[label], 3),
                       "tracks": len(members), "shots": sorted({t["shot"] for t in members}),
                       "seconds_on_screen": round(sum(len(t["rows"]) for t in members) / args.fps, 1)})

    track_out = []
    for track, label in zip(tracks, labels):
        rows_ = track["rows"]
        mean_emotions = {c: float(np.mean([r["emotions"][c] for r in rows_])) for c in emotion_cols}
        track_out.append({
            "person": names[label], "shot": track["shot"], "start": rows_[0]["t"], "end": rows_[-1]["t"],
            "samples": len(rows_), "mean_box": [round(float(np.mean([r["box"][i] for r in rows_])), 1) for i in range(4)],
            "head_yaw": round(float(np.mean([r["yaw"] for r in rows_])), 3),  # radians
            "gaze_yaw": round(float(np.mean([r["gaze_yaw"] for r in rows_])), 3),
            "gaze_pitch": round(float(np.mean([r["gaze_pitch"] for r in rows_])), 3),
            "dominant_emotion": max(mean_emotions, key=mean_emotions.get),
            "emotions": {k: round(v, 3) for k, v in mean_emotions.items()},
            "series": [{"t": r["t"], "box": [round(v, 1) for v in r["box"]], "mouth": round(r["mouth"], 3),
                        "gaze_yaw": round(r["gaze_yaw"], 3), "gaze_pitch": round(r["gaze_pitch"], 3)} for r in rows_],
        })

    out = {"model": info, "threshold": args.threshold, "frontal_yaw": FRONTAL_YAW, "licence": "py-feat Detectorv2 weights: non-commercial; ArcFace: research-only",
           "sample_fps": args.fps, "detections": len(rows), "people": people, "tracks": track_out}
    (args.run_dir / "faces.json").write_text(json.dumps(out, indent=2))
    print(f"{len(rows)} face detections -> {len(tracks)} tracks -> {len(people)} people")
    for p in people:
        print(f"  {p['label']}: first {p['first_seen']}s, {p['tracks']} tracks, shots {p['shots']}, "
              f"{p['seconds_on_screen']}s on screen")


if __name__ == "__main__":
    main()
