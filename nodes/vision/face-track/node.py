"""vision.face-track: who is on screen, across the whole video, as the bundle's faces layer.

py-feat Detectorv2 (RetinaFace + multitask model + ArcFace) on frames sampled at `fps`; faces linked into tracks
within each shot and clustered into people (tracking.py). Writes one crop per person (thumbnails.py: shortlisted by size and
frontality, chosen on exposure, colour cast and sharpness) and one per track (its largest face), referenced from the
layer as assets/crops/...

Licence: py-feat's code is MIT, but the Detectorv2 multitask weights are non-commercial and InsightFace-derived
ArcFace weights are research-only, so this layer is academic-only.
"""

import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from perception_node import Node, NodeError
from frames import crop_pixels, longest_side, save_crop, width
from loader import decode_workers
from thumbnails import QUALITY_PX, choose, shortlist
from tracking import FRONTAL_YAW, build_tracks, cluster_tracks, detection_rows, name_people, person_pairs

PERSON_CROP_PAD, PERSON_CROP_PX = 0.25, 256   # padding as a fraction of the box; longest side of the saved crop
TRACK_CROP_PAD, TRACK_CROP_WIDTH = 0.4, 320


def run_jobs(jobs) -> list:
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(lambda job: job(), jobs))


with Node() as node:
    from feat.detector_v2 import Detectorv2

    c = node.config
    video, media = node.input("video"), node.input_json("media")
    shots = [{"index": s["index"], "start": s["start"], "end": s["end"]} for s in node.input_json("shots")["shots"]]
    fps = media["fps"]
    device = {"mps": "mps", "cuda": "cuda"}.get(node.accelerator, "cpu")

    try:
        workers = decode_workers(c["num_workers"], multiprocessing.get_start_method(), os.cpu_count() or 1)
    except ValueError as e:
        raise NodeError("input", str(e)) from None

    t0 = time.time()
    detector = Detectorv2(device=device)
    fex = detector.detect(str(video), data_type="video", skip_frames=max(1, round(fps / c["fps"])), batch_size=c["batch_size"],
                          num_workers=workers, pin_memory=device == "cuda")
    detect_s = round(time.time() - t0, 1)
    node.model("py-feat Detectorv2 (RetinaFace + multitask)", licence_mode="academic-only", source="hf://py-feat",
               licence="py-feat code MIT; v2 weights non-commercial")
    node.model("ArcFace (InsightFace, via py-feat)", licence_mode="academic-only", licence="research-only weights")

    rows = detection_rows(fex, fps)
    tracks = build_tracks(rows, shots)
    labels = cluster_tracks(tracks, c["cluster_threshold"]) if tracks else []
    order, first_seen, names = name_people(tracks, labels)
    emotion_cols = list(fex.emotion_columns)

    assets = node.path("crops", "assets")
    crops = assets / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    W, H = media["width"], media["height"]
    t0 = time.time()
    members = {label: [t for t, l in zip(tracks, labels) if l == label] for label in order}
    candidates = {label: shortlist(members[label]) for label in order}
    flat = [(label, r) for label in order for r in candidates[label]]
    raw = run_jobs([partial(crop_pixels, video, r["t"], r["box"], 0.0, W, H, QUALITY_PX) for _, r in flat])   # the face box only
    images = {label: [] for label in order}
    for (label, _), pixels in zip(flat, raw):
        images[label].append(pixels)
    thumbnails_s = round(time.time() - t0, 1)

    jobs, people, thumbnail_measures = [], [], {}
    for label in order:
        index, measures = choose(candidates[label], images[label])
        best = candidates[label][index]
        thumbnail_measures[names[label]] = {"t": best["t"], **measures[index]}
        name = f"person_{names[label].replace(' ', '_')}.jpg"
        jobs.append(partial(save_crop, video, best["t"], best["box"], PERSON_CROP_PAD, W, H, crops / name, longest_side(PERSON_CROP_PX)))
        people.append({"id": f"person:{names[label]}", "first_seen": round(first_seen[label], 3),
                       "seconds_on_screen": round(sum(len(t["rows"]) for t in members[label]) / c["fps"], 1),
                       "crop": f"assets/crops/{name}", "shots": [f"shot:{s}" for s in sorted({t["shot"] for t in members[label]})]})

    out_tracks = []
    for i, (track, label) in enumerate(zip(tracks, labels)):
        rs = track["rows"]
        biggest = max(rs, key=lambda r: r["box"][2] * r["box"][3])
        jobs.append(partial(save_crop, video, biggest["t"], biggest["box"], TRACK_CROP_PAD, W, H, crops / f"face_{i}.jpg", width(TRACK_CROP_WIDTH)))
        mean_emotions = {col: float(np.mean([r["emotions"][col] for r in rs])) for col in emotion_cols}
        out_tracks.append({
            "id": f"face:{i}", "person": f"person:{names[label]}" if label != -1 else None, "shot": f"shot:{track['shot']}",
            "start": rs[0]["t"], "end": rs[-1]["t"], "linked_body": None, "crop": f"assets/crops/face_{i}.jpg",
            "samples": [{"t": r["t"], "box": [round(v, 1) for v in r["box"]], "mouth": round(r["mouth"], 3),
                         "gaze": [round(r["gaze_yaw"], 3), round(r["gaze_pitch"], 3)]} for r in rs],
            "summary": {"head_yaw": round(float(np.mean([r["yaw"] for r in rs])), 3),
                        "gaze_yaw": round(float(np.mean([r["gaze_yaw"] for r in rs])), 3),
                        "gaze_pitch": round(float(np.mean([r["gaze_pitch"] for r in rs])), 3),
                        "dominant_emotion": max(mean_emotions, key=mean_emotions.get),
                        "emotions": {k: round(v, 3) for k, v in mean_emotions.items()},
                        "mean_box": [round(float(np.mean([r["box"][k] for r in rs])), 1) for k in range(4)]},
        })
    t0 = time.time()
    run_jobs(jobs)
    crops_s = round(time.time() - t0, 1)

    pairs = [{**p, "a": f"person:{p['a']}", "b": f"person:{p['b']}"} for p in person_pairs(tracks, labels, names)]
    node.write_json("faces", "faces.json", {
        "layer": "faces", "schema_version": "0.1.0",
        "producer": node.producer(timings={"detect_s": detect_s, "thumbnails_s": thumbnails_s, "crops_s": crops_s}, device=device),
        "sample_fps": c["fps"], "angles": "radians; yaw + = subject's right (image left)",
        "people": people, "tracks": out_tracks, "person_pairs": pairs, "detail": None})
    node.usage(detections=len(rows), tracks=len(tracks), people=len(people), frontal_yaw=FRONTAL_YAW, thumbnails=thumbnail_measures,
               decode_workers=workers)
