"""Stand-in body stage for runs whose pose_rtmw.json does not exist yet: RTMW-l whole-body keypoints via rtmlib.

hands.py needs whole-body wrist and hand keypoints. The body stage (pose/pose.py) owns pose_rtmw.json; when a run
has none, this runs pose.py's own RTMW path (same YOLOX person boxes, RTMW-l model, face linking and tracking,
imported from pose/pose.py, not copied) in the hands env and writes runs/<name>/hands_body_rtmw.json in the same
schema, so pose_rtmw.json stays the body stage's file. Use it with: hands.py --pose hands_body_rtmw.json.

Usage: uv run python body_rtmw.py ../runs/<name> [--device mps|cpu]
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "pose"))
import pose  # noqa: E402  (pose/pose.py; needs pose/vendor/sapiens2 for the COCO-WholeBody keypoint names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = parser.parse_args()

    cfg = pose.MODELS["rtmw"]
    names, skeleton, groups = pose.keypoint_format(cfg["format"])
    extract = json.loads((args.run_dir / "extract.json").read_text())
    faces = json.loads((args.run_dir / "faces.json").read_text())
    face_tracks = dict(enumerate(faces["tracks"]))
    video = Path(extract["video"]["source"])
    video = video if video.is_absolute() else pose.BAKEOFF / video
    fps, shots, duration = extract["video"]["fps"], extract["shots"], extract["video"]["duration"]
    step = max(1, round(fps / pose.POSE_FPS))

    detector, model = pose.RtmlibYolox(args.device), pose.Rtmw(args.device)
    first = next(pose.sample_frames(video, fps, shots, step))[2]
    model([first], detector([first]))  # warm-up
    stats = {"box_candidates": 0, "keypoint_check_failed": 0, "linked_samples": 0, "unlinked_samples": 0}
    timings = {"detect_s": 0.0, "pose_s": 0.0}
    frames_out, frame_shots = [], []
    start = time.perf_counter()
    for t, shot, frame in pose.sample_frames(video, fps, shots, step):
        tick = time.perf_counter()
        boxes = detector([frame])
        timings["detect_s"] += time.perf_counter() - tick
        tick = time.perf_counter()
        kpts = model([frame], boxes)
        timings["pose_s"] += time.perf_counter() - tick
        bodies = [{"box": [x1, y1, x2 - x1, y2 - y1], "kpts": k} for ((x1, y1, x2, y2), _), k in zip(boxes[0], kpts[0])]
        pose.link_faces(bodies, pose.face_boxes_at(t, shot, face_tracks), cfg["score_threshold"], stats)
        frames_out.append((t, bodies))
        frame_shots.append(shot)
    tracks = pose.track_bodies(frames_out, frame_shots)
    total_s = time.perf_counter() - start
    tracks.sort(key=lambda tr: (tr["shot"], tr["samples"][0]["t"]))
    out = {"model": "rtmw", "weights": f"{cfg['weights']}; person detector {pose.DETECTORS['yolox']['weights']}",
           "licence_mode": "academic-only", "licence": f"{cfg['licence']}. {pose.DETECTORS['yolox']['licence']}.",
           "keypoints": names, "skeleton": skeleton, "groups": groups, "score_threshold": cfg["score_threshold"],
           "fps": pose.POSE_FPS, "device": "coreml" if "CoreMLExecutionProvider" in model.providers else "cpu",
           "timings": {"detect_s": round(timings["detect_s"], 2), "pose_s": round(timings["pose_s"], 2),
                       "total_s": round(total_s, 2), "realtime_factor": round(total_s / duration, 3)},
           "tracks": [{"id": f"pose:{i}", "shot": tr["shot"], "face_track": tr["face_track"],
                       "link": "face-in-box+keypoints" if tr["face_track"] is not None else None,
                       "start": tr["samples"][0]["t"], "end": tr["samples"][-1]["t"],
                       "samples": [{"t": s["t"], "box": [round(float(v), 1) for v in s["box"]],
                                    "kpts": [[round(float(x), 1), round(float(y), 1), round(float(c), 2)] for x, y, c in s["kpts"]]}
                                   for s in tr["samples"]]} for i, tr in enumerate(tracks)]}
    path = args.run_dir / "hands_body_rtmw.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{path.name}: {len(frames_out)} frames, {len(tracks)} body tracks; link checks {stats}; timings {out['timings']}")


if __name__ == "__main__":
    main()
