"""vision.body-pose: each person's 2D skeleton, tracked within shots and linked to their face track, as the bundle's
bodies layer.

Frames are sampled at the faces node's frame indices (every round(fps / `fps`) frames), so a body sample and a face
sample share a timestamp. Per frame: YOLOX-m finds people, RTMW-l estimates the 133 COCO-WholeBody keypoints in each
box (models.py), and bodies.py drops second detections, links bodies to face tracks and tracks them.

Beside the layer, two side files keep what the layer drops of the models' output (detections.py):
layers/bodies_scores.parquet, YOLOX-m's score for every body sample, and layers/bodies_duplicates.parquet, every
detection dropped as a second detection of a person (its score, box and keypoints, and the body sample it duplicated),
so the de-duplication can be revisited. The layer points at them with raw.scores and raw.duplicates.

Licence: rtmlib's code and the RTMW and YOLOX weights are Apache-2.0, but RTMW is trained on Cocktail14, which includes
COCO-WholeBody (annotations research/non-commercial only) and Human-Art (non-commercial), and YOLOX-m on Human-Art: so
this layer is academic-only.
"""

import json
import time
from pathlib import Path

import cv2

from perception_node import Node, NodeError
from bodies import (DUPLICATE_HEAD_DIST, HEAD_JOINTS, drop_duplicate_bodies, face_boxes_at, layer_tracks, link_faces,
                    side_rows, track_bodies)
import detections
from models import SCORE_THRESHOLD, WEIGHTS, YOLOX_SCORE, Detector, Pose

KEYPOINTS = json.loads((Path(__file__).parent / "keypoints.json").read_text())
YOLOX_ID = "YOLOX-m person detector (yolox_m_8xb8-300e_humanart)"
LICENCE = ("rtmlib code and RTMW weights Apache-2.0, but RTMW is trained on Cocktail14, which includes COCO-WholeBody "
           "(annotations research/non-commercial only) and Human-Art (non-commercial)")


def sampled(video: Path, fps: float, shots: list[dict], step: int):
    """(t, shot id, BGR frame) for every step-th frame that lies inside a shot, decoded in order."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise NodeError("input", f"OpenCV cannot open {video.name}")
    index = 0
    try:
        while True:
            if index % step:
                if not capture.grab():
                    break
            else:
                ok, frame = capture.read()
                if not ok:
                    break
                t = round(index / fps, 3)
                shot = next((s for s in shots if s["start"] <= t < s["end"]), None)
                if shot is not None:
                    yield t, f"shot:{shot['index']}", frame
            index += 1
    finally:
        capture.release()


with Node() as node:
    c = node.config
    video, media = node.input("video"), node.input_json("media")
    shots = node.input_json("shots")["shots"]
    face_tracks = node.input_json("faces")["tracks"]
    fps = media["fps"]
    step = max(1, round(fps / c["fps"]))
    names = KEYPOINTS["names"]
    head_idx = [names.index(n) for n in HEAD_JOINTS]

    t0 = time.time()
    detector, pose = Detector(node.accelerator), Pose(node.accelerator)
    load_s = round(time.time() - t0, 1)
    device = "coreml" if "CoreMLExecutionProvider" in pose.providers else "cuda" if "CUDAExecutionProvider" in pose.providers else "cpu"
    node.model("RTMW-l 384x288 (rtmw-dw-x-l_simcc-cocktail14_270e)", licence_mode="academic-only", source=WEIGHTS["rtmw"]["url"],
               revision=WEIGHTS["rtmw"]["sha256"], licence=LICENCE)
    node.model(YOLOX_ID, licence_mode="academic-only", source=WEIGHTS["yolox"]["url"],
               revision=WEIGHTS["yolox"]["sha256"], licence="Apache-2.0 weights trained on Human-Art (non-commercial) and COCO")

    timings = {"detect_s": 0.0, "pose_s": 0.0}
    stats = {"duplicates_dropped": 0, "box_candidates": 0, "keypoint_check_failed": 0, "linked_samples": 0, "unlinked_samples": 0}
    frames, frame_shots, duplicates = [], [], []
    t_all = time.time()
    for t, shot, image in sampled(video, fps, shots, step):
        tick = time.time()
        boxes = detector(image)
        timings["detect_s"] += time.time() - tick
        tick = time.time()
        kpts = pose(image, boxes)
        timings["pose_s"] += time.time() - tick
        # det: the detection's index in this frame, by which a dropped duplicate names the body it duplicated (side_rows)
        bodies = [{"box": [x1, y1, x2 - x1, y2 - y1], "score": score, "kpts": k, "det": n}
                  for n, (((x1, y1, x2, y2), score), k) in enumerate(zip(boxes, kpts))]
        dropped = []
        kept = drop_duplicate_bodies(bodies, head_idx, SCORE_THRESHOLD, dropped)
        duplicates += [{**d, "t": t} for d in dropped]
        stats["duplicates_dropped"] += len(bodies) - len(kept)
        link_faces(kept, face_boxes_at(t, shot, face_tracks), SCORE_THRESHOLD, stats)
        frames.append((t, kept))
        frame_shots.append(shot)
    body_tracks = track_bodies(frames, frame_shots)
    tracks = layer_tracks(body_tracks)
    total_s = round(time.time() - t_all, 1)

    node.write_json("bodies", "bodies.json", {
        "layer": "bodies", "schema_version": "0.1.0",
        "producer": node.producer(timings={"load_s": load_s, "detect_s": round(timings["detect_s"], 1),
                                           "pose_s": round(timings["pose_s"], 1), "total_s": total_s}, device=device),
        "keypoint_names": names, "skeleton": KEYPOINTS["skeleton"], "score_threshold": SCORE_THRESHOLD, "fps": c["fps"],
        "tracks": tracks,
        "raw": {"scores": {"file": detections.SCORES_FILE, "format": "parquet", "description": detections.SCORES_DESCRIPTION},
                "duplicates": {"file": detections.DUPLICATES_FILE, "format": "parquet",
                               "description": detections.DUPLICATES_DESCRIPTION}}})
    score_rows, duplicate_rows = side_rows(body_tracks, duplicates)
    try:
        detections.write_scores(node.path("scores", "bodies_scores.parquet"), score_rows, YOLOX_ID, YOLOX_SCORE)
        detections.write_duplicates(node.path("duplicates", "bodies_duplicates.parquet"), duplicate_rows, YOLOX_ID,
                                    YOLOX_SCORE, DUPLICATE_HEAD_DIST, SCORE_THRESHOLD)
    except ValueError as e:
        raise NodeError("internal", f"the models' output: {e}") from None
    node.usage(frames=len(frames), bodies=sum(len(b) for _, b in frames), tracks=len(tracks),
               linked_tracks=sum(1 for tr in tracks if tr["face"]), realtime_factor=round(total_s / media["duration"], 3)
               if media.get("duration") else None, **stats)
