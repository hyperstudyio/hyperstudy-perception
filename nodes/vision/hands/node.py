"""vision.hands: each hand's 21 keypoints, tracked within shots and linked to its body and face tracks, as the bundle's
hands layer.

Frames are sampled on the bodies layer's grid (every round(fps / bodies fps) frames, inside a shot), so every body
sample has its frame; frames with no body still go through the hand detector. Per frame: hand boxes come from the
bodies' hand keypoints and from RTMDet-nano hand, RTMPose-m Hand5 estimates the keypoints in each (models.py), and
hand_tracks.py links and tracks them.

Beside the layer, a side file keeps every detection RTMDet-nano hand returned, kept or not, with its score and why
(hand_detections.py), so hand assignment can be revisited. The layer points at it with raw.detections.

Licence: rtmlib's code and the RTMPose-hand and RTMDet-hand weights are Apache-2.0, but Hand5 is trained on
COCO-WholeBody-Hand (non-commercial) and FreiHAND (research only), RTMDet-nano hand on FreiHAND too, and the hand boxes
come from RTMW (academic-only): so this layer is academic-only.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from perception_node import Node, NodeError
import hand_detections
from hand_tracks import KEYPOINTS, SKELETON, body_indices, bodies_by_time, frame_hands, layer_hands, propose, track_hands
from models import WEIGHTS, Hands

POSE_ID = "RTMPose-m Hand5 256x256 (rtmpose-m_simcc-hand5_pt-aic-coco_210e)"
POSE_LICENCE = ("rtmlib code and RTMPose-m Hand5 weights Apache-2.0, but Hand5 is trained on COCO-WholeBody-Hand "
                "(non-commercial) and FreiHAND (research only), plus OneHand10K, RHD and Halpe (licences not checked); "
                "its hand boxes also come from RTMW (academic-only)")
DET_ID = "RTMDet-nano hand 320x320 (rtmdet_nano_8xb32-300e_hand)"
DET_LICENCE = ("Apache-2.0 weights trained on OneHand10K, FreiHAND (research only), RHD and Halpe-Hand "
               "(mmpose's rtmdet_nano_320-8xb32_hand.py)")


def sampled(video: Path, fps: float, shots: list[dict], step: int):
    """(t, shot id, BGR frame) for every step-th frame that lies inside a shot, decoded in order (body-pose's grid)."""
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
    video, media = node.input("video"), node.input_json("media")
    shots = node.input_json("shots")["shots"]
    bodies_layer = node.input_json("bodies")
    fps = media["fps"]
    step = max(1, round(fps / bodies_layer["fps"]))
    try:
        idx = body_indices(bodies_layer["keypoint_names"])
    except ValueError as e:
        raise NodeError("input", str(e)) from None
    threshold = bodies_layer["score_threshold"]
    bodies = bodies_by_time(bodies_layer["tracks"])

    t0 = time.time()
    model = Hands(node.accelerator)
    # the first inference compiles the CoreML model: done before the run is timed, as the bake-off did
    blank = np.zeros((media["height"], media["width"], 3), np.uint8)
    model.detect(blank)
    model.keypoints(blank, [800.0, 500.0, 200.0, 200.0])
    model.detect_s = model.pose_s = 0.0
    load_s = round(time.time() - t0, 1)
    device = "coreml" if "CoreMLExecutionProvider" in model.providers else "cuda" if "CUDAExecutionProvider" in model.providers else "cpu"
    node.model(POSE_ID, licence_mode="academic-only", source=WEIGHTS["rtmpose_hand"]["url"],
               revision=WEIGHTS["rtmpose_hand"]["sha256"], licence=POSE_LICENCE)
    node.model(DET_ID, licence_mode="academic-only", source=WEIGHTS["rtmdet_hand"]["url"],
               revision=WEIGHTS["rtmdet_hand"]["sha256"], licence=DET_LICENCE)

    stats = {"wrist_proposals": 0, "wrist_below_gate": 0, "detector_below_score": 0, "detector_below_gate": 0,
             "detector_too_large": 0, "detector_duplicate": 0, "detector_linked": 0, "detector_unlinked": 0}
    frames, seen, sizes = [], set(), set()
    detections = hand_detections.Detections()   # every detection and what became of it (the side file)
    t_all = time.time()
    for t, shot, image in sampled(video, fps, shots, step):
        seen.add(t)
        height, width = image.shape[:2]   # the frame as decoded (upright): the coordinates body-pose's keypoints are in
        sizes.add((width, height))
        posed = [propose(b, idx, threshold, width, height) for b in bodies.get(t, [])]
        stats["wrist_proposals"] += sum(len(b["proposals"]) for b in posed)
        record = []
        frames.append((t, shot, frame_hands(image, posed, model, stats, record)))
        try:
            detections.add(t, record)
        except ValueError as e:
            raise NodeError("internal", f"the hand detector's output: {e}") from None
    total_s = round(time.time() - t_all, 1)
    missing = sorted(set(bodies) - seen)
    if missing:
        raise NodeError("input", f"{len(missing)} body samples are not on this video's sampling grid (e.g. {missing[:3]}): "
                                 "the bodies layer is not this video's")
    if len(sizes) > 1:   # the side file names one frame size, which a re-run of the gates and matching needs
        raise NodeError("input", f"the frames change size ({sorted(sizes)[:3]}): the side file names one frame size")
    tracks = track_hands(frames)
    hands = layer_hands(tracks)

    node.write_json("hands", "hands.json", {
        "layer": "hands", "schema_version": "0.1.0",
        "producer": node.producer(timings={"load_s": load_s, "detect_s": round(model.detect_s, 1), "pose_s": round(model.pose_s, 1),
                                           "total_s": total_s}, device=device),
        "keypoint_names": KEYPOINTS, "skeleton": SKELETON, "score_threshold": None, "fps": bodies_layer["fps"], "hands": hands,
        "raw": {"detections": {"file": hand_detections.FILE, "format": "parquet", "description": hand_detections.DESCRIPTION}}})
    width, height = sizes.pop() if sizes else (media["width"], media["height"])
    side_path = node.path("detections", "hands_detections.parquet")
    try:
        hand_detections.write(side_path, detections.columns(tracks), DET_ID, width, height)
    except ValueError as e:
        raise NodeError("internal", f"the detections side file: {e}") from None
    # the stored detections must give the layer's hands through hand_tracks' own gates, matching and tracking (as
    # decode checks its stored scores give its cuts): about 0.3 s on OOO clip 1
    problem = hand_detections.check(side_path, hands, bodies_layer, shots)
    if problem:
        raise NodeError("internal", f"the detections side file does not give the layer: {problem}")
    samples = lambda pred: sum(len(h["samples"]) for h in hands if pred(h))
    node.usage(frames=len(frames), hands=len(hands), samples=samples(lambda h: True),
               linked_samples=samples(lambda h: h["body"] is not None), face_linked_samples=samples(lambda h: h["face"] is not None),
               realtime_factor=round(total_s / media["duration"], 3) if media.get("duration") else None, **stats)
