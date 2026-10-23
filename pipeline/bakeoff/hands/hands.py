"""Hand pose stage of the bake-off: 21 keypoints per hand, tracked within shots and linked to body and face tracks.

Hands are a gated second stage on top of the body stage (pose/pose.py), as recommended in
research_notes/.../hand_pose_estimation.md (HaMeR's design):
  1. Hand proposals come from each body's whole-body hand keypoints in pose_rtmw.json: a hand slot with at
     least MIN_HAND_KPTS keypoints above the body model's score threshold and inside the frame gives a square box around those
     keypoints and the body wrist. These hands inherit the body's pose track, face track and side
     (COCO-WholeBody left_hand = the subject's left hand). box_source "wrist".
  2. A full-frame hand detector (RTMDet-nano hand for rtmpose_hand, MediaPipe's palm detector for mediapipe)
     finds hands whose body was not posed. Detector hands are matched to body wrists of every skeleton in the
     frame by Hungarian assignment (cost = wrist distance outside the box / box side, gated by WRIST_GATE).
     A detector hand matched to a wrist that already produced a hand is the same hand and is dropped; one
     matched to a wrist with no hand inherits that person and side; an unmatched one is kept with
     person null (side from MediaPipe's handedness, else null). box_source "detector".
  3. The second-stage model runs only on hands whose box side is at least MIN_HAND_PX. Smaller hands are
     left out of the second-stage files (they keep only their RTMW keypoints in hands_rtmw.json).
  4. Hands are tracked within the shot: linked hands by (pose track, side, box source), unlinked hands by
     box IoU; a gap longer than MAX_TRACK_GAP_S ends a track, and so does a cut.

Models (--model):
  rtmw          the body stage's own RTMW-l hand keypoints (no extra model, no detector); every hand slot that
                passes step 1 is kept, whatever its size. Timings are the body stage's, read from pose_rtmw.json.
  rtmpose_hand  rtmlib Hand: RTMPose-m Hand5 256x256 on the step-1 boxes, RTMDet-nano hand 320x320 on the full frame.
  mediapipe     MediaPipe Hand Landmarker (float16 v1) in IMAGE mode on a crop around each step-1 box, and on the
                full frame as the detector. Always the CPU delegate (the Metal delegate loses hands; see MediapipeHand).
                Its handedness label is used as is for unlinked hands: measured against the body slot, it already
                names the subject's side on non-mirrored video (MEDIAPIPE_LABEL_TO_SIDE).
  sam3dbody     reads pose_sam3dbody.json's hand keypoints, as for rtmw. Exits with the reason when that file
                does not exist or has no hand keypoints.

Keypoint scores are each model's own and not comparable across models: RTMW and RTMPose-hand give SimCC peak
values (RTMW 0-12, RTMPose-hand 0-1); MediaPipe gives no per-landmark confidence, so its keypoints carry 1.0
(it only emits a hand whose presence score passed min_hand_presence_confidence).

--device only affects RTMPose-hand (mps = CoreML EP); RTMDet-hand and MediaPipe always run on the CPU.
When a run has no pose_rtmw.json yet, body_rtmw.py writes hands_body_rtmw.json; pass it with --pose.

Usage: uv run python hands.py ../runs/<name> --model rtmw|rtmpose_hand|mediapipe|sam3dbody [--device mps|cpu] [--pose pose_rtmw.json]
Writes runs/<name>/hands_<model>.json.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).parent
BAKEOFF = HERE.parent  # extract.json's video.source is relative to this
MEDIAPIPE_TASK = HERE / "vendor/hand_landmarker.task"  # curl -L -o vendor/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task (sha256 fbc2a300...cde1)

# The body model's per-keypoint visibility threshold. pose.py calibrated RTMW's (raw SimCC peaks, 0-12) at 3.0
# against ViTPose; older pose_rtmw.json files predate its "score_threshold" field, so it is pinned here and
# checked against the file when the file has it.
BODY_SCORE_THRESHOLD = {"rtmw": 3.0}
MIN_HAND_KPTS = 4        # HaMeR's demo builds a hand box only from more than 3 valid hand keypoints
HAND_BOX_PAD = 1.2       # square box side = this x the larger extent of the valid hand keypoints + wrist (tips sit on the extent edge)
# Second-stage gate on the hand box side. Contact sheets of 24 random OOO RTMW hand boxes per size bin: at 24-32 px
# the hands are blurred blobs (wide shots, hands in pockets or at the sides) with no fingers separable by eye; at
# 32-48 px finger separation is visible on some (clasped hands at a mouth, hands on a chest). 32 px is also the top
# of the research note's 24-32 px "no model resolves fingers below this" range.
MIN_HAND_PX = 32
WRIST_GATE = 1.0         # a detector box matches a wrist at most one box side outside it ("about one hand-length", research note Q4)
DEDUP_IOU = 0.3          # a detector box overlapping an accepted hand box this much is that hand again
TRACK_IOU = 0.3          # unlinked hands: box overlap with a track's last sample to continue it (same rule as faces.py / pose.py)
MAX_TRACK_GAP_S = 0.5    # a hand unseen for longer than this (4 samples at 8 fps) starts a new track, as in pose.py
# RTMDet-nano hand scores stay low on this footage (none above 0.4 on OOO apart from one 0.71 on a tiny graphic).
# Contact sheets of 30 random OOO detections per score bin, judged by eye: 0.15-0.2 about 13/30 are hands (the rest
# faces, necklaces, elbows), 0.2-0.25 about 17/30, >= 0.25 about 24/30. rtmlib's default 0.3 drops visible
# gesturing hands that score 0.25-0.3, so 0.25.
RTMDET_SCORE = 0.25
# RTMDet-nano hand also returns frame-sized boxes (1900 px wide) on black frames and bokeh shots of nes-biking;
# a detector box whose longer side exceeds this fraction of the frame height is not a hand.
MAX_DETECTOR_BOX = 0.5
MEDIAPIPE_FULL_HANDS = 8 # max hands in a full frame (panel shows have up to 4 people on screen)
MEDIAPIPE_CROP_HANDS = 2 # a 3x crop around one hand often holds that person's other hand too
# The MediaPipe hand kept for a box is the one whose landmark centroid is nearest the box centre, and only if within
# this x box side of it. Without the gate, a crop where MediaPipe finds only the OTHER hand assigns that hand to this
# slot (seen on OOO: clasped and crossed hands); MediaPipe's label agreed with the body slot on 826/1316 OOO crops
# without the gate and 780/1172 with it.
MEDIAPIPE_CENTRE_GATE = 0.5
# Crop side = this x hand box side. The palm detector needs the palm plus context: on 150 OOO wrist boxes with a
# visible hand (RTMPose-hand mean score >= 0.3), CPU recall was 87 at 1.5x, 98 at 2x, 109 at 3x (256 px; 384 px no better).
MEDIAPIPE_CROP_SCALE = 3.0
MEDIAPIPE_CROP_PX = 256  # crops are resized to this before detection (the models run at 192 and 224 px)
# MediaPipe's guide says handedness "assumes the input image is mirrored", which would make its "Left" the subject's
# right on this non-mirrored video. Measured on OOO with mediapipe 1.1.1, it is not: on wrist crops whose MediaPipe
# wrist lies within half a box side of the RTMW body wrist, the raw label equals the body slot's side for 251/310
# hands >= 128 px, 92/104 at 64-128 px and 19/30 at 32-64 px (85% overall; the swap would give 15%). So no swap.
# main() prints the agreement for every run.
MEDIAPIPE_LABEL_TO_SIDE = {"Left": "left", "Right": "right"}
# RTMPose-hand, one 256x256 crop, M5 Max: CPU EP 11.9 ms, CoreML NeuralNetwork 4.2 ms, CoreML MLProgram 5.3 ms, so
# rtmlib's own device='mps' (CoreML EP, NeuralNetwork format) is used as is.
# rtmlib's RTMDet-nano hand ONNX ends in NMS; ONNX Runtime 1.31's CoreML EP fails on it ("CoreML static output shape
# ({1,1,1,2100,2100}) and inferred shape ({1,2100}) have different ranks"), the same failure as YOLOX in pose.py,
# so the hand detector always runs on the CPU EP (about 7 ms per frame).

RTMDET_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip"
RTMPOSE_HAND_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"

KEYPOINTS = ["wrist", "thumb1", "thumb2", "thumb3", "thumb4", "forefinger1", "forefinger2", "forefinger3", "forefinger4",
             "middle_finger1", "middle_finger2", "middle_finger3", "middle_finger4", "ring_finger1", "ring_finger2",
             "ring_finger3", "ring_finger4", "pinky_finger1", "pinky_finger2", "pinky_finger3", "pinky_finger4"]
SKELETON = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
# COCO-WholeBody's 21 hand keypoints, MediaPipe's 21 landmarks and RTMPose Hand5 share this order
# (wrist, then thumb, index, middle, ring, pinky from base to tip), so one list serves every model.

BODY_WRIST = {"left": 9, "right": 10}  # COCO-17 body indices, shared by COCO-WholeBody

MODELS = {
    "rtmw": {
        "licence_mode": "academic-only",
        "licence": ("RTMW-l hand keypoints from the body stage: rtmlib code and RTMW weights Apache-2.0, but RTMW is "
                    "trained on Cocktail14, which includes COCO-WholeBody (annotations research/non-commercial only) and "
                    "Human-Art (non-commercial); its person detector YOLOX-m is trained on Human-Art"),
    },
    "rtmpose_hand": {
        "licence_mode": "academic-only",
        "licence": ("rtmlib code and RTMPose-m Hand5 / RTMDet-nano hand weights Apache-2.0, but Hand5 is trained on "
                    "COCO-WholeBody-Hand (non-commercial) and FreiHAND (research only), plus OneHand10K, RHD and Halpe "
                    "(licences not checked); RTMDet-nano hand's training data is unverified. Hand boxes also come from "
                    "RTMW (academic-only)"),
    },
    "mediapipe": {
        "licence_mode": "academic-only",
        "licence": ("MediaPipe Hand Landmarker: code Apache-2.0 and model card 'licensed under Apache License 2.0' "
                    "(training data ~30K real images + synthetic, provenance undocumented; card says experimental use). "
                    "Commercial-ok on its own, but the hand boxes here come from RTMW (academic-only); a commercial "
                    "pipeline would need commercial body keypoints or full-frame detection only"),
    },
    "sam3dbody": {
        "licence_mode": "gated-review",
        "licence": "SAM 3D Body: SAM License (commercial use allowed with conditions); hand output is the MHR rig; needs legal review",
    },
}


# ---------------------------------------------------------------- geometry

def iou(a, b):
    """IoU of two [x, y, w, h] boxes."""
    w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def square_box(points, pad):
    """Square [x, y, w, h] around points, side = pad x the larger extent."""
    lo, hi = points.min(0), points.max(0)
    centre, side = (lo + hi) / 2, max(float((hi - lo).max()) * pad, 1.0)
    return [float(centre[0] - side / 2), float(centre[1] - side / 2), side, side]


def outside_distance(point, box):
    """Distance from a point to a [x, y, w, h] box (0 inside)."""
    dx = max(box[0] - point[0], 0.0, point[0] - box[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[1] - box[3])
    return float(np.hypot(dx, dy))


# ---------------------------------------------------------------- body proposals

def visible(kpts, threshold, width, height):
    """Keypoints above the score threshold AND inside the frame. RTMW's SimCC heads put off-frame hands (arms
    leaving the bottom of a medium shot) just outside the image with scores well above the threshold."""
    return (kpts[:, 2] >= threshold) & (kpts[:, 0] >= 0) & (kpts[:, 0] < width) & (kpts[:, 1] >= 0) & (kpts[:, 1] < height)


def body_frames(pose, threshold, width, height):
    """{t: [body]} with body = {pose_track, face_track, shot, kpts (133x3), wrists {side: xy or None}, proposals {side: box, kpts}}."""
    frames = {}
    for track in pose["tracks"]:
        for s in track["samples"]:
            kpts = np.array(s["kpts"], dtype=float)
            body = {"pose_track": track["id"], "face_track": track["face_track"], "shot": track["shot"],
                    "kpts": kpts, "wrists": {}, "proposals": {}}
            for side in ("left", "right"):
                wrist = kpts[BODY_WRIST[side]]
                body["wrists"][side] = wrist[:2] if visible(wrist[None], threshold, width, height)[0] else None
                hand = kpts[pose["groups"][f"{side}_hand"]]
                valid = visible(hand, threshold, width, height)
                if valid.sum() < MIN_HAND_KPTS:
                    continue
                points = hand[valid, :2]
                if body["wrists"][side] is not None:
                    points = np.vstack([points, body["wrists"][side]])
                body["proposals"][side] = {"box": square_box(points, HAND_BOX_PAD), "kpts": hand}
            frames.setdefault(round(s["t"], 3), []).append(body)
    return frames


# ---------------------------------------------------------------- models

class RtmposeHand:
    """RTMDet-nano hand (full frame, CPU EP) + RTMPose-m Hand5 (boxes; CoreML EP on --device mps)."""

    def __init__(self, device):
        from rtmlib import RTMDet, RTMPose
        self.det = RTMDet(RTMDET_URL, model_input_size=(320, 320), backend="onnxruntime", device="cpu")
        self.pose = RTMPose(RTMPOSE_HAND_URL, model_input_size=(256, 256), backend="onnxruntime", device=device)
        self.det_providers = self.det.session.get_providers()
        self.pose_providers = self.pose.session.get_providers()

    def detect(self, frame):
        """[(box [x, y, w, h], side None)]. rtmlib's RTMDet drops scores, so its pre/post-processing is used directly."""
        image, ratio = self.det.preprocess(frame)
        dets = self.det.inference(image)[0][0]  # (N, 5): x1, y1, x2, y2, score, NMS baked into the ONNX
        out = []
        for x1, y1, x2, y2, score in dets:
            if score >= RTMDET_SCORE:
                x1, y1, x2, y2 = (v / ratio for v in (x1, y1, x2, y2))
                out.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], None, None, float(score)))
        return sorted(out, key=lambda d: -d[3])

    def keypoints(self, frame, box, wrist):
        x, y, w, h = box
        kpts, scores = self.pose(frame, bboxes=[[x, y, x + w, y + h]])  # rtmlib pads the box by 1.25 itself
        return np.concatenate([kpts[0], scores[0][:, None]], axis=1)


class MediapipeHand:
    """MediaPipe Hand Landmarker, IMAGE mode, on the full frame (detector) and on crops around step-1 boxes."""

    def __init__(self, device):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        # Always the CPU (XNNPACK) delegate, whatever --device says. The GPU (Metal) delegate runs (with SRGBA input;
        # it rejects RGB: "Only 4-channel texture input formats are supported") but on the same 150 OOO crops it found
        # 76 hands vs the CPU delegate's 109, and its landmarks on hands both found differ by 14.2 px on average. The CPU
        # delegate gives identical results for RGB and RGBA input, so the loss is the delegate, not the input format.
        make = lambda n: vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MEDIAPIPE_TASK), delegate=BaseOptions.Delegate.CPU), num_hands=n))
        self.full, self.crop = make(MEDIAPIPE_FULL_HANDS), make(MEDIAPIPE_CROP_HANDS)
        self.rgba = False
        self.det_providers = self.pose_providers = ["cpu-xnnpack"]
        self.label_stats = {"agree": 0, "disagree": 0}

    def _run(self, landmarker, image_bgr):
        code, fmt = (cv2.COLOR_BGR2RGBA, self.mp.ImageFormat.SRGBA) if self.rgba else (cv2.COLOR_BGR2RGB, self.mp.ImageFormat.SRGB)
        result = landmarker.detect(self.mp.Image(image_format=fmt, data=np.ascontiguousarray(cv2.cvtColor(image_bgr, code))))
        h, w = image_bgr.shape[:2]
        return [(np.array([[p.x * w, p.y * h] for p in lms]), MEDIAPIPE_LABEL_TO_SIDE[hd[0].category_name])
                for lms, hd in zip(result.hand_landmarks, result.handedness)]

    def detect(self, frame):
        out = []
        for xy, side in self._run(self.full, frame):
            out.append((square_box(xy, HAND_BOX_PAD), side, np.concatenate([xy, np.ones((21, 1))], axis=1), 1.0))
        return out

    def keypoints(self, frame, box, wrist):
        """(landmarks, MediaPipe side) of the hand in a crop around box, or None when MediaPipe finds no hand there
        or only hands centred elsewhere (MEDIAPIPE_CENTRE_GATE)."""
        side = box[2] * MEDIAPIPE_CROP_SCALE
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        s = int(round(side))
        crop = np.zeros((s, s, 3), np.uint8)  # black outside the frame
        fx0, fy0 = max(x0, 0), max(y0, 0)
        fx1, fy1 = min(x0 + s, frame.shape[1]), min(y0 + s, frame.shape[0])
        crop[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] = frame[fy0:fy1, fx0:fx1]
        scale = s / MEDIAPIPE_CROP_PX
        hands = self._run(self.crop, cv2.resize(crop, (MEDIAPIPE_CROP_PX, MEDIAPIPE_CROP_PX), interpolation=cv2.INTER_LINEAR))
        if not hands:
            return None
        hands = [(xy * scale + [x0, y0], s_) for xy, s_ in hands]
        centre = np.array([cx, cy])
        distance = lambda h: float(np.hypot(*(h[0].mean(0) - centre)))
        xy, mp_side = min(hands, key=distance)
        if distance((xy, mp_side)) > MEDIAPIPE_CENTRE_GATE * box[2]:
            return None
        return np.concatenate([xy, np.ones((21, 1))], axis=1), mp_side


# ---------------------------------------------------------------- per-frame assembly

def match_detections(detections, bodies):
    """Hungarian match of detector boxes to body wrists. Returns {detection index: (body index, side)}."""
    wrists = [(b, side, xy) for b, body in enumerate(bodies) for side, xy in body["wrists"].items() if xy is not None]
    if not detections or not wrists:
        return {}
    cost = np.array([[outside_distance(xy, box) / box[2] for _, _, xy in wrists] for box, *_ in detections])
    rows, cols = linear_sum_assignment(cost)
    return {r: wrists[c][:2] for r, c in zip(rows, cols) if cost[r, c] <= WRIST_GATE}


def frame_hands(frame, bodies, model, stats):
    """Hands of one frame: [{box, side, box_source, person, kpts}] for a second-stage model."""
    hands, done = [], set()  # done: (body index, side) already holding a hand
    for b, body in enumerate(bodies):
        for side, prop in body["proposals"].items():
            if prop["box"][2] < MIN_HAND_PX:
                stats["wrist_below_gate"] += 1
                continue
            tick = time.perf_counter()
            result = model.keypoints(frame, prop["box"], body["wrists"][side])
            stats["pose_s"] += time.perf_counter() - tick
            if isinstance(model, MediapipeHand):
                if result is None:
                    stats["crop_no_hand"] += 1  # no hand, or only hands centred away from this box
                    continue
                result, mp_side = result
                model.label_stats["agree" if mp_side == side else "disagree"] += 1
            hands.append({"box": prop["box"], "side": side, "box_source": "wrist", "kpts": result,
                          "person": {"pose_track": body["pose_track"], "face_track": body["face_track"]}})
            done.add((b, side))

    tick = time.perf_counter()
    detections = model.detect(frame)
    stats["detect_s"] += time.perf_counter() - tick
    matches = match_detections(detections, bodies)
    for d, (box, det_side, det_kpts, _) in enumerate(detections):  # highest score first
        if min(box[2], box[3]) < MIN_HAND_PX:
            stats["detector_below_gate"] += 1
            continue
        if max(box[2], box[3]) > MAX_DETECTOR_BOX * frame.shape[0]:
            stats["detector_too_large"] += 1
            continue
        # RTMDet's baked-in NMS leaves overlapping boxes on one hand (IoU ~0.5 seen on OOO), so any box overlapping
        # an accepted hand is that hand again.
        if any(iou(box, h["box"]) >= DEDUP_IOU for h in hands):
            stats["detector_duplicate"] += 1
            continue
        if d in matches:
            b, side = matches[d]
            if (b, side) in done:
                stats["detector_duplicate"] += 1
                continue
            person = {"pose_track": bodies[b]["pose_track"], "face_track": bodies[b]["face_track"]}
            done.add((b, side))
            stats["detector_linked"] += 1
        else:
            side, person = det_side, None
            stats["detector_unlinked"] += 1
        if det_kpts is None:
            tick = time.perf_counter()
            det_kpts = model.keypoints(frame, box, None)
            stats["pose_s"] += time.perf_counter() - tick
        hands.append({"box": box, "side": side, "box_source": "detector", "person": person, "kpts": det_kpts})
    return hands


def track_hands(frames):
    """frames: [(t, shot, [hand])]. Linked hands continue by (pose track, side, source); unlinked ones by IoU."""
    tracks, open_keyed, open_free = [], {}, {}
    for t, shot, hands in frames:
        free_candidates = [tr for tr in open_free.get(shot, []) if t - tr["samples"][-1]["t"] <= MAX_TRACK_GAP_S + 1e-6]
        used = set()
        for hand in sorted(hands, key=lambda h: -h["box"][2]):
            if hand["person"] is not None:
                key = (shot, hand["person"]["pose_track"], hand["side"], hand["box_source"])
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


def sample_frames(video_path, fps, step, times):
    """(t, BGR frame) for every step-th frame whose timestamp is in times (pose.py's sampling), decoded sequentially."""
    capture = cv2.VideoCapture(str(video_path))
    index = 0
    while True:
        if index % step:
            if not capture.grab():
                break
        else:
            ok, frame = capture.read()
            if not ok:
                break
            t = round(index / fps, 3)
            if t in times:
                yield t, frame
        index += 1
    capture.release()


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--pose", default="pose_rtmw.json", help="body stage file whose wrist/hand keypoints make the hand boxes")
    args = parser.parse_args()

    pose_path = args.run_dir / ("pose_sam3dbody.json" if args.model == "sam3dbody" else args.pose)
    if not pose_path.exists():
        reason = (" pose.py reports SAM 3D Body cannot run here (gated checkpoints, CUDA-only inference)"
                  if args.model == "sam3dbody" else " run pose/pose.py first")
        raise SystemExit(f"{pose_path} does not exist;{reason}")
    pose = json.loads(pose_path.read_text())
    if not pose["groups"].get("left_hand"):
        raise SystemExit(f"{pose_path.name} has no hand keypoints (groups.left_hand is empty)")
    threshold = BODY_SCORE_THRESHOLD[pose["model"]]
    if "score_threshold" in pose and pose["score_threshold"] != threshold:
        raise SystemExit(f"{pose_path.name} score_threshold {pose['score_threshold']} != pinned {threshold}")
    extract = json.loads((args.run_dir / "extract.json").read_text())
    duration = extract["video"]["duration"]
    bodies = body_frames(pose, threshold, extract["video"]["width"], extract["video"]["height"])
    shot_of = {round(s["t"], 3): tr["shot"] for tr in pose["tracks"] for s in tr["samples"]}

    stats = {"wrist_proposals": sum(len(b["proposals"]) for bs in bodies.values() for b in bs), "wrist_below_gate": 0,
             "crop_no_hand": 0, "detector_below_gate": 0, "detector_too_large": 0, "detector_duplicate": 0, "detector_linked": 0,
             "detector_unlinked": 0, "detect_s": 0.0, "pose_s": 0.0}
    if args.model in ("rtmw", "sam3dbody"):
        frames = [(t, shot_of[t], [{"box": p["box"], "side": side, "box_source": "wrist", "kpts": p["kpts"],
                                    "person": {"pose_track": b["pose_track"], "face_track": b["face_track"]}}
                                   for b in bs for side, p in b["proposals"].items()])
                  for t, bs in sorted(bodies.items())]
        device, timings = pose["device"], dict(pose["timings"])
        print(f"{args.model}: hands read from {pose_path.name}; timings are the body stage's")
    else:
        # Frames with no posed body still go through the detector: every sampled frame inside a shot.
        fps, shots = extract["video"]["fps"], extract["shots"]
        step = max(1, round(fps / pose["fps"]))
        n_frames = int(duration * fps) + 1
        times = {round(i / fps, 3) for i in range(0, n_frames, step)}
        shot_at = lambda t: next((s["index"] for s in shots if s["start"] <= t < s["end"]), None)
        times = {t for t in times if shot_at(t) is not None}
        missing = set(bodies) - times
        if missing:
            raise SystemExit(f"{len(missing)} pose timestamps are not on this sampling grid (e.g. {sorted(missing)[:3]})")
        video = Path(extract["video"]["source"])
        video = video if video.is_absolute() else BAKEOFF / video
        load = time.perf_counter()
        model = RtmposeHand(args.device) if args.model == "rtmpose_hand" else MediapipeHand(args.device)
        print(f"loaded in {time.perf_counter() - load:.1f} s; detector {model.det_providers}, hand model {model.pose_providers}")
        first_t, first = next(sample_frames(video, fps, step, times))  # warm-up: CoreML / Metal compile on first inference
        model.detect(first)
        model.keypoints(first, [800.0, 500.0, 200.0, 200.0], None)
        frames = []
        start = time.perf_counter()
        for t, frame in sample_frames(video, fps, step, times):
            frames.append((t, shot_at(t), frame_hands(frame, bodies.get(t, []), model, stats)))
        total_s = time.perf_counter() - start
        providers = model.pose_providers
        device = ("coreml" if "CoreMLExecutionProvider" in providers else providers[0] if providers[0] in ("metal-gpu", "cpu-xnnpack")
                  else "cpu")
        timings = {"detect_s": round(stats["detect_s"], 2), "pose_s": round(stats["pose_s"], 2),
                   "total_s": round(total_s, 2), "realtime_factor": round(total_s / duration, 3)}
        if isinstance(model, MediapipeHand):
            print(f"MediaPipe handedness vs body slot on wrist crops: {model.label_stats}")

    tracks = track_hands(frames)
    tracks.sort(key=lambda tr: (tr["shot"], tr["samples"][0]["t"]))
    out_hands = [{
        "id": f"hand:{i}", "shot": tr["shot"], "side": tr["side"], "box_source": tr["box_source"], "person": tr["person"],
        "start": tr["samples"][0]["t"], "end": tr["samples"][-1]["t"],
        "samples": [{"t": s["t"], "box": [round(float(v), 1) for v in s["box"]],
                     "kpts": [[round(float(x), 1), round(float(y), 1), round(float(c), 2)] for x, y, c in s["kpts"]]}
                    for s in tr["samples"]],
    } for i, tr in enumerate(tracks)]
    out = {"model": args.model, "licence_mode": MODELS[args.model]["licence_mode"], "licence": MODELS[args.model]["licence"],
           "keypoints": KEYPOINTS, "skeleton": SKELETON, "fps": pose["fps"], "device": device, "timings": timings,
           "hands": out_hands}
    path = args.run_dir / f"hands_{args.model}.json"
    path.write_text(json.dumps(out, separators=(",", ":")))

    n = lambda pred: sum(len(h["samples"]) for h in out_hands if pred(h))
    print(f"{path.name}: {len(out_hands)} hand tracks, {n(lambda h: True)} hand samples; "
          f"linked {n(lambda h: h['person'] is not None)} (to a face track {n(lambda h: h['person'] is not None and h['person']['face_track'] is not None)}), "
          f"unlinked {n(lambda h: h['person'] is None)}; wrist {n(lambda h: h['box_source'] == 'wrist')}, "
          f"detector {n(lambda h: h['box_source'] == 'detector')}; device {device}")
    print(f"stats: { {k: (round(v, 2) if isinstance(v, float) else v) for k, v in stats.items()} }")
    print(f"timings: {timings}")


if __name__ == "__main__":
    main()
