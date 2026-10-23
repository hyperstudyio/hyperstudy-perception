"""Semantic co-speech gesture stage of the bake-off: GRW's per-word "is a depictive gesture made here" probability.

GRW (Hegde, Prajwal, Zisserman, "Recognizing Co-speech Gestures in-the-Wild", ECCV 2026, arXiv 2605.31589;
github.com/Sindhu-Hegde/grw @ 8d36846) classifies a 4 s query window (100 frames at 25 fps, centred on a word)
inside up to 10 s of motion context (250 frames: the window plus up to 6 s before it) as containing a clear
semantic (iconic / deictic / metaphoric) gesture or not. The classifier never sees the word itself: the word
only places the window. Its input is SHuBERT features (sign-language foundation model) computed from
  - DINOv2-S/14 features (hand-finetuned, GRW's hand_dinov2.pth) of 224x224 left- and right-hand crops,
  - 7 MediaPipe Holistic pose keypoints (nose, shoulders, elbows, wrists) in a shoulder-scaled "signing space",
  - no face (zeros), as in GRW.

What runs per word (words in transcript_parakeet.json whose on-screen speaker can be found):
  1. Speaker: the face track whose mean LR-ASD score over the word is highest and > 0 within the word's shot
     (annotator/build.py's rule); then the pose_rtmw.json body track linked to that face and covering the word.
  2. Person crop from OUR body track instead of GRW's YOLO person stage (Ultralytics, AGPL-3.0, not installed).
     GRW crops twice: inference_preprocess.py takes a square around the union of the track's (10 px padded)
     person boxes and, when the boxes look full-body, trims it to the arms and waist from pose keypoints;
     inference_shubert.py then crops that to the mean person box widened 30% on each side and 1% top/bottom.
     person_crop() does the same arithmetic on the RTMW detector boxes and RTMW keypoints. One fixed crop per
     decoded track range (GRW: one per track, then one per clip).
  3. GRW's own MediaPipe stage, unchanged, inside that crop at 25 fps: Holistic (pose) and HandLandmarker
     (GRW's hand_landmarker.task, 6 hands, detection confidence 0.05); each detected hand is matched to the
     nearer Holistic wrist; hand crops, gap filling and pose normalisation follow inference_shubert.py.
     A word is skipped when Holistic tracked someone else in its clip (HOLISTIC_MIN_AGREE: its nose is checked
     against our body track's RTMW nose), since GRW's crop often holds a neighbour on a panel show.
  4. DINOv2 on the hand crops, SHuBERT on each word's <= 250-frame clip, then GRW's semantic classifier with
     the 100-frame query window; prob = sigmoid of its logit.
Words of one speaker in one body track share the decoded frames, MediaPipe output and DINOv2 features;
SHuBERT and the classifier run once per word because SHuBERT's features depend on the clip they see.

Deviations from the authors' preprocessing (all affect accuracy; see DEVIATIONS for the list written to stdout):
person boxes from RTMW's YOLOX-m at 8 fps interpolated, not YOLO at 25 fps; crops clamped to the frame instead
of grey-padded; no lossy MP4 re-encode of the person and hand crops; query windows are clamped inside the body
track (a cut ends a track), so a word near a cut is not centred and a track shorter than 4 s gets a shorter
query; context is shorter than 6 s when the track starts later.

Events: consecutive transcript words of one speaker with prob >= EVENT_THRESHOLD, at most MERGE_GAP_S apart,
are merged; score = the highest word prob.

Device: --device mps runs DINOv2, SHuBERT and the classifier on Metal; MediaPipe always runs on the CPU.
check_sample.py compares the two devices and checks this pipeline against GRW's bundled sample features.

Weights (GRW's README, thor.robots.ox.ac.uk/grw/checkpoints): semantic_classification.pth,
shubert/hand_dinov2.pth, shubert/hand_landmarker.task, shubert/shubert.pt into vendor/checkpoints/
(yolov8n.pt is not downloaded). Code: git clone https://github.com/Sindhu-Hegde/grw vendor/grw (8d36846) and
https://github.com/facebookresearch/dinov2 vendor/dinov2 (7764ea0) for torch.hub's local load.

Usage: uv run python grw.py ../runs/<name> [--device mps|cpu]
Writes runs/<name>/grw.json.
"""

import argparse
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).parent
BAKEOFF = HERE.parent  # extract.json's video.source is relative to this
VENDOR = HERE / "vendor"
GRW_REPO = VENDOR / "grw"
CHECKPOINTS = VENDOR / "checkpoints"
CLASSIFIER_CKPT = CHECKPOINTS / "semantic_classification.pth"
SHUBERT_CKPT = CHECKPOINTS / "shubert.pt"
DINO_CKPT = CHECKPOINTS / "hand_dinov2.pth"
HAND_TASK = CHECKPOINTS / "hand_landmarker.task"

FPS = 25                 # GRW resamples every video to 25 fps (ffmpeg -r 25); frame k of a run is at k / 25 s
QUERY_FRAMES = 100       # Q: the 4 s query interval (paper 6.2; 4-second window "centered around the word timestamp")
CONTEXT_FRAMES = 150     # T - Q: up to 6 s of motion context before the query (T = 250 frames, 10 s)
EDGE_PAD_S = 0.0625      # a body track is used up to half its 8 fps sample gap beyond its first and last sample, within the shot
EVENT_THRESHOLD = 0.7    # GRW inference.py's default --semantic_thr; the paper's high-confidence point (>= 0.9) is reported too
HIGH_CONFIDENCE = 0.9    # the paper's 93.2% accuracy operating point
MERGE_GAP_S = 0.5        # words further apart than a phrase-level pause (half a second) start a new event

# --- GRW's person-crop arithmetic (preprocess/inference_preprocess.py and preprocess/inference_shubert.py) ---
STAGE1_BOX_PAD = 10      # detect_speaker pads every person box by 10 px
LEG_ASPECT = 1.6         # check_leg_visibility: box height / width above this means legs are visible ...
LEG_FRACTION = 0.8       # ... in at least this fraction of frames
KP_FRACTION = 0.7        # adjust_bbox_kps: arm / hip limits are used only if seen in more than 70% of frames
ARM_PAD_X = 25           # adjust_bbox_kps padding_x (px)
WAIST_PAD_Y = -15        # adjust_bbox_kps padding_y (px): the crop ends 15 px above the mean hip height
SIDE_EXPAND = 0.3        # crop_clip widens the person box by 30% of its width on each side ...
VERTICAL_EXPAND = 0.01   # ... and 1% of its height at the top and bottom
MIN_PERSON_PX = 64       # track_speakers drops tracks whose mean box side is not above 64 px
# RTMW's body keypoints stand in for MediaPipe's in adjust_bbox_kps. MediaPipe visibility > 0.7 there; RTMW's
# raw SimCC scores are on another scale, and pose.py calibrated 3.0 as RTMW's "visible" threshold.
RTMW_VISIBLE = 3.0
# adjust_bbox_kps' LEFT_KP_INDICES are MediaPipe's subject-right joints (on the image left); COCO-17 equivalents:
SUBJECT_RIGHT_KPS = [6, 8, 10, 12]  # right shoulder, elbow, wrist, hip
SUBJECT_LEFT_KPS = [5, 7, 9, 11]
HIPS = (11, 12)
RTMW_NOSE, RTMW_SHOULDERS = 0, (5, 6)

# --- GRW's MediaPipe / hand-crop settings (preprocess/inference_shubert.py) ---
HOLISTIC_MIN_DETECTION = 0.1
HAND_NUM = 6
HAND_MIN_DETECTION = 0.05
HAND_WRIST_MATCH = 0.1   # select_hands: a hand belongs to a pose wrist within 0.1 (crop-normalised units)
HAND_BOX_SCALE = 1.5     # get_hand_crops: square around the 21 landmarks, padded x1.5
HAND_PX = 224
POSE_INDICES = [0, 11, 12, 13, 14, 15, 16]  # nose, shoulders, elbows, wrists
MISSING_POSE = -9999.0   # get_body_embeddings' value before the first detected pose
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DINO_BATCH = 64          # hand crops per DINOv2 batch (GRW uses 128 on CUDA; 64 keeps MPS memory modest)
HOLISTIC_AGREE = 0.5     # Holistic's nose within this x RTMW shoulder width of the target's RTMW nose = same person
# GRW's crop holds neighbours (panel shows), and Holistic tracks ONE person: on OOO it locked onto the person
# behind the speaker for whole tracks (body 176: nose and a shoulder on the neighbour in every frame, checked by
# eye). Such a clip's features describe someone else, so a word is skipped (counted) when Holistic is on the
# target in fewer than this fraction of the clip's checkable frames, or when no frame can be checked.
HOLISTIC_MIN_AGREE = 0.5

DEVIATIONS = [
    "person boxes: pose_rtmw.json (RTMW's YOLOX-m detector at 8 fps, linearly interpolated) instead of YOLOv9/YOLOv8n at 25 fps",
    "leg-visibility trim: RTMW keypoints (score > 3.0) instead of MediaPipe Holistic keypoints (visibility > 0.7) on the stage-1 crop",
    "one crop per decoded body-track range (all words of that speaker in that track), not one per 10 s clip",
    "crops are clamped to the frame; GRW pads the frame with grey (110) when the square crop leaves it",
    "person and hand crops are not re-encoded (GRW writes and re-reads MJPG/MP4V videos between stages)",
    "query window clamped inside the body track: a word within 2 s of a cut is off-centre; a track < 4 s gives a shorter query",
    "context is the up-to-6 s before the query within the same body track (GRW: within the source clip)",
]

LICENCE = (
    "GRW code Apache-2.0 (LICENSE, github.com/Sindhu-Hegde/grw); checkpoints carry no separate licence beyond the "
    "repo's; GRW annotations Apache-2.0, 'intended for research purposes', videos are YouTube (MultiVSR, extending "
    "AVSpeech) 'subject to their original terms of use' (huggingface.co/datasets/sindhuhegde/grw); SHuBERT 'primarily "
    "under the MIT license' (README, github.com/ShesterG/SHuBERT; no LICENSE file), pre-trained on YouTube-ASL; "
    "DINOv2 code Apache-2.0, hand finetune data not stated; MediaPipe Apache-2.0. Inputs come from pose_rtmw.json "
    "(academic-only: COCO-WholeBody / Human-Art training data) and LR-ASD. GRW alone would be commercial-review "
    "(YouTube-derived training data); this stage as built is academic-only because of its RTMW inputs."
)


# ----------------------------------------------------------------------------------------------- models

def load_models(device):
    """GRW's three networks: hand DINOv2, SHuBERT, semantic classifier (as in inference_shubert.py / inference.py)."""
    sys.path.insert(0, str(GRW_REPO))
    sys.path.insert(0, str(GRW_REPO / "models/fairseq"))
    from models.fairseq.examples.shubert.models.shubert import SHubertModel, SHubertConfig
    from models.grw_models import Semantic_Classifier

    dino = torch.hub.load(str(VENDOR / "dinov2"), "dinov2_vits14_reg", source="local", pretrained=False)
    teacher = torch.load(DINO_CKPT, map_location="cpu", weights_only=False)["teacher"]
    state = {k.replace("backbone.", ""): v for k, v in teacher.items() if "dino_head" not in k}
    dino.pos_embed = torch.nn.Parameter(torch.zeros(1, 257, 384))  # finetuned at 224 px: 16x16 patches + CLS
    dino.load_state_dict(state, strict=True)

    shubert = SHubertModel(SHubertConfig())
    ckpt = torch.load(SHUBERT_CKPT, map_location="cpu", weights_only=False)
    result = shubert.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    # GRW loads with strict=False for training-only modules; every module the feature path uses must be loaded.
    missing = [k for k in result.missing_keys if not k.startswith("heads.")]
    assert not missing, f"SHuBERT checkpoint lacks feature-path weights: {missing}"

    classifier = Semantic_Classifier(input_dim=768, num_classes=1, use_layer_weights=True)
    state = torch.load(CLASSIFIER_CKPT, map_location="cpu", weights_only=True)["state_dict"]
    classifier.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
    return dino.to(device).eval(), shubert.to(device).eval(), classifier.to(device).eval()


def mediapipe_models():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    options = vision.HandLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(HAND_TASK)),
                                           num_hands=HAND_NUM, min_hand_detection_confidence=HAND_MIN_DETECTION)
    hands = vision.HandLandmarker.create_from_options(options)
    return mp, hands


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- GRW's feature path

def hand_box(landmarks, shape):
    """inference_shubert.get_bounding_box + adjust_bounding_box: padded square around normalised landmarks."""
    ih, iw = shape[:2]
    px = np.array([(int(l[0] * iw), int(l[1] * ih)) for l in landmarks])
    cx, cy = np.mean(px, axis=0, dtype=int)
    _, _, wb, hb = cv2.boundingRect(px)
    size = max(wb, hb)
    x, y, w, h = cx - size // 2, cy - size // 2, size, size
    pad_w, pad_h = int((HAND_BOX_SCALE - 1) * w / 2), int((HAND_BOX_SCALE - 1) * h / 2)
    x, y, w, h = x - pad_w, y - pad_h, w + 2 * pad_w, h + 2 * pad_h
    if x + w > iw:
        x = iw - w
    if y + h > ih:
        y = ih - h
    return max(x, 0), max(y, 0), w, h


def select_hands(pose, hands):
    """inference_shubert.select_hands: the detected hand whose wrist is nearest each pose wrist, if within 0.1."""
    if hands is None:
        return None, None
    chosen = []
    for wrist in (pose[15], pose[16]):  # MediaPipe pose 15 = subject's left wrist, 16 = right
        dists = [np.linalg.norm(np.array(wrist[:2]) - np.array(h[0][:2])) for h in hands]
        best = int(np.argmin(dists))
        chosen.append(hands[best] if dists[best] < HAND_WRIST_MATCH else None)
    return chosen[0], chosen[1]


def signing_space(pose):
    """inference_shubert.normalize_pose_keypoints, then the 7 GRW joints flattened to 14 values."""
    pose = np.array(pose)[:25, :2]
    head_unit = np.linalg.norm(pose[12] - pose[11]) / 2
    width, height = 6 * head_unit, 7 * head_unit
    top = pose[2][1] - 0.5 * head_unit
    left = pose[0][0] - width / 2
    norm = np.stack([(pose[:, 0] - left) / width - 0.5, (pose[:, 1] - top) / height - 0.5], axis=1)
    return norm[POSE_INDICES].flatten()


class TrackFeatures:
    """Runs GRW's MediaPipe stage frame by frame on one person crop and keeps hand crops + pose vectors.

    Gap filling follows get_hand_crops / get_body_embeddings: a frame without a pose reuses the previous frame's
    crops and pose; a hand that is not found reuses its previous crop; before the first find a hand is black and
    the pose is MISSING_POSE.
    """

    def __init__(self, mp, hand_detector):
        self.mp = mp
        self.hand_detector = hand_detector
        self.holistic = mp.solutions.holistic.Holistic(min_detection_confidence=HOLISTIC_MIN_DETECTION)
        self.left, self.right, self.pose = [], [], []
        self.nose_px = []  # Holistic nose in crop pixels (None when no pose), for the identity diagnostic
        self.counts = {"frames": 0, "pose": 0, "left": 0, "right": 0}
        self.black = np.zeros((HAND_PX, HAND_PX, 3), dtype=np.uint8)

    def add(self, rgb):
        self.counts["frames"] += 1
        result = self.holistic.process(rgb)
        detected = self.hand_detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        hands = [[[l.x, l.y, l.z] for l in h] for h in detected.hand_landmarks] or None
        prev_left = self.left[-1] if self.left else self.black
        prev_right = self.right[-1] if self.right else self.black
        if result.pose_landmarks is None:
            self.left.append(prev_left)
            self.right.append(prev_right)
            self.pose.append(self.pose[-1] if self.pose else np.full(14, MISSING_POSE))
            self.nose_px.append(None)
            return
        pose = [[l.x, l.y, l.z] for l in result.pose_landmarks.landmark]
        self.counts["pose"] += 1
        left, right = select_hands(pose, hands)
        for side, lms, prev, out in (("left", left, prev_left, self.left), ("right", right, prev_right, self.right)):
            if lms is None:
                out.append(prev)
                continue
            x, y, w, h = hand_box(lms, rgb.shape)
            crop = rgb[y:y + h, x:x + w]
            out.append(cv2.resize(crop, (HAND_PX, HAND_PX), interpolation=cv2.INTER_AREA) if crop.size else prev)
            self.counts[side] += 1
        self.pose.append(signing_space(pose))
        self.nose_px.append((pose[0][0] * rgb.shape[1], pose[0][1] * rgb.shape[0]))

    def close(self):
        self.holistic.close()


@torch.no_grad()
def dino_features(dino, crops, device):
    feats = []
    for i in range(0, len(crops), DINO_BATCH):
        batch = (np.stack(crops[i:i + DINO_BATCH]).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        feats.append(dino(torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)).cpu())
    return torch.cat(feats).float()


@torch.no_grad()
def shubert_features(shubert, left, right, body, device):
    """inference_shubert.get_shubert_feats: all 12 layer outputs, returned as (T, 12, 768) for the classifier."""
    length = left.shape[0]
    zeros = torch.zeros((length, 1), device=device)
    source = [{"face": torch.zeros_like(left).to(device), "left_hand": left.to(device), "right_hand": right.to(device),
               "body_posture": body.to(device), "label_face": zeros, "label_left_hand": zeros,
               "label_right_hand": zeros, "label_body_posture": zeros}]
    result = shubert.extract_features(source, padding_mask=None, kmeans_labels=None, mask=False)
    layers = torch.stack([layer[-1].squeeze(1) for layer in result["layer_results"]])  # (12, T, 768)
    return layers.permute(1, 0, 2)


@torch.no_grad()
def classify(classifier, feats, query, device):
    """inference.py's semantic classification on one clip: sigmoid of the logit for query = [start, end] (inclusive)."""
    x = feats.unsqueeze(0).to(device)
    mask = torch.ones((1, 1, feats.shape[0]), device=device)
    logit, _ = classifier(x, target_intervals=torch.tensor([list(query)]), x_mask=mask)
    return float(torch.sigmoid(logit).cpu()[0, 0])


# ----------------------------------------------------------------------------------------------- our tracks

def shot_of(shots, t):
    for s in shots:
        if s["start"] <= t < s["end"]:
            return s
    return None


def mean_asd(track, start, end):
    """annotator/build.py mean_score: mean LR-ASD score of a face track over [start, end]."""
    i0, i1 = int((start - track["start"]) * track["fps"]), int((end - track["start"]) * track["fps"]) + 1
    window = track["scores"][max(0, i0):max(0, i1)]
    return sum(window) / len(window) if window else None


def track_span(track, shot):
    """25 fps frames a body track covers: its samples plus EDGE_PAD_S, inside its shot."""
    first = max(track["start"] - EDGE_PAD_S, shot["start"])
    last = min(track["end"] + EDGE_PAD_S, shot["end"] - 1 / FPS)
    return math.ceil(first * FPS), math.floor(last * FPS)


def interp_samples(track, frames, key):
    """Linear interpolation of a per-sample array field (box or kpts) to 25 fps frame indices (clamped at the ends)."""
    t = np.array([s["t"] for s in track["samples"]])
    values = np.array([s[key] for s in track["samples"]], dtype=np.float64)
    flat = values.reshape(len(t), -1)
    out = np.stack([np.interp(frames / FPS, t, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return out.reshape((len(frames),) + values.shape[1:])


def person_crop(track, f0, f1, width, height):
    """GRW's two-stage person crop (see module docstring) from RTMW boxes and keypoints; returns x0, y0, x1, y1 ints."""
    samples = [s for s in track["samples"] if f0 / FPS - EDGE_PAD_S <= s["t"] <= f1 / FPS + EDGE_PAD_S]
    boxes = np.array([s["box"] for s in samples], dtype=np.float64)  # x, y, w, h
    xyxy = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]], axis=1)
    # Stage 1 (inference_preprocess.detect_speaker + crop_video): padded boxes, square around their union.
    pad = np.clip(xyxy + [-STAGE1_BOX_PAD, -STAGE1_BOX_PAD, STAGE1_BOX_PAD, STAGE1_BOX_PAD], 0, [width, height, width, height])
    ux0, uy0, ux1, uy1 = pad[:, 0].min(), pad[:, 1].min(), pad[:, 2].max(), pad[:, 3].max()
    half = max(ux1 - ux0, uy1 - uy0) / 2
    cx, cy = (ux0 + ux1) / 2, (uy0 + uy1) / 2
    s1 = [max(0, cx - half), max(0, cy - half), min(width, cx + half), min(height, cy + half)]
    aspect = (pad[:, 3] - pad[:, 1]) / np.maximum(pad[:, 2] - pad[:, 0], 1)
    legs = np.mean(aspect > LEG_ASPECT) >= LEG_FRACTION
    if legs:  # adjust_bbox_kps: trim to the arms' extent and to just above the hips
        kpts = np.array([s["kpts"] for s in samples], dtype=np.float64)  # (N, 133, 3)
        lefts, rights, waists = [], [], []
        for k in kpts:
            r = k[SUBJECT_RIGHT_KPS]
            r = r[r[:, 2] > RTMW_VISIBLE]
            l = k[SUBJECT_LEFT_KPS]
            l = l[l[:, 2] > RTMW_VISIBLE]
            if len(r):
                lefts.append(r[:, 0].min())
            if len(l):
                rights.append(l[:, 0].max())
            if k[HIPS[0], 2] > RTMW_VISIBLE and k[HIPS[1], 2] > RTMW_VISIBLE:
                waists.append((k[HIPS[0], 1] + k[HIPS[1], 1]) / 2)
        n = len(kpts)
        if len(lefts) / n > KP_FRACTION:
            s1[0] = max(s1[0], min(lefts) - ARM_PAD_X)
        if len(rights) / n > KP_FRACTION:
            s1[2] = min(s1[2], max(rights) + ARM_PAD_X)
        if len(waists) / n > KP_FRACTION:
            s1[3] = min(s1[3], np.mean(waists) + WAIST_PAD_Y)
    # Stage 2 (inference_shubert.crop_clip): mean person box inside stage 1, widened, clamped to stage 1.
    inner = np.stack([np.maximum(xyxy[:, 0], s1[0]), np.maximum(xyxy[:, 1], s1[1]),
                      np.minimum(xyxy[:, 2], s1[2]), np.minimum(xyxy[:, 3], s1[3])], axis=1)
    w, h = inner[:, 2] - inner[:, 0] + 1, inner[:, 3] - inner[:, 1] + 1
    grown = np.stack([inner[:, 0] - w * SIDE_EXPAND, inner[:, 1] - h * VERTICAL_EXPAND,
                      inner[:, 2] + w * SIDE_EXPAND, inner[:, 3] + h * VERTICAL_EXPAND], axis=1).mean(axis=0)
    # Even coordinates: ffmpeg's crop on yuv420p video silently rounds odd sizes down (867 rows -> 866), which
    # would misalign every raw frame read from the pipe.
    x0, y0 = 2 * math.ceil(max(grown[0], s1[0]) / 2), 2 * math.ceil(max(grown[1], s1[1]) / 2)
    x1, y1 = 2 * math.floor(min(grown[2], s1[2]) / 2), 2 * math.floor(min(grown[3], s1[3]) / 2)
    return (x0, y0, x1, y1), bool(legs), float(np.mean(np.maximum(boxes[:, 2], boxes[:, 3])))


def decode_crop(video, f0, f1, box):
    """Frames f0..f1 (25 fps grid) of the crop box, RGB, streamed from ffmpeg (only this range is decoded)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{f0 / FPS:.4f}", "-i", str(video), "-vf", f"fps={FPS},crop={w}:{h}:{x0}:{y0}",
           "-frames:v", str(f1 - f0 + 1), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    for _ in range(f1 - f0 + 1):
        buf = proc.stdout.read(size)
        if len(buf) < size:
            proc.kill()
            raise SystemExit(f"ffmpeg returned fewer frames than frames {f0}-{f1} of {video}")
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.stdout.close()
    proc.wait()


def assign_words(transcript, shots, asd, pose):
    """Words with an on-screen speaker and a body track; returns (jobs by body track, skip counts)."""
    jobs, skipped = {}, {"no_shot": 0, "no_speaking_face": 0, "no_body_track": 0}
    words = [w for s in transcript["segments"] for w in s["words"]]
    for order, w in enumerate(words):
        shot = shot_of(shots, (w["start"] + w["end"]) / 2)
        if shot is None:
            skipped["no_shot"] += 1
            continue
        scored = [(mean_asd(t, w["start"], w["end"]), t["track"]) for t in asd["tracks"] if t["shot"] == shot["index"]]
        scored = sorted(((s, i) for s, i in scored if s is not None), reverse=True)
        if not scored or scored[0][0] <= 0:
            skipped["no_speaking_face"] += 1
            continue
        face = scored[0][1]
        bodies = [i for i, t in enumerate(pose["tracks"]) if t["face_track"] == face and t["shot"] == shot["index"]
                  and t["start"] - EDGE_PAD_S <= w["start"] and w["end"] <= t["end"] + EDGE_PAD_S]
        if not bodies:
            skipped["no_body_track"] += 1
            continue
        jobs.setdefault(bodies[0], {"shot": shot, "face": face, "words": []})["words"].append({**w, "order": order})
    return jobs, skipped


def word_windows(word, span):
    """(clip_start, clip_end, query_start, query_end) in absolute 25 fps frames for one word."""
    s0, s1 = span
    mid = round((word["start"] + word["end"]) / 2 * FPS)
    q0 = mid - QUERY_FRAMES // 2
    if s1 - s0 + 1 >= QUERY_FRAMES:
        q0 = min(max(q0, s0), s1 - QUERY_FRAMES + 1)
        q1 = q0 + QUERY_FRAMES - 1
    else:
        q0, q1 = s0, s1
    return max(s0, q0 - CONTEXT_FRAMES), q1, q0, q1


def events_from(words, threshold=EVENT_THRESHOLD):
    """Merge consecutive transcript words (by "order") of one speaker at or above threshold."""
    events, current = [], None
    for w in sorted(words, key=lambda w: w["order"]):
        hit = w["prob"] >= threshold
        joins = (current is not None and hit and w["face_track"] == current["face_track"]
                 and w["order"] == current["last_order"] + 1 and w["start"] - current["end"] <= MERGE_GAP_S)
        if joins:
            current["end"] = w["end"]
            current["words"].append(w["word"])
            current["score"] = max(current["score"], w["prob"])
            current["last_order"] = w["order"]
            continue
        if current is not None:
            events.append(current)
        current = ({"kind": "semantic_gesture", "face_track": w["face_track"], "start": w["start"], "end": w["end"],
                    "words": [w["word"]], "score": w["prob"], "last_order": w["order"]} if hit else None)
    if current is not None:
        events.append(current)
    return [{k: v for k, v in e.items() if k != "last_order"} for e in events]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    args = parser.parse_args()
    run = args.run
    asd_path = run / "asd_lrasd.json"
    if not asd_path.exists():
        raise SystemExit(f"{asd_path} does not exist: without LR-ASD scores no word has an on-screen speaker "
                         "(run asd/asd_scores.py --model lrasd first)")
    extract = json.loads((run / "extract.json").read_text())
    transcript = json.loads((run / "transcript_parakeet.json").read_text())
    asd = json.loads(asd_path.read_text())
    pose = json.loads((run / "pose_rtmw.json").read_text())
    video_info = extract["video"]
    video = BAKEOFF / video_info["source"]
    width, height = video_info["width"], video_info["height"]

    device = torch.device(args.device)
    dino, shubert, classifier = load_models(device)
    mp, hand_detector = mediapipe_models()
    print("deviations from GRW's preprocessing:\n  - " + "\n  - ".join(DEVIATIONS))

    jobs, skipped = assign_words(transcript, extract["shots"], asd, pose)
    n_words = sum(len(j["words"]) for j in jobs.values())
    print(f"{n_words} words to score in {len(jobs)} body tracks; skipped {skipped}")

    decode_s = model_s = 0.0
    start_all = time.perf_counter()
    out_words, short_queries, small_tracks, agree_all, off_target = [], 0, 0, [], 0
    for body_id, job in sorted(jobs.items()):
        track = pose["tracks"][body_id]
        span = track_span(track, job["shot"])
        windows = [word_windows(w, span) for w in job["words"]]
        f0, f1 = min(w[0] for w in windows), max(w[1] for w in windows)
        box, legs, mean_side = person_crop(track, f0, f1, width, height)
        if mean_side <= MIN_PERSON_PX:
            small_tracks += 1
            print(f"  body {body_id}: mean person box {mean_side:.0f} px <= {MIN_PERSON_PX}: GRW drops such tracks; "
                  f"{len(job['words'])} words skipped")
            continue
        feats = TrackFeatures(mp, hand_detector)
        frames = decode_crop(video, f0, f1, box)
        while True:
            t0 = time.perf_counter()
            rgb = next(frames, None)
            decode_s += time.perf_counter() - t0
            if rgb is None:
                break
            t0 = time.perf_counter()
            feats.add(np.ascontiguousarray(rgb))
            model_s += time.perf_counter() - t0
        feats.close()
        # Identity diagnostic: is Holistic's person the target body (RTMW nose, interpolated)?
        idx = np.arange(f0, f1 + 1)
        kp = interp_samples(track, idx, "kpts")
        on_target = []  # per frame: True / False, or None when it cannot be checked
        for k, nose in zip(kp, feats.nose_px):
            if nose is None or k[RTMW_NOSE, 2] <= RTMW_VISIBLE:
                on_target.append(None)
                continue
            shoulder = np.linalg.norm(k[RTMW_SHOULDERS[0], :2] - k[RTMW_SHOULDERS[1], :2])
            on_target.append(bool(np.linalg.norm(np.array(nose) + box[:2] - k[RTMW_NOSE, :2]) < HOLISTIC_AGREE * max(shoulder, 1)))
        agree = [a for a in on_target if a is not None]
        agree_all.extend(agree)

        t0 = time.perf_counter()
        left = dino_features(dino, feats.left, device)
        right = dino_features(dino, feats.right, device)
        body = torch.from_numpy(np.stack(feats.pose)).float()
        for w, (c0, c1, q0, q1) in zip(job["words"], windows):
            sl = slice(c0 - f0, c1 - f0 + 1)
            checked = [a for a in on_target[sl] if a is not None]
            if not checked or sum(checked) / len(checked) < HOLISTIC_MIN_AGREE:
                off_target += 1
                continue
            short_queries += (q1 - q0 + 1) < QUERY_FRAMES
            clip = shubert_features(shubert, left[sl], right[sl], body[sl], device)
            prob = classify(classifier, clip, (q0 - c0, q1 - c0), device)
            out_words.append({"word": w["word"], "start": w["start"], "end": w["end"], "face_track": job["face"],
                              "prob": round(prob, 4), "order": w["order"]})
        model_s += time.perf_counter() - t0
        c = feats.counts
        print(f"  body {body_id} (face {job['face']}, shot {job['shot']['index']}): frames {f0}-{f1}, crop {box}, "
              f"legs {legs}, pose {c['pose']}/{c['frames']}, hands L {c['left']} R {c['right']}, "
              f"holistic on target {sum(agree)}/{len(agree)}, {len(job['words'])} words")

    total_s = time.perf_counter() - start_all
    duration = video_info["duration"]
    events = events_from(out_words)
    words_sorted = sorted(out_words, key=lambda w: w["order"])
    out = {
        "model": "grw semantic_classification (SHuBERT hand+body features; GRW 8d36846)",
        "checkpoint": (f"semantic_classification.pth sha256:{sha256(CLASSIFIER_CKPT)[:12]}; shubert.pt sha256:"
                       f"{sha256(SHUBERT_CKPT)[:12]}; hand_dinov2.pth sha256:{sha256(DINO_CKPT)[:12]}; "
                       f"hand_landmarker.task sha256:{sha256(HAND_TASK)[:12]}"),
        "licence_mode": "academic-only",
        "licence": LICENCE,
        "fps": FPS,
        "device": f"{args.device} (DINOv2, SHuBERT, classifier); MediaPipe Holistic + HandLandmarker on CPU",
        "timings": {"decode_s": round(decode_s, 2), "model_s": round(model_s, 2), "total_s": round(total_s, 2),
                    "realtime_factor": round(total_s / duration, 3)},
        "words": [{k: v for k, v in w.items() if k != "order"} for w in words_sorted],
        "events": events,
    }
    (run / "grw.json").write_text(json.dumps(out, indent=1))
    probs = np.array([w["prob"] for w in out_words])
    print(f"wrote {run / 'grw.json'}: {len(out_words)} words, {len(events)} events (>= {EVENT_THRESHOLD}); "
          f"words >= {EVENT_THRESHOLD}: {int((probs >= EVENT_THRESHOLD).sum())}, >= {HIGH_CONFIDENCE}: "
          f"{int((probs >= HIGH_CONFIDENCE).sum())}; short queries {short_queries}; small tracks {small_tracks}; "
          f"holistic on target {sum(agree_all)}/{len(agree_all)} frames; words skipped as Holistic off target {off_target}")
    print(f"timings {out['timings']}; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
