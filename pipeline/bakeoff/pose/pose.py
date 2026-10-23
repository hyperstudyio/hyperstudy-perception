"""Body pose stage of the bake-off: 2D skeletons per person, tracked within shots and linked to face tracks.

Frames are sampled at the same frame indices as faces.py (every round(fps / POSE_FPS)
frames), so a body sample and a face sample share a timestamp. Per frame:
  1. people are detected (a person detector; the pose models are top-down),
  2. keypoints are estimated in each person box; a second detection of the same person
     (same head and shoulder joints) is dropped (drop_duplicate_bodies),
  3. each body is linked to a face track from faces.json: the face box (interpolated
     to the frame time) must sit inside the top part of the body box, AND the body's
     own nose/eye keypoints must fall inside that face box. A body that passes the
     box test but fails the keypoint test is not linked (counted as a failed check).
     Links are one-to-one per frame,
  4. bodies are tracked within the shot (face-linked bodies first, onto their face's
     track; the rest by box IoU); a cut always ends a track. A face track is never
     linked to two body tracks at once (asserted; check_links.py checks the files).

Models (--model):
  rtmw      RTMW-l 384x288 (COCO-WholeBody 133 kp) via rtmlib / ONNX Runtime, person
            boxes from rtmlib's YOLOX-m (trained on Human-Art + COCO).
  vitpose   ViTPose-B simple (COCO 17 kp) via transformers, person boxes from RT-DETRv2-R50 (COCO).
  sapiens2  Sapiens2 pose 0.4B (308 kp) via transformers, same RT-DETRv2 boxes.
  sam3dbody SAM 3D Body: does not run here; see SAM3DBODY_BLOCKED. The command exits with that reason.
--detector yolox|rtdetr overrides the model's default person detector (the output file then
gets a -<detector> suffix), so pose models can be compared on identical boxes.

Keypoint names and bones for COCO-WholeBody and Sapiens2's 308-point format are read from
Sapiens2's own config (mmpose-style dataset_info): git clone https://github.com/facebookresearch/sapiens2
into pose/vendor/sapiens2 (tested at 7e5bae88).

Keypoint scores are each model's own (not comparable across models). The output adds "score_threshold",
the per-model score above which a joint is treated as visible (used for the face-link check; drawing
should use it too, since RTMW's raw SimCC scores make a fixed 0.3 draw every joint, including off-frame legs).

Timings exclude model loading and one warm-up frame. realtime_factor = total_s / video duration
(below 1 is faster than real time), the same convention as audio_events.py's RTF.

Usage: uv run python pose.py ../runs/<name> --model rtmw|vitpose|sapiens2 [--device mps|cpu] [--detector yolox|rtdetr]
Writes runs/<name>/pose_<model>.json.
"""

import argparse
import json
import resource
import runpy
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
BAKEOFF = HERE.parent  # extract.json's video.source is relative to this
KEYPOINT_CONFIG = HERE / "vendor/sapiens2/sapiens/pose/configs/_base_/keypoints308.py"

POSE_FPS = 8.0              # matches faces.json's sample rate, so body and face samples share frames
CHUNK_FRAMES = 8            # frames per batch for the torch models
PROGRESS_EVERY = 100        # print progress every this many sampled frames
RTDETR_SCORE = 0.5          # person score for RT-DETRv2; transformers' post-processing example default
YOLOX_SCORE = 0.3           # the score filter rtmlib applies to this ONNX, which has its own NMS (rtmlib's score_thr=0.7 never runs)
DUPLICATE_HEAD_DIST = 0.1   # two bodies are one person if their head/shoulder joints lie this close (x box scale); see drop_duplicate_bodies
HEAD_JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder")
TRACK_IOU = 0.3             # box overlap with a track's last sample to continue it (same rule as faces.py)
MAX_TRACK_GAP_S = 0.5       # a body unseen for longer than this (4 samples at 8 fps) starts a new track
MAX_FACE_GAP_S = 0.5        # face boxes are interpolated only between face samples at most this far apart
FACE_INSIDE_BODY = 0.7      # fraction of the face box's area that must lie inside the body box
FACE_CENTRE_TOP = 0.5       # the face centre must lie in this top fraction of the body box
LINK_MIN_KPTS = 2           # of nose, left eye, right eye that must fall inside the face box
FACE_BOX_MARGIN = 0.15      # RetinaFace boxes are tight; eyes on profile faces sit on the edge, so the box is grown by this much per side
NOSE_EYES = (0, 1, 2)       # nose, left_eye, right_eye are indices 0-2 in COCO-17, COCO-WholeBody and Sapiens2-308
# RTMW-l on CoreML, one 384x288 crop, M5 Max: CPU EP 41.8 ms, CoreML NeuralNetwork 18.9 ms, MLProgram 10.7 ms
# (405 of 456 nodes on CoreML, the rest on CPU).
COREML_OPTIONS = {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}
# rtmlib's YOLOX ONNX ends in NMS with data-dependent output shapes. ONNX Runtime 1.31's CoreML EP fails on it:
# NeuralNetwork format "static output shape ({1,1,1,8400,8400}) and inferred shape ({1,8400}) have different ranks";
# MLProgram "Input has a dynamic shape ({-1}) but the runtime shape ({0}) has zero elements" (any frame with no
# person). So YOLOX always runs on the CPU EP (67 ms per 640x640 frame); the output's device is the pose model's.

# Sapiens2-0.4B on MPS, 1024x768 crops, M5 Max (random input, warm): fp32 469 / 545 / 685 / 1270 ms per crop at
# 1 / 2 / 4 / 8 crops per pass (driver memory 3.2 / 5.5 / 8.5 / 14.9 GiB). Bigger passes are SLOWER per crop, and
# a whole frame chunk in one pass (~30 crops) grew the process to 125 GB and swapped. fp16: 232 / 228 / 308 ms.
# fp16 vs fp32 on 44 real OOO crops: keypoint shift median 0.0000, p95 0.0001 of box size, one outlier joint
# (a two-peak heatmap); scores differ by <= 0.001 at p95. So fp16, 2 crops per pass.
SAPIENS2_PASS = {"crops_per_pass": 2, "dtype": "float16"}

SAM3DBODY_BLOCKED = (
    "SAM 3D Body cannot run on this machine: (1) its checkpoints are gated on Hugging Face and the "
    "HF_TOKEN account is not on the authorized list ('Access to model facebook/sam-3d-body-dinov3 is "
    "restricted and you are not in the authorized list', same for -vith; access must be requested at "
    "https://huggingface.co/facebook/sam-3d-body-dinov3); (2) its inference code hard-codes CUDA "
    "(sam_3d_body_estimator.py:160 recursive_to(batch, 'cuda'); models/meta_arch/sam3d_body.py:1034, "
    "1246, 1282, 1289, 1292, 1508, 1512 .cuda() / 'cuda'), so it would need patching before MPS could "
    "even be attempted; its demo detector is Detectron2 ViTDet.")

MODELS = {
    "rtmw": {
        "weights": "rtmw-dw-x-l_simcc-cocktail14_270e-384x288 (RTMW-l 384x288, rtmlib 'performance' mode)",
        "pose_url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
        "format": "coco_wholebody", "detector": "yolox",
        # RTMW scores are raw SimCC peaks (0-12 on OOO), not probabilities. Against ViTPose on 2786 paired OOO bodies,
        # >= 3 keeps 98% of joints ViTPose scores > 0.5 and 4% of those it scores < 0.2; >= 0.3 keeps every joint.
        "score_threshold": 3.0,
        "licence_mode": "academic-only",
        "licence": ("rtmlib code and RTMW weights Apache-2.0, but RTMW is trained on Cocktail14, which includes "
                    "COCO-WholeBody (annotations research/non-commercial only) and Human-Art (non-commercial)"),
    },
    "vitpose": {
        "weights": "usyd-community/vitpose-base-simple", "format": "coco17", "detector": "rtdetr",
        "score_threshold": 0.3,  # heatmap peak in [0, 1]; the annotator's default draw threshold
        "crops_per_pass": 64, "dtype": "float32",  # 256x192 crops: a frame chunk's crops fit in one pass
        "licence_mode": "commercial-ok",
        "licence": "ViTPose code and weights Apache-2.0, trained on COCO only (MAE ImageNet-1K initialisation)",
    },
    "sapiens2": {
        "weights": "facebook/sapiens2-pose-0.4b", "format": "sapiens308", "detector": "rtdetr",
        "score_threshold": 0.3,
        **SAPIENS2_PASS,
        "licence_mode": "gated-review",
        "licence": ("Sapiens2 License: commercial use allowed, but prohibits biometric processing and identifying or "
                    "re-identifying individuals; linking skeletons to ArcFace identities needs legal review"),
    },
}

DETECTORS = {
    "yolox": {"weights": "yolox_m_8xb8-300e_humanart (rtmlib)",
              "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
              "licence": "YOLOX-m person detector trained on Human-Art + COCO (Human-Art: non-commercial)"},
    "rtdetr": {"weights": "PekingU/rtdetr_v2_r50vd", "licence": "RT-DETRv2-R50 person detector, Apache-2.0, trained on COCO"},
}


# ---------------------------------------------------------------- keypoint formats

def keypoint_format(name):
    """Names, bones and groups for one format, from Sapiens2's mmpose-style dataset_info."""
    config = runpy.run_path(str(KEYPOINT_CONFIG))
    if name == "sapiens308":
        info = config["dataset_info"]
        names = [info["keypoint_info"][i]["name"] for i in range(len(info["keypoint_info"]))]
        index = {n: i for i, n in enumerate(names)}
        group = lambda key: sorted(index[n] for n in info[key])
        groups = {"body": sorted(group("body_keypoint_names") + group("extra_keypoint_names")),
                  "feet": group("foot_keypoint_names"), "face": group("face_keypoint_names"),
                  "left_hand": group("left_hand_keypoint_names"), "right_hand": group("right_hand_keypoint_names")}
    else:
        info = config["coco_wholebody_info"]
        names = [info["keypoint_info"][i]["name"] for i in range(len(info["keypoint_info"]))]
        index = {n: i for i, n in enumerate(names)}
        groups = {"body": list(range(17)), "feet": list(range(17, 23)), "face": list(range(23, 91)),
                  "left_hand": list(range(91, 112)), "right_hand": list(range(112, 133))}
        if name == "coco17":
            names = names[:17]
            groups = {"body": list(range(17)), "feet": [], "face": [], "left_hand": [], "right_hand": []}
    skeleton = [[index[a], index[b]] for a, b in (info["skeleton_info"][i]["link"] for i in sorted(info["skeleton_info"]))
                if index[a] < len(names) and index[b] < len(names)]
    assert sum(len(g) for g in groups.values()) == len(names), "groups do not partition the keypoints"
    assert names[:3] == ["nose", "left_eye", "right_eye"], names[:3]
    return names, skeleton, groups


# ---------------------------------------------------------------- detectors and pose models

class RtmlibYolox:
    def __init__(self, device):
        from rtmlib import YOLOX
        # Always the CPU EP: the CoreML EP cannot run this model (see the note above COREML_OPTIONS). The device argument is accepted so both detectors share a signature.
        self.model = YOLOX(DETECTORS["yolox"]["url"], model_input_size=(640, 640), backend="onnxruntime", device="cpu")
        self.providers = self.model.session.get_providers()

    def __call__(self, frames_bgr):
        out = []
        for frame in frames_bgr:  # rtmlib's __call__ drops the scores, so its preprocess/inference are called directly
            image, ratio = self.model.preprocess(frame)
            dets = self.model.inference(image)[0][0]  # 'dets' (N, 5): x1, y1, x2, y2, score, after the model's own NMS
            out.append([((d[:4] / ratio).tolist(), float(d[4])) for d in dets if d[4] > YOLOX_SCORE])
        return out


class RtDetr:
    def __init__(self, device):
        import torch
        from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection
        self.torch, self.device = torch, device
        self.processor = AutoImageProcessor.from_pretrained(DETECTORS["rtdetr"]["weights"])
        self.model = RTDetrV2ForObjectDetection.from_pretrained(DETECTORS["rtdetr"]["weights"]).to(device).eval()
        self.person = self.model.config.label2id["person"]
        self.providers = [device]

    def __call__(self, frames_bgr):
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=[f.shape[:2] for f in rgb], threshold=RTDETR_SCORE)
        return [[(r_box.tolist(), float(score)) for r_box, score, label in zip(r["boxes"].cpu(), r["scores"].cpu(), r["labels"].cpu())
                 if int(label) == self.person] for r in results]


class Rtmw:
    def __init__(self, device):
        import onnxruntime as ort
        from rtmlib import RTMPose
        self.model = RTMPose(MODELS["rtmw"]["pose_url"], model_input_size=(288, 384), backend="onnxruntime", device="cpu")
        if device == "mps":  # rtmlib's device='mps' builds a NeuralNetwork-format CoreML session; MLProgram is ~2x faster (COREML_OPTIONS)
            self.model.session = ort.InferenceSession(self.model.onnx_model, providers=[("CoreMLExecutionProvider", COREML_OPTIONS)])
        self.providers = self.model.session.get_providers()

    def __call__(self, frames_bgr, boxes_per_frame):
        out = []
        for frame, boxes in zip(frames_bgr, boxes_per_frame):
            if not boxes:
                out.append([])
                continue
            kpts, scores = self.model(frame, bboxes=[b for b, _ in boxes])
            out.append([np.concatenate([k, s[:, None]], axis=1) for k, s in zip(kpts, scores)])
        return out


class TransformersPose:
    """ViTPose and Sapiens2 share the transformers top-down interface: images + COCO (x, y, w, h) boxes in, keypoints out.
    Crops are run crops_per_pass at a time (see SAPIENS2_PASS); heatmaps are decoded in float32."""

    def __init__(self, weights, device, crops_per_pass, dtype):
        import torch
        from types import SimpleNamespace
        from transformers import AutoImageProcessor, AutoModelForPoseEstimation, VitPoseForPoseEstimation
        self.torch, self.device, self.crops_per_pass, self.namespace = torch, device, crops_per_pass, SimpleNamespace
        self.dtype = getattr(torch, dtype)
        self.processor = AutoImageProcessor.from_pretrained(weights)
        cls = VitPoseForPoseEstimation if "vitpose" in weights else AutoModelForPoseEstimation
        self.model = cls.from_pretrained(weights, dtype=self.dtype).to(device).eval()
        self.providers = [device]

    def __call__(self, frames_bgr, boxes_per_frame):
        keep = [i for i, b in enumerate(boxes_per_frame) if b]
        out = [[] for _ in frames_bgr]
        if not keep:
            return out
        rgb = [cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB) for i in keep]
        coco = [[[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2), _ in boxes_per_frame[i]] for i in keep]
        pixels = self.processor(images=rgb, boxes=coco, return_tensors="pt")["pixel_values"]
        heatmaps = []
        with self.torch.inference_mode():
            for start in range(0, len(pixels), self.crops_per_pass):
                batch = pixels[start:start + self.crops_per_pass].to(self.device, self.dtype)
                heatmaps.append(self.model(pixel_values=batch).heatmaps.float().cpu())
        outputs = self.namespace(heatmaps=self.torch.cat(heatmaps))
        results = self.processor.post_process_pose_estimation(outputs, boxes=coco)
        for i, people in zip(keep, results):
            out[i] = [np.concatenate([p["keypoints"].cpu().numpy(), p["scores"].cpu().numpy()[:, None]], axis=1) for p in people]
        return out


# ---------------------------------------------------------------- face linking and tracking

def iou(a, b):
    """IoU of two [x, y, w, h] boxes."""
    w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def face_boxes_at(t, shot, face_tracks):
    """{face track id: [x, y, w, h]} for the face tracks of this shot on screen at t (linear interpolation, no extrapolation)."""
    boxes = {}
    for track_id, track in face_tracks.items():
        if track["shot"] != shot or not track["start"] - 1e-6 <= t <= track["end"] + 1e-6:
            continue
        series = track["series"]
        times = [s["t"] for s in series]
        j = int(np.searchsorted(times, t))
        if j < len(times) and abs(times[j] - t) < 1e-3:
            boxes[track_id] = series[j]["box"]
        elif 0 < j < len(times) and times[j] - times[j - 1] <= MAX_FACE_GAP_S:
            w = (t - times[j - 1]) / (times[j] - times[j - 1])
            boxes[track_id] = [(1 - w) * a + w * b for a, b in zip(series[j - 1]["box"], series[j]["box"])]
    return boxes


def face_in_body(face, body):
    """The face box lies inside the top part of the body box."""
    fx, fy, fw, fh = face
    bx, by, bw, bh = body
    inside = max(0.0, min(fx + fw, bx + bw) - max(fx, bx)) * max(0.0, min(fy + fh, by + bh) - max(fy, by))
    return inside >= FACE_INSIDE_BODY * fw * fh and by <= fy + fh / 2 <= by + FACE_CENTRE_TOP * bh


def keypoints_in_face(kpts, face, score_threshold, nose_eyes=NOSE_EYES):
    """nose_eyes: indices of nose, left eye, right eye (pose3d.py passes MediaPipe's, which are not 0-2)."""
    fx, fy, fw, fh = face
    mx, my = FACE_BOX_MARGIN * fw, FACE_BOX_MARGIN * fh
    return sum(1 for i in nose_eyes if kpts[i][2] >= score_threshold
               and fx - mx <= kpts[i][0] <= fx + fw + mx and fy - my <= kpts[i][1] <= fy + fh + my)


def drop_duplicate_bodies(bodies, head_idx, score_threshold):
    """Remove second detections of the same person: of two bodies whose head and shoulder joints (visible in both)
    lie within DUPLICATE_HEAD_DIST x the smaller box's sqrt(area), keep the higher detector score.

    Why: both detectors can return two boxes on one person. rtmlib's YOLOX ONNX ends in its own NMS, so rtmlib's
    nms_thr=0.45 never runs; on OOO at 69.002 s it returned Jason as [440, 183, 1542, 1070] (0.74) and
    [487, 233, 1496, 846] (0.50, IoU 0.63, fully inside the first). RT-DETR has no NMS. The duplicate became a second
    body linked to his face. Box IoU cannot tell duplicates from real overlapping people (an IoU 0.45 NMS removed
    the woman standing behind another at 0.8 s), but the joints can: across the six pre-fix outputs, same-frame pairs
    with box IoU >= 0.2 had a median head/shoulder distance below 0.05 (Jason's pair among them) or above 0.16
    (the two women at 0.8 s among them), with 2 of 1004 pairs in between.
    """
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


def link_faces(bodies, faces, score_threshold, stats, nose_eyes=NOSE_EYES):
    """One-to-one body -> face track links for one frame. A face passing the box test whose keypoints disagree is not linked.
    When two bodies pass for one face, the one with more nose/eye keypoints in the face wins, then the higher detector score;
    the other stays unlinked."""
    candidates = []
    for b, body in enumerate(bodies):
        for face_id, face in faces.items():
            if not face_in_body(face, body["box"]):
                continue
            stats["box_candidates"] += 1
            hits = keypoints_in_face(body["kpts"], face, score_threshold, nose_eyes)
            if hits >= LINK_MIN_KPTS:
                candidates.append((hits, body["score"], b, face_id))
            else:
                stats["keypoint_check_failed"] += 1
    used_bodies, used_faces = set(), set()
    for hits, _, b, face_id in sorted(candidates, reverse=True):
        if b in used_bodies or face_id in used_faces:
            continue
        bodies[b]["face_track"] = face_id
        used_bodies.add(b)
        used_faces.add(face_id)
    stats["linked_samples"] += len(used_bodies)
    stats["unlinked_samples"] += len(bodies) - len(used_bodies)


def track_bodies(frames, shots_of_frames):
    """Greedy within-shot tracking. A body joins the open track holding its face link; else the best-IoU track
    whose face link does not conflict. Face-linked bodies are placed first, then the rest largest first (as in
    faces.py). Placing linked bodies first matters: largest-first let an unlinked duplicate box take the track of
    the face its twin was linked to, and the linked twin then opened a second track on the same face."""
    tracks, open_tracks = [], {}
    for (t, bodies), shot in zip(frames, shots_of_frames):
        candidates = [tr for tr in open_tracks.setdefault(shot, []) if t - tr["samples"][-1]["t"] <= MAX_TRACK_GAP_S + 1e-6]
        used = set()
        for body in sorted(bodies, key=lambda b: (b.get("face_track") is None, -b["box"][2] * b["box"][3])):
            face = body.get("face_track")
            free = [tr for tr in candidates if id(tr) not in used]
            same_face = [tr for tr in free if face is not None and tr["face_track"] == face]
            compatible = [tr for tr in free if face is None or tr["face_track"] in (None, face)]
            best = same_face[0] if same_face else max(compatible, key=lambda tr: iou(tr["samples"][-1]["box"], body["box"]), default=None)
            if best is None or (not same_face and iou(best["samples"][-1]["box"], body["box"]) < TRACK_IOU):
                best = {"shot": shot, "face_track": None, "samples": []}
                open_tracks[shot].append(best)
                tracks.append(best)
            if face is not None:
                best["face_track"] = face
            best["samples"].append({"t": t, **body})
            used.add(id(best))
    assert_one_body_per_face(tracks)
    return tracks


def assert_one_body_per_face(tracks):
    """Invariant: tracks linked to the same face track never overlap in time."""
    spans = sorted((tr["face_track"], tr["samples"][0]["t"], tr["samples"][-1]["t"]) for tr in tracks if tr["face_track"] is not None)
    for (face_a, _, end_a), (face_b, start_b, _) in zip(spans, spans[1:]):
        assert face_a != face_b or start_b > end_a, f"face track {face_a} linked to two bodies at once ({start_b} <= {end_a})"


# ---------------------------------------------------------------- main

def sample_frames(video_path, fps, shots, step):
    """(t, shot, BGR frame) for every step-th frame that lies inside a shot, decoded sequentially."""
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
            shot = next((s["index"] for s in shots if s["start"] <= t < s["end"]), None)
            if shot is not None:
                yield t, shot, frame
        index += 1
    capture.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", required=True, choices=[*MODELS, "sam3dbody"])
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--detector", choices=list(DETECTORS))
    parser.add_argument("--fps", type=float, default=POSE_FPS)
    args = parser.parse_args()
    if args.model == "sam3dbody":
        raise SystemExit(SAM3DBODY_BLOCKED)

    cfg = MODELS[args.model]
    detector_name = args.detector or cfg["detector"]
    names, skeleton, groups = keypoint_format(cfg["format"])
    extract = json.loads((args.run_dir / "extract.json").read_text())
    faces = json.loads((args.run_dir / "faces.json").read_text())
    face_tracks = dict(enumerate(faces["tracks"]))  # face track id = index in faces.json
    video = Path(extract["video"]["source"])
    video = video if video.is_absolute() else BAKEOFF / video
    fps, shots, duration = extract["video"]["fps"], extract["shots"], extract["video"]["duration"]
    step = max(1, round(fps / args.fps))

    load_start = time.perf_counter()
    detector = RtmlibYolox(args.device) if detector_name == "yolox" else RtDetr(args.device)
    pose_model = Rtmw(args.device) if args.model == "rtmw" else TransformersPose(cfg["weights"], args.device, cfg["crops_per_pass"], cfg["dtype"])
    print(f"loaded in {time.perf_counter() - load_start:.1f} s; detector providers {detector.providers}, pose providers {pose_model.providers}")
    torch_mps = args.device == "mps" and (args.model != "rtmw" or detector_name == "rtdetr")
    if torch_mps:
        import torch
    sync = (lambda: torch.mps.synchronize()) if torch_mps else (lambda: None)

    first = next(sample_frames(video, fps, shots, step))[2]  # warm-up: first inference compiles kernels / CoreML partitions
    pose_model([first], detector([first]))
    sync()

    timings = {"decode_s": 0.0, "detect_s": 0.0, "pose_s": 0.0}
    head_idx = [names.index(n) for n in HEAD_JOINTS]
    stats = {"duplicates_dropped": 0, "box_candidates": 0, "keypoint_check_failed": 0, "linked_samples": 0, "unlinked_samples": 0}
    peak_mps = 0
    frames_out, frame_shots = [], []
    total_start = time.perf_counter()
    chunk_size = 1 if args.model == "rtmw" and detector_name == "yolox" else CHUNK_FRAMES
    frames = sample_frames(video, fps, shots, step)
    while True:
        tick = time.perf_counter()
        chunk = [x for _, x in zip(range(chunk_size), frames)]
        timings["decode_s"] += time.perf_counter() - tick
        if not chunk:
            break
        images = [f for _, _, f in chunk]
        tick = time.perf_counter()
        boxes = detector(images)
        sync()
        timings["detect_s"] += time.perf_counter() - tick
        tick = time.perf_counter()
        kpts = pose_model(images, boxes)
        sync()
        timings["pose_s"] += time.perf_counter() - tick
        if torch_mps:
            peak_mps = max(peak_mps, torch.mps.driver_allocated_memory())
        if len(frames_out) // PROGRESS_EVERY != (len(frames_out) + len(chunk)) // PROGRESS_EVERY:
            print(f"  t={chunk[-1][0]:.1f} s, {time.perf_counter() - total_start:.0f} s elapsed", flush=True)
        for (t, shot, _), frame_boxes, frame_kpts in zip(chunk, boxes, kpts):
            bodies = [{"box": [x1, y1, x2 - x1, y2 - y1], "score": score, "kpts": k}
                      for ((x1, y1, x2, y2), score), k in zip(frame_boxes, frame_kpts)]
            kept = drop_duplicate_bodies(bodies, head_idx, cfg["score_threshold"])
            stats["duplicates_dropped"] += len(bodies) - len(kept)
            bodies = kept
            link_faces(bodies, face_boxes_at(t, shot, face_tracks), cfg["score_threshold"], stats)
            frames_out.append((t, bodies))
            frame_shots.append(shot)
    tracks = track_bodies(frames_out, frame_shots)
    total_s = time.perf_counter() - total_start

    tracks.sort(key=lambda tr: (tr["shot"], tr["samples"][0]["t"]))
    out_tracks = [{
        "id": f"pose:{i}", "shot": tr["shot"], "face_track": tr["face_track"],
        "link": "face-in-box+keypoints" if tr["face_track"] is not None else None,
        "start": tr["samples"][0]["t"], "end": tr["samples"][-1]["t"],
        "samples": [{"t": s["t"], "box": [round(float(v), 1) for v in s["box"]],
                     "kpts": [[round(float(x), 1), round(float(y), 1), round(float(c), 2)] for x, y, c in s["kpts"]]}
                    for s in tr["samples"]],
    } for i, tr in enumerate(tracks)]

    providers = pose_model.providers  # the pose model's device; the detector's is printed at load
    device = "mps" if "mps" in providers else "coreml" if "CoreMLExecutionProvider" in providers else "cpu"
    weights = f"{cfg['weights']}; person detector {DETECTORS[detector_name]['weights']}"
    licence = f"{cfg['licence']}. {DETECTORS[detector_name]['licence']}."
    licence_mode = "academic-only" if detector_name == "yolox" else cfg["licence_mode"]
    out = {"model": args.model if detector_name == cfg["detector"] else f"{args.model}+{detector_name}",
           "weights": weights, "licence_mode": licence_mode, "licence": licence,
           "keypoints": names, "skeleton": skeleton, "groups": groups, "score_threshold": cfg["score_threshold"],
           "fps": args.fps, "device": device,
           "timings": {"detect_s": round(timings["detect_s"], 2), "pose_s": round(timings["pose_s"], 2),
                       "total_s": round(total_s, 2), "realtime_factor": round(total_s / duration, 3)},
           "tracks": out_tracks}
    suffix = "" if detector_name == cfg["detector"] else f"-{detector_name}"
    path = args.run_dir / f"pose_{args.model}{suffix}.json"
    path.write_text(json.dumps(out, separators=(",", ":")))

    linked = [tr for tr in out_tracks if tr["face_track"] is not None]
    n_samples = sum(len(tr["samples"]) for tr in out_tracks)
    print(f"{path.name}: {len(frames_out)} frames, {n_samples} bodies, {len(out_tracks)} tracks, "
          f"{len(linked)} linked to {len({tr['face_track'] for tr in linked})} of {len(face_tracks)} face tracks; device {device}")
    print(f"link checks: {stats}")
    print(f"timings: decode {timings['decode_s']:.1f} s, {out['timings']}; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GiB"
          + (f", peak MPS driver memory {peak_mps / 2**30:.2f} GiB" if torch_mps else ""))


if __name__ == "__main__":
    main()
