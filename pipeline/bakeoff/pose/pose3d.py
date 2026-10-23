"""3D body pose stage of the bake-off: RTMW3D and MediaPipe Pose Landmarker, plus torso facing direction per body.

Same pipeline as pose.py (imported, not copied): the same sampled frames, rtmlib's YOLOX-m person boxes,
drop_duplicate_bodies, face linking (face box in the top of the body box AND nose/eyes in the face box,
one body per face) and within-shot tracking. The output has pose_rtmw.json's schema, plus:
  "frame_3d"   the 3D frame of "kpts3d" (axes, units, origin), per model,
  "facing_method"  how "facing" is computed,
and per sample:
  "kpts3d"      [[x, y, z], ...] in the model's 3D frame,
  "facing"      [yaw_deg, pitch_deg] of the torso normal relative to the camera, or null; pitch is null when only
                the shoulders were usable,
  "facing_from" "torso" | "shoulders" | null.

Models (--model):
  rtmw3d     RTMW3D-x 384x288 (COCO-WholeBody 133 kp with a depth axis) via rtmlib's RTMPose3d / ONNX Runtime
             (CoreML EP on --device mps: identical output to the CPU EP on 457 OOO crops, max 1.1 px, 17.6 vs
             56.8 ms per crop). rtmlib's z decode is wrong and is redone here (see Rtmw3d).
  mediapipe  MediaPipe Pose Landmarker heavy (BlazePose GHUM 3D, 33 landmarks) in IMAGE mode on a crop around
             each YOLOX box; the CPU (XNNPACK) delegate always (see MediapipePose).

Facing (torso_facing). The camera frame is x right, y down, z away from the camera (both models' 3D frames
are camera-aligned; MediaPipe's world landmarks were checked on OOO: a frontal subject's left shoulder has
x > 0 and the nose z < 0). With a = unit(L shoulder - R shoulder) + unit(L hip - R hip) (the left-right axis)
and u = mid-shoulder - mid-hip (the trunk's up axis), the torso normal is f = a x u, normalised. A subject
facing the camera has a = +x, u = -y and f = -z. Then
    yaw   = atan2(-f_x, -f_z)               0 = facing the camera, + = turned to the subject's right (image
                                             left), as py-feat's head yaw; +-180 = back to the camera
    pitch = atan2(-f_y, hypot(f_x, f_z))    + = torso tilted back (normal pointing up)
When a hip is not visible, a = L shoulder - R shoulder and u = -y (image up), "facing_from" is "shoulders" and
pitch is null (no trunk axis). When a shoulder is not visible, facing is null.

RTMW3D's x and y are image pixels and its z is metres, so x and y are scaled to metres per body first:
s = TORSO_M / (2D mid-shoulder to mid-hip distance in pixels). Shoulders only: s makes the 3D shoulder
length SHOULDER_M (s |dx_px| = sqrt(SHOULDER_M^2 - dz^2)), which amounts to yaw = asin(-dz / SHOULDER_M)
with the front/back sign from the 2D shoulder order. MediaPipe's world landmarks are already metric.

Usage: uv run python pose3d.py ../runs/<name> --model rtmw3d|mediapipe [--device mps|cpu]
Writes runs/<name>/pose_<model>.json.
"""

import argparse
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np

import pose

HERE = Path(__file__).parent
MEDIAPIPE_TASK = HERE / "vendor/pose_landmarker_heavy.task"
# curl -L -o vendor/pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
# (sha256 64437af8...011b1; bundles pose_detector.tflite and pose_landmarks_detector.tflite, built from
# blazepose_ghum_39kp_heavy_oss_2021_07_02, the model the BlazePose GHUM 3D model card describes)

# RTMW3D's SimCC z axis has input_size[2] / 2 = 144 bins per z_range on each side of the root (mmpose
# projects/rtmpose3d codec: input_size=(288, 384, 288), simcc_split_ratio 2, z_range 2.1744869, root = mean of the
# hips 11/12). rtmlib 0.0.16 divides by model_input_size[-1] / 2 = 192 (the crop HEIGHT), so its z is
# (4/3) too small and offset by -z_range/4: on 457 OOO crops its mid-hip z was -0.544, the corrected one 0.000.
RTMW3D_Z_HALF_BINS = 144.0
RTMW3D_Z_RANGE = 2.1744869
# Adult trunk length, glenohumeral joint centre to greater trochanter, is 0.288 x stature (Winter, Biomechanics and
# Motor Control of Human Movement, segment-length figure): 0.49 m at 1.70 m.
TORSO_M = 0.49
# On 136 OOO keyframe bodies RTMW3D placed both shoulders at equal depth (|dz| < 0.03 m), the 2D shoulder width was
# a median 0.64 of the 2D trunk length (IQR 0.59-0.71); 0.64 x TORSO_M.
SHOULDER_M = 0.31

MEDIAPIPE_MARGIN = 1.1   # crop = YOLOX box grown by this factor, the rest of the square crop black (see MediapipePose)
MEDIAPIPE_POSES = 2      # poses asked of the landmarker per crop; the one centred nearest the box centre is kept

# COCO-WholeBody / MediaPipe indices of left shoulder, right shoulder, left hip, right hip.
TORSO_JOINTS = {"coco_wholebody": (5, 6, 11, 12), "mediapipe33": (11, 12, 23, 24)}

MEDIAPIPE_NAMES = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
                   "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
                   "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
                   "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
                   "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"]
MEDIAPIPE_GROUPS = {"body": [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28], "feet": [29, 30, 31, 32],
                    "face": [1, 3, 4, 6, 9, 10], "left_hand": [17, 19, 21], "right_hand": [18, 20, 22]}

MODELS = {
    "rtmw3d": {
        "weights": ("rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626 (RTMW3D-x 384x288): mmpose checkpoint converted to "
                    "ONNX by a third party, https://huggingface.co/Soykaf/RTMW3D-x (rtmlib's Wholebody3d default; "
                    "sha256 4a289c0e...85ebcd)"),
        "url": "https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx",
        "format": "coco_wholebody",
        # Scores are min(x peak, y peak) of the SimCC outputs (rtmlib), 0-1 here, not RTMW's 0-12. Against RTMW >= 3.0 on the same
        # 457 OOO keyframe boxes (17 body joints), Youden's J (kept visible - kept invisible) is 0.80 at 0.4 and 0.81 at 0.5
        # (88% of RTMW-visible joints kept, 7% of RTMW-invisible), 0.70 at 0.3 and 0.6.
        "score_threshold": 0.5,
        "frame_3d": ("x, y: full-frame image pixels (the 2D keypoints); z: metres along the camera's viewing axis relative to the "
                     "mid-hip (+ = farther from the camera), as RTMW3D is trained (H3WB and UBody3D camera-space depth, root "
                     "= mean of the hips). x/y and z are in different units; facing scales x/y per body (facing_method)"),
        "licence_mode": "academic-only",
        "licence": ("rtmlib code Apache-2.0; mmpose (RTMW3D code, config, checkpoint) Apache-2.0; the ONNX conversion's HF card says "
                    "apache-2.0. Training data (mmpose config rtmw3d-x_8xb32_cocktail14-384x288.py): COCO-WholeBody (annotations "
                    "non-commercial), AIC, CrowdPose, MPII, JHMDB, Halpe, PoseTrack18, Human-Art (non-commercial), WFLW, 300W, COFW, "
                    "LaPa, InterHand2.6M (non-commercial), UBody, and H3WB on Human3.6M (academic use only)"),
    },
    "mediapipe": {
        "weights": "pose_landmarker_heavy.task float16 v1 (BlazePose GHUM 3D heavy, blazepose_ghum_39kp_heavy_oss_2021_07_02)",
        "format": "mediapipe33",
        # Scores are MediaPipe's per-landmark visibility (probability). Against RTMW >= 3.0 on the same OOO boxes, Youden's J is
        # 0.74-0.75 for 0.3-0.5 (0.5: 86% of RTMW-visible joints kept, 11% of RTMW-invisible); 0.5 is also MediaPipe's default
        # for its detection and presence confidences.
        "score_threshold": 0.5,
        "frame_3d": ("MediaPipe pose_world_landmarks: metres, origin at the mid-hip, x right, y down, z away from the camera "
                     "(checked: frontal subjects have left shoulder x > 0, nose z < 0). The model card says z comes from GHUM fits "
                     "to 2D annotations and is 'not metric but up to scale'"),
        "licence_mode": "academic-only",
        "licence": ("MediaPipe code Apache-2.0; the BlazePose GHUM 3D model card (Model Card BlazePose GHUM 3D.pdf, the card the "
                    "Pose Landmarker page links for lite/full/heavy) says 'LICENSED UNDER Apache License, Version 2.0'. "
                    "Commercial-ok on its own, but the person boxes here are YOLOX (Human-Art, non-commercial). The card lists "
                    "multiple people in an image, people further than 4 m and 'head is not visible' as out of scope"),
    },
}


# ---------------------------------------------------------------- facing

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def torso_facing(points, visible, joints):
    """(yaw_deg, pitch_deg or None, "torso" | "shoulders") from metric camera-frame points, or (None, None, None).
    See the module docstring for the formula."""
    left, right, left_hip, right_hip = joints
    if not (visible[left] and visible[right]):
        return None, None, None
    shoulders = points[left] - points[right]
    if visible[left_hip] and visible[right_hip]:
        normal = np.cross(unit(shoulders) + unit(points[left_hip] - points[right_hip]),
                          (points[left] + points[right]) / 2 - (points[left_hip] + points[right_hip]) / 2)
        source = "torso"
    else:
        normal = np.cross(shoulders, np.array([0.0, -1.0, 0.0]))
        source = "shoulders"
    if np.linalg.norm(normal) == 0:
        return None, None, None
    f = unit(normal)
    yaw = float(np.degrees(np.arctan2(-f[0], -f[2])))
    pitch = float(np.degrees(np.arctan2(-f[1], np.hypot(f[0], f[2])))) if source == "torso" else None
    return yaw, pitch, source


def rtmw3d_metric(kpts, kpts3d, visible, joints):
    """RTMW3D points in metres (x, y scaled per body; see the module docstring), or None when the scale is undefined."""
    left, right, left_hip, right_hip = joints
    if not (visible[left] and visible[right]):
        return None
    xy = kpts[:, :2]
    if visible[left_hip] and visible[right_hip]:
        trunk_px = np.linalg.norm((xy[left] + xy[right]) / 2 - (xy[left_hip] + xy[right_hip]) / 2)
        if trunk_px == 0:
            return None
        scale = TORSO_M / trunk_px
    else:
        shoulder_px = np.linalg.norm(xy[left] - xy[right])
        if shoulder_px == 0:
            return None
        dz = kpts3d[left, 2] - kpts3d[right, 2]
        scale = np.sqrt(max(SHOULDER_M ** 2 - dz ** 2, 0.0)) / shoulder_px
    return np.c_[xy * scale, kpts3d[:, 2]]


# ---------------------------------------------------------------- models

class Rtmw3d:
    def __init__(self, device):
        import onnxruntime as ort
        from rtmlib import RTMPose3d
        self.model = RTMPose3d(MODELS["rtmw3d"]["url"], model_input_size=(288, 384), backend="onnxruntime", device="cpu")
        if device == "mps":  # MLProgram CoreML, as pose.Rtmw; 510 of 513 nodes on CoreML
            self.model.session = ort.InferenceSession(self.model.onnx_model, providers=[("CoreMLExecutionProvider", pose.COREML_OPTIONS)])
        self.providers = self.model.session.get_providers()

    def __call__(self, frame, boxes):
        """[(kpts (K, 3): x, y, score in frame pixels, kpts3d (K, 3): x, y pixels, z metres)] per box."""
        if not boxes:
            return []
        _, scores, simcc, kpts2d = self.model(frame, bboxes=boxes)
        z = (simcc[..., 2] / RTMW3D_Z_HALF_BINS - 1) * RTMW3D_Z_RANGE
        return [(np.c_[k, s], np.c_[k, zz]) for k, s, zz in zip(kpts2d, scores, z)]


class MediapipePose:
    """MediaPipe Pose Landmarker heavy, IMAGE mode, one crop per YOLOX box.

    The crop is the box grown by MEDIAPIPE_MARGIN, centred in a square whose rest is black. The landmarker is
    single-person (its model card lists multiple people as out of scope) and picks the most prominent person in its
    input, so a square crop around a tall box in a row of standing people (OOO) holds two or three people: on two OOO
    frames a full square crop (1.25 x the box's longer side) posed the wrong person for 2 of 9 boxes. Masking outside
    the box removes the neighbours. On 457 OOO keyframe boxes a pose was found for 351 / 365 / 385 boxes at margin
    1.0 / 1.1 / 1.25, of which 316 / 320 / 322 match RTMW's skeleton on the same box (median joint distance < 0.1 box
    scale): 1.25 adds poses that mostly disagree with RTMW, so 1.1.

    Always the CPU (XNNPACK) delegate, whatever --device says: on the same 457 boxes the GPU (Metal) delegate found a
    pose for 317 boxes vs the CPU's 365, and on boxes both posed, joints differed by a median 0.022 box scale (p90
    0.082) with the left/right shoulder order swapped on 11 of 311 (as hands.py found for the hand landmarker).
    """

    def __init__(self, device):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        self.landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MEDIAPIPE_TASK), delegate=BaseOptions.Delegate.CPU),
            num_poses=MEDIAPIPE_POSES))
        self.providers = ["cpu-xnnpack"]
        self.skeleton = [[c.start, c.end] for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]

    def __call__(self, frame, boxes):
        """[(kpts (33, 3): x, y, visibility in frame pixels, world landmarks (33, 3) in metres) or None] per box."""
        out = []
        h, w = frame.shape[:2]
        for x1, y1, x2, y2 in boxes:
            mx, my = (x2 - x1) * (MEDIAPIPE_MARGIN - 1) / 2, (y2 - y1) * (MEDIAPIPE_MARGIN - 1) / 2
            bx0, by0 = int(max(x1 - mx, 0)), int(max(y1 - my, 0))
            bx1, by1 = int(min(x2 + mx, w)), int(min(y2 + my, h))
            side = max(bx1 - bx0, by1 - by0)
            x0, y0 = (bx0 + bx1) // 2 - side // 2, (by0 + by1) // 2 - side // 2
            crop = np.zeros((side, side, 3), np.uint8)
            crop[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = frame[by0:by1, bx0:bx1]
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            result = self.landmarker.detect(image)
            poses = [(np.array([[p.x * side + x0, p.y * side + y0, p.visibility] for p in lms]),
                      np.array([[p.x, p.y, p.z] for p in world]))
                     for lms, world in zip(result.pose_landmarks, result.pose_world_landmarks)]
            centre = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            out.append(min(poses, key=lambda p: np.linalg.norm(p[0][:, :2].mean(0) - centre)) if poses else None)
        return out


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = parser.parse_args()

    cfg = MODELS[args.model]
    threshold = cfg["score_threshold"]
    if args.model == "rtmw3d":
        names, skeleton, groups = pose.keypoint_format("coco_wholebody")
        nose_eyes = pose.NOSE_EYES
    else:
        names, groups = MEDIAPIPE_NAMES, MEDIAPIPE_GROUPS
        nose_eyes = tuple(names.index(n) for n in ("nose", "left_eye", "right_eye"))
    joints = TORSO_JOINTS[cfg["format"]]
    extract = json.loads((args.run_dir / "extract.json").read_text())
    faces = json.loads((args.run_dir / "faces.json").read_text())
    face_tracks = dict(enumerate(faces["tracks"]))
    video = Path(extract["video"]["source"])
    video = video if video.is_absolute() else pose.BAKEOFF / video
    fps, shots, duration = extract["video"]["fps"], extract["shots"], extract["video"]["duration"]
    step = max(1, round(fps / pose.POSE_FPS))

    load_start = time.perf_counter()
    detector = pose.RtmlibYolox(args.device)
    model = Rtmw3d(args.device) if args.model == "rtmw3d" else MediapipePose(args.device)
    if args.model == "mediapipe":
        skeleton = model.skeleton
    print(f"loaded in {time.perf_counter() - load_start:.1f} s; detector providers {detector.providers}, pose providers {model.providers}")
    first = next(pose.sample_frames(video, fps, shots, step))[2]
    model(first, [b for b, _ in detector([first])[0]])  # warm-up

    head_idx = [names.index(n) for n in pose.HEAD_JOINTS]
    stats = {"no_pose": 0, "duplicates_dropped": 0, "box_candidates": 0, "keypoint_check_failed": 0, "linked_samples": 0, "unlinked_samples": 0}
    facing_counts = {"torso": 0, "shoulders": 0, "none": 0}
    timings = {"decode_s": 0.0, "detect_s": 0.0, "pose_s": 0.0}
    frames_out, frame_shots = [], []
    total_start = time.perf_counter()
    frames = pose.sample_frames(video, fps, shots, step)
    while True:
        tick = time.perf_counter()
        item = next(frames, None)
        timings["decode_s"] += time.perf_counter() - tick
        if item is None:
            break
        t, shot, frame = item
        tick = time.perf_counter()
        boxes = detector([frame])[0]
        timings["detect_s"] += time.perf_counter() - tick
        tick = time.perf_counter()
        results = model(frame, [b for b, _ in boxes])
        timings["pose_s"] += time.perf_counter() - tick
        if len(frames_out) % pose.PROGRESS_EVERY == 0:
            print(f"  t={t:.1f} s, {time.perf_counter() - total_start:.0f} s elapsed", flush=True)
        bodies = []
        for ((x1, y1, x2, y2), score), result in zip(boxes, results):
            if result is None:
                stats["no_pose"] += 1
                continue
            kpts, kpts3d = result
            visible = kpts[:, 2] >= threshold
            points = rtmw3d_metric(kpts, kpts3d, visible, joints) if args.model == "rtmw3d" else kpts3d
            yaw, pitch, source = torso_facing(points, visible, joints) if points is not None else (None, None, None)
            facing_counts[source or "none"] += 1
            bodies.append({"box": [x1, y1, x2 - x1, y2 - y1], "score": score, "kpts": kpts, "kpts3d": kpts3d,
                           "facing": None if yaw is None else [yaw, pitch], "facing_from": source})
        kept = pose.drop_duplicate_bodies(bodies, head_idx, threshold)
        stats["duplicates_dropped"] += len(bodies) - len(kept)
        pose.link_faces(kept, pose.face_boxes_at(t, shot, face_tracks), threshold, stats, nose_eyes)
        frames_out.append((t, kept))
        frame_shots.append(shot)
    tracks = pose.track_bodies(frames_out, frame_shots)
    total_s = time.perf_counter() - total_start

    tracks.sort(key=lambda tr: (tr["shot"], tr["samples"][0]["t"]))
    z_digits = 3
    xy3_digits = 1 if args.model == "rtmw3d" else 3
    out_tracks = [{
        "id": f"pose:{i}", "shot": tr["shot"], "face_track": tr["face_track"],
        "link": "face-in-box+keypoints" if tr["face_track"] is not None else None,
        "start": tr["samples"][0]["t"], "end": tr["samples"][-1]["t"],
        "samples": [{"t": s["t"], "box": [round(float(v), 1) for v in s["box"]],
                     "kpts": [[round(float(x), 1), round(float(y), 1), round(float(c), 2)] for x, y, c in s["kpts"]],
                     "kpts3d": [[round(float(x), xy3_digits), round(float(y), xy3_digits), round(float(z), z_digits)] for x, y, z in s["kpts3d"]],
                     "facing": None if s["facing"] is None else [round(s["facing"][0], 1), None if s["facing"][1] is None else round(s["facing"][1], 1)],
                     "facing_from": s["facing_from"]}
                    for s in tr["samples"]],
    } for i, tr in enumerate(tracks)]

    device = "coreml" if "CoreMLExecutionProvider" in model.providers else model.providers[0] if args.model == "mediapipe" else "cpu"
    out = {"model": args.model, "weights": f"{cfg['weights']}; person detector {pose.DETECTORS['yolox']['weights']}",
           "licence_mode": cfg["licence_mode"], "licence": f"{cfg['licence']}. {pose.DETECTORS['yolox']['licence']}.",
           "keypoints": names, "skeleton": skeleton, "groups": groups, "score_threshold": threshold,
           "fps": pose.POSE_FPS, "device": device, "frame_3d": cfg["frame_3d"],
           "facing_method": ("torso normal f = (unit(Lsh - Rsh) + unit(Lhip - Rhip)) x (mid-shoulder - mid-hip) in a camera frame "
                             "(x right, y down, z away); yaw = atan2(-f_x, -f_z), pitch = atan2(-f_y, hypot(f_x, f_z)), degrees; "
                             "0, 0 = facing the camera, yaw + = turned to the subject's right (image left). Without both hips: "
                             "f = (Lsh - Rsh) x (0, -1, 0), pitch null. Joints used only when score >= score_threshold."
                             + (f" RTMW3D x/y scaled to metres by {TORSO_M} m / 2D trunk length (shoulders only: 3D shoulder width "
                                f"{SHOULDER_M} m)." if args.model == "rtmw3d" else "")),
           "timings": {"detect_s": round(timings["detect_s"], 2), "pose_s": round(timings["pose_s"], 2),
                       "total_s": round(total_s, 2), "realtime_factor": round(total_s / duration, 3)},
           "tracks": out_tracks}
    path = args.run_dir / f"pose_{args.model}.json"
    path.write_text(json.dumps(out, separators=(",", ":")))

    linked = [tr for tr in out_tracks if tr["face_track"] is not None]
    n_samples = sum(len(tr["samples"]) for tr in out_tracks)
    print(f"{path.name}: {len(frames_out)} frames, {n_samples} bodies, {len(out_tracks)} tracks, "
          f"{len(linked)} linked to {len({tr['face_track'] for tr in linked})} of {len(face_tracks)} face tracks; device {device}")
    print(f"link checks: {stats}; facing from (before duplicate removal): {facing_counts}")
    print(f"timings: decode {timings['decode_s']:.1f} s, {out['timings']}; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
