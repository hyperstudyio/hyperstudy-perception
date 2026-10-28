"""The hand models: RTMDet-nano hand finds hands in the full frame, RTMPose-m Hand5 (top-down, 21 keypoints) estimates
each hand's keypoints in a box, both through rtmlib on ONNX Runtime, as the bake-off's hands.py --model rtmpose_hand.
Their weights come from OpenMMLab's server, fetched once into the shared weights cache and checked by sha256
(perception_node.weights); the manifest lists the same files.

Where they run (the request's accelerator):
  mps   RTMPose-hand on CoreML in the NeuralNetwork format, the format rtmlib's device='mps' builds and the bake-off
        ran: one 256x256 crop took 4.2 ms on an M5 Max, against 5.3 ms as an MLProgram and 11.9 ms on the CPU
        (bake-off, hands.py). RTMDet runs on the CPU (about 7 ms a frame): its ONNX ends in NMS, which ONNX Runtime
        1.31's CoreML provider fails on in both formats ("CoreML static output shape ({1,1,1,2100,2100}) and inferred
        shape ({1,2100}) have different ranks"), as YOLOX in body-pose.
  cuda  both on CUDA (onnxruntime-gpu, with CUDA 13 and cuDNN 9 from pip wheels; an NVIDIA driver >= 580).
  cpu   both on the CPU.

A provider ONNX Runtime cannot load (CUDA without its libraries, CoreML off a Mac) is not quietly replaced by the CPU:
the node fails.
"""

from __future__ import annotations

import time

from perception_node import NodeError
from perception_node.weights import fetch

WEIGHTS = {
    "rtmpose_hand": {"url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip",
                     "sha256": "39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29"},
    "rtmdet_hand": {"url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip",
                    "sha256": "568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2"},
}
COREML = ("CoreMLExecutionProvider", {"ModelFormat": "NeuralNetwork", "MLComputeUnits": "ALL"})
CUDA = "CUDAExecutionProvider"
# RTMDet-nano hand scores stay low on this footage (none above 0.4 on OOO apart from one 0.71 on a tiny graphic).
# Contact sheets of 30 random OOO detections per score bin, judged by eye: 0.15-0.2 about 13/30 are hands (the rest
# faces, necklaces, elbows), 0.2-0.25 about 17/30, >= 0.25 about 24/30. rtmlib's default 0.3 drops visible gesturing
# hands that score 0.25-0.3, so 0.25 (bake-off, hands.py).
RTMDET_SCORE = 0.25


def weights(name: str):
    w = WEIGHTS[name]
    return fetch(w["url"], w["sha256"], member="end2end.onnx")


def session(onnx, accelerator: str, use_coreml: bool):
    import onnxruntime as ort
    if accelerator == "cuda":
        ort.preload_dlls()   # the CUDA 13 runtime and cuDNN 9 from their pip wheels (envs/vision), not the system's
    wanted = CUDA if accelerator == "cuda" else COREML if accelerator == "mps" and use_coreml else "CPUExecutionProvider"
    s = ort.InferenceSession(str(onnx), providers=[wanted, "CPUExecutionProvider"] if wanted != "CPUExecutionProvider" else [wanted])
    name = wanted if isinstance(wanted, str) else wanted[0]
    if name not in s.get_providers():
        raise NodeError("resources", f"ONNX Runtime could not use {name} here (it has {', '.join(ort.get_available_providers())}): "
                                     "run Hands on another device")
    return s


class Hands:
    """The two models behind hand_tracks.frame_hands' `detect` and `keypoints`, timing each."""

    def __init__(self, accelerator: str):
        from rtmlib import RTMDet, RTMPose
        self.det = RTMDet(str(weights("rtmdet_hand")), model_input_size=(320, 320), backend="onnxruntime", device="cpu")
        self.det.session = session(self.det.onnx_model, accelerator, use_coreml=False)
        self.pose = RTMPose(str(weights("rtmpose_hand")), model_input_size=(256, 256), backend="onnxruntime", device="cpu")
        self.pose.session = session(self.pose.onnx_model, accelerator, use_coreml=True)
        self.providers = self.pose.session.get_providers()
        self.detect_s = self.pose_s = 0.0

    def detect(self, frame):
        """[(box [x, y, w, h], score)], highest score first. rtmlib's RTMDet drops the scores, so its preprocess and
        inference are called directly; its ONNX ends in its own NMS."""
        tick = time.perf_counter()
        image, ratio = self.det.preprocess(frame)
        dets = self.det.inference(image)[0][0]   # (N, 5): x1, y1, x2, y2, score
        out = []
        for x1, y1, x2, y2, score in dets:
            if score >= RTMDET_SCORE:
                x1, y1, x2, y2 = (v / ratio for v in (x1, y1, x2, y2))
                out.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(score)))
        self.detect_s += time.perf_counter() - tick
        return sorted(out, key=lambda d: -d[1])

    def keypoints(self, frame, box):
        """A (21, 3) array of x, y, score: RTMPose-hand on the [x, y, w, h] box (rtmlib pads it by 1.25 itself)."""
        import numpy as np
        tick = time.perf_counter()
        x, y, w, h = box
        kpts, scores = self.pose(frame, bboxes=[[x, y, x + w, y + h]])
        self.pose_s += time.perf_counter() - tick
        return np.concatenate([kpts[0], scores[0][:, None]], axis=1)
