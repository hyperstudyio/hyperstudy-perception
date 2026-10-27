"""The body-pose models: YOLOX-m finds people, RTMW-l (top-down, COCO-WholeBody's 133 keypoints) estimates each one's
pose, both through rtmlib on ONNX Runtime. Their weights come from OpenMMLab's server, fetched once into the shared
weights cache and checked by sha256 (perception_node.weights); the manifest lists the same files.

Where they run (the request's accelerator):
  mps   RTMW on CoreML, as an MLProgram: one 384x288 crop takes 10.7 ms on an M5 Max, against 18.9 ms for the
        NeuralNetwork format rtmlib's device='mps' builds and 41.8 ms on the CPU. 405 of its 456 nodes run on CoreML.
  cuda  both on CUDA (onnxruntime-gpu).
  cpu   both on the CPU.
YOLOX runs on the CPU on a Mac: its ONNX ends in NMS with data-dependent output shapes, which ONNX Runtime 1.31's CoreML
provider fails on in both formats (static/inferred rank mismatch; zero-element dynamic shape on a frame with no person).

A provider ONNX Runtime cannot load (CUDA without its libraries) is not quietly replaced by the CPU: the node fails.
"""

from __future__ import annotations

from perception_node import NodeError
from perception_node.weights import fetch

WEIGHTS = {
    "rtmw": {"url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
             "sha256": "bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054"},
    "yolox": {"url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
              "sha256": "3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23"},
}
COREML = ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"})
CUDA = "CUDAExecutionProvider"
YOLOX_SCORE = 0.3   # the score filter for this ONNX, which has its own NMS (rtmlib's score_thr=0.7 is never applied)
# RTMW scores are raw SimCC peaks (0-12 on OOO), not probabilities. Against ViTPose on 2786 paired OOO bodies, >= 3 keeps
# 98% of the joints ViTPose scores > 0.5 and 4% of those it scores < 0.2: the bodies layer's score_threshold
SCORE_THRESHOLD = 3.0


def weights(name: str):
    w = WEIGHTS[name]
    return fetch(w["url"], w["sha256"], member="end2end.onnx")


def session(onnx, accelerator: str, use_coreml: bool):
    import onnxruntime as ort
    wanted = CUDA if accelerator == "cuda" else COREML if accelerator == "mps" and use_coreml else "CPUExecutionProvider"
    s = ort.InferenceSession(str(onnx), providers=[wanted, "CPUExecutionProvider"] if wanted != "CPUExecutionProvider" else [wanted])
    name = wanted if isinstance(wanted, str) else wanted[0]
    if name not in s.get_providers():
        raise NodeError("resources", f"ONNX Runtime could not use {name} here (it has {', '.join(ort.get_available_providers())}): "
                                     "run Body pose on another device")
    return s


class Detector:
    """YOLOX-m person boxes: [((x1, y1, x2, y2), score)] per frame. rtmlib's __call__ drops the scores, so its
    preprocess and inference are called directly."""

    def __init__(self, accelerator: str):
        from rtmlib import YOLOX
        self.model = YOLOX(str(weights("yolox")), model_input_size=(640, 640), backend="onnxruntime", device="cpu")
        self.model.session = session(self.model.onnx_model, accelerator, use_coreml=False)

    def __call__(self, frame):
        image, ratio = self.model.preprocess(frame)
        dets = self.model.inference(image)[0][0]   # (N, 5): x1, y1, x2, y2, score, after the model's own NMS
        return [((d[:4] / ratio).tolist(), float(d[4])) for d in dets if d[4] > YOLOX_SCORE]


class Pose:
    """RTMW-l keypoints for each box: an (133, 3) array of x, y, score per person."""

    def __init__(self, accelerator: str):
        from rtmlib import RTMPose
        self.model = RTMPose(str(weights("rtmw")), model_input_size=(288, 384), backend="onnxruntime", device="cpu")
        self.model.session = session(self.model.onnx_model, accelerator, use_coreml=True)
        self.providers = self.model.session.get_providers()

    def __call__(self, frame, boxes):
        import numpy as np
        if not boxes:
            return []
        kpts, scores = self.model(frame, bboxes=[b for b, _ in boxes])
        return [np.concatenate([k, s[:, None]], axis=1) for k, s in zip(kpts, scores)]
