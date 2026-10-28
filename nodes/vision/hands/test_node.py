"""node.py run whole on a video whose decoded frames change size, with OpenCV's decoder and the models stood in for:
the side file names one frame size, so such a video is refused as an input error, not written with the wrong one."""

import json
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
NAMES = json.loads((HERE.parent / "body-pose" / "keypoints.json").read_text())["names"]
FPS = 8


def stand_ins(monkeypatch, sizes):
    """cv2 whose VideoCapture decodes one frame of each (width, height) in `sizes`, and models whose detector finds
    nothing."""
    class VideoCapture:
        def __init__(self, path):
            self.frames = [np.zeros((h, w, 3), np.uint8) for w, h in sizes]

        def isOpened(self):
            return True

        def read(self):
            return (True, self.frames.pop(0)) if self.frames else (False, None)

        def grab(self):
            return bool(self.frames) and self.frames.pop(0) is not None

        def release(self):
            pass

    class Hands:
        providers = ["CPUExecutionProvider"]

        def __init__(self, accelerator):
            self.detect_s = self.pose_s = 0.0

        def detect(self, frame):
            return []

        def keypoints(self, frame, box):
            return np.zeros((21, 3))

    cv2 = types.ModuleType("cv2")
    cv2.VideoCapture = VideoCapture
    models = types.ModuleType("models")
    models.Hands = Hands
    models.WEIGHTS = {name: {"url": f"https://example.org/{name}.onnx", "sha256": "0" * 64} for name in ("rtmpose_hand", "rtmdet_hand")}
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "models", models)


def run(tmp_path, monkeypatch, sizes) -> dict:
    """node.py's response for a video of these frame sizes (one shot, no bodies)."""
    stand_ins(monkeypatch, sizes)
    inputs = {"video": ("clip.mp4", None),
              "media": ("media.json", {"fps": FPS, "width": sizes[0][0], "height": sizes[0][1], "duration": len(sizes) / FPS}),
              "shots": ("shots.json", {"shots": [{"index": 0, "start": 0.0, "end": 10.0}]}),
              "bodies": ("bodies.json", {"keypoint_names": NAMES, "score_threshold": 3.0, "fps": FPS, "tracks": []})}
    for file, doc in inputs.values():
        (tmp_path / file).write_text(json.dumps(doc) if doc is not None else "")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"protocol": 1, "out_dir": str(tmp_path / "out"), "accelerator": "cpu",
                                   "inputs": {name: {"path": str(tmp_path / file)} for name, (file, _) in inputs.items()}}))
    monkeypatch.setattr(sys, "argv", ["node.py", "--request", str(request)])
    try:
        runpy.run_path(str(HERE / "node.py"), run_name="__main__")
    except SystemExit:   # how the node ends after writing a failed response
        pass
    return json.loads((tmp_path / "response.json").read_text())


def test_frames_of_one_size_are_a_layer_and_a_side_file(tmp_path, monkeypatch):
    response = run(tmp_path, monkeypatch, [(64, 48)] * 4)
    assert response["ok"], response["error"]
    assert set(response["outputs"]) == {"hands", "detections"}


def test_frames_that_change_size_are_an_input_error(tmp_path, monkeypatch):
    response = run(tmp_path, monkeypatch, [(64, 48), (64, 48), (32, 24), (32, 24)])
    assert not response["ok"] and response["outputs"] == {}
    assert response["error"]["code"] == "input"
    assert response["error"]["message"] == "the frames change size ([(32, 24), (64, 48)]): the side file names one frame size"
