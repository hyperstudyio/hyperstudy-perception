"""MediaPipe Pose Landmarker heavy (BlazePose GHUM 3D, 33 landmarks), as the bake-off ran it (pose/pose3d.py,
MediapipePose): IMAGE mode, one crop per body box (facing.crop), up to facing.POSES poses per crop, the one nearest the
box centre kept (facing.nearest_pose). Its world landmarks are metres in a camera-aligned frame with the origin at the
mid-hip, which is what the torso direction needs.

Weights: Google's model storage, fetched once into the shared weights cache and checked by sha256
(perception_node.weights); the manifest lists the same file. The .task bundles pose_detector.tflite and
pose_landmarks_detector.tflite, built from blazepose_ghum_39kp_heavy_oss_2021_07_02.

The CPU (XNNPACK) delegate only, so the node has only cpu runners: on the bake-off's 457 OOO boxes MediaPipe's GPU
(Metal) delegate found a pose for 317 boxes against the CPU's 365, joints differed by a median 0.022 box scale on boxes
both posed, and the left and right shoulders were swapped on 11 of 311. A request for another accelerator fails the
node rather than quietly running on the CPU.
"""

from __future__ import annotations

from perception_node import NodeError
from perception_node.weights import fetch

WEIGHTS = {
    "pose_landmarker_heavy": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        "sha256": "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b"},
}
MODEL_ID = "MediaPipe Pose Landmarker heavy (pose_landmarker_heavy.task float16 v1, BlazePose GHUM 3D)"
LICENCE = ("MediaPipe code Apache-2.0; the BlazePose GHUM 3D model card (the card the Pose Landmarker page links for "
           "lite/full/heavy) says 'LICENSED UNDER Apache License, Version 2.0'. The card lists multiple people in an "
           "image, people further than 4 m and 'head is not visible' as out of scope")


def weights() -> str:
    w = WEIGHTS["pose_landmarker_heavy"]
    return str(fetch(w["url"], w["sha256"]))


class Landmarker:
    """__call__(BGR square crop) -> [(landmarks (33, 3): x, y in [0, 1] of the crop and visibility, world (33, 3) in
    metres)] for every pose found in it, at most `poses`."""

    def __init__(self, accelerator: str, poses: int):
        if accelerator != "cpu":
            raise NodeError("resources", f"MediaPipe's Pose Landmarker runs here on the CPU (XNNPACK) only, not {accelerator}: "
                                         "run Facing on the CPU")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        self.landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=weights(), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE, num_poses=poses))

    def __call__(self, square):
        import cv2
        import numpy as np
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB)))
        result = self.landmarker.detect(image)
        return [(np.array([[p.x, p.y, p.visibility] for p in lms]), np.array([[p.x, p.y, p.z] for p in world]))
                for lms, world in zip(result.pose_landmarks, result.pose_world_landmarks)]

    def close(self):
        self.landmarker.close()
