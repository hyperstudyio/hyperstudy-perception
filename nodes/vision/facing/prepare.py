"""Fetch the facing weights ahead of a run (`perception setup`, and the driver before a remote run): MediaPipe's Pose
Landmarker heavy from Google's model storage into the shared weights cache, checked by sha256 (models.py)."""

from models import weights

print(f"pose_landmarker_heavy: {weights()}")
