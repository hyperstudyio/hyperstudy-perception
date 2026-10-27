"""Fetch the body-pose weights ahead of a run (`perception setup`, and the driver before a remote run): RTMW-l and
YOLOX-m from OpenMMLab's server into the shared weights cache, checked by sha256 (models.py)."""

from models import weights

for name in ("rtmw", "yolox"):
    print(f"{name}: {weights(name)}")
