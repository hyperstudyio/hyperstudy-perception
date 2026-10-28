"""Fetch the hands weights ahead of a run (`perception setup`, and the driver before a remote run): RTMPose-m Hand5 and
RTMDet-nano hand from OpenMMLab's server into the shared weights cache, checked by sha256 (models.py)."""

from models import weights

for name in ("rtmpose_hand", "rtmdet_hand"):
    print(f"{name}: {weights(name)}")
