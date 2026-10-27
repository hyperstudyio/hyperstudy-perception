"""Fetch the gaze weights ahead of a run (`perception setup`, and the driver before a remote run): the Gaze-LLE decoder
from its GitHub release and DINOv2 ViT-L/14 from Meta's server, into the shared weights cache, checked by sha256
(models.py)."""

from models import weights

for name in ("gazelle", "dinov2"):
    print(f"{name}: {weights(name)}")
