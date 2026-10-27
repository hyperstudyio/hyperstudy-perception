"""Fetch the LR-ASD weights ahead of a run (`perception setup`, and the driver before a remote run): from the pinned
commit of its GitHub repository, into the shared weights cache, checked by sha256 (models.py)."""

from models import weights

print(f"lrasd: {weights('lrasd')}")
