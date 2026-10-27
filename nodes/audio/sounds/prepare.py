"""Fetch the sound models ahead of a run (`perception setup`, and the driver before a remote run): PANNs and EfficientAT
mn40 into the shared weights cache, checked by sha256, and PE-A-Frame large into the Hugging Face hub cache (models.py)."""

from models import peaframe_files, weights

for name in ("panns", "mn40"):
    print(f"{name}: {weights(name)}")
print(f"pe-a-frame-large: {peaframe_files()}")
