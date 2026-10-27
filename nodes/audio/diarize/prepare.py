"""Fetch the diarization pipeline ahead of a run (`perception setup`, and the driver before a remote run): pyannote
downloads its config, segmentation, embedding and PLDA files from Hugging Face (gated: HF_TOKEN) into the Hugging Face
cache the node then reads, at the pinned revision (diarizer.py). Loaded on the CPU: nothing runs.
    python prepare.py CONFIG_JSON      (the node's settings, as its request would carry them; it has none)"""

import os

from diarizer import REPO, REVISION, load

load(os.environ.get("HF_TOKEN"), "cpu")
print(f"{REPO}@{REVISION[:8]} is in the cache")
