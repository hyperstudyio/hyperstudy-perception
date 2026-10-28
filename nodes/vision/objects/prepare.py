"""Fetch the object node's weights ahead of a run (`perception setup`, and the driver before a remote run): SAM 3 and
DINOv3 ViT-L/16 from Hugging Face (both gated: HF_TOKEN), at the pinned revisions, into the Hugging Face cache the node
then reads (models.py). Loaded on the CPU: nothing runs."""

import os

import torch

from models import EMBEDDER, EMBEDDER_REVISION, SEGMENTER, SEGMENTER_REVISION, load_embedder, load_segmenter

token = os.environ.get("HF_TOKEN")
load_segmenter(torch.device("cpu"), token)
load_embedder(torch.device("cpu"), token)
print(f"{SEGMENTER}@{SEGMENTER_REVISION[:8]} and {EMBEDDER}@{EMBEDDER_REVISION[:8]} are in the cache")
