"""Fetch the ASR node's weights ahead of a run (`perception setup`): parakeet-mlx downloads the model from Hugging Face
the first time it is loaded, into the cache the node then reads.
    python prepare.py CONFIG_JSON      (the node's settings, as its request would carry them)"""

import json
import sys

from parakeet_mlx import from_pretrained

model_id = json.loads(sys.argv[1])["model"]
from_pretrained(model_id)
print(f"{model_id} is in the cache")
