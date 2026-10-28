"""Fetch the ASR node's weights ahead of a run (`perception setup`, and the driver before a remote run), into the
Hugging Face cache the node then reads: in envs/audio (the MLX runner) parakeet-mlx's conversion, which it downloads
the first time it is loaded; in envs/asr-cuda (the CUDA and CPU runners) NVIDIA's .nemo checkpoint at its pinned
commit (nemo_model.py), only downloaded: nothing is loaded.
    python prepare.py CONFIG_JSON      (the node's settings, as its request would carry them)"""

import importlib.util
import json
import sys

model_id = json.loads(sys.argv[1])["model"]
if importlib.util.find_spec("parakeet_mlx") is not None:   # envs/audio, on Apple silicon
    from parakeet_mlx import from_pretrained
    from_pretrained(model_id)
    print(f"{model_id} is in the cache")
else:                                                      # envs/asr-cuda
    import nemo_model
    nemo_model.checkpoint(model_id)
    print(f"{nemo_model.REPO}@{nemo_model.REVISION[:8]} ({nemo_model.FILE}) is in the cache")
