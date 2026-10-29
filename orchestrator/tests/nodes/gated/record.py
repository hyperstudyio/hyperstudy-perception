"""What test.gated's prepare and node record: the key-like variables in their environment, the sha256 of HF_TOKEN
(never the token), HF_HOME and HF_HUB_OFFLINE, for test_scheduler and test_remote; and where its weights go."""

import hashlib
import json
import os
from pathlib import Path


def record(folder: str, who: str, **more) -> None:
    token = os.environ.get("HF_TOKEN")
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / f"{who}-{os.getpid()}").write_text(json.dumps(
        {"keys": sorted(k for k in os.environ if k.endswith(("_KEY", "_TOKEN")) and k != "HF_HUB_DISABLE_IMPLICIT_TOKEN"),
         "sha256": hashlib.sha256(token.encode()).hexdigest() if token else None,
         "hf_home": os.environ.get("HF_HOME"), "hub_cache": os.environ.get("HF_HUB_CACHE"),
         "offline": os.environ.get("HF_HUB_OFFLINE"), "implicit_token": os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") != "1",
         **more}))


def weights() -> Path:
    """Its weights' file: in the shared weights cache here (PERCEPTION_WEIGHTS, which plan.weights_cached reads), and in
    the Hugging Face cache on a host (HF_HOME, the account's there)."""
    folder = os.environ.get("PERCEPTION_WEIGHTS") or os.path.join(os.environ.get("HF_HOME") or "/nowhere", "weights")
    return Path(folder) / "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08" / "weights.bin"
