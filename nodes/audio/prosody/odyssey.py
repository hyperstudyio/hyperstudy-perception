"""Speech emotion: the Odyssey 2024 challenge baseline, 3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes (WavLM-large
and attentive statistics pooling; arousal, dominance and valence, about 0..1), as the bake-off chose and ran it
(prosody/prosody.py: load_odyssey, run_odyssey; the annotator's default emotion row). The bake-off's audEERING and
emotion2vec+ models are not ported.

Input: each utterance's own audio, from its start to its analysed end (prosody_units.word_end), no context, as float
samples (int16 / 32768), normalised with the card's config mean and std as the card does ((x - mean) / (std + 1e-6)),
with a mask of ones. Output: the model's raw regression output, not clipped or rescaled.

Code: the card's classes, copied into third_party/odyssey (its README: why, from which commit, under what licence).
Weights: the checkpoint at the pinned revision from the Hugging Face cache (or the hub, the first time), loaded
strictly (every tensor present, none left over); the backbone is built from microsoft/wavlm-large's config.json at the
revision the card's code would have fetched it at. Not gated.

Where it runs (the request's accelerator): mps, cuda or cpu, in float32. A device PyTorch cannot use here fails the
node; it never runs elsewhere instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from perception_node import NodeError

REPO = "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes"
REVISION = "00d0e12ba9bf957f5aeea36e8663c8c61cb50ac9"   # main on 2026-10-23 (last change 2024-06-12)
WAVLM = "microsoft/wavlm-large"
WAVLM_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
MODEL_ID = f"{REPO}@{REVISION[:8]}"   # as the bake-off named it, which the app shows (shortModel: "Odyssey")
LICENCE = "MIT (code and weights, MSP-UTD / 3loi); WavLM-large backbone MIT (Microsoft)"
NOTE = ("Trained on MSP-Podcast, whose data licence is academic-only; whether that carries over to MIT-licensed weights "
        "is unresolved.")
EPSILON = 0.000001
RATE = 16000
# WavLM's convolutional feature encoder needs 400 samples (25 ms) for one frame: a shorter clip has no output
MIN_SAMPLES = 400
PLACES = 6   # the layer's rounding, the bake-off's
THIRD_PARTY = Path(__file__).parent / "third_party" / "odyssey"


def device_for(accelerator: str):
    """The torch device for the request's accelerator, or NodeError when PyTorch cannot use it here."""
    import torch
    if accelerator == "mps" and not torch.backends.mps.is_available():
        raise NodeError("resources", "PyTorch cannot use the Apple GPU (MPS) here: run Prosody on another device")
    if accelerator == "cuda" and not torch.cuda.is_available():
        raise NodeError("resources", f"PyTorch {torch.__version__} (CUDA {torch.version.cuda}) finds no usable CUDA GPU here: "
                                     "run Prosody on another device")
    return torch.device(accelerator if accelerator in ("mps", "cuda") else "cpu")


def files() -> tuple[Path, Path, Path]:
    """(config.json, model.safetensors, WavLM's config.json) at the pinned revisions, from the cache or the hub."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
    try:
        return (Path(hf_hub_download(REPO, "config.json", revision=REVISION)),
                Path(hf_hub_download(REPO, "model.safetensors", revision=REVISION)),
                Path(hf_hub_download(WAVLM, "config.json", revision=WAVLM_REVISION)))
    except LocalEntryNotFoundError as e:   # offline, and not in the cache
        raise NodeError("input", f"the Odyssey weights are not in the Hugging Face cache and could not be fetched: {e}") from None
    except HfHubHTTPError as e:
        raise NodeError("transient", f"Hugging Face: {e}") from None


def load(device):
    """(model on `device`, eval mode; config dict) with the checkpoint loaded strictly."""
    from safetensors.torch import load_file
    from transformers import WavLMConfig

    if str(THIRD_PARTY) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY))
    from ser_model import SERModel

    config_path, weights_path, wavlm_config = files()
    config = json.loads(config_path.read_text())
    model = SERModel(config, WavLMConfig.from_json_file(str(wavlm_config)))
    state = load_file(str(weights_path))
    # the checkpoint was saved by transformers 4.34, whose weight-normed positional conv stored weight_g/weight_v;
    # transformers 5's WavLM keeps them as parametrizations (original0 = g, original1 = v)
    for old, new in (("weight_g", "parametrizations.weight.original0"), ("weight_v", "parametrizations.weight.original1")):
        key = f"ssl_model.encoder.pos_conv_embed.conv.{old}"
        state[key.replace(old, new)] = state.pop(key)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config


def labels(config: dict) -> list[str]:
    """The output order, from the card's config: arousal, dominance, valence."""
    return [config["id2label"][str(i)] for i in range(config["num_classes"])]


def clip_span(unit: dict) -> tuple[int, int]:
    """The samples [a, b) of an utterance's own audio: start to analysed end, as the bake-off cut them."""
    return int(unit["start"] * RATE), int(unit["analysed_end"] * RATE)


def run(model, config: dict, clips: list[np.ndarray], device) -> list[list[float] | None]:
    """The raw float32 outputs per clip (labels() order), or None for a clip under MIN_SAMPLES."""
    import torch
    mean, std = config["mean"], config["std"]
    out = []
    with torch.no_grad():
        for clip in clips:
            if len(clip) < MIN_SAMPLES:
                out.append(None)
                continue
            norm = (clip - mean) / (std + EPSILON)   # the card's normalisation, epsilon included
            x = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(device)
            y = model(x, torch.ones(1, len(norm), device=device))[0].cpu().numpy()
            out.append([float(v) for v in y])
    return out
