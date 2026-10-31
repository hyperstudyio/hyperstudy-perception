"""The sound models, run on PyTorch 2.8 (envs/audio-events), as the bake-off ran them (audio/audio_events.py):
- PANNs Cnn14_DecisionLevelMax (code vendored in panns/, MIT; weights from Zenodo, CC-BY-4.0): frame-wise AudioSet
  scores at 100 frames/s, from 32 kHz audio.
- EfficientAT mn40_as (code vendored in efficientat/, MIT; weights from its GitHub release, MIT): AudioSet scores for
  a 5 s window centred on each 1 s step, from 32 kHz audio.
- PE-A-Frame large (facebook/pe-a-frame-large on Hugging Face, Apache-2.0, through transformers 5.20.2): text queries
  scored against 40 ms frames of 48 kHz audio.
The PANNs and EfficientAT weights are fetched once into the shared weights cache and checked by sha256
(perception_node.weights); the manifest lists the same files. PE-A-Frame comes from the Hugging Face hub cache, at the
revision the bake-off used.

Where they run (the request's accelerator): mps, cuda or cpu, all three models. A device PyTorch cannot use here is an
error, not a quiet fall back to the CPU.

torch is imported inside the functions: the orchestrator's tests read WEIGHTS from this file without it.
"""

from __future__ import annotations

import contextlib
import io

from perception_node import NodeError
from perception_node.weights import fetch

WEIGHTS = {
    "panns": {"url": "https://zenodo.org/records/3987831/files/Cnn14_DecisionLevelMax_mAP=0.385.pth",
              "sha256": "dd3b4043a87d4ec13df8082c0fcfee3fb5084151808e47e060987a95eabdd142"},
    "mn40": {"url": "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn40_as_mAP_484.pt",
             "sha256": "09624dca1c53128255c65c5a52ff49ae91bb01fe17f0e13b3c3a0c9f24a8478b"},
}
PEAFRAME = {"repo": "facebook/pe-a-frame-large", "revision": "d7fd29e06effc0fe5630de3b6e4ca784876a1a16"}
TAGGER_SR = 32000    # PANNs and EfficientAT
PEAFRAME_SR = 48000


def weights(name: str):
    w = WEIGHTS[name]
    return fetch(w["url"], w["sha256"])


def peaframe_files() -> str:
    """The PE-A-Frame snapshot in the Hugging Face hub cache (downloaded the first time)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(PEAFRAME["repo"], revision=PEAFRAME["revision"])


def device(accelerator: str):
    import torch
    if accelerator == "mps" and not torch.backends.mps.is_available():
        raise NodeError("resources", "PyTorch cannot use the Apple GPU (MPS) here: run Sound events on another device")
    if accelerator == "cuda" and not torch.cuda.is_available():
        raise NodeError("resources", "PyTorch cannot use CUDA here (no NVIDIA GPU, or a driver older than CUDA 12.8 "
                                     "needs, >= 570): run Sound events on another device")
    return torch.device(accelerator)


def sync(dev):
    import torch
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def load_panns(dev):
    import torch
    from panns.cnn14 import Cnn14_DecisionLevelMax
    # hyperparameters from the repository's scripts/0_inference.sh for this checkpoint
    model = Cnn14_DecisionLevelMax(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                                   classes_num=527)
    model.load_state_dict(torch.load(weights("panns"), map_location="cpu", weights_only=True)["model"])
    return model.to(dev).eval()


def run_panns(model, wave, dev):
    """Frame-wise scores (frames, 527) at 100 frames/s for 32 kHz samples."""
    import torch
    with torch.no_grad():
        return model(torch.from_numpy(wave)[None].to(dev))["framewise_output"][0].cpu().numpy()


def free(dev):
    """Give a finished stage's memory back to the GPU. Freed tensors stay reserved by PyTorch's caching allocator, in
    blocks the next stage's largest allocation may not fit: on a 24 GB card, PE-A-Frame large asked for 7.38 GiB with
    6.5 GiB of the earlier stages' left reserved but unused, and 6.9 GiB free (a RunPod pod, 2026-10-31)."""
    import gc
    import torch
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

def load_mn40(dev):
    import torch
    from efficientat.mn.model import get_model
    from efficientat.preprocess import AugmentMelSTFT
    with contextlib.redirect_stdout(io.StringIO()):   # get_model prints the whole network; AugmentMelSTFT its fmax
        model = get_model(width_mult=4.0, pretrained_name=None)   # mn40: NAME_TO_WIDTH("mn40_as")
        mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320)   # EfficientAT's EATagger defaults
    model.load_state_dict(torch.load(weights("mn40"), map_location="cpu", weights_only=True))
    return model.to(dev).eval(), mel.to(dev).eval()


def run_mn40(loaded, windows, dev):
    """AudioSet scores (windows, 527) for a batch of 32 kHz windows."""
    import torch
    model, mel = loaded
    with torch.no_grad():
        spec = mel(torch.from_numpy(windows).to(dev))
        logits, _ = model(spec.unsqueeze(1))
        return torch.sigmoid(logits).cpu().numpy()


def load_peaframe(dev):
    import torch
    from transformers import PeAudioFrameLevelModel, PeAudioProcessor
    path = peaframe_files()
    model = PeAudioFrameLevelModel.from_pretrained(path)
    # transformers 5.20.2's port embeds the text from ModernBERT's 22nd layer and then applies its final norm; Meta's
    # PEAudioFrame takes hidden_states[22] of the 28-layer ModernBERT, layer 22's raw output. The extra norm shifts every
    # text embedding (~95% relative error) and inflates the probabilities. Without it the port matches Meta's reference
    # (perception_models @ 3e352cc) to < 1e-4 (the bake-off's check, on the small and large models).
    model.text_model.final_norm = torch.nn.Identity()
    return model.to(dev).eval(), PeAudioProcessor.from_pretrained(path)


def run_peaframe(loaded, wave, queries, dev):
    """Each query's probability per 40 ms frame of 48 kHz samples: (queries, frames)."""
    import torch
    model, processor = loaded
    inputs = processor(text=queries, audio=wave, sampling_rate=PEAFRAME_SR, return_tensors="pt", padding=True).to(dev)
    if processor.feature_extractor.hop_length != 1920:
        raise NodeError("internal", f"PE-A-Frame's hop is {processor.feature_extractor.hop_length} samples, not 1920")
    with torch.no_grad():
        return model(**inputs).logits_audio_text[0].sigmoid().cpu().numpy()
