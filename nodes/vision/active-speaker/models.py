"""LR-ASD (Liao et al., IJCV 2025; github.com/Junhua-Liao/LR-ASD), as the bake-off chose and ran it: the weights
fine-tuned on TalkSet (weight/finetuning_TalkSet.model). Given a 25 fps grey face clip and the 13 MFCCs (100 Hz) of the
audio under it, it scores every frame: > 0 means this face is speaking then.

Code: vendored in third_party/ (its README: LR-ASD at the bake-off's commit, with its licence), never cloned at run time.
Weights: from the same commit of the GitHub repository (3.4 MB, inside the repo, no release), fetched once into the
shared weights cache and checked by sha256 (perception_node.weights); the manifest lists the same file. PLAN.md §4 does
not clear LR-ASD's weights for a mirror of ours (they state no licence; TalkSet comes from VoxCeleb2 and LRS3), so they
come from source.

Where it runs (the request's accelerator): mps (the Apple GPU), cuda, or cpu, in float32 as in the bake-off. A device
PyTorch cannot use here fails the node; it never runs elsewhere instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

from perception_node import NodeError
from perception_node.weights import fetch

COMMIT = "1b6dcd2d8fc2895683de6508ec6294ec47d388ca"
WEIGHTS = {
    "lrasd": {"url": f"https://raw.githubusercontent.com/Junhua-Liao/LR-ASD/{COMMIT}/weight/finetuning_TalkSet.model",
              "sha256": "6b4ef53694e874e96cf630198dc479c78aebb3993bbf166aee3d926dfe7d9342"},
}
LRASD_ID = "LR-ASD (finetuning_TalkSet: AVA-ActiveSpeaker, then TalkSet)"
LRASD_LICENCE = ("LR-ASD code MIT (Junhua Liao 2025); the weights in its repository (weight/finetuning_TalkSet.model) "
                 "state no licence of their own. Training data (per its README and TalkNet's): AVA-ActiveSpeaker, then "
                 "TalkSet, built from VoxCeleb2 and LRS3; what those datasets' terms mean for trained weights is unverified")
THIRD_PARTY = Path(__file__).parent / "third_party" / "lr_asd"


def weights(name: str) -> Path:
    w = WEIGHTS[name]
    return fetch(w["url"], w["sha256"])


def device_for(accelerator: str):
    """The torch device for the request's accelerator, or NodeError when PyTorch cannot use it here."""
    import torch
    if accelerator == "mps" and not torch.backends.mps.is_available():
        raise NodeError("resources", "PyTorch cannot use the Apple GPU (MPS) here: run Active speakers on another device")
    if accelerator == "cuda" and not torch.cuda.is_available():
        raise NodeError("resources", f"PyTorch {torch.__version__} (CUDA {torch.version.cuda}) finds no usable CUDA GPU here: "
                                     "run Active speakers on another device")
    return torch.device(accelerator if accelerator in ("mps", "cuda") else "cpu")


# Model code and numpy are imported where used: this module is also read for its WEIGHTS alone (orchestrator tests)


def mfcc(samples):
    """13 MFCCs every 10 ms over 25 ms windows of 16 kHz int16 samples, as LR-ASD was trained on them."""
    import python_speech_features
    from asd_crops import AUDIO_RATE
    return python_speech_features.mfcc(samples, AUDIO_RATE, numcep=13, winlen=0.025, winstep=0.010)


class ASDModel:
    """score(faces uint8 [frames, 112, 112], mfcc [n, 13]) -> (one score per frame, the clip cut to its audio; the
    scores per window length it is the mean of, [len(DURATIONS), frames]), all float32."""

    def __init__(self, accelerator: str):
        import torch
        self.device = device_for(accelerator)
        if str(THIRD_PARTY) not in sys.path:
            sys.path.insert(0, str(THIRD_PARTY))
        from loss import lossAV
        from model.Model import ASD_Model

        net = torch.nn.Module()
        net.model, net.lossAV = ASD_Model(), lossAV()
        state = torch.load(weights("lrasd"), map_location="cpu", weights_only=True)
        # the checkpoint also holds the audio-only and visual-only training heads (lossA, lossV), unused in scoring
        result = net.load_state_dict({k: v for k, v in state.items() if not k.startswith(("lossA.", "lossV."))})
        if result.missing_keys or result.unexpected_keys:
            raise NodeError("internal", f"the LR-ASD checkpoint does not match the model: missing {result.missing_keys[:5]}, "
                                        f"unexpected {result.unexpected_keys[:5]}")
        self.net = net.to(self.device).eval()

    def score(self, faces, features):
        import numpy as np
        import torch
        from asd_crops import clip_windows
        net, runs = self.net, []
        with torch.no_grad():
            for windows in clip_windows(len(features), len(faces)):
                scores = []
                for a0, a1, v0, v1 in windows:
                    a = torch.FloatTensor(features[a0:a1]).unsqueeze(0).to(self.device)
                    v = torch.FloatTensor(faces[v0:v1]).unsqueeze(0).to(self.device)
                    ea, ev = net.model.forward_audio_frontend(a), net.model.forward_visual_frontend(v)
                    scores.extend(net.lossAV(net.model.forward_audio_visual_backend(ea, ev)))
                runs.append(scores)
        windows = np.array(runs)
        return np.mean(windows, axis=0), windows
