"""Gaze-LLE (Ryan et al., CVPR 2025; github.com/fkryan/gazelle), as the bake-off chose and ran it: the checkpoint
fine-tuned on VideoAttentionTarget (gazelle_dinov2_vitl14_inout: GazeFollow, then TV footage), which also scores
whether each person looks at something in the frame. It encodes the frame once with a frozen DINOv2 ViT-L/14 at
448x448 and, for each head box, decodes a 64x64 heatmap of where that person looks and an in-frame score in [0, 1].
One forward per sampled frame carries every head in that frame.

Code: vendored in third_party/ (its README: gazelle and dinov2 at the bake-off's commits, with their licences), never
cloned or fetched by torch hub at run time.
Weights: the decoder from the Gaze-LLE GitHub release and the backbone from Meta's server, fetched once into the shared
weights cache and checked by sha256 (perception_node.weights); the manifest lists the same files. PLAN.md §4 does not
clear Gaze-LLE's weights for a mirror of ours (its training data's terms are unverified), so both come from source.

Where it runs (the request's accelerator): mps (the Apple GPU), cuda, or cpu, in float32 as in the bake-off. A device
PyTorch cannot use here fails the node; it never runs elsewhere instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

from perception_node import NodeError
from perception_node.weights import fetch

WEIGHTS = {
    "gazelle": {"url": "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14_inout.pt",
                "sha256": "e48136e4efad6bb91182a51a2fb365c2202d974e5e449145a21348aca7b5d0b1"},
    "dinov2": {"url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
               "sha256": "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"},
}
GAZELLE_ID = "Gaze-LLE ViT-L/14 in/out (gazelle_dinov2_vitl14_inout, GazeFollow -> VideoAttentionTarget)"
DINOV2_ID = "DINOv2 ViT-L/14 (dinov2_vitl14_pretrain, LVD-142M)"
GAZELLE_LICENCE = ("Gaze-LLE code MIT (Fiona Ryan 2024); the release v1.0.0 checkpoints state no licence of their own. "
                   "Training data: GazeFollow (Recasens et al. 2015; terms unverified, gazefollow.csail.mit.edu unreachable "
                   "2026-10-22) and VideoAttentionTarget (YouTube TV clips; its repository's MIT licence covers the code "
                   "and states no terms for the dataset)")
DINOV2_LICENCE = "Apache-2.0, code and weights (facebookresearch/dinov2 README)"
INPUT_SIZE = (448, 448)
THIRD_PARTY = Path(__file__).parent / "third_party"


def weights(name: str) -> Path:
    w = WEIGHTS[name]
    return fetch(w["url"], w["sha256"])


def device_for(accelerator: str):
    """The torch device for the request's accelerator, or NodeError when PyTorch cannot use it here."""
    import torch
    if accelerator == "mps" and not torch.backends.mps.is_available():
        raise NodeError("resources", "PyTorch cannot use the Apple GPU (MPS) here: run Gaze targets on another device")
    if accelerator == "cuda" and not torch.cuda.is_available():
        raise NodeError("resources", f"PyTorch {torch.__version__} (CUDA {torch.version.cuda}) finds no usable CUDA GPU here: "
                                     "run Gaze targets on another device")
    return torch.device(accelerator if accelerator in ("mps", "cuda") else "cpu")


def load(device):
    """Gaze-LLE with its DINOv2 ViT-L/14 backbone on `device`, in eval mode; and the frame transform."""
    import torch
    for path in (THIRD_PARTY / "gazelle", THIRD_PARTY / "dinov2"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from dinov2.models import vision_transformer as vits
    from gazelle.backbone import Backbone, DinoV2Backbone
    from gazelle.model import GazeLLE

    class Dinov2(DinoV2Backbone):
        def __init__(self, state):
            Backbone.__init__(self)   # DinoV2Backbone.__init__ would torch.hub.load the code from GitHub
            # dinov2/hub/backbones.py's _make_dinov2_model("vit_large", patch 14), without its download
            self.model = vits.vit_large(img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp", block_chunks=0,
                                        num_register_tokens=0, interpolate_antialias=False, interpolate_offset=0.1)
            self.model.load_state_dict(state, strict=True)

    backbone = Dinov2(torch.load(weights("dinov2"), weights_only=True, map_location="cpu"))
    model = GazeLLE(backbone, inout=True)
    state = torch.load(weights("gazelle"), weights_only=True, map_location="cpu")
    decoder = {k for k in model.state_dict() if not k.startswith("backbone")}
    if decoder - set(state) or set(state) - decoder:
        raise NodeError("internal", f"the Gaze-LLE checkpoint does not match the model: missing {sorted(decoder - set(state))[:5]}, "
                                    f"unexpected {sorted(set(state) - decoder)[:5]}")
    model.load_gazelle_state_dict(state)
    model.to(device).eval()
    return model, backbone.get_transform(INPUT_SIZE)


class GazeModel:
    """predict(frame BGR, [(face, head box [x1, y1, x2, y2] px)]) -> [(64x64 heatmap, in-frame score)], one forward."""

    def __init__(self, accelerator: str):
        self.device = device_for(accelerator)
        self.model, self.transform = load(self.device)

    def predict(self, frame_bgr, heads):
        import cv2
        import torch
        from PIL import Image
        height, width = frame_bgr.shape[:2]
        image = self.transform(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0)
        boxes = [(x1 / width, y1 / height, x2 / width, y2 / height) for _, (x1, y1, x2, y2) in heads]
        with torch.no_grad():
            out = self.model({"images": image.to(self.device), "bboxes": [boxes]})
        heat = out["heatmap"][0].float().cpu().numpy()
        inout = out["inout"][0].float().cpu().numpy()
        return [(heat[k], float(inout[k])) for k in range(len(heads))]
