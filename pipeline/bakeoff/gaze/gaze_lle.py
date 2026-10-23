"""Gaze-LLE adapter for the gaze-target stage (gaze_stage.py has the targets, events and JSON shared with gaze_page.py).

Gaze-LLE (Ryan et al., CVPR 2025; github.com/fkryan/gazelle) encodes the frame once with a frozen DINOv2 ViT-L/14
at 448x448 and, for each head box given, decodes a 64x64 heatmap of where that person is looking. The checkpoint
fine-tuned on VideoAttentionTarget (TV footage) also returns an in-frame score in [0, 1]. One forward per sampled
frame carries every head in that frame (the head boxes go in as normalised [x1, y1, x2, y2]).

Checkpoint (--checkpoint): vat (default, gazelle_dinov2_vitl14_inout: GazeFollow -> VideoAttentionTarget) or
gazefollow (gazelle_dinov2_vitl14, no in/out head, so no "out" targets and inout null). check_device.py compares
the two on sampled frames.

The DINOv2 code and weights are pinned in vendor/ rather than pulled from torch hub at run time:
  git clone https://github.com/fkryan/gazelle vendor/gazelle            (tested at 2d78f9f3)
  git clone https://github.com/facebookresearch/dinov2 vendor/dinov2    (tested at 7764ea0f)
  curl -L -o vendor/checkpoints/<name>.pt https://github.com/fkryan/gazelle/releases/download/v1.0.0/<name>.pt
  curl -L -o vendor/checkpoints/dinov2_vitl14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
(sha256 in CHECKPOINTS / DINO_SHA256; checked at load.)

Usage: uv run python gaze_lle.py ../runs/<name> [--checkpoint vat|gazefollow] [--device mps|cpu] [--heatmaps out.npz]
Writes runs/<name>/gaze_lle.json (gaze_lle_gazefollow.json for --checkpoint gazefollow).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

from gaze_stage import VENDOR, run_stage, sha256

sys.path.insert(0, str(VENDOR / "gazelle"))
sys.path.insert(0, str(VENDOR / "dinov2"))

CHECKPOINTS = {
    "vat": {"file": "gazelle_dinov2_vitl14_inout.pt", "inout": True,
            "sha256": "e48136e4efad6bb91182a51a2fb365c2202d974e5e449145a21348aca7b5d0b1",
            "name": "gazelle_dinov2_vitl14_inout (GazeFollow -> VideoAttentionTarget)"},
    "gazefollow": {"file": "gazelle_dinov2_vitl14.pt", "inout": False,
                   "sha256": "abb2950b3f953d145465eb81811ef61abeb61874e009c5a712dc89036f3c4bde",
                   "name": "gazelle_dinov2_vitl14 (GazeFollow)"},
}
DINO_WEIGHTS = VENDOR / "checkpoints/dinov2_vitl14_pretrain.pth"
DINO_SHA256 = "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"
LICENCE = ("Gaze-LLE code MIT (LICENSE, Fiona Ryan 2024); the decoder checkpoints (GitHub release v1.0.0) state no "
           "licence of their own. Training data: GazeFollow (Recasens et al. 2015; gazefollow.csail.mit.edu unreachable "
           "2026-10-22, terms unverified) and VideoAttentionTarget (YouTube TV clips; distributed via Dropbox from "
           "github.com/ejcgt/attention-target-detection, whose MIT licence covers the code, and whose README states no "
           "terms for the dataset). DINOv2 ViT-L/14: code and weights Apache-2.0 (dinov2 README). Head boxes come "
           "from py-feat RetinaFace (non-commercial weights). Academic use only")


def load_model(checkpoint, device):
    """Gaze-LLE ViT-L/14 with the vendored DINOv2 (no torch hub fetch at run time); returns (model, transform)."""
    from dinov2.hub.backbones import dinov2_vitl14
    from gazelle.backbone import Backbone, DinoV2Backbone
    from gazelle.model import GazeLLE

    class VendoredDinoV2(DinoV2Backbone):
        def __init__(self):
            Backbone.__init__(self)  # DinoV2Backbone.__init__ would torch.hub.load from GitHub
            self.model = dinov2_vitl14(pretrained=False)  # weights=<path> would copy 1.2 GB into ~/.cache/torch/hub
            self.model.load_state_dict(torch.load(DINO_WEIGHTS, weights_only=True, map_location="cpu"), strict=True)

    cfg = CHECKPOINTS[checkpoint]
    ckpt = VENDOR / "checkpoints" / cfg["file"]
    for path, expected in ((ckpt, cfg["sha256"]), (DINO_WEIGHTS, DINO_SHA256)):
        if sha256(path) != expected:
            raise SystemExit(f"{path} does not match its pinned sha256; re-download it (see the module docstring)")
    backbone = VendoredDinoV2()
    model = GazeLLE(backbone, inout=cfg["inout"])
    state = torch.load(ckpt, weights_only=True, map_location="cpu")
    missing = {k for k in model.state_dict() if not k.startswith("backbone")} - set(state)
    if missing:
        raise SystemExit(f"{ckpt.name} lacks decoder weights {sorted(missing)[:5]}")
    model.load_gazelle_state_dict(state)
    model.to(device).eval()
    return model, backbone.get_transform((448, 448))


class GazeModel:
    """gaze_stage contract: predict(frame, [(face_track, head box px)]) -> [(heatmap 64x64, inout)].

    One forward per frame: all heads of the frame share the scene encoding."""

    def __init__(self, checkpoint, device):
        self.device = device
        self.inout = CHECKPOINTS[checkpoint]["inout"]
        self.model, self.transform = load_model(checkpoint, device)
        self.sync = torch.mps.synchronize if device == "mps" else (lambda: None)
        self.forward_s = 0.0

    def predict(self, frame_bgr, heads):
        height, width = frame_bgr.shape[:2]
        image = self.transform(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0)
        norm = [(x1 / width, y1 / height, x2 / width, y2 / height) for _, (x1, y1, x2, y2) in heads]
        tick = time.perf_counter()
        with torch.no_grad():
            out = self.model({"images": image.to(self.device), "bboxes": [norm]})
        self.sync()
        heat = out["heatmap"][0].float().cpu().numpy()
        inout = out["inout"][0].float().cpu().numpy() if self.inout else None
        self.forward_s += time.perf_counter() - tick
        return [(heat[k], None if inout is None else float(inout[k])) for k in range(len(heads))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="vat", choices=list(CHECKPOINTS))
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--heatmaps", type=Path, help="also save every heatmap (float16) to this .npz, for draw_gaze.py")
    args = parser.parse_args()

    load_start = time.perf_counter()
    model = GazeModel(args.checkpoint, args.device)
    print(f"loaded {CHECKPOINTS[args.checkpoint]['name']} on {args.device} in {time.perf_counter() - load_start:.1f} s")
    header = {"model": "gaze-lle ViT-L/14 (frozen DINOv2 ViT-L/14 + gaze decoder), github.com/fkryan/gazelle @2d78f9f3",
              "checkpoint": CHECKPOINTS[args.checkpoint]["name"], "licence_mode": "academic-only", "licence": LICENCE}
    name = "gaze_lle.json" if args.checkpoint == "vat" else f"gaze_lle_{args.checkpoint}.json"
    run_stage(args.run_dir, model, header, args.run_dir / name, args.heatmaps)


if __name__ == "__main__":
    main()
