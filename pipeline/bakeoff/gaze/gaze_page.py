"""PaGE adapter for the gaze-target stage (gaze_stage.py has the targets, events and JSON shared with gaze_lle.py).

PaGE (Ye et al., arXiv 2607.04860, Jul 2026; github.com/OctopusWen/PaGE) has two DINOv3 branches: the scene at
512x512 and a crop of each head at 256x256, joined by 5 scene-head cross-attention layers; per head it returns a
64x64 heatmap over the whole frame and an in-frame probability (both sigmoids). Same contract as Gaze-LLE: image +
head box in, heatmap + in/out out.

Preprocessing follows the authors' released inference path (scripts/inference.py + PaGEImageProcessor in
modeling_page.py) exactly:
  - head box: pixel xyxy rounded to integers and clipped to the frame (inference.py parse_bbox); the head crop is
    scene.crop(box) of the RGB frame (inference.py crop_head) and the normalised box handed to the model is the
    integer box / frame size. We pass our own head box (gaze_stage.HEAD_BOX_SCALE x the py-feat face box, as for
    Gaze-LLE); PaGE does no head detection or box expansion of its own. Its training crops are the datasets'
    annotated head boxes (page/dataloader.py: img.crop(bbox), with jitter only in training), the same kind of box
    Gaze-LLE is trained on, so the same 1.2x scale applies.
  - scene and crop: PIL bilinear resize (preprocessor_config resample=2) to 512x512 and 256x256, aspect ratio NOT
    kept (the frame is squashed, as Gaze-LLE squashes to 448x448), /255, ImageNet mean/std.
    (Training and the authors' eval scripts instead resize after normalising with torchvision Resize on the tensor;
    the released processor is what the released checkpoints ship with, so that is what is used here.)
  - several heads in one forward: PaGEModel.get_logits runs the scene backbone once per image and repeats the
    scene features per head; the head backbone, the interaction layers and both heads run batched over the heads.
    check_device.py --model page-vitb checks batched against one-head-at-a-time.

The model code (modeling_page.py, from the HF repo Octopus1/PaGE that every weight repo's auto_map points at) and the
weights are pinned in vendor/ instead of trust_remote_code at run time, and the model is built from that file and
loaded with a strict load_state_dict (the authors' from_pretrained override wraps its backbone reload in a broad
except that only warns, so a mismatch would leave a random backbone; a strict load fails instead):
  git clone https://github.com/OctopusWen/PaGE vendor/page        (tested at 4352bead; only read, not imported)
  curl -L -o vendor/page_hf/modeling_page.py https://huggingface.co/Octopus1/PaGE/resolve/7147c67c440a110f050eb1017de50c7a1925c496/modeling_page.py
  for m in page-vitb page-vits: curl -L -o vendor/checkpoints/$m/<f> https://huggingface.co/Octopus1/$m/resolve/<revision>/<f>
    for f in config.json preprocessor_config.json model.safetensors LICENSE DINOv3_LICENSE.md README.md
(revisions and sha256 in CHECKPOINTS / MODELING_SHA256; checked at load.) Needs transformers==5.6.2 (modeling_page.py
builds DINOv3 from transformers' built-in dinov3_vit; the authors pin 5.6.2).

Usage: uv run python gaze_page.py ../runs/<name> [--checkpoint vitb|vits] [--device mps|cpu] [--heatmaps out.npz]
Writes runs/<name>/gaze_page.json (gaze_page_vits.json for --checkpoint vits).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

from gaze_stage import VENDOR, run_stage, sha256

MODELING = VENDOR / "page_hf/modeling_page.py"
MODELING_SHA256 = "5f21eb9a4f2ff10400b8819e21acd6b3503533dcb4d01e954acedc3566af43f2"  # Octopus1/PaGE @7147c67c
CHECKPOINTS = {
    "vitb": {"dir": "page-vitb", "revision": "b991d8bcb2890a97db611a2080ca250ecd943952",
             "sha256": "fb6571bdaf629524df24265b5734ac0c12aad70f05e3b43635b412c1c93af97f",
             "name": "Octopus1/page-vitb @b991d8bc (PaGE ViT-B distill: DINOv3 ViT-B/16 scene + head branches)"},
    "vits": {"dir": "page-vits", "revision": "aa8ddb4e5251b27e63a5365dc479b765c11c7296",
             "sha256": "6de1c3777b51f0a13745392046bd72812d9869f8034b9df6534b3183c43818a4",
             "name": "Octopus1/page-vits @aa8ddb4e (PaGE ViT-S distill: DINOv3 ViT-S/16 scene + head branches)"},
}
LICENCE = ("PaGE code MIT (github.com/OctopusWen/PaGE LICENSE; modeling_page.py MIT, HF Octopus1/PaGE). Weights: HF repos "
           "tagged MIT for the decoder and heads; the bundled DINOv3 backbones are derivative weights of DINOv3 under "
           "Meta's DINOv3 Licence (DINOv3_LICENSE.md, 2025-08-14: royalty-free use/modify/redistribute, no "
           "non-commercial clause; must ship the licence with redistribution, acknowledge DINO in publications, no "
           "ITAR/military/weapons uses, comply with trade controls). Training data: GazeFollow (terms unverified), "
           "VideoAttentionTarget (no dataset licence stated), ChildPlay (CC BY-NC), plus pseudo-labelled MPII and "
           "OpenImages heads (YOLOv5-CrowdHuman detector) for distillation. Head boxes come from py-feat RetinaFace "
           "(non-commercial weights). Academic use only")


def load_model(checkpoint, device):
    """PaGE from the pinned modeling_page.py and safetensors; returns (model, processor)."""
    from safetensors.torch import load_file

    cfg = CHECKPOINTS[checkpoint]
    ckpt_dir = VENDOR / "checkpoints" / cfg["dir"]
    for path, expected in ((MODELING, MODELING_SHA256), (ckpt_dir / "model.safetensors", cfg["sha256"])):
        if sha256(path) != expected:
            raise SystemExit(f"{path} does not match its pinned sha256; re-download it (see the module docstring)")
    sys.path.insert(0, str(MODELING.parent))
    from modeling_page import PaGEConfig, PaGEImageProcessor, PaGEModel

    config = PaGEConfig.from_json_file(str(ckpt_dir / "config.json"))
    model = PaGEModel(config)
    # strict: every parameter and persistent buffer must come from the checkpoint. PaGEBackbone's pre-hook remaps the
    # DINOv3 keys to this transformers version's naming during load_state_dict.
    model.load_state_dict(load_file(str(ckpt_dir / "model.safetensors")), strict=True)
    for module in model.modules():  # as the authors' from_pretrained does: RoPE inv_freq buffers are non-persistent
        if hasattr(module, "reset_inv_freq"):
            module.reset_inv_freq()
    model.to(device).eval()
    pre = json.loads((ckpt_dir / "preprocessor_config.json").read_text())
    processor = PaGEImageProcessor(scene_size=pre["scene_size"], head_size=pre["head_size"], image_mean=pre["image_mean"],
                                   image_std=pre["image_std"], resample=pre["resample"])
    return model, processor


def pixel_box(box, width, height):
    """inference.py parse_bbox for a pixel box: round to integers, clip to the frame; (pixel xyxy, normalised xyxy)."""
    x1 = max(0, min(width - 1, int(round(box[0]))))
    y1 = max(0, min(height - 1, int(round(box[1]))))
    x2 = max(0, min(width, int(round(box[2]))))
    y2 = max(0, min(height, int(round(box[3]))))
    if x2 <= x1 or y2 <= y1:
        raise SystemExit(f"head box {box} is empty after rounding to pixels")
    return (x1, y1, x2, y2), (x1 / width, y1 / height, x2 / width, y2 / height)


class GazeModel:
    """gaze_stage contract: predict(frame, [(face_track, head box px)]) -> [(heatmap 64x64, inout)].

    One forward per frame: the scene branch runs once, the head branch and decoder batched over the heads."""

    def __init__(self, checkpoint, device):
        self.device = device
        self.model, self.processor = load_model(checkpoint, device)
        self.sync = torch.mps.synchronize if device == "mps" else (lambda: None)
        self.forward_s = 0.0

    def predict(self, frame_bgr, heads):
        height, width = frame_bgr.shape[:2]
        scene = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes = [pixel_box(box, width, height) for _, box in heads]
        inputs = self.processor(scene, head_crops=[scene.crop(px) for px, _ in boxes], bboxes=[[norm for _, norm in boxes]])
        tick = time.perf_counter()
        with torch.inference_mode():
            out = self.model({"images": inputs["images"].to(self.device),
                              "head_images": [h.to(self.device) for h in inputs["head_images"]],
                              "bboxes": inputs["bboxes"]})
        self.sync()
        heat = out["heatmap"][0].float().cpu().numpy()
        inout = out["inout"][0].float().cpu().numpy()
        self.forward_s += time.perf_counter() - tick
        return [(heat[k], float(inout[k])) for k in range(len(heads))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="vitb", choices=list(CHECKPOINTS))
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--heatmaps", type=Path, help="also save every heatmap (float16) to this .npz, for draw_gaze.py")
    args = parser.parse_args()

    load_start = time.perf_counter()
    model = GazeModel(args.checkpoint, args.device)
    print(f"loaded {CHECKPOINTS[args.checkpoint]['name']} on {args.device} in {time.perf_counter() - load_start:.1f} s")
    header = {"model": "PaGE (DINOv3 scene 512 + head-crop 256 branches, 5 scene-head interaction layers), "
                       "github.com/OctopusWen/PaGE @4352bead, modeling_page.py HF Octopus1/PaGE @7147c67c",
              "checkpoint": CHECKPOINTS[args.checkpoint]["name"], "licence_mode": "academic-only", "licence": LICENCE}
    name = "gaze_page.json" if args.checkpoint == "vitb" else f"gaze_page_{args.checkpoint}.json"
    run_stage(args.run_dir, model, header, args.run_dir / name, args.heatmaps)


if __name__ == "__main__":
    main()
