"""Checks grw.py's feature path against GRW's bundled samples, and MPS against CPU.

GRW ships two preprocessed clips with the SHuBERT features its own pipeline extracted from them
(vendor/grw/samples/{bye,below}.mp4 and *_shubert.npy; README: bye -> 'Semantic'). This script runs grw.py's
reimplementation (MediaPipe stage, hand crops, signing-space pose, DINOv2, SHuBERT) on each clip with the whole
frame as the person crop (the clips are already GRW person crops; GRW's second YOLO crop is not run), then
reports, per SHuBERT layer, the mean cosine similarity to the authors' features, and the classifier's
probability (query = whole clip, as inference.py's default) on our features and on theirs.
With --devices cpu,mps it repeats the run on each device and reports the largest differences.

Usage: uv run python grw_check.py [--devices cpu,mps]
"""

import argparse

import cv2
import numpy as np
import torch

import grw

SAMPLES = ["bye", "below"]


def clip_frames(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.get(cv2.CAP_PROP_FPS) == grw.FPS, f"{path} is not {grw.FPS} fps"
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def our_features(models, mp, hands, frames, device):
    dino, shubert, _ = models
    feats = grw.TrackFeatures(mp, hands)
    for rgb in frames:
        feats.add(rgb)
    feats.close()
    left = grw.dino_features(dino, feats.left, device)
    right = grw.dino_features(dino, feats.right, device)
    body = torch.from_numpy(np.stack(feats.pose)).float()
    return grw.shubert_features(shubert, left, right, body, device).cpu(), feats.counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", default="cpu,mps")
    args = parser.parse_args()
    mp, hands = grw.mediapipe_models()
    results = {}
    for name in args.devices.split(","):
        device = torch.device(name)
        models = grw.load_models(device)
        for sample in SAMPLES:
            frames = clip_frames(grw.GRW_REPO / "samples" / f"{sample}.mp4")
            ours, counts = our_features(models, mp, hands, frames, device)
            theirs = torch.from_numpy(np.load(grw.GRW_REPO / "samples" / f"{sample}_shubert.npy")).permute(1, 0, 2)
            assert ours.shape == theirs.shape, (ours.shape, theirs.shape)
            cos = torch.nn.functional.cosine_similarity(ours, theirs, dim=-1).mean(dim=0)  # per layer
            query = (0, ours.shape[0] - 1)
            p_ours = grw.classify(models[2], ours, query, device)
            p_theirs = grw.classify(models[2], theirs, query, device)
            results[(name, sample)] = (ours, p_ours, p_theirs)
            print(f"[{name}] {sample}: MediaPipe pose {counts['pose']}/{counts['frames']}, hands L {counts['left']} "
                  f"R {counts['right']}; cosine to GRW's features per layer {np.round(cos.numpy(), 3).tolist()}; "
                  f"P(semantic) ours {p_ours:.4f}, on GRW's features {p_theirs:.4f}")
    names = args.devices.split(",")
    for a, b in zip(names, names[1:]):
        for sample in SAMPLES:
            fa, pa, ta = results[(a, sample)]
            fb, pb, tb = results[(b, sample)]
            print(f"{a} vs {b}, {sample}: max |feature diff| {float((fa - fb).abs().max()):.2e} "
                  f"(feature std {float(fa.std()):.3f}); |P diff| ours {abs(pa - pb):.2e}, on GRW's features {abs(ta - tb):.2e}")


if __name__ == "__main__":
    main()
