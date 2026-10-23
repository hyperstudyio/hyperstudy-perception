"""Checks behind gaze_lle.py's and gaze_page.py's choices, on a few frames of one run: MPS against CPU, the Gaze-LLE
VAT checkpoint against the GazeFollow one, the head-box scale, and (PaGE) several heads per forward against one.

  device      the model on --device and on CPU, same inputs: max |heatmap difference|, same peak cell, |inout difference|
  checkpoint  (lle) VAT vs GazeFollow (both on --device): peak distance (fraction of the frame diagonal), same target
  scales      head boxes at 1.0 / 1.4 x the face box against 1.2: peak distance and target agreement
  batch       (page) all heads of a frame in one forward vs one forward per head: max |heatmap / inout difference|

--model: lle (Gaze-LLE VAT), page-vitb, page-vits. Frames are the --frames sampled frames with the most heads, evenly
spread over the clip.

Usage: uv run python check_device.py ../runs/<name> [--model lle] [--frames 12] [--device mps] [--skip device,checkpoint,scales,batch]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import gaze_lle
import gaze_page
from gaze_stage import BAKEOFF, candidates_at, frame_heads, frames_at, heads_by_frame, target_for


def make_model(name, device):
    if name == "lle":
        return gaze_lle.GazeModel("vat", device)
    return gaze_page.GazeModel(name.removeprefix("page-"), device)


def pick_frames(heads, n):
    """n frame indices spread over the clip, each the most crowded frame of its stretch."""
    ordered = sorted(heads)
    chunks = np.array_split(np.array(ordered), n)
    return [int(max(c, key=lambda i: len(heads[int(i)]))) for c in chunks if len(c)]


def run_model(model, frames, heads, scale, width, height):
    """frame index -> (heatmaps [N, 64, 64], inout [N] or None)."""
    out = {}
    for index, frame in frames:
        pred = model.predict(frame, frame_heads(heads[index], width, height, scale))
        inout = None if pred[0][1] is None else np.array([io for _, io in pred])
        out[index] = (np.stack([h for h, _ in pred]), inout)
    return out


def targets(results, heads, faces, objects_by_shot, width, height, scale):
    """[(peak, target)] per head, in frame then head order."""
    face_tracks = dict(enumerate(faces["tracks"]))
    rows = []
    for index, (heat, inout) in results.items():
        for k, (tid, j, _) in enumerate(heads[index]):
            t = face_tracks[tid]["series"][j]["t"]
            cands = candidates_at(tid, face_tracks[tid]["shot"], t, face_tracks, objects_by_shot, width, height, scale)
            target, _, peak = target_for(heat[k], None if inout is None else float(inout[k]), cands, width, height)
            rows.append((peak, target))
    return rows


def compare(name, a, b, diag):
    dists = [math.dist(pa, pb) / diag for (pa, _), (pb, _) in zip(a, b)]
    same = sum(ta == tb for (_, ta), (_, tb) in zip(a, b))
    print(f"{name}: {len(a)} heads; peak distance / diagonal median {np.median(dists):.3f}, "
          f"p90 {np.percentile(dists, 90):.3f}; same target {same}/{len(a)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="lle", choices=["lle", "page-vitb", "page-vits"])
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--skip", default="")
    args = parser.parse_args()
    skip = set(args.skip.split(",")) - {""}

    run = args.run_dir
    extract = json.loads((run / "extract.json").read_text())
    faces = json.loads((run / "faces.json").read_text())
    objects = json.loads((run / "objects.json").read_text()) if (run / "objects.json").exists() else {"tracks": []}
    objects_by_shot = {}
    for obj in objects["tracks"]:
        objects_by_shot.setdefault(obj["shot"], []).append(obj)
    video = Path(extract["video"]["source"])
    video = video if video.is_absolute() else BAKEOFF / video
    fps, width, height = extract["video"]["fps"], extract["video"]["width"], extract["video"]["height"]
    diag = math.hypot(width, height)
    heads = heads_by_frame(faces, fps)
    chosen = pick_frames(heads, args.frames)
    frames = list(frames_at(video, chosen))
    print(f"{len(frames)} frames, {sum(len(heads[i]) for i in chosen)} heads: t = {[round(i / fps, 2) for i in chosen]}")

    vat = make_model(args.model, args.device)
    base = run_model(vat, frames, heads, 1.2, width, height)
    base_rows = targets(base, heads, faces, objects_by_shot, width, height, 1.2)

    if "device" not in skip:
        cpu = make_model(args.model, "cpu")
        on_cpu = run_model(cpu, frames, heads, 1.2, width, height)
        del cpu
        heat_diff = max(float(np.abs(base[i][0] - on_cpu[i][0]).max()) for i in base)
        inout_diff = max(float(np.abs(base[i][1] - on_cpu[i][1]).max()) for i in base)
        same_peak = sum(int(h1.argmax()) == int(h2.argmax()) for i in base for h1, h2 in zip(base[i][0], on_cpu[i][0]))
        print(f"device {args.device} vs cpu: max |heatmap diff| {heat_diff:.2e}, max |inout diff| {inout_diff:.2e}, "
              f"same peak cell {same_peak}/{sum(len(base[i][0]) for i in base)}")
        compare(f"device {args.device} vs cpu", base_rows, targets(on_cpu, heads, faces, objects_by_shot, width, height, 1.2), diag)

    if args.model == "lle" and "checkpoint" not in skip:
        gf = gaze_lle.GazeModel("gazefollow", args.device)
        gf_out = run_model(gf, frames, heads, 1.2, width, height)
        del gf
        gf_rows = targets(gf_out, heads, faces, objects_by_shot, width, height, 1.2)
        inout = [float(x) for i in base for x in base[i][1]]
        in_frame = [k for k, v in enumerate(inout) if v >= 0.5]
        print(f"VAT inout: {len(in_frame)}/{len(inout)} heads in frame (>= 0.5); values {np.round(sorted(inout), 2).tolist()}")
        compare("VAT vs GazeFollow (all heads)", base_rows, gf_rows, diag)
        compare("VAT vs GazeFollow (VAT in-frame heads)", [base_rows[k] for k in in_frame], [gf_rows[k] for k in in_frame], diag)
        # per-head listing, to look at the disagreements on the frames
        n = 0
        for index in chosen:
            for tid, j, _ in heads[index]:
                print(f"  t={index / fps:.3f} face {tid}: VAT {base_rows[n][1]} inout {inout[n]:.2f} peak {np.round(base_rows[n][0]).tolist()} | "
                      f"GF {gf_rows[n][1]} peak {np.round(gf_rows[n][0]).tolist()}")
                n += 1

    if args.model != "lle" and "batch" not in skip:
        heat_diff = inout_diff = 0.0
        for index, frame in frames:
            one = [vat.predict(frame, [h])[0] for h in frame_heads(heads[index], width, height)]
            heat_diff = max(heat_diff, max(float(np.abs(base[index][0][k] - h).max()) for k, (h, _) in enumerate(one)))
            inout_diff = max(inout_diff, max(abs(float(base[index][1][k]) - io) for k, (_, io) in enumerate(one)))
        print(f"batch: all heads in one forward vs one head per forward: max |heatmap diff| {heat_diff:.2e}, "
              f"max |inout diff| {inout_diff:.2e}")

    if "scales" not in skip:
        for scale in (1.0, 1.4):
            out = run_model(vat, frames, heads, scale, width, height)
            compare(f"scale {scale} vs 1.2", base_rows, targets(out, heads, faces, objects_by_shot, width, height, scale), diag)
    if args.device == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
