"""Active speaker detection: TalkNet or LR-ASD scores for every face track in faces.json.

TalkNet (Tao et al. 2021) and LR-ASD (Liao et al. 2025) share one input format
(MIT code; both run with their TalkSet-trained weights): a 25 fps grayscale
face clip plus the matching audio (13 MFCCs at 100 Hz). Each scores every video
frame: > 0 means this face is speaking now. Instead of the repo's own S3FD
detection we reuse our py-feat tracks, so scores attach to Person A, B, ...
directly. Tracks were sampled at 4 fps; boxes are interpolated to 25 fps and
smoothed the way TalkNet's demo does before cropping.

Usage: uv run python asd_scores.py ../runs/<name> [--model talknet|lrasd] [--device mps|cpu]
Writes asd_<model>.json next to faces.json.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import python_speech_features
import torch
from scipy import signal

# Both repos ship top-level `loss` and `model` modules, so only one is imported per process.
MODELS = {
    "talknet": {"repo": "TalkNet-ASD", "weights": "pretrain_TalkSet.model", "cross_attention": True},
    "lrasd": {"repo": "LR-ASD", "weights": "weight/finetuning_TalkSet.model", "cross_attention": False},
}

FPS = 25                 # TalkNet's video rate; audio features are 4 per video frame
CROP_SCALE = 0.40        # TalkNet demo default: context kept around the face box
EDGE_PAD = 0.125         # seconds added at each end of a 4 fps track (half a sample gap), within its shot
MIN_TRACK_SECONDS = 0.4  # TalkNet demo minTrack is 10 frames
DURATIONS = (1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6)  # window lengths (s) averaged per frame, as in the demo


def load_model(name, device):
    cfg = MODELS[name]
    vendor = Path(__file__).parent / "vendor" / cfg["repo"]
    sys.path.insert(0, str(vendor))
    from loss import lossAV
    if name == "talknet":
        from model.talkNetModel import talkNetModel as Model
    else:
        from model.Model import ASD_Model as Model

    net = torch.nn.Module()
    net.model, net.lossAV = Model(), lossAV()
    state = torch.load(vendor / cfg["weights"], map_location="cpu", weights_only=True)
    result = net.load_state_dict({k: v for k, v in state.items() if not k.startswith(("lossA.", "lossV."))})
    assert not result.missing_keys and not result.unexpected_keys, result
    net.cross_attention = cfg["cross_attention"]
    return net.to(device).eval()


def track_frames(track, shot):
    """25 fps frame indices and interpolated [x, y, w, h] boxes for one 4 fps track."""
    start = max(shot["start"], track["start"] - EDGE_PAD)
    end = min(shot["end"] - 1 / FPS, track["end"] + EDGE_PAD)
    frames = np.arange(math.ceil(start * FPS), math.floor(end * FPS) + 1)
    t = np.array([s["t"] for s in track["series"]])
    boxes = np.array([s["box"] for s in track["series"]])
    interp = np.stack([np.interp(frames / FPS, t, boxes[:, i]) for i in range(4)], axis=1)
    return frames, interp


def crop_boxes(boxes):
    """TalkNet demo cropping: square around the face centre, median-smoothed, with extra room below the chin."""
    size = signal.medfilt(np.maximum(boxes[:, 2], boxes[:, 3]) / 2, kernel_size=13)
    cx = signal.medfilt(boxes[:, 0] + boxes[:, 2] / 2, kernel_size=13)
    cy = signal.medfilt(boxes[:, 1] + boxes[:, 3] / 2, kernel_size=13)
    return [(cy_ - s, cy_ + s * (1 + 2 * CROP_SCALE), cx_ - s * (1 + CROP_SCALE), cx_ + s * (1 + CROP_SCALE))
            for s, cx_, cy_ in zip(size, cx, cy)]


def crop(gray, y0, y1, x0, x1):
    """Crop with constant grey (110) outside the frame, then TalkNet's 224 -> centre 112 face patch."""
    h, w = gray.shape
    y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
    out = np.full((y1 - y0, x1 - x0), 110, np.uint8)
    sy0, sy1, sx0, sx1 = max(0, y0), min(h, y1), max(0, x0), min(w, x1)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = gray[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (224, 224))[56:168, 56:168]


def decode(video, width, height):
    """Yield (frame index, grayscale frame) at 25 fps."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={FPS}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    index = 0
    while chunk := proc.stdout.read(width * height):
        yield index, np.frombuffer(chunk, np.uint8).reshape(height, width)
        index += 1
    proc.wait()


def audio_16k(video):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16)


@torch.no_grad()
def score_clip(net, device, faces, mfcc):
    length = min((mfcc.shape[0] - mfcc.shape[0] % 4) / 100, faces.shape[0] / FPS)
    mfcc, faces = mfcc[:int(round(length * 100))], faces[:int(round(length * FPS))]
    runs = []
    for duration in DURATIONS:
        scores = []
        for i in range(math.ceil(length / duration)):
            a = torch.FloatTensor(mfcc[i * duration * 100:(i + 1) * duration * 100]).unsqueeze(0).to(device)
            v = torch.FloatTensor(faces[i * duration * FPS:(i + 1) * duration * FPS]).unsqueeze(0).to(device)
            ea, ev = net.model.forward_audio_frontend(a), net.model.forward_visual_frontend(v)
            if net.cross_attention:
                ea, ev = net.model.forward_cross_attention(ea, ev)
            scores.extend(net.lossAV(net.model.forward_audio_visual_backend(ea, ev)))
        runs.append(scores)
    return np.mean(np.array(runs), axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", choices=MODELS, default="talknet")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    extract = json.loads((args.run_dir / "extract.json").read_text())
    faces = json.loads((args.run_dir / "faces.json").read_text())
    video = Path(extract["video"]["source"])
    video = video if video.is_absolute() else args.run_dir.parent.parent / video
    shots = {s["index"]: s for s in extract["shots"]}

    plans = []  # (track index, frame indices, crop rects)
    for i, track in enumerate(faces["tracks"]):
        frames, boxes = track_frames(track, shots[track["shot"]])
        if len(frames) >= MIN_TRACK_SECONDS * FPS:
            plans.append((i, frames, crop_boxes(boxes)))
    wanted = {}
    for slot, (_, frames, _) in enumerate(plans):
        for j, f in enumerate(frames):
            wanted.setdefault(int(f), []).append((slot, j))

    t0 = time.time()
    clips = [np.zeros((len(frames), 112, 112), np.uint8) for _, frames, _ in plans]
    for index, gray in decode(video, extract["video"]["width"], extract["video"]["height"]):
        for slot, j in wanted.get(index, ()):
            clips[slot][j] = crop(gray, *plans[slot][2][j])
    t_crop = time.time() - t0

    audio = audio_16k(video)
    net = load_model(args.model, args.device)
    t0 = time.time()
    out = []
    for (i, frames, _), clip in zip(plans, clips):
        a = audio[int(frames[0] / FPS * 16000):int((frames[-1] + 1) / FPS * 16000)]
        mfcc = python_speech_features.mfcc(a, 16000, numcep=13, winlen=0.025, winstep=0.010)
        scores = score_clip(net, args.device, clip, mfcc)
        track = faces["tracks"][i]
        out.append({"track": i, "person": track["person"], "shot": track["shot"], "start": round(frames[0] / FPS, 3),
                    "fps": FPS, "scores": [round(float(s), 2) for s in scores]})
    t_score = time.time() - t0

    result = {"model": f"{MODELS[args.model]['repo']} {MODELS[args.model]['weights']} (MIT code)", "fps": FPS, "crop_scale": CROP_SCALE,
              "source_tracks": "faces.json (py-feat, 4 fps, interpolated)", "skipped_short_tracks":
              len(faces["tracks"]) - len(plans), "seconds": {"crop": round(t_crop, 1), "score": round(t_score, 1)},
              "tracks": out}
    (args.run_dir / f"asd_{args.model}.json").write_text(json.dumps(result))
    speaking = sum(sum(1 for s in t["scores"] if s > 0) for t in out)
    total = sum(len(t["scores"]) for t in out)
    print(f"{len(out)} tracks scored ({result['skipped_short_tracks']} too short), {speaking}/{total} face-frames "
          f"speaking; crop {t_crop:.0f}s, score {t_score:.0f}s on {args.device}")


if __name__ == "__main__":
    main()
