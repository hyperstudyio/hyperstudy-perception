"""Audio events for the bake-off: AudioSet taggers plus free-text sound localization.

Three open model families, all run on the clip's soundtrack:
- EfficientAT MobileNets (mn10_as, mn40_as; MIT code and weights): clip-level
  AudioSet taggers (527 classes). They have no frame-level head, so they are
  run on 5 s windows centred on every 1 s step.
- PANNs Cnn14_DecisionLevelMax (MIT code, CC-BY-4.0 weights): frame-wise AudioSet
  sound event detection at 100 frames/s, max-pooled into 1 s steps.
- Meta PE-A-Frame (Apache-2.0, via transformers plus a one-line port fix, see
  load_peaframe): text queries scored against 40 ms audio frames; frames above
  the model's threshold become time spans.

The audio is decoded once per sample rate with ffmpeg. Every model x device run
is timed (load, cold inference, warm inference) and merged into
runs/<name>/audio_events.json, so models can be run one per process (which keeps
peak-memory measurements under /usr/bin/time -l separate).

Usage: uv run python audio_events.py ../runs/<name> [--device mps|cpu]
           [--models mn10,mn40,panns,peaframe,peaframe_small]  (efficientat = mn10,mn40)
"""

import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

VENDOR = Path(__file__).parent / "vendor"
BAKEOFF = Path(__file__).parent.parent  # extract.json's video.source is relative to this

STEP_S = 1.0               # output resolution for the taggers: one row of labels per second
# Context per 1 s step. The models were trained on 10 s clips; mn10 collapses to saturated nonsense
# (Sawing/Cowbell/Hammer ~1.0) on windows of 3 s or less, and is stable from 5 s, so 5 s it is.
EFFICIENTAT_WINDOW_S = 5.0
TAG_THRESHOLD = 0.2        # AudioSet sigmoid scores are poorly calibrated; 0.2 keeps clear secondary sounds
TOP_K = 5                  # labels kept per step
PE_THRESHOLD = 0.3         # the default in Meta's PEAudioFrame.forward (perception_models)

MODELS = {
    "mn10": {"family": "efficientat", "weights": "EfficientAT/resources/mn10_as_mAP_471.pt", "sr": 32000},
    "mn40": {"family": "efficientat", "weights": "EfficientAT/resources/mn40_as_mAP_484.pt", "sr": 32000},
    "panns": {"family": "panns", "weights": "panns/Cnn14_DecisionLevelMax_mAP=0.385.pth", "sr": 32000},
    "peaframe": {"family": "peaframe", "weights": "pe-a-frame-large", "sr": 48000},
    "peaframe_small": {"family": "peaframe", "weights": "pe-a-frame-small", "sr": 48000},
}
GROUPS = {"efficientat": ["mn10", "mn40"]}

# Sounds expected in each clip, plus sounds absent from both (a first look at false alarms).
QUERIES = {
    "nes-biking": ["music", "bicycle riding on dirt", "wind", "tires skidding", "crowd cheering", "speech"],
    "ooo-genz-clip1": ["speech", "laughter", "music", "applause"],
}
ABSENT_PROBES = ["dog barking", "gunshot", "glass breaking", "car horn", "baby crying"]


def decode(video, sr):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def audioset_labels():
    import csv
    with open(VENDOR / "EfficientAT/metadata/class_labels_indices.csv") as f:
        return [row[2] for row in list(csv.reader(f))[1:]]


# --- EfficientAT ---------------------------------------------------------------

def load_efficientat(name, device):
    # The repo resolves metadata/ and resources/ relative to the working directory at import and load time.
    with contextlib.chdir(VENDOR / "EfficientAT"), contextlib.redirect_stdout(io.StringIO()):
        sys.path.insert(0, ".")
        from helpers.utils import NAME_TO_WIDTH
        from models.mn.model import get_model
        from models.preprocess import AugmentMelSTFT
        model = get_model(width_mult=NAME_TO_WIDTH(f"{name}_as"), pretrained_name=f"{name}_as")
        mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320)  # EATagger defaults
    return model.to(device).eval(), mel.to(device).eval()


@torch.no_grad()
def run_efficientat(loaded, wave, sr, device):
    model, mel = loaded
    n_steps = int(np.ceil(len(wave) / sr / STEP_S))
    win, step = int(EFFICIENTAT_WINDOW_S * sr), int(STEP_S * sr)
    pad = (win - step) // 2  # zero-pad so the window is centred on each step, edges included
    padded = np.pad(wave, (pad, n_steps * step + win - len(wave) - pad))
    windows = np.stack([padded[i * step:i * step + win] for i in range(n_steps)])
    spec = mel(torch.from_numpy(windows).to(device))
    logits, _ = model(spec.unsqueeze(1))
    return torch.sigmoid(logits).cpu().numpy()  # (steps, 527)


# --- PANNs ---------------------------------------------------------------------

def load_panns(device):
    import importlib.util
    # PANNs' pytorch/models.py would clash with EfficientAT's `models` package, so load it under its own name.
    repo = VENDOR / "audioset_tagging_cnn/pytorch"
    sys.path.insert(0, str(repo))  # for its `pytorch_utils` import
    spec = importlib.util.spec_from_file_location("panns_models", repo / "models.py")
    panns_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panns_models)
    Cnn14_DecisionLevelMax = panns_models.Cnn14_DecisionLevelMax
    # Hyperparameters from the repo's scripts/0_inference.sh for this checkpoint.
    model = Cnn14_DecisionLevelMax(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                                   fmin=50, fmax=14000, classes_num=527)
    checkpoint = torch.load(VENDOR / MODELS["panns"]["weights"], map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


@torch.no_grad()
def run_panns(model, wave, sr, device):
    frames = model(torch.from_numpy(wave)[None].to(device))["framewise_output"][0].cpu().numpy()  # 100 fps
    per_step = int(STEP_S * 100)
    n_steps = int(np.ceil(frames.shape[0] / per_step))
    # Max over each 1 s step, matching the model's own max-pooled clip decision.
    return np.stack([frames[i * per_step:(i + 1) * per_step].max(0) for i in range(n_steps)])


# --- PE-A-Frame ----------------------------------------------------------------

def load_peaframe(name, device):
    from transformers import PeAudioFrameLevelModel, PeAudioProcessor
    path = VENDOR / MODELS[name]["weights"]
    model = PeAudioFrameLevelModel.from_pretrained(path)
    # Port bug in transformers 5.20.2: Meta's PEAudioFrame embeds text from hidden_states[22] of a 28-layer
    # ModernBERT, i.e. layer 22's raw output, but the port truncates to 22 layers and applies ModernBERT's
    # final_norm on top. That shifts every text embedding (~95% relative error) and inflates the probabilities
    # (absent probes like "gunshot" fired across most of the NES clip). Removing final_norm reproduces Meta's
    # reference (perception_models @ 3e352cc, its own checkpoint) to <1e-4 on both small and large.
    model.text_model.final_norm = torch.nn.Identity()
    return model.to(device).eval(), PeAudioProcessor.from_pretrained(path)


@torch.no_grad()
def run_peaframe(loaded, wave, sr, device, queries):
    model, processor = loaded
    inputs = processor(text=queries, audio=wave, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
    out = model(**inputs)
    probs = out.logits_audio_text[0].sigmoid().cpu()  # (queries, frames) for our single clip
    hop_s = processor.feature_extractor.hop_length / sr  # 1920 / 48000 = 40 ms
    results = {}
    for query, p in zip(queries, probs):
        on = torch.nn.functional.pad(p > PE_THRESHOLD, (1, 1)).int()
        edges = torch.diff(on).nonzero().flatten().tolist()  # same span rule as Meta's PEAudioFrame.forward
        spans = [[round(a * hop_s, 2), round(b * hop_s, 2)] for a, b in zip(edges[::2], edges[1::2])]
        results[query] = {"spans": spans, "n_spans": len(spans),
                          "total_s": round(sum(b - a for a, b in spans), 2),
                          "max_prob": round(float(p.max()), 3)}
    return results


# --- output --------------------------------------------------------------------

def tag_rows(scores, labels):
    rows = []
    for i, s in enumerate(scores):
        order = np.argsort(s)[::-1][:TOP_K]
        rows.append({"start": i * STEP_S, "end": (i + 1) * STEP_S,
                     "labels": [[labels[j], round(float(s[j]), 3)] for j in order if s[j] >= TAG_THRESHOLD]})
    return rows


def summarize_tags(rows):
    """Merge consecutive steps that share a top label; show its mean score and the runners-up."""
    lines, run = [], []
    for row in rows + [None]:
        top = row["labels"][0][0] if row and row["labels"] else None
        if run and (row is None or top != run[0]["top"]):
            scores = {}
            for r in run:
                for label, score in r["labels"]:
                    scores.setdefault(label, []).append(score)
            means = sorted(((sum(v) / len(run), k) for k, v in scores.items()), reverse=True)[:3]
            body = ", ".join(f"{k} {m:.2f}" for m, k in means) or "(nothing above threshold)"
            lines.append(f"  {run[0]['start']:.0f}-{run[-1]['end']:.0f} s: {body}")
            run = []
        if row is not None:
            run.append({**row, "top": top})
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--models", default="efficientat,panns,peaframe")
    args = parser.parse_args()

    names = [m for g in args.models.split(",") for m in GROUPS.get(g, [g])]
    unknown = [m for m in names if m not in MODELS]
    if unknown:
        sys.exit(f"unknown models {unknown}; choose from {list(MODELS)} or {list(GROUPS)}")

    extract = json.loads((args.run_dir / "extract.json").read_text())
    video = BAKEOFF / extract["video"]["source"]
    clip = args.run_dir.name
    queries = QUERIES[clip] + ABSENT_PROBES
    device = torch.device(args.device)

    out_path = args.run_dir / "audio_events.json"
    result = json.loads(out_path.read_text()) if out_path.exists() else {"models": {}, "timings": {}}
    result["video"] = extract["video"]
    labels = audioset_labels()

    waves, decode_s = {}, {}
    for sr in sorted({MODELS[m]["sr"] for m in names}):
        t0 = time.time()
        waves[sr] = decode(video, sr)
        decode_s[sr] = round(time.time() - t0, 2)
    sr0 = min(waves)
    duration = len(waves[sr0]) / sr0

    for name in names:
        cfg, sr = MODELS[name], MODELS[name]["sr"]
        t0 = time.time()
        if cfg["family"] == "efficientat":
            loaded = load_efficientat(name, device)
            infer = lambda: run_efficientat(loaded, waves[sr], sr, device)
        elif cfg["family"] == "panns":
            loaded = load_panns(device)
            infer = lambda: run_panns(loaded, waves[sr], sr, device)
        else:
            loaded = load_peaframe(name, device)
            infer = lambda: run_peaframe(loaded, waves[sr], sr, device, queries)
        sync(device)
        load_s = time.time() - t0

        infer_s = []
        for _ in range(2):  # cold (includes MPS kernel compilation), then warm
            t0 = time.time()
            output = infer()
            sync(device)
            infer_s.append(time.time() - t0)

        timing = {"device": args.device, "load_s": round(load_s, 2), "infer_cold_s": round(infer_s[0], 2),
                  "infer_warm_s": round(infer_s[1], 2), "rtf_cold": round(infer_s[0] / duration, 4),
                  "rtf_warm": round(infer_s[1] / duration, 4), "decode_s": decode_s[sr], "audio_s": round(duration, 2)}
        if device.type == "mps":
            timing["mps_driver_allocated_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
        result["timings"][f"{name}@{args.device}"] = timing

        entry = {"device": args.device, "weights": cfg["weights"]}
        if cfg["family"] == "peaframe":
            entry.update({"threshold": PE_THRESHOLD, "frame_s": 0.04, "queries": output,
                          "absent_probes": ABSENT_PROBES})
        else:
            entry.update({"step_s": STEP_S, "threshold": TAG_THRESHOLD, "top_k": TOP_K,
                          "steps": tag_rows(output, labels)})
            if cfg["family"] == "efficientat":
                entry["window_s"] = EFFICIENTAT_WINDOW_S
        result["models"][name] = entry
        del loaded, infer
        if device.type == "mps":
            torch.mps.empty_cache()

        print(f"\n{name} on {args.device}: load {timing['load_s']} s, infer {timing['infer_cold_s']} s cold / "
              f"{timing['infer_warm_s']} s warm, RTF {timing['rtf_warm']}")
        if cfg["family"] == "peaframe":
            for query, r in output.items():
                tag = " (absent probe)" if query in ABSENT_PROBES else ""
                spans = ", ".join(f"{a:.1f}-{b:.1f}" for a, b in r["spans"][:8]) + (" ..." if r["n_spans"] > 8 else "")
                print(f"  {query!r}{tag}: {r['n_spans']} spans, {r['total_s']} s, max p {r['max_prob']}  {spans}")
        else:
            print("\n".join(summarize_tags(entry["steps"])))

    out_path.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
