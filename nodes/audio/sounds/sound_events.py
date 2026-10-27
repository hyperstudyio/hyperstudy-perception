"""The sounds layer from the models' scores: numpy only, so the node tests can run it without torch.

Two kinds of event, as the bake-off's annotator built them (annotator/build.py: tag_rows in audio_events.py,
tagger_events, peaframe_events; pipeline/bundle/convert_bakeoff.py's sounds_layer):
- tagger events: the AudioSet taggers (PANNs Cnn14_DecisionLevelMax, EfficientAT mn40) score every 1 s step; each step
  keeps its top 5 labels scoring >= 0.2. Consecutive steps with a label become one event, and the two taggers' events
  of one label merge where they overlap. Speech and Silence are left out (speech is the transcript's job).
- query events: PE-A-Frame scores each 40 ms frame against a text query; frames above 0.3 become spans. An event's
  `agreement` says whether a tagger event with a matching AudioSet label overlaps it.

Long audio is cut into chunks (plan_chunks), because PANNs and PE-A-Frame take their whole input at once. A chunk
carries context on both sides, and only its core is kept (stitch). Audio no longer than one chunk is one piece, as the
bake-off ran it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STEP_S = 1.0              # the taggers' resolution: one row of labels per second
TAG_THRESHOLD = 0.2       # AudioSet sigmoid scores are poorly calibrated; 0.2 keeps clear secondary sounds
TOP_K = 5                 # labels kept per step
# EfficientAT has no frame-level head, so mn40 scores a window centred on each step. It was trained on 10 s clips: mn10
# collapsed to saturated labels on windows of 3 s or less in the bake-off, and was stable from 5 s.
EFFICIENTAT_WINDOW_S = 5.0
PANNS_FPS = 100           # Cnn14_DecisionLevelMax's frame-wise output (hop 320 at 32 kHz)
PE_THRESHOLD = 0.3        # the default in Meta's PEAudioFrame.forward (perception_models)
PE_HOP = 1920             # PE-A-Frame's frame, in samples at 48 kHz: 40 ms

# Chunks (plan_chunks). Up to 140 s is one piece for every model, as the bake-off ran its clips (119 and 137.6 s).
# - PANNs: 600 s with 9.6 s of context each side, far more than Cnn14's receptive field (about 2 s), and both whole
#   multiples of its 32-frame pooling grid (0.32 s, 10240 samples at 32 kHz), so a chunk's pooling lines up with one
#   pass's: on the CPU, chunks of an 825 s soundtrack give one pass's scores to 5e-10. A chunk must also stay under
#   PANNS_MPS_FRAMES: on the Apple GPU (torch 2.8), Cnn14 over more than about 65,536 frames (655 s) returns wrong
#   scores (off by up to 0.79 from the CPU's, against 1e-6 below it), with no error.
# - PE-A-Frame: 120 s with 10 s of context each side, in whole 40 ms frames. Its scores depend on how much audio it
#   attends over, not only near a chunk's edges: one pass over 240 s of the bake-off's clips and 100 s chunks of it
#   differed by up to 0.94, and the one pass found far more spans (car horn 198 against 71, crowd chatter 361 against
#   103). So no input is longer than the bake-off's longest one pass (137.6 s), and its memory stays bounded.
PANNS_CORE_S, PANNS_CONTEXT_S, PANNS_GRID, PANNS_MPS_FRAMES = 600, 9.6, 10240, 65536
PE_CORE_S, PE_CONTEXT_S = 120, 10

TAGGERS = ("panns", "mn40")
IGNORED_TAGS = {"Speech", "Silence"}   # speech is the transcript's job; silence is not an event

# Text queries for PE-A-Frame. The bake-off asked per clip (laughter, music and applause on OOO, plus five sounds
# absent from both clips to look for false alarms). A node runs on any video, so this is one list for videos of people
# interacting: the bake-off's eight (whose AudioSet matches are kept as they were, but for "Honk", a goose's call,
# dropped from car horn), sounds people make to and around each other, and sounds that interrupt a scene. "speech" is
# not asked: the transcript covers it (the bake-off asked and then skipped it). Each query lists the AudioSet labels
# (audioset_labels.csv) that count as a tagger agreeing with it.
AGREEING_TAGS = {
    # the bake-off's queries
    "laughter": {"Laughter", "Giggle", "Chuckle, chortle", "Belly laugh", "Snicker", "Baby laughter"},
    "music": {"Music", "Musical instrument", "Background music"},
    "applause": {"Applause", "Clapping", "Cheering"},
    "dog barking": {"Dog", "Bark", "Bow-wow", "Yip"},
    "gunshot": {"Gunshot, gunfire", "Machine gun", "Cap gun", "Explosion"},
    "glass breaking": {"Glass", "Shatter", "Breaking"},
    "car horn": {"Vehicle horn, car horn, honking", "Car alarm"},
    "baby crying": {"Baby cry, infant cry", "Crying, sobbing"},
    # voices other than speech
    "cheering": {"Cheering", "Applause", "Crowd"},
    "crowd chatter": {"Chatter", "Crowd", "Hubbub, speech noise, speech babble"},
    "singing": {"Singing", "Choir", "Chant", "Male singing", "Female singing", "Child singing", "Humming", "Vocal music"},
    "crying": {"Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan"},
    "screaming": {"Screaming", "Yell", "Shout", "Children shouting", "Battle cry"},
    "shouting": {"Shout", "Yell", "Bellow", "Children shouting", "Screaming"},
    "whispering": {"Whispering"},
    "sighing": {"Sigh", "Breathing", "Gasp"},
    "coughing": {"Cough", "Throat clearing", "Sneeze"},
    # sounds of the room
    "footsteps": {"Walk, footsteps"},
    "door knocking": {"Knock", "Door"},
    "door slamming": {"Slam", "Door", "Sliding door"},
    "phone ringing": {"Telephone", "Telephone bell ringing", "Ringtone"},
}
QUERIES = list(AGREEING_TAGS)

# AudioSet's class_labels_indices.csv (Google, CC-BY-4.0), as EfficientAT's metadata/ carries it (efficientat/UPSTREAM.md)
LABELS_CSV = Path(__file__).parent / "audioset_labels.csv"


def audioset_labels() -> list[str]:
    """The 527 AudioSet class names, in the taggers' output order."""
    with open(LABELS_CSV, newline="") as f:
        return [row[2] for row in list(csv.reader(f))[1:]]


# --- chunks ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    start: int        # the samples a model is given: [start, end)
    end: int
    core_start: int   # the samples whose output is kept: [core_start, core_end)
    core_end: int


def plan_chunks(n: int, core: int, context: int) -> list[Chunk]:
    """Cut n samples into cores of `core` samples, each given `context` samples more on both sides where there are any.
    Audio no longer than one chunk with its context is one chunk, whole."""
    if n <= core + 2 * context:
        return [Chunk(0, n, 0, n)]
    return [Chunk(max(0, s - context), min(n, s + core + context), s, min(n, s + core)) for s in range(0, n, core)]


def stitch(pieces: list[tuple[Chunk, np.ndarray]], hop: int, axis: int = -1) -> np.ndarray:
    """One output from each chunk's output: frames every `hop` samples, the first at the chunk's start, kept from its
    core. The last chunk keeps every frame to its end (a model may add a frame for the remainder). Chunk boundaries
    must fall on frames: core and context in whole frames."""
    kept = []
    for i, (chunk, out) in enumerate(pieces):
        last = i == len(pieces) - 1
        if (chunk.core_start - chunk.start) % hop or (not last and (chunk.core_end - chunk.start) % hop):
            raise ValueError(f"chunk {chunk} is not on {hop}-sample frames")
        a = (chunk.core_start - chunk.start) // hop
        b = None if last else (chunk.core_end - chunk.start) // hop
        kept.append(np.take(out, np.arange(a, out.shape[axis] if b is None else b), axis=axis))
    return np.concatenate(kept, axis=axis)


# --- taggers -----------------------------------------------------------------------------------------------------

def tag_steps(n_samples: int, sr: int) -> int:
    """The number of 1 s steps the taggers score: one for each started second."""
    return int(np.ceil(n_samples / sr / STEP_S))


def padded_for_windows(wave: np.ndarray, sr: int) -> np.ndarray:
    """The audio zero-padded so that the EfficientAT window of every step is centred on it, edges included."""
    n_steps = tag_steps(len(wave), sr)
    win, step = int(EFFICIENTAT_WINDOW_S * sr), int(STEP_S * sr)
    pad = (win - step) // 2
    return np.pad(wave, (pad, n_steps * step + win - len(wave) - pad))


def windows(padded: np.ndarray, sr: int, first: int, last: int) -> np.ndarray:
    """The EfficientAT windows of steps first..last-1, from padded_for_windows: (steps, samples)."""
    win, step = int(EFFICIENTAT_WINDOW_S * sr), int(STEP_S * sr)
    return np.stack([padded[i * step:i * step + win] for i in range(first, last)])


def step_max(frames: np.ndarray, fps: int = PANNS_FPS) -> np.ndarray:
    """Frame-wise scores (frames, classes) as one row per 1 s step: the max over the step, as the model's own clip
    decision is the max over frames."""
    per_step = int(STEP_S * fps)
    n_steps = int(np.ceil(frames.shape[0] / per_step))
    return np.stack([frames[i * per_step:(i + 1) * per_step].max(0) for i in range(n_steps)])


def tag_rows(scores: np.ndarray, labels: list[str]) -> list[dict]:
    """Each step's labels: its top TOP_K, those scoring >= TAG_THRESHOLD, as [label, score to 3 places]."""
    rows = []
    for i, s in enumerate(scores):
        order = np.argsort(s)[::-1][:TOP_K]
        rows.append({"start": i * STEP_S, "end": (i + 1) * STEP_S,
                     "labels": [[labels[j], round(float(s[j]), 3)] for j in order if s[j] >= TAG_THRESHOLD]})
    return rows


def tagger_events(steps: dict[str, list[dict]]) -> list[dict]:
    """Consecutive 1 s steps with the same label become one event; panns and mn40 merge when they overlap."""
    spans = []
    for model in TAGGERS:
        if model not in steps:
            continue
        open_ = {}
        for step in steps[model]:
            labels = {name: p for name, p in step["labels"] if name not in IGNORED_TAGS}
            for name in list(open_):
                if name not in labels:
                    spans.append(open_.pop(name))
            for name, p in labels.items():
                if name in open_ and abs(open_[name]["end"] - step["start"]) < 1e-6:
                    open_[name]["end"] = step["end"]
                    open_[name]["score"] = max(open_[name]["score"], p)
                else:
                    open_[name] = {"label": name, "start": step["start"], "end": step["end"], "score": p, "sources": [model]}
        spans.extend(open_.values())
    spans.sort(key=lambda s: (s["label"], s["start"]))
    merged = []
    for s in spans:
        last = merged[-1] if merged else None
        if last and last["label"] == s["label"] and s["start"] <= last["end"]:
            last["end"] = max(last["end"], s["end"])
            last["score"] = max(last["score"], s["score"])
            last["sources"] = sorted(set(last["sources"]) | set(s["sources"]))
        else:
            merged.append(dict(s))
    return merged


# --- queries -----------------------------------------------------------------------------------------------------

def spans(probs: np.ndarray, sr: int, threshold: float = PE_THRESHOLD, hop: int = PE_HOP) -> list[list[float]]:
    """Runs of frames scoring above threshold, as [start, end] seconds to 2 places (Meta's PEAudioFrame.forward rule)."""
    hop_s = hop / sr
    on = np.pad(probs > threshold, (1, 1)).astype(np.int8)
    edges = np.flatnonzero(np.diff(on)).tolist()
    return [[round(a * hop_s, 2), round(b * hop_s, 2)] for a, b in zip(edges[::2], edges[1::2])]


def query_events(query_spans: dict[str, list[list[float]]], taggers: list[dict]) -> list[dict]:
    """One event per span of each query, with whether a tagger event of a matching label overlaps it."""
    events = []
    for query, found in query_spans.items():
        agreeing = AGREEING_TAGS.get(query, set())
        for start, end in found:
            agree = any(t["label"] in agreeing and t["start"] < end and t["end"] > start for t in taggers)
            events.append({"label": query, "start": start, "end": end, "score": None, "sources": ["pe-a-frame"], "agreement": agree})
    return events


def sound_events(steps: dict[str, list[dict]], query_spans: dict[str, list[list[float]]]) -> list[dict]:
    """The layer's events, by start time (tagger events first among equal starts, in label order; then queries in
    query order), numbered in that order."""
    taggers = tagger_events(steps)
    events = sorted([{**t, "agreement": None} for t in taggers] + query_events(query_spans, taggers), key=lambda e: e["start"])
    return [{"id": f"sound:{i}", "label": e["label"], "start": e["start"], "end": e["end"], "sources": e["sources"],
             "agreement": e["agreement"], "score": e["score"]} for i, e in enumerate(events)]

