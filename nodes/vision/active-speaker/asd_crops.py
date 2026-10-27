"""What LR-ASD is given for one face track, as the bake-off's asd/asd_scores.py prepared it: the 25 fps frames the track
spans, a smoothed square crop per frame, the audio under them, and the windows each frame is scored in. numpy only.

LR-ASD (like TalkNet, whose demo pipeline this follows) scores a 25 fps 112x112 grey face clip against 13 MFCCs at
100 Hz. Instead of the demo's own face detector, the faces layer's tracks are used, so scores attach to its face tracks
(and through them to people) directly. A track's samples (8 fps) are interpolated to 25 fps and widened by EDGE_PAD at
each end, within its shot; tracks shorter than MIN_TRACK_SECONDS are not scored.
"""

from __future__ import annotations

import math

import numpy as np

FPS = 25                 # LR-ASD's video rate; audio features are 4 per video frame
AUDIO_RATE = 16000       # decode's audio, as python_speech_features is given it
MFCC_RATE = 100          # MFCC frames per second (winstep 10 ms)
CROP_SCALE = 0.40        # TalkNet demo default: context kept around the face box
EDGE_PAD = 0.125         # seconds added at each end of a track, within its shot (the bake-off's constant)
MIN_TRACK_SECONDS = 0.4  # TalkNet demo minTrack is 10 frames
MEDIAN_KERNEL = 13       # the demo's smoothing of the crop's size and centre (scipy.signal.medfilt)
DURATIONS = (1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6)  # window lengths (s) whose scores are averaged per frame, as in the demo
OUTSIDE_GREY = 110       # what a crop holds outside the frame
CROP_SIZE, FACE_SIZE = 224, 112   # the crop is resized to 224 x 224 and its centre 112 x 112 kept


def track_frames(track: dict, shot: dict) -> tuple[np.ndarray, np.ndarray]:
    """25 fps frame indices (frame i at i / FPS s) and [x, y, w, h] boxes interpolated from the track's samples, for
    the track widened by EDGE_PAD and kept within its shot (the last frame before the shot's end)."""
    start = max(shot["start"], track["start"] - EDGE_PAD)
    end = min(shot["end"] - 1 / FPS, track["end"] + EDGE_PAD)
    frames = np.arange(math.ceil(start * FPS), math.floor(end * FPS) + 1)
    t = np.array([s["t"] for s in track["samples"]], dtype=float)
    boxes = np.array([s["box"] for s in track["samples"]], dtype=float).reshape(-1, 4)
    interp = np.stack([np.interp(frames / FPS, t, boxes[:, i]) for i in range(4)], axis=1).reshape(-1, 4)
    return frames, interp


def medfilt(x: np.ndarray, kernel: int = MEDIAN_KERNEL) -> np.ndarray:
    """scipy.signal.medfilt(x, kernel) of a 1-D array: the median of each odd window, zero-padded at both ends."""
    pad = kernel // 2
    padded = np.concatenate([np.zeros(pad), np.asarray(x, dtype=float), np.zeros(pad)])
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, kernel), axis=1)


def crop_boxes(boxes: np.ndarray) -> list[tuple[float, float, float, float]]:
    """TalkNet demo cropping: (y0, y1, x0, x1) of a square around the face centre, median-smoothed over time, with
    extra room below the chin."""
    size = medfilt(np.maximum(boxes[:, 2], boxes[:, 3]) / 2)
    cx = medfilt(boxes[:, 0] + boxes[:, 2] / 2)
    cy = medfilt(boxes[:, 1] + boxes[:, 3] / 2)
    return [(cy_ - s, cy_ + s * (1 + 2 * CROP_SCALE), cx_ - s * (1 + CROP_SCALE), cx_ + s * (1 + CROP_SCALE))
            for s, cx_, cy_ in zip(size, cx, cy)]


def padded_region(gray: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> np.ndarray:
    """The crop's pixels (corners truncated to integers), OUTSIDE_GREY where it leaves the frame. Then resized to
    CROP_SIZE and its centre kept (video25.face_patch)."""
    h, w = gray.shape
    y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
    out = np.full((y1 - y0, x1 - x0), OUTSIDE_GREY, np.uint8)
    sy0, sy1, sx0, sx1 = max(0, y0), min(h, y1), max(0, x0), min(w, x1)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = gray[sy0:sy1, sx0:sx1]
    return out


def plans_for(tracks: list[dict], shots: dict[str, dict]) -> list[tuple[int, np.ndarray, list]]:
    """(track position, frame indices, crop rectangles) for every track long enough to score, in track order."""
    plans = []
    for i, track in enumerate(tracks):
        if track["shot"] not in shots:
            raise ValueError(f"{track['id']} is in {track['shot']}, which the shots layer does not have")
        frames, boxes = track_frames(track, shots[track["shot"]])
        if len(frames) >= MIN_TRACK_SECONDS * FPS:
            plans.append((i, frames, crop_boxes(boxes)))
    return plans


def wanted_frames(plans) -> dict[int, list[tuple[int, int]]]:
    """frame index -> [(plan slot, position in that plan's clip)]: what to cut from each decoded frame."""
    wanted: dict[int, list[tuple[int, int]]] = {}
    for slot, (_, frames, _) in enumerate(plans):
        for j, f in enumerate(frames):
            wanted.setdefault(int(f), []).append((slot, j))
    return wanted


def audio_span(frames: np.ndarray) -> tuple[int, int]:
    """The audio samples under frames[0] .. frames[-1] (to the end of the last frame)."""
    return int(frames[0] / FPS * AUDIO_RATE), int((frames[-1] + 1) / FPS * AUDIO_RATE)


def clip_windows(n_mfcc: int, n_frames: int, durations=DURATIONS) -> list[list[tuple[int, int, int, int]]]:
    """Per window length: the (MFCC start, end, frame start, end) windows a clip is scored in. The clip is first cut
    to the shorter of its audio (whole video frames of MFCCs) and its frames; each length tiles it from the start, the
    last window shorter. Every frame is scored once per length; its score is the mean over the lengths."""
    length = min((n_mfcc - n_mfcc % 4) / MFCC_RATE, n_frames / FPS)
    n_a, n_v = int(round(length * MFCC_RATE)), int(round(length * FPS))
    return [[(min(i * d * MFCC_RATE, n_a), min((i + 1) * d * MFCC_RATE, n_a), min(i * d * FPS, n_v), min((i + 1) * d * FPS, n_v))
             for i in range(math.ceil(length / d))] for d in durations]
