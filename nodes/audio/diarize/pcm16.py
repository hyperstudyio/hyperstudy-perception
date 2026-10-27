"""Decode's audio (16 kHz mono 16-bit WAV) as the waveform pyannote is given: float32 in [-1, 1), samples / 32768.

What the bake-off gave it (WhisperX's load_audio: `ffmpeg -f s16le -ac 1 -ar 16000`, then / 32768), read with the
standard library rather than the ffmpeg program, which an app started from the Finder does not find, or pyannote's own
loader (torchcodec, which needs FFmpeg's shared libraries).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from perception_node import NodeError

RATE = 16000


def read_waveform(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        got = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        if got != (1, 2, RATE):
            raise NodeError("input", f"the audio is {got[0]} channel(s), {8 * got[1]}-bit, {got[2]} Hz; "
                                     f"decode writes 1 channel, 16-bit, {RATE} Hz")
        samples = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return samples.astype(np.float32) / 32768.0
