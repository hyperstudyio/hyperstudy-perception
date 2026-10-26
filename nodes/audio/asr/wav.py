"""Decode's audio (16 kHz mono 16-bit WAV) as samples, read here rather than by parakeet-mlx: its load_audio runs the
ffmpeg program, which an app started from the Finder does not find (its PATH is /usr/bin:/bin:/usr/sbin:/sbin)."""

import wave
from pathlib import Path

import numpy as np

from perception_node import NodeError


def read_wav(path: Path, rate: int) -> np.ndarray:
    """The samples, as int16: what `ffmpeg -i path -f s16le -ac 1 -ar rate -` printed for a WAV already in that form."""
    with wave.open(str(path)) as w:
        got = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        if got != (1, 2, rate):
            raise NodeError("input", f"the audio is {got[0]} channel(s), {8 * got[1]}-bit, {got[2]} Hz; "
                                     f"decode writes 1 channel, 16-bit, {rate} Hz")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)
