"""The waveform layer's peaks from a 16-bit mono WAV, as the bake-off's annotator/build.py waveform() computed them:
the peak absolute sample per 1/HZ s (the last bin may be shorter), scaled 0-100 relative to the loudest bin.

Standard library only (the node runs in envs/core). The file is read in blocks of whole bins, so a long video's audio
is never held in memory at once; the result is the same as reading it whole.
"""

import sys
import wave
from array import array
from pathlib import Path

from perception_node import NodeError

HZ = 50                  # bins per second of audio (build.py WAVEFORM_HZ)
BLOCK_BINS = 4096        # bins read from the file at a time


def bin_peaks(samples: array, step: int) -> list[int]:
    """The peak absolute value of each run of `step` samples; a shorter last run is a bin of its own. In Python
    integers, so abs(-32768) is 32768 (an int16 array's abs would wrap it back to -32768)."""
    return [max(map(abs, samples[i:i + step])) for i in range(0, len(samples), step)]


def scale(peaks: list[int]) -> list[int]:
    """0-100 relative to the loudest bin; all zeros for silence. Python's round(), as build.py: halves go to even."""
    top = max(peaks, default=0) or 1
    return [round(100 * p / top) for p in peaks]


def read_peaks(path: Path, hz: int = HZ, block_bins: int = BLOCK_BINS) -> list[int]:
    """Unscaled bin peaks of decode's audio (audio.wav.v1). Refuses audio whose bins would not be exactly 1/hz s."""
    with wave.open(str(path)) as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        if (channels, width) != (1, 2):
            raise NodeError("input", f"the audio is {channels} channel(s), {8 * width}-bit; decode writes 1 channel, 16-bit")
        if rate % hz:
            raise NodeError("input", f"the audio's rate, {rate} Hz, is not a multiple of {hz} bins per second")
        step = rate // hz
        peaks = []
        while True:
            samples = array("h", w.readframes(step * block_bins))   # WAV samples are little-endian
            if sys.byteorder == "big":
                samples.byteswap()
            if not samples:
                return peaks
            peaks += bin_peaks(samples, step)


def waveform(path: Path, hz: int = HZ, block_bins: int = BLOCK_BINS) -> list[int]:
    """The layer's peaks: what build.py waveform(path) returns (but [] for audio with no samples, where it fails)."""
    return scale(read_peaks(path, hz, block_bins))
