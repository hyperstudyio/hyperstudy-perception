"""The waveform layer's peaks (waveform_peaks.py) against the bake-off's annotator/build.py waveform(), copied below.
Run: uv run --no-project --with pytest --with numpy --with-editable ./sdk pytest nodes/audio/waveform"""

import wave
from array import array

import numpy as np
import pytest

from perception_node import NodeError
from waveform_peaks import HZ, bin_peaks, scale, waveform


def bakeoff_waveform(path):
    """build.py waveform() as the bake-off ran it (WAVEFORM_HZ = 50), its SystemExit aside."""
    with wave.open(str(path)) as w:
        samples = array("h", w.readframes(w.getnframes()))
        step = w.getframerate() // 50
    peaks = [max(map(abs, samples[i:i + step]), default=0) for i in range(0, len(samples), step)]
    top = max(peaks) or 1
    return [round(100 * p / top) for p in peaks]


def write(path, samples, rate=16000, channels=1, width=2):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(np.asarray(samples, np.int16 if width == 2 else np.uint8).tobytes())
    return path


def speech_like(n, seed=0):
    """Bursts of noise at different loudness, the extremes included."""
    rng = np.random.default_rng(seed)
    loud = np.repeat(rng.uniform(0, 1, n // 800 + 1), 800)[:n]
    s = (rng.uniform(-32768, 32767, n) * loud).astype(np.int16)
    s[n // 3], s[n // 2] = -32768, 32767
    return s


@pytest.mark.parametrize("n", [16000 * 3 + 123, 320 * 7, 1, 319, 321])
@pytest.mark.parametrize("block_bins", [1, 3, 4096])
def test_same_as_the_bakeoff(tmp_path, n, block_bins):
    path = write(tmp_path / "a.wav", speech_like(n))
    got = waveform(path, block_bins=block_bins)
    assert got == bakeoff_waveform(path)
    assert len(got) == -(-n // 320)   # a shorter last bin is a bin


def test_the_most_negative_sample_is_the_loudest(tmp_path):
    """-32768 is louder than 32767: in int16 arithmetic its abs wraps to -32768 and it would be the quietest."""
    s = np.zeros(320 * 3, np.int16)
    s[0], s[320], s[640] = -32768, 32767, -16384
    assert waveform(write(tmp_path / "a.wav", s)) == [100, 100, 50]
    assert bin_peaks(array("h", [-32768, 1]), 2) == [32768]


def test_halves_round_to_even_as_the_bakeoff_did():
    assert scale([200, 1, 3, 5]) == [100, 0, 2, 2]   # 0.5 -> 0, 1.5 -> 2, 2.5 -> 2


def test_silence_is_zeros_and_no_audio_is_no_bins(tmp_path):
    assert waveform(write(tmp_path / "s.wav", np.zeros(1000, np.int16))) == [0, 0, 0, 0]
    assert waveform(write(tmp_path / "e.wav", np.zeros(0, np.int16))) == []


def test_bins_are_one_fiftieth_of_a_second_at_other_rates(tmp_path):
    path = write(tmp_path / "a.wav", speech_like(48000 + 5), rate=48000)
    assert HZ == 50 and len(waveform(path)) == 51 and waveform(path) == bakeoff_waveform(path)


@pytest.mark.parametrize("rate, channels, width, message", [
    (16000, 2, 2, "2 channel"), (16000, 1, 1, "8-bit"), (44110, 1, 2, "not a multiple of 50")])
def test_refuses_audio_it_cannot_bin_exactly(tmp_path, rate, channels, width, message):
    path = write(tmp_path / "a.wav", np.zeros(1000 * channels), rate, channels, width)
    with pytest.raises(NodeError, match=message):
        waveform(path)
