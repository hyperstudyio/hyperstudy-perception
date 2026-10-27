"""Decode's audio read as the waveform pyannote is given (pcm16.py): the bake-off's samples / 32768, with nothing on
PATH (WhisperX ran the ffmpeg program, which an app started from the Finder does not find)."""

import wave

import numpy as np
import pytest

from pcm16 import read_waveform
from perception_node import NodeError


@pytest.fixture(autouse=True)
def no_tools_on_path(monkeypatch):
    monkeypatch.setenv("PATH", "")


def write(path, samples, rate=16000, channels=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype(np.int16).tobytes())


def test_reads_decodes_audio_as_whisperx_scaled_it(tmp_path):
    samples = np.array([0, 1, -1, 16384, -32768, 32767] * 100, np.int16)
    write(tmp_path / "a.wav", samples)
    out = read_waveform(tmp_path / "a.wav")
    assert out.dtype == np.float32 and out.shape == (600,)
    assert np.array_equal(out, samples.astype(np.float32) / 32768.0)
    assert out.min() == -1.0 and out.max() < 1.0


@pytest.mark.parametrize("rate, channels", [(44100, 1), (16000, 2)])
def test_refuses_audio_decode_does_not_write(tmp_path, rate, channels):
    write(tmp_path / "a.wav", np.zeros(1000 * channels), rate, channels)
    with pytest.raises(NodeError, match="decode writes 1 channel, 16-bit, 16000 Hz"):
        read_waveform(tmp_path / "a.wav")
