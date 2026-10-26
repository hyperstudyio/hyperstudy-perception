"""Decode's audio read as samples (wav.py), with nothing on PATH: parakeet-mlx's own loader runs ffmpeg, which an app
started from the Finder does not find. Run: uv run --no-project --with pytest --with numpy --with-editable ./sdk
pytest nodes/audio/asr"""

import wave

import numpy as np
import pytest

from perception_node import NodeError
from wav import read_wav


@pytest.fixture(autouse=True)
def no_tools_on_path(monkeypatch):
    monkeypatch.setenv("PATH", "")


def write(path, samples, rate=16000, channels=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype(np.int16).tobytes())


def test_reads_decodes_audio_as_it_was_written(tmp_path):
    samples = (np.sin(np.arange(16000) / 5) * 12000).astype(np.int16)
    write(tmp_path / "a.wav", samples)
    out = read_wav(tmp_path / "a.wav", 16000)
    assert out.dtype == np.int16 and np.array_equal(out, samples)


@pytest.mark.parametrize("rate, channels", [(44100, 1), (16000, 2)])
def test_refuses_audio_decode_does_not_write(tmp_path, rate, channels):
    write(tmp_path / "a.wav", np.zeros(1000 * channels), rate, channels)
    with pytest.raises(NodeError, match="decode writes 1 channel, 16-bit, 16000 Hz"):
        read_wav(tmp_path / "a.wav", 16000)
