"""The decode node's media reading (media.py), with nothing on PATH: the app runs nodes from the Finder, where ffmpeg
and ffprobe are not found. Run: uv run --project envs/core --with pytest pytest nodes/core/decode"""

import wave

import av
import numpy as np
import pytest
from PIL import Image

from media import audio, even_height, frame_at, keyframes, probe


@pytest.fixture(autouse=True)
def no_tools_on_path(monkeypatch):
    monkeypatch.setenv("PATH", "")


def test_probe(clip):
    m = probe(clip.path)
    assert (m["width"], m["height"], m["r_frame_rate"], m["avg_frame_rate"], m["fps"]) == (clip.w, clip.h, "10/1", "10/1", 10.0)
    assert m["duration"] == pytest.approx(clip.seconds + clip.late / clip.fps, abs=0.05)   # the container's: from the audio's start
    assert m["audio"] == {"sample_rate": 48000, "channels": 2}
    assert not m["variable_frame_rate"]


def test_frame_at_counts_from_the_container_start(clip):
    with av.open(str(clip.path)) as c:
        v = c.streams.video[0]
        assert float(v.start_time * v.time_base) == clip.late / clip.fps and not c.start_time
        for t in (0.0, 0.1, 0.3, 1.0, 2.0):   # before the video starts, then frames `late` apart from the stream's own count
            assert np.array_equal(frame_at(c, t).to_ndarray(format="rgb24"), clip.shown_at(t)), t


def test_frame_at_past_the_end(clip):
    with av.open(str(clip.path)) as c, pytest.raises(Exception, match="no frame at 5.000 s"):
        frame_at(c, 5.0)


def test_keyframes(clip, tmp_path):
    shots = [{"index": 0, "start": 0.0, "end": 1.0}, {"index": 1, "start": 1.0, "end": 2.0}]
    keyframes(clip.path, shots, tmp_path, 33)
    assert [k["t"] for k in shots[1]["keyframes"]] == [1.15, 1.5, 1.85]
    img = Image.open(tmp_path / "shot0001_50.jpg")
    assert img.size == (33, even_height(33, clip.w, clip.h)) == (33, 24)   # ffmpeg's scale=33:-2
    assert abs(np.asarray(img).mean() - clip.shown_at(1.5).mean()) < 3   # frames are 10 grey levels apart


def test_audio(clip, tmp_path):
    audio(clip.path, tmp_path / "a.wav")
    with wave.open(str(tmp_path / "a.wav")) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.getnframes() == pytest.approx(16000 * clip.seconds, abs=1600)
        samples = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    assert np.abs(samples).max() > 5000   # the tone, downmixed, not silence
