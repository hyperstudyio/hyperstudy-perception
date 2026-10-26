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


def test_frame_at_counts_from_the_video_streams_first_frame(clip):
    with av.open(str(clip.path)) as c:
        v = c.streams.video[0]
        assert float(v.start_time * v.time_base) == clip.late / clip.fps and not c.start_time   # video after the audio
        for i in (0, 1, 7, 19):   # frame i at i / fps, as face-track and scenedetect count (not ffmpeg -ss's i - late)
            for off in (0, 0.0004, -0.0004):   # rounded to the millisecond, either way (29.97 fps frame times are)
                assert np.array_equal(frame_at(c, i / clip.fps + off).to_ndarray(format="rgb24"), clip.nth(i)), (i, off)


def test_frame_at_past_the_end(clip):
    with av.open(str(clip.path)) as c, pytest.raises(Exception, match="no frame at 5.000 s"):
        frame_at(c, 5.0)


def test_keyframes(clip, tmp_path):
    shots = [{"index": 0, "start": 0.0, "end": 1.0}, {"index": 1, "start": 1.0, "end": 2.0}]
    keyframes(clip.path, shots, tmp_path, 33)
    assert [k["t"] for k in shots[1]["keyframes"]] == [1.15, 1.5, 1.85]
    img = Image.open(tmp_path / "shot0001_50.jpg")
    assert img.size == (33, even_height(33, clip.w, clip.h)) == (33, 24)   # ffmpeg's scale=33:-2
    assert abs(np.asarray(img).mean() - clip.nth(15).mean()) < 3   # frames are 10 grey levels apart


def test_audio(clip, tmp_path):
    audio(clip.path, tmp_path / "a.wav")
    with wave.open(str(tmp_path / "a.wav")) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.getnframes() == pytest.approx(16000 * clip.seconds, abs=1600)
        samples = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    assert np.abs(samples).max() > 5000   # the tone, downmixed, not silence


def test_keyframes_are_upright(rotated, tmp_path):
    shots = [{"index": 0, "start": 0.0, "end": 1.0}]
    keyframes(rotated, shots, tmp_path, 16)
    img = np.asarray(Image.open(tmp_path / "shot0000_50.jpg")).astype(int)
    assert img.shape[:2] == (32, 16)   # 16 wide, as ffmpeg shows it: 32x64 upright
    top, bottom = img[:12].mean((0, 1)), img[20:].mean((0, 1))
    assert top[2] > 200 > top[0] and bottom[0] > 200 > bottom[2], (top, bottom)
