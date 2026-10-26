"""A small synthetic video for the node tests that read one (decode's media.py, face-track's frames.py)."""

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

FPS, SECONDS, W, H = 10, 2, 64, 48
LATE = 3   # frames the video starts after the audio (and the container)


def level(i: int) -> int:
    """Frame i's grey level: frames tell apart by brightness, through lossy encoding."""
    return 10 + 10 * i


def make(path):
    """2 s of 64x48 H.264 at 10 fps, each frame one grey level, starting LATE frames after its 48 kHz stereo AAC (the
    container's start): files cut from a longer one often start that way."""
    with av.open(str(path), "w") as out:
        v = out.add_stream("libx264", rate=FPS)
        v.width, v.height, v.pix_fmt = W, H, "yuv420p"
        a = out.add_stream("aac", rate=48000, layout="stereo")
        for i in range(FPS * SECONDS):
            frame = av.VideoFrame.from_ndarray(np.full((H, W, 3), level(i), np.uint8), format="rgb24")
            frame.pts, frame.time_base = LATE + i, Fraction(1, FPS)
            out.mux(v.encode(frame))
        out.mux(v.encode())
        t = np.arange(48000 * SECONDS) / 48000
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        for start in range(0, len(tone), 1024):
            chunk = np.stack([tone[start:start + 1024]] * 2)
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(chunk), format="fltp", layout="stereo")
            frame.sample_rate, frame.pts = 48000, start
            out.mux(a.encode(frame))
        out.mux(a.encode())


def decoded(path) -> list[tuple[float, np.ndarray]]:
    """Every frame in order: (seconds from the container's start, RGB)."""
    with av.open(str(path)) as c:
        return [(float(f.pts * f.time_base), f.to_ndarray(format="rgb24")) for f in c.decode(c.streams.video[0])]


def nth(path, i: int) -> np.ndarray:
    """Frame i of the video stream, counted from its first: the frame a detector sampling frame i saw."""
    return [rgb for _, rgb in decoded(path)][i]


@pytest.fixture
def rotated():
    """A clip stored sideways with a 90° rotation, as phones record (nodes/testdata/README.md): shown 32x64, blue on top
    and red below."""
    return Path(__file__).parent / "testdata" / "rotated-90.mp4"


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    make(path)
    return SimpleNamespace(path=path, fps=FPS, seconds=SECONDS, w=W, h=H, late=LATE, nth=lambda i: nth(path, i))
