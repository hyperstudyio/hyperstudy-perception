"""The video's length for a plan, read without ffprobe: an app started from the Finder has none on its PATH, and a plan
without the length has no time estimates and prices a cloud GPU for a fixed hour rather than the video."""

import av
import pytest

from perception.estimates import probe_seconds


@pytest.fixture(autouse=True)
def no_tools_on_path(monkeypatch):
    monkeypatch.setenv("PATH", "")


def test_a_videos_length_is_read_from_the_file(tmp_path):
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as out:
        v = out.add_stream("mpeg4", rate=10)
        v.width, v.height, v.pix_fmt = 32, 32, "yuv420p"
        for _ in range(20):
            out.mux(v.encode(av.VideoFrame(32, 32, "yuv420p")))
        out.mux(v.encode())
    assert probe_seconds(path) == pytest.approx(2.0, abs=0.1)


def test_a_file_that_is_not_a_video_has_no_length(tmp_path):
    (tmp_path / "notes.mp4").write_text("not a video")
    assert probe_seconds(tmp_path / "notes.mp4") is None
    assert probe_seconds(tmp_path / "missing.mp4") is None
