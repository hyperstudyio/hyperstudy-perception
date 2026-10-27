"""soundtrack: the video's audio as mono float samples at a model's rate."""

from fractions import Fraction

import av
import numpy as np
import pytest

from perception_node import NodeError
from soundtrack import samples


@pytest.mark.parametrize("rate", [32000, 48000])
def test_the_soundtrack_is_mono_float_at_the_rate_asked(clip, rate):
    wave = samples(clip.path, rate)
    assert wave.dtype == np.float32 and wave.ndim == 1
    assert abs(len(wave) / rate - clip.seconds) < 0.05            # the 2 s stereo tone (AAC adds a frame of priming)
    middle = wave[len(wave) // 4:3 * len(wave) // 4]
    # both channels carry the 0.3 tone: FFmpeg's stereo-to-mono downmix (as `ffmpeg -ac 1`) adds them at -3 dB each
    assert abs(np.abs(middle).max() - 0.3 * np.sqrt(2)) < 0.02
    spectrum = np.abs(np.fft.rfft(middle))
    assert abs(np.argmax(spectrum) * rate / len(middle) - 440) < 5   # still 440 Hz after resampling


def test_a_video_without_sound_is_an_input_error(tmp_path):
    path = tmp_path / "silent.mp4"
    with av.open(str(path), "w") as out:
        v = out.add_stream("libx264", rate=10)
        v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
        for i in range(10):
            frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 10)
            out.mux(v.encode(frame))
        out.mux(v.encode())
    with pytest.raises(NodeError, match="no audio stream") as e:
        samples(path, 32000)
    assert e.value.code == "input"
