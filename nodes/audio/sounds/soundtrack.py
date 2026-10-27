"""The video's soundtrack as mono float samples at the rate a model wants, read with PyAV (its wheels carry FFmpeg's
libraries: an app started from the Finder has no ffmpeg on its PATH). Not decode's 16 kHz WAV: the taggers take
32 kHz and PE-A-Frame 48 kHz, and 16 kHz has nothing above 8 kHz.

The samples are those `ffmpeg -i VIDEO -ac 1 -ar RATE -f f32le -` printed, which the bake-off read (identical on its
three clips). Times count from the first audio sample, as the transcript's do (decode's WAV starts there too).
"""

from pathlib import Path

import av
import numpy as np

from perception_node import NodeError


def samples(video: Path, rate: int) -> np.ndarray:
    """The first audio stream, downmixed to mono, resampled to `rate`, as float32."""
    try:
        container = av.open(str(video))
    except av.FFmpegError as e:
        raise NodeError("input", f"cannot read the video: {str(e)[:300]}") from None
    with container:
        if not container.streams.audio:
            raise NodeError("input", "the video has no audio stream")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        parts = []
        for frame in container.decode(container.streams.audio[0]):
            parts += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
        parts += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    if not parts:
        raise NodeError("input", "the audio stream has no samples")
    return np.concatenate(parts)
