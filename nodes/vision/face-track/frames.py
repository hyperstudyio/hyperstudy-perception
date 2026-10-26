"""Face crops, cut with PyAV: its wheels carry FFmpeg's libraries, so the node needs no ffmpeg on the machine (an app
started from the Finder has only /usr/bin:/bin:/usr/sbin:/sbin on its PATH). Each crop is what the ffmpeg command it
replaced wrote (`ffmpeg -ss {t:.3f} -i video -frames:v 1 -vf crop=...,scale=...`), at the frame the detector saw."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from perception_node import NodeError


def frame_at(container, t: float) -> av.VideoFrame:
    """The frame nearest `t` seconds after the video stream's first frame. Callers' times are frame times counted from
    that frame (face-track's frame index / fps, decode's shots from scenedetect's frame counts), rounded to the
    millisecond: on either side of the frame's own time at 29.97 fps, so neither the first frame at or after t nor the
    one shown at t is always it. (`ffmpeg -ss` counted from the container's start instead: when the video starts after
    the audio, it cut crops and keyframes early.) The decode node has the same function."""
    v = container.streams.video[0]
    target = Fraction(v.start_time or 0) * v.time_base + Fraction(round(t * 1000), 1000)
    container.seek(int(target / v.time_base), stream=v, backward=True)
    before = None
    for frame in container.decode(v):
        if frame.pts is None:
            continue
        at = frame.pts * v.time_base
        if at >= target:
            return before if before is not None and target - before.pts * v.time_base < at - target else frame
        before = frame
    if before is not None and target - before.pts * v.time_base < 1 / v.average_rate:   # within the last frame
        return before
    raise NodeError("input", f"the video has no frame at {t:.3f} s")


def upright(frame: av.VideoFrame) -> np.ndarray:
    """The frame as RGB, turned the way players show it. Phone recordings store the picture sideways with a rotation
    to apply: ffmpeg applied it, and so does torchcodec, which reads the frames face-track's detector sees; PyAV only
    reports it. The decode node has the same function."""
    return np.ascontiguousarray(np.rot90(frame.to_ndarray(format="rgb24"), round((frame.rotation or 0) / 90) % 4))


def crop_box(box, pad: float, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """x, y, width, height of the padded box within the frame; even, as ffmpeg's crop makes them on 4:2:0 video."""
    x, y, w, h = box
    x0, y0 = max(0, x - pad * w), max(0, y - pad * h)
    x1, y1 = min(frame_w, x + (1 + pad) * w), min(frame_h, y + (1 + pad) * h)
    return int(x0) & ~1, int(y0) & ~1, int(x1 - x0) & ~1, int(y1 - y0) & ~1


def even(n: Fraction) -> int:
    """ffmpeg's -2 in a scale: the nearest even number."""
    return max(2, round(n / 2) * 2)


def crop(video: Path, t: float, box, pad: float, size) -> av.VideoFrame:
    """The padded box at `t` (in the upright frame, as the detector saw it, and within it), scaled (bicubic, as
    ffmpeg's scale) to size(width, height) -> (width, height)."""
    with av.open(str(video)) as container:
        rgb = upright(frame_at(container, t))
    x, y, w, h = crop_box(box, pad, rgb.shape[1], rgb.shape[0])
    out_w, out_h = size(w, h)
    return av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb[y:y + h, x:x + w]), format="rgb24").reformat(
        width=out_w, height=out_h, interpolation="BICUBIC")


def longest_side(px: int):
    """scale='if(gt(iw,ih),px,-2)':'if(gt(iw,ih),-2,px)'"""
    return lambda w, h: (px, even(Fraction(px * h, w))) if w > h else (even(Fraction(px * w, h)), px)


def width(px: int):
    """scale=px:-2"""
    return lambda w, h: (px, even(Fraction(px * h, w)))


def square(px: int):
    """scale=px:px"""
    return lambda w, h: (px, px)


def save_crop(video: Path, t: float, box, pad: float, out: Path, size) -> None:
    crop(video, t, box, pad, size).to_image().save(out, quality=90)


def crop_pixels(video: Path, t: float, box, pad: float, px: int) -> np.ndarray:
    """The crop as px x px x 3 RGB, for scoring."""
    return crop(video, t, box, pad, square(px)).to_ndarray(format="rgb24")
