"""Face crops, cut with PyAV: its wheels carry FFmpeg's libraries, so the node needs no ffmpeg on the machine (an app
started from the Finder has only /usr/bin:/bin:/usr/sbin:/sbin on its PATH). Each crop is what the ffmpeg command it
replaced wrote: `ffmpeg -ss {t:.3f} -i video -frames:v 1 -vf crop=...,scale=...`."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from perception_node import NodeError


def frame_at(container, t: float) -> av.VideoFrame:
    """The first frame shown at or after `t` seconds (to the millisecond) from the file's start, as `ffmpeg -ss` picks
    it: from the container's start time, not the video stream's. The decode node has the same function."""
    v = container.streams.video[0]
    target = Fraction(container.start_time or 0, av.time_base) + Fraction(round(t * 1000), 1000)
    container.seek(int(target / v.time_base), stream=v, backward=True)
    for frame in container.decode(v):
        if frame.pts is not None and frame.pts * v.time_base >= target:
            return frame
    raise NodeError("input", f"the video has no frame at {t:.3f} s")


def crop_box(box, pad: float, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """x, y, width, height of the padded box within the frame; even, as ffmpeg's crop makes them on 4:2:0 video."""
    x, y, w, h = box
    x0, y0 = max(0, x - pad * w), max(0, y - pad * h)
    x1, y1 = min(frame_w, x + (1 + pad) * w), min(frame_h, y + (1 + pad) * h)
    return int(x0) & ~1, int(y0) & ~1, int(x1 - x0) & ~1, int(y1 - y0) & ~1


def even(n: Fraction) -> int:
    """ffmpeg's -2 in a scale: the nearest even number."""
    return max(2, round(n / 2) * 2)


def crop(video: Path, t: float, box, pad: float, frame_w: int, frame_h: int, size) -> av.VideoFrame:
    """The padded box at `t`, scaled (bicubic, as ffmpeg's scale) to size(width, height) -> (width, height)."""
    with av.open(str(video)) as container:
        rgb = frame_at(container, t).to_ndarray(format="rgb24")
    x, y, w, h = crop_box(box, pad, frame_w, frame_h)
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


def save_crop(video: Path, t: float, box, pad: float, frame_w: int, frame_h: int, out: Path, size) -> None:
    crop(video, t, box, pad, frame_w, frame_h, size).to_image().save(out, quality=90)


def crop_pixels(video: Path, t: float, box, pad: float, frame_w: int, frame_h: int, px: int) -> np.ndarray:
    """The crop as px x px x 3 RGB, for scoring."""
    return crop(video, t, box, pad, frame_w, frame_h, square(px)).to_ndarray(format="rgb24")
