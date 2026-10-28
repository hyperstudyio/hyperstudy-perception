"""A shot's frames at a fixed rate, decoded with PyAV: its wheels carry FFmpeg's libraries, so the node needs no ffmpeg
on the machine (an app started from the Finder has only /usr/bin:/bin:/usr/sbin:/sbin on its PATH).

The bake-off decoded each shot with `ffmpeg -ss START -t LENGTH -i VIDEO -vf fps=FPS -pix_fmt rgb24` (START and LENGTH
printed to the millisecond) and called output frame k the frame at START + k / FPS. These are the same frames, chosen
the way those options choose them:

- `-ss` (an input option): time counts from START, and frames before START are dropped (accurate seek). ffmpeg counted
  from the file's start; here time counts from the video stream's first frame, as the shots, faces and bodies layers
  count it (decode's and face-track's frame_at). The two are the same unless the video starts after the audio, as on
  none of the bake-off's clips. `-t` (also an input option, a trim filter): LENGTH counts from the first frame kept, in the video
  stream's time base, and frames from there on are dropped.
- `fps=FPS` (rounding "near"): each kept frame's time is rounded to the nearest 1 / FPS tick (halves away from zero);
  output tick n shows the last frame whose tick is at most n, from the first frame's tick up to the end of the input,
  rounded the same way. That end is the end (time + duration) of the last frame kept.

test_shot_frames.py checks the choice on a synthetic clip; the PR that added this node also compared it with ffmpeg 8.1
on every shot of the bake-off's clips, pixel for pixel.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

MICRO = 1_000_000   # FFmpeg's AV_TIME_BASE: -ss and -t are parsed to microseconds


def round_half_away(x: Fraction) -> int:
    """av_rescale_q_rnd's AV_ROUND_NEAR_INF: to the nearest integer, halves away from zero."""
    n = abs(x)
    whole = int(n)
    if n - whole >= Fraction(1, 2):
        whole += 1
    return whole if x >= 0 else -whole


def micros(seconds: float) -> int:
    """Seconds as `f"{seconds:.3f}"` prints them, in microseconds (what ffmpeg parses from that text)."""
    return int(round(float(f"{seconds:.3f}") * 1000)) * 1000


class Ticker:
    """ffmpeg's fps filter (rounding "near") over frames arriving in order: push() each frame's time and get how many
    output ticks the frame before it fills; finish() gives the last frame's count."""

    def __init__(self, fps: float):
        self.rate = Fraction(fps).limit_denominator(1_000_000)
        self.next = None   # the next output tick

    def _fill(self, upto: int) -> int:
        n = max(0, upto - self.next)
        self.next += n
        return n

    def push(self, t: Fraction) -> int:
        tick = round_half_away(t * self.rate)
        if self.next is None:
            self.next = tick
            return 0
        return self._fill(tick)   # the earlier frame is shown until this one's tick; if it is due already, never

    def finish(self, end: Fraction) -> int:
        return 0 if self.next is None else self._fill(round_half_away(end * self.rate))


def select(times: list[Fraction], end: Fraction, fps: float) -> list[int]:
    """The frames (indices into times) the fps filter outputs, one per tick, for frames at these times (from START)
    and an input ending at `end`."""
    ticker, out = Ticker(fps), []
    for i, t in enumerate(times):
        out += [i - 1] * ticker.push(t)
    return out + [len(times) - 1] * ticker.finish(end) if times else []


def upright(frame) -> np.ndarray:
    """The frame as full-range RGB, turned the way players show it (ffmpeg applies a stored rotation; PyAV only reports
    it). Full range as ffmpeg's rgb24: PyAV's default keeps a limited-range (TV) frame's range, up to 16 levels off."""
    rgb = frame.reformat(format="rgb24", dst_color_range="JPEG").to_ndarray()
    return np.ascontiguousarray(np.rot90(rgb, round((frame.rotation or 0) / 90) % 4))


def shot_frames(container, start: float, end: float, fps: float, wanted: set[float] | None = None) -> tuple[list, list[float]]:
    """[start, end) at `fps` as upright RGB frames, and their times start + k / fps (to the millisecond), as the bake-off
    decoded and timed them (the module says how). Times at or after `end` are left out: fps rounding can emit one frame
    past the cut. `container` is an open PyAV container. With `wanted` (times), only those frames are converted and
    held; the others are None."""
    v = container.streams.video[0]
    tb = v.time_base
    video_start = Fraction(v.start_time or 0) * tb
    s, length = Fraction(micros(start), MICRO), Fraction(micros(end - start), MICRO)
    container.seek(int((video_start + s) / tb), stream=v, backward=True)   # the keyframe at or before START
    span = round_half_away(length / tb) * tb   # -t in the stream's time base, as the trim filter holds it
    ticker, frames, previous, first, stop = Ticker(fps), [], None, None, None

    def emit(frame, repeat):
        for _ in range(repeat):
            t = round(start + len(frames) / fps, 3)
            frames.append(upright(frame) if wanted is None or t in wanted else None)

    for frame in container.decode(v):
        if frame.pts is None:
            continue
        t = frame.pts * tb - video_start - s
        if t < 0:
            continue
        first = t if first is None else first
        if t - first >= span:
            break
        emit(previous, ticker.push(t))
        previous, stop = frame, t + (frame.duration * tb if frame.duration else 1 / Fraction(v.average_rate))
    if previous is not None:
        emit(previous, ticker.finish(stop))
    times = [round(start + k / fps, 3) for k in range(len(frames))]
    keep = sum(1 for t in times if t < end)
    return frames[:keep], times[:keep]
