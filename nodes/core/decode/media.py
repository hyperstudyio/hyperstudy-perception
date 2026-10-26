"""The video's facts, keyframes and audio, read with PyAV: its wheels carry FFmpeg's libraries, so the node needs no
ffmpeg or ffprobe on the machine (an app started from the Finder has only /usr/bin:/bin:/usr/sbin:/sbin on its PATH).
Each function does what the ffprobe or ffmpeg command it replaced did; the comments say which."""

import wave
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from perception_node import NodeError

KEYFRAME_POSITIONS = (0.15, 0.5, 0.85)


def ratio(r: Fraction | None) -> str:
    """A rate or time base as ffprobe printed it ("30000/1001", "30/1"; "0/0" when unknown)."""
    return f"{r.numerator}/{r.denominator}" if r else "0/0"


def probe(video: Path) -> dict:
    """ffprobe's format=duration and the first video and audio streams' fields."""
    try:
        container = av.open(str(video))
    except av.FFmpegError as e:
        raise NodeError("input", f"cannot read the video: {str(e)[:300]}") from None
    with container:
        if not container.streams.video:
            raise NodeError("input", "the file has no video stream")
        v = container.streams.video[0]
        a = container.streams.audio[0] if container.streams.audio else None
        r, avg = v.base_rate, v.average_rate or v.base_rate
        return {"duration": container.duration / av.time_base, "width": v.codec_context.width, "height": v.codec_context.height,
                "fps": float(avg or r), "r_frame_rate": ratio(r), "avg_frame_rate": ratio(v.average_rate),
                "time_base": ratio(v.time_base), "start_pts": v.start_time,
                # containers report r_frame_rate as the base rate and avg_frame_rate from the frame count; they differ
                # when frame durations vary
                "variable_frame_rate": bool(avg) and abs(float(r) - float(avg)) > 1e-3,
                "audio": {"sample_rate": a.codec_context.sample_rate, "channels": a.codec_context.channels} if a else None}


def frame_at(container, t: float) -> av.VideoFrame:
    """The frame nearest `t` seconds after the video stream's first frame. Callers' times are frame times counted from
    that frame (face-track's frame index / fps, decode's shots from scenedetect's frame counts), rounded to the
    millisecond: on either side of the frame's own time at 29.97 fps, so neither the first frame at or after t nor the
    one shown at t is always it. (`ffmpeg -ss` counted from the container's start instead: when the video starts after
    the audio, it cut crops and keyframes early.) The face-track node has the same function."""
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
    reports it. The face-track node has the same function."""
    return np.ascontiguousarray(np.rot90(frame.to_ndarray(format="rgb24"), round((frame.rotation or 0) / 90) % 4))


def even_height(w: int, src_w: int, src_h: int) -> int:
    """ffmpeg's `scale=w:-2`: the height that keeps the aspect ratio, to the nearest even number."""
    return max(2, round(Fraction(w * src_h, src_w * 2)) * 2)


def keyframes(video: Path, shots: list[dict], out_dir: Path, width: int) -> None:
    """Three frames per shot, upright, `width` wide (ffmpeg's scale=width:-2, bicubic), as JPEG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with av.open(str(video)) as container:
        for shot in shots:
            shot["keyframes"] = []
            for pos in KEYFRAME_POSITIONS:
                t = shot["start"] + pos * (shot["end"] - shot["start"])
                name = f"shot{shot['index']:04d}_{int(pos * 100):02d}.jpg"
                rgb = upright(frame_at(container, t))
                h, w = rgb.shape[:2]
                av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(
                    width=width, height=even_height(width, w, h), interpolation="BICUBIC").to_image().save(out_dir / name, quality=90)
                shot["keyframes"].append({"t": round(t, 3), "file": f"assets/keyframes/{name}"})


def audio(video: Path, wav: Path) -> None:
    """The first audio stream as 16 kHz mono 16-bit WAV (ffmpeg -vn -ac 1 -ar 16000 -sample_fmt s16)."""
    with av.open(str(video)) as container, wave.open(str(wav), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(container.streams.audio[0]):
            for f in resampler.resample(frame):
                out.writeframes(f.to_ndarray().tobytes())
        for f in resampler.resample(None):
            out.writeframes(f.to_ndarray().tobytes())
