"""The video as 25 fps grey frames, and the face patches cut from them, as the bake-off made them with the ffmpeg program
(`ffmpeg -i video -vf fps=25 -f rawvideo -pix_fmt gray -`, then OpenCV): here through PyAV, whose wheels carry FFmpeg's
libraries and so its own fps filter, since an app started from the Finder finds no ffmpeg program.

Frame i is the one FFmpeg's fps filter shows at i / 25 s after the video stream's first frame (the time face-track's
samples count from), turned upright as players show it (ffmpeg rotates a phone recording; PyAV only reports it).
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

from perception_node import NodeError
from asd_crops import CROP_SIZE, FACE_SIZE, FPS, padded_region


def _drain(sink):
    while True:
        try:
            yield sink.pull()
        except (av.error.BlockingIOError, av.error.EOFError):
            return


def grey_frames(video: Path):
    """(frame index, upright grey frame) at 25 fps, in order."""
    try:
        container = av.open(str(video))
    except av.FFmpegError as e:
        raise NodeError("input", f"cannot read the video: {str(e)[:300]}") from None
    with container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        graph = av.filter.Graph()
        source = graph.add_buffer(template=stream)
        fps, grey, sink = graph.add("fps", str(FPS)), graph.add("format", "gray"), graph.add("buffersink")
        source.link_to(fps)
        fps.link_to(grey)
        grey.link_to(sink)
        graph.configure()
        first, turns = None, 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if first is None:
                first, turns = frame.pts, round((frame.rotation or 0) / 90) % 4
            frame.pts -= first   # time from the stream's first frame
            source.push(frame)
            for out in _drain(sink):
                yield int(out.pts), np.ascontiguousarray(np.rot90(out.to_ndarray(), turns))
        source.push(None)
        for out in _drain(sink):
            yield int(out.pts), np.ascontiguousarray(np.rot90(out.to_ndarray(), turns))


def face_patch(gray: np.ndarray, rect) -> np.ndarray:
    """The 112 x 112 face patch LR-ASD sees: the crop resized to 224 x 224 (OpenCV's bilinear), its centre."""
    import cv2
    lo, hi = (CROP_SIZE - FACE_SIZE) // 2, (CROP_SIZE + FACE_SIZE) // 2
    return cv2.resize(padded_region(gray, *rect), (CROP_SIZE, CROP_SIZE))[lo:hi, lo:hi]


def face_clips(video: Path, plans, wanted) -> list[np.ndarray]:
    """Each plan's clip (frames x 112 x 112, uint8), cut from one pass over the video. Plans stay within their shots,
    so a frame the video does not have means the faces or shots are not this video's: NodeError."""
    clips = [np.zeros((len(frames), FACE_SIZE, FACE_SIZE), np.uint8) for _, frames, _ in plans]
    count = 0
    for index, gray in grey_frames(video):
        for slot, j in wanted.get(index, ()):
            clips[slot][j] = face_patch(gray, plans[slot][2][j])
        count = index + 1
    if wanted and max(wanted) >= count:
        raise NodeError("input", f"{video.name} has {count} frames at {FPS} fps, but face tracks need frame {max(wanted)}")
    return clips
