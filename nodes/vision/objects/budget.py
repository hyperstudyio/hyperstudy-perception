"""The memory a shot's SAM 3 session needs, checked before the node uploads or spends anything.

SAM 3 video (transformers' Sam3VideoProcessor.init_video_session) takes every frame of a shot at once: the processor
stacks the RGB frames, resizes them to 1008x1008 in float32, and the session keeps them in its dtype on the CPU for the
whole pass. A session never spans a cut, as in the bake-off, so the memory grows with the longest shot: a 10-minute
take at 6 fps is 3600 frames, ~110 GB at 1080p. Windowing a long shot (overlapping sessions, reconciled tracks) would
bound it but would change the tracks; that is a follow-up, not this port. Until then a shot that cannot fit in this
machine's memory fails the node at the start, naming the shot, rather than swapping or being killed halfway.
"""

from __future__ import annotations

import math

GB = 1 << 30
SIDE = 1008   # SAM 3's input size (its processor_config.json)


def shot_frame_count(start: float, end: float, fps: float) -> int:
    """At most how many frames shot_frames gives for [start, end) at fps (times start + k / fps before end)."""
    return max(0, math.ceil(round((end - start) * fps, 6)))


def segment_bytes(frames: int, width: int, height: int, stored_bytes: int) -> int:
    """Peak bytes of one SAM 3 session's frames: each RGB frame and the processor's stacked copy, its float32 resize,
    and the session's copy at stored_bytes a value (2 for bfloat16, 4 for float32)."""
    return frames * (2 * width * height * 3 + SIDE * SIDE * 3 * 4 + SIDE * SIDE * 3 * stored_bytes)


def too_long(shots: list[dict], fps: float, width: int, height: int, stored_bytes: int, memory: int) -> dict | None:
    """The first shot whose SAM 3 session needs more than `memory` bytes ({index, frames, gb}), or None."""
    for shot in shots:
        n = shot_frame_count(shot["start"], shot["end"], fps)
        need = segment_bytes(n, width, height, stored_bytes)
        if need > memory:
            return {"index": shot["index"], "frames": n, "gb": round(need / GB, 1)}
    return None


def physical_memory() -> int:
    """This machine's RAM in bytes (macOS and Linux)."""
    import os
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
