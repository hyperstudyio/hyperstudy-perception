"""What LR-ASD is given per face track (asd_crops.py), and the 25 fps grey frames it is cut from (video25.py), without
the model. Run: uv run --project envs/gaze --with pytest pytest nodes/vision/active-speaker"""

import hashlib
import re
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from asd_crops import (CROP_SCALE, FACE_SIZE, FPS, OUTSIDE_GREY, VideoTooShort, audio_span, clip_windows, clips_as_completed,
                       crop_boxes, medfilt, padded_region, plans_for, track_frames)
from video25 import grey_frames


def track(start, end, boxes, shot="shot:0", id="face:0"):
    """A face track sampled evenly from start to end with these [x, y, w, h] boxes."""
    ts = np.linspace(start, end, len(boxes))
    return {"id": id, "shot": shot, "start": start, "end": end, "samples": [{"t": float(t), "box": b} for t, b in zip(ts, boxes)]}


def test_a_track_is_widened_by_an_eighth_of_a_second_each_side_within_its_shot():
    frames, _ = track_frames(track(1.0, 2.0, [[0, 0, 10, 10]] * 2), {"start": 0.0, "end": 10.0})
    assert frames[0] == 22 and frames[-1] == 53          # ceil(0.875 * 25), floor(2.125 * 25)
    frames, _ = track_frames(track(1.0, 2.0, [[0, 0, 10, 10]] * 2), {"start": 0.95, "end": 2.05})
    assert frames[0] == 24 and frames[-1] == 50          # ceil(0.95 * 25); floor((2.05 - 0.04) * 25): the shot's last frame


def test_boxes_are_interpolated_to_25_fps_and_held_beyond_the_samples():
    frames, boxes = track_frames(track(1.0, 2.0, [[0, 0, 10, 10], [100, 50, 30, 20]]), {"start": 0.0, "end": 10.0})
    at = dict(zip(frames.tolist(), boxes.tolist()))
    assert at[25] == [0, 0, 10, 10] and at[50] == [100, 50, 30, 20]
    assert at[22] == [0, 0, 10, 10] and at[53] == [100, 50, 30, 20]      # the padding holds the end boxes
    assert at[30] == pytest.approx([20, 10, 14, 12])                   # 0.2 of the way


def test_the_median_filter_is_scipys_zero_padded_one():
    x = np.array([5.0, 1, 4, 2, 3])
    assert medfilt(x, 3).tolist() == [1, 4, 2, 3, 2]                   # [0 5 1] -> 1 ... [2 3 0] -> 2
    assert medfilt(x, 13).tolist() == [0, 0, 0, 0, 0]                 # wider than the series: mostly zeros
    assert medfilt(np.arange(1.0, 9)).tolist() == [1, 2, 2, 2, 2, 2, 2, 2]   # kernel 13 (the values scipy gives)


def test_a_crop_is_a_square_about_the_face_centre_with_room_below_the_chin():
    boxes = np.tile([100.0, 200.0, 40.0, 60.0], (30, 1))   # long enough that the middle frames see no padding
    y0, y1, x0, x1 = crop_boxes(boxes)[15]
    s = 30.0                                               # half the box's longer side
    assert (y0, y1) == pytest.approx((230 - s, 230 + s * (1 + 2 * CROP_SCALE)))
    assert (x0, x1) == pytest.approx((120 - s * (1 + CROP_SCALE), 120 + s * (1 + CROP_SCALE)))


def test_a_crop_leaving_the_frame_is_grey_there_with_its_corners_truncated():
    gray = np.arange(100, dtype=np.uint8).reshape(10, 10)
    out = padded_region(gray, -2.7, 3.9, 8.2, 12.5)            # int(): -2, 3, 8, 12
    assert out.shape == (5, 4)
    assert (out[:2] == OUTSIDE_GREY).all() and (out[:, 2:] == OUTSIDE_GREY).all()
    assert out[2:, :2].tolist() == [[8, 9], [18, 19], [28, 29]]


def test_tracks_under_0_4_s_are_not_scored_and_a_track_in_an_unknown_shot_is_refused():
    shots = {"shot:0": {"start": 0.0, "end": 10.0}}
    short = track(5.0, 5.1, [[0, 0, 10, 10]] * 2, id="face:0")       # 4.875 .. 5.225 s: 9 frames
    long = track(6.0, 6.2, [[0, 0, 10, 10]] * 2, id="face:1")        # 5.875 .. 6.325 s: frames 147 .. 158
    plans = plans_for([short, long], shots)
    assert [p[0] for p in plans] == [1] and plans[0][1].tolist() == list(range(147, 159))
    with pytest.raises(ValueError, match="face:0 is in shot:9"):
        plans_for([track(1, 2, [[0, 0, 1, 1]], shot="shot:9")], shots)


def numbered_frames(n, seen):
    """n tiny frames, each filled with its index; `seen` records how far the consumer has read."""
    for i in range(n):
        seen.append(i)
        yield i, np.full((4, 4), i, np.uint8)


def patch_of(gray, rect):
    """A face patch holding the frame's number plus the rectangle's tag, so the test can tell what was cut where."""
    return np.full((FACE_SIZE, FACE_SIZE), gray[0, 0] + rect, np.uint8)


def test_each_clip_is_handed_over_as_soon_as_its_last_frame_has_gone_by():
    plans = [(0, np.arange(2, 5), [100] * 3), (1, np.arange(3, 9), [200] * 6), (2, np.arange(0, 2), [50] * 2)]
    seen = []
    got = []
    for slot, clip in clips_as_completed(numbered_frames(12, seen), plans, patch_of):
        got.append((slot, seen[-1]))                      # the frame being read when the clip came
        assert clip.shape == (len(plans[slot][1]), FACE_SIZE, FACE_SIZE) and clip.dtype == np.uint8
        assert clip[:, 0, 0].tolist() == [plans[slot][2][0] + int(f) for f in plans[slot][1]]
    assert got == [(2, 1), (0, 4), (1, 8)]                # not all at the end: their clips are not held to the end


def test_a_video_ending_before_a_tracks_frames_is_refused():
    plans = [(0, np.arange(2, 5), [100] * 3), (1, np.arange(3, 9), [200] * 6)]
    run = clips_as_completed(numbered_frames(7, []), plans, patch_of)
    assert next(run)[0] == 0
    with pytest.raises(VideoTooShort, match="has 7 frames at 25 fps, but face tracks need frame 8"):
        next(run)


def test_the_audio_under_frames_runs_to_the_end_of_the_last():
    assert audio_span(np.arange(25, 51)) == (16000, 32640)


def test_a_clip_is_cut_to_whole_frames_of_audio_and_tiled_by_each_window_length():
    windows = clip_windows(n_mfcc=503, n_frames=130, durations=(1, 2, 6))
    # 503 MFCCs -> 500 (whole frames of 4) = 5 s = 125 frames, under the 130 frames
    assert windows[0] == [(i * 100, (i + 1) * 100, i * 25, (i + 1) * 25) for i in range(5)]
    assert windows[1] == [(0, 200, 0, 50), (200, 400, 50, 100), (400, 500, 100, 125)]   # the last window shorter
    assert windows[2] == [(0, 500, 0, 125)]
    assert clip_windows(n_mfcc=1000, n_frames=30, durations=(1,)) == [[(0, 100, 0, 25), (100, 120, 25, 30)]]   # video-bound


def test_frames_are_25_per_second_from_the_video_streams_first_frame(clip):
    """The clip's video starts 3 frames (0.3 s) after its audio; frame 0 is still its first frame. At 10 fps in, frame
    n shows what FFmpeg's fps filter picks: the last input frame whose time, rounded to the nearest 25 fps tick (halves
    up), is at or before n."""
    got = list(grey_frames(clip.path))
    assert [i for i, _ in got] == list(range(clip.seconds * FPS))
    for n, gray in got:
        assert gray.shape == (clip.h, clip.w)
        source = max(i for i in range(clip.seconds * clip.fps) if np.floor(i * FPS / clip.fps + 0.5) <= n)
        assert abs(float(gray.mean()) - (10 + 10 * source)) < 4, (n, source, gray.mean())   # conftest's level(source)
    assert [round((g.mean() - 10) / 10) for _, g in got[:9]] == [0, 0, 0, 1, 1, 2, 2, 2, 3]


def test_frames_are_turned_upright(rotated):
    """Stored sideways with a 90° rotation: shown 32 wide and 64 high, blue (darker in grey) above red."""
    (_, gray), *_ = list(grey_frames(rotated))
    assert gray.shape == (64, 32)
    assert gray[:24].mean() + 20 < gray[40:].mean()


def bt709_clip(path):
    """Three 32x16 frames of limited-range yuv420p tagged BT.709 (as OOO and most H.264 files are), with varied
    chroma, stored raw (no lossy codec: every FFmpeg decodes the same bytes)."""
    with av.open(str(path), "w", format="matroska") as out:
        s = out.add_stream("rawvideo", rate=10)
        s.width, s.height, s.pix_fmt = 32, 16, "yuv420p"
        cc = s.codec_context
        cc.color_range, cc.colorspace, cc.color_primaries, cc.color_trc = 1, 1, 1, 1
        rng = np.random.default_rng(0)
        for i in range(3):
            y = (16 + (np.arange(32 * 16).reshape(16, 32) * 7 + i * 13) % 220).astype(np.uint8)
            u, v = (rng.integers(16, 241, (8, 16), dtype=np.uint8).reshape(-1, 32) for _ in range(2))
            frame = av.VideoFrame.from_ndarray(np.concatenate([y, u, v]), format="yuv420p")
            frame.pts, frame.time_base = i, Fraction(1, 10)
            out.mux(s.encode(frame))
        out.mux(s.encode())


def ffmpeg_version() -> tuple[int, ...]:
    return tuple(int(x) for x in av.ffmpeg_version_info.split("-")[0].split(".")[:2])


@pytest.mark.skipif(ffmpeg_version() < (8, 1), reason=f"PyAV's FFmpeg {av.ffmpeg_version_info} is older than envs/gaze's 8.1 "
                                                       "(CI runs this test again with PyAV 17.1.0)")
def test_grey_frames_are_ffmpeg_8_1s_the_programs_the_bake_off_cut_its_crops_with(tmp_path):
    """FFmpeg converts BT.709 limited-range YUV to grey through its chroma, and 8.1 rounds it otherwise than 8.0: these
    are the bytes of 8.1.1 (PyAV 17.1.0, and the ffmpeg 8.1.1 program the bake-off cut its crops with; 8.1.2 and 9.0.2
    give the same). PyAV 16.1.0's 8.0.1 starts [22, 27, 41, 46, 41, 49, 72, 77]; on OOO it differed in 82% of pixels,
    by up to 3 levels."""
    bt709_clip(tmp_path / "bt709.mkv")
    frames = list(grey_frames(tmp_path / "bt709.mkv"))
    assert len(frames) == 8 and frames[0][1][0, :8].tolist() == [23, 28, 42, 47, 43, 51, 72, 79]
    assert hashlib.sha256(b"".join(g.tobytes() for _, g in frames)).hexdigest() == \
        "226b64fd5fdb9a1c7ba878c199c0d69e6c6bc2c42f8ee293c6012b0af12fef64"


def test_envs_gaze_locks_a_pyav_with_ffmpeg_8_1_or_later():
    """PyAV 17.1.0 is the first release carrying FFmpeg 8.1 (the test above): a relock back to 16.1.0, the other
    environments' version, would silently change LR-ASD's input."""
    lock = (Path(__file__).resolve().parents[3] / "envs" / "gaze" / "uv.lock").read_text()
    version = re.search(r'\[\[package\]\]\nname = "av"\nversion = "([0-9.]+)"', lock).group(1)
    assert tuple(int(x) for x in version.split(".")) >= (17, 1), version
