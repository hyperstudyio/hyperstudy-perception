"""The object node's shot frames (shot_frames.py): the frames `ffmpeg -ss -t -i -vf fps` chose, read with PyAV.
Run: uv run --no-project --with pytest --with numpy --with av==17.1.0 --with pillow --with-editable ./sdk pytest -q nodes/vision/objects"""

from fractions import Fraction

import av
import numpy as np

from conftest import level
from shot_frames import Ticker, micros, round_half_away, select, shot_frames


def which(rgb) -> int:
    """The synthetic clip's frame index a decoded frame shows (each frame is one grey level)."""
    return int(np.argmin([abs(float(rgb.mean()) - level(i)) for i in range(20)]))


def test_rounding_is_ffmpegs_near_halves_away_from_zero():
    assert [round_half_away(Fraction(x, 2)) for x in (1, 3, 5, -1, -3)] == [1, 2, 3, -1, -2]
    assert round_half_away(Fraction(249, 100)) == 2 and round_half_away(Fraction(251, 100)) == 3


def test_start_and_length_are_parsed_as_printed_to_the_millisecond():
    assert micros(32.7334999) == 32_733_000 and micros(2.8035) == 2_804_000 and micros(0.0) == 0


def test_each_tick_shows_the_last_frame_due_by_then():
    """30 fps frames into 6 fps ticks: tick n shows the last frame whose rounded tick is <= n; frames overtaken by a
    later one before their tick are never shown."""
    times = [Fraction(i, 30) for i in range(12)]   # ticks round(i / 5): 0 0 0 1 1 1 1 1 2 2 2 2
    assert select(times, Fraction(12, 30), 6) == [2, 7]   # the end, 12/30 s, rounds to tick 2: ticks 0 and 1 only
    assert select(times, Fraction(14, 30), 6) == [2, 7, 11]   # an end at tick 3 (2.8) shows the last frame at tick 2


def test_a_slow_video_repeats_frames():
    times = [Fraction(i, 2) for i in range(3)]   # 2 fps into 6 fps
    assert select(times, Fraction(3, 2), 6) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_ticker_counts_nothing_before_the_first_frame():
    t = Ticker(6)
    assert t.push(Fraction(1, 10)) == 0 and t.finish(Fraction(1, 10)) == 0   # a lone frame ending at tick 1 rounds to 1
    assert Ticker(6).finish(Fraction(1)) == 0


def test_frames_of_a_shot_are_chosen_and_timed_as_ffmpeg_chose_them(clip):
    """[0.5, 1.5) of the 10 fps clip at 6 fps: frames 5..14 kept, their ticks 0 1 1 2 2 3 4 4 5 5, the input ending at
    frame 14's end (tick 6): six frames, the last at each tick. Counted from the video's first frame, though the video
    starts 0.3 s after the file."""
    with av.open(str(clip.path)) as c:
        frames, times = shot_frames(c, 0.5, 1.5, 6)
    assert [which(f) for f in frames] == [5, 7, 9, 10, 12, 14]
    assert times == [0.5, 0.667, 0.833, 1.0, 1.167, 1.333]
    assert frames[0].shape == (clip.h, clip.w, 3) and frames[0].dtype == np.uint8


def test_times_at_or_after_the_shot_end_are_left_out(clip):
    """[0.2, 0.55): frames 2..5 (ticks 0 1 1 2), the input ending at frame 5's end (0.4 s, tick 2.4 -> 2): two ticks,
    both before the end."""
    with av.open(str(clip.path)) as c:
        frames, times = shot_frames(c, 0.2, 0.55, 6)
    assert [which(f) for f in frames] == [2, 4] and times == [0.2, 0.367]


def test_the_last_shot_ends_with_the_video(clip):
    with av.open(str(clip.path)) as c:
        frames, times = shot_frames(c, 1.5, 2.0, 6)
    assert [which(f) for f in frames] == [15, 17, 19] and times == [1.5, 1.667, 1.833]


def test_shots_in_any_order_from_one_container(clip):
    """Each shot seeks: a later shot read first does not change an earlier one's frames."""
    with av.open(str(clip.path)) as c:
        late = shot_frames(c, 1.5, 2.0, 6)
        early = shot_frames(c, 0.5, 1.5, 6)
    assert [which(f) for f in early[0]] == [5, 7, 9, 10, 12, 14] and [which(f) for f in late[0]] == [15, 17, 19]


def test_frames_are_upright(rotated):
    """Stored sideways with a 90 degree rotation: shown 32 wide and 64 tall, blue on top and red below."""
    with av.open(str(rotated)) as c:
        frames, _ = shot_frames(c, 0.0, 0.5, 6)
    f = frames[0].astype(int)
    assert f.shape[:2] == (64, 32)
    assert f[8, 16, 2] > 150 and f[8, 16, 0] < 100 and f[56, 16, 0] > 150 and f[56, 16, 2] < 100
