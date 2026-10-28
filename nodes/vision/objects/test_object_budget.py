"""What the object node checks before it spends: the memory a shot's SAM 3 session needs, the room the name
classification's reply has, and replies cut short (budget.py, haiku.py). Without the models or the API.
Run: uv run --no-project --with pytest --with numpy --with av==16.1.0 --with pillow --with-editable ./sdk pytest -q nodes/vision/objects"""

import json
from types import SimpleNamespace

import av
import numpy as np
import pytest

from budget import GB, segment_bytes, shot_frame_count, too_long
from conftest import level
from haiku import name_classes_max_tokens, reply_json
from perception_node import NodeError
from shot_frames import shot_frames


def reply(text, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, stop_details=None, content=[SimpleNamespace(type="text", text=text)])


def test_a_reply_is_its_json():
    assert reply_json(reply('{"objects": ["bicycle"]}')) == {"objects": ["bicycle"]}


def test_a_reply_cut_short_says_so():
    """A reply that reached max_tokens holds cut-off JSON: say so, not a JSON error."""
    with pytest.raises(NodeError, match="max_tokens") as e:
        reply_json(reply('{"names": [{"name": "yellow card', stop_reason="max_tokens"))
    assert e.value.code == "internal"


def test_a_refusal_is_the_inputs():
    with pytest.raises(NodeError) as e:
        reply_json(reply("", stop_reason="refusal"))
    assert e.value.code == "input"


def test_the_name_classification_has_room_for_every_name():
    """The bake-off's 4000 tokens for a clip's few dozen names; about 40 tokens a name beyond that (an entry is ~25),
    so a film's vocabulary is not cut off."""
    assert name_classes_max_tokens(27) == 4000   # OOO's vocabulary: the bake-off's call, unchanged
    for n in (150, 400, 2000):
        assert name_classes_max_tokens(n) >= 40 * n
    entry = json.dumps({"name": "yellow cardigan", "family": "upper-body clothing", "kind": "individual", "worn": True})
    assert len(entry) / 3 < 40   # well under 40 tokens, at ~3 characters a token


def test_a_shots_frames_are_counted_as_shot_frames_counts_them(clip):
    with av.open(str(clip.path)) as c:
        for start, end in ((0.5, 1.5), (0.2, 0.55), (1.5, 2.0)):
            assert shot_frame_count(start, end, 6) >= len(shot_frames(c, start, end, 6)[1])
    assert shot_frame_count(0.0, 600.0, 6) == 3600


def test_a_sam3_session_holds_every_frame_of_its_shot():
    """Per 1080p frame: the RGB frame and the processor's stacked copy (6.2 MB each), its 1008x1008 float32 resize
    (12.2 MB), and the session's bfloat16 store (6.1 MB)."""
    per_frame = segment_bytes(1, 1920, 1080, 2)
    assert per_frame == 2 * 1920 * 1080 * 3 + 1008 * 1008 * 3 * 4 + 1008 * 1008 * 3 * 2
    assert segment_bytes(3600, 1920, 1080, 2) / GB == pytest.approx(3600 * per_frame / GB)   # a 10-minute take: ~110 GB


def test_a_shot_too_long_for_this_machine_is_named_before_anything_runs():
    shots = [{"index": 0, "start": 0.0, "end": 5.0}, {"index": 1, "start": 5.0, "end": 605.0}, {"index": 2, "start": 605.0, "end": 610.0}]
    long = too_long(shots, 6, 1920, 1080, 2, memory=64 * GB)
    assert long is not None and long["index"] == 1 and long["frames"] == 3600 and long["gb"] > 64
    assert too_long(shots[:1] + shots[2:], 6, 1920, 1080, 2, memory=64 * GB) is None
    assert too_long(shots, 1, 1920, 1080, 2, memory=64 * GB) is None   # at 1 fps the long take fits


def test_only_the_frames_asked_for_are_kept(clip):
    """The embedding pass needs a few frames per track, not the shot: the others are not converted or held."""
    with av.open(str(clip.path)) as c:
        frames, times = shot_frames(c, 0.5, 1.5, 6, wanted={0.667, 1.333})
    assert times == [0.5, 0.667, 0.833, 1.0, 1.167, 1.333]
    assert [f is None for f in frames] == [True, False, True, True, True, False]
    assert int(np.argmin([abs(float(frames[1].mean()) - level(i)) for i in range(20)])) == 7
