"""Speaker turns onto the transcript's words and utterances (voices.py), as the bake-off assigned them. Run:
uv run --no-project --with pytest --with numpy --with-editable ./sdk pytest nodes/audio/diarize"""

import pytest

from perception_node import NodeError
from voices import diarize, speaker_for


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def word(w, start, end):
    return {"w": w, "start": start, "end": end, "voice": None}


def transcript(words, spans):
    """asr's layer: no voices, every voice null."""
    return {"layer": "transcript", "schema_version": "0.1.0", "producer": {"stage": "transcribe"}, "voices": [],
            "words": words, "utterances": [{"id": f"utt:{i}", "start": words[a]["start"], "end": words[b]["end"], "text": "",
                                            "shot": "shot:0", "words": [a, b], "voice": None} for i, (a, b) in enumerate(spans)]}


def test_a_word_takes_the_speaker_whose_turns_overlap_it_longest_summed_over_their_turns():
    w = word("hello", 1.0, 2.0)
    assert speaker_for(w, [turn(0.0, 1.4, "A"), turn(1.4, 2.0, "B")]) == "B"                         # 0.6 s against 0.4 s
    assert speaker_for(w, [turn(1.0, 1.3, "A"), turn(1.3, 1.65, "B"), turn(1.65, 2.0, "A")]) == "A"  # 0.65 s, in two turns


def test_a_word_no_turn_overlaps_has_no_speaker_and_touching_is_not_overlapping():
    w = word("hm", 1.0, 2.0)
    assert speaker_for(w, []) is None
    assert speaker_for(w, [turn(0.0, 1.0, "A"), turn(2.0, 3.0, "B")]) is None


def test_a_tie_goes_to_the_speaker_of_the_earlier_turn():
    w = word("so", 1.0, 2.0)
    assert speaker_for(w, [turn(1.5, 2.5, "B"), turn(0.5, 1.5, "A")]) == "B"
    assert speaker_for(w, [turn(0.5, 1.5, "A"), turn(1.5, 2.5, "B")]) == "A"


def test_overlapping_turns_each_count_for_their_speaker():
    """pyannote's speaker_diarization lets two speakers' turns overlap (crosstalk): each gets the whole overlap."""
    w = word("yeah", 1.0, 2.0)
    assert speaker_for(w, [turn(0.0, 3.0, "A"), turn(1.2, 1.8, "B")]) == "A"


def test_an_utterance_takes_its_words_most_common_voice_ties_to_the_first_word():
    words = [word("a", 0.0, 0.5), word("b", 0.5, 1.0), word("c", 1.0, 1.5),   # A, B, B
             word("d", 2.0, 2.5), word("e", 2.5, 3.0)]                       # B, A: a tie
    turns = [turn(0.0, 0.5, "A"), turn(0.5, 1.5, "B"), turn(2.0, 2.5, "B"), turn(2.5, 3.0, "A")]
    out = diarize(transcript(words, [(0, 2), (3, 4)]), turns)
    assert [w["voice"] for w in out["words"]] == ["voice:A", "voice:B", "voice:B", "voice:B", "voice:A"]
    assert [u["voice"] for u in out["utterances"]] == ["voice:B", "voice:B"]


def test_an_utterance_most_of_whose_words_have_no_voice_has_none():
    words = [word("a", 0.0, 0.5), word("b", 5.0, 5.5), word("c", 6.0, 6.5)]
    out = diarize(transcript(words, [(0, 2)]), [turn(0.0, 0.5, "A")])
    assert [w["voice"] for w in out["words"]] == ["voice:A", None, None]
    assert out["utterances"][0]["voice"] is None
    assert out["voices"] == [{"id": "voice:A"}]   # a voice is listed when a word has it, not only an utterance


def test_voices_are_listed_once_sorted_and_everything_else_is_kept():
    words = [word("a", 0.0, 0.5), word("b", 0.5, 1.0), word("c", 1.0, 1.5)]
    before = transcript(words, [(0, 0), (1, 2)])
    out = diarize(before, [turn(0.0, 0.5, "SPEAKER_01"), turn(0.5, 1.5, "SPEAKER_00")])
    assert out["voices"] == [{"id": "voice:SPEAKER_00"}, {"id": "voice:SPEAKER_01"}]
    assert {k: v for k, v in out.items() if k not in ("voices", "words", "utterances")} == \
           {k: v for k, v in before.items() if k not in ("voices", "words", "utterances")}
    strip = lambda items: [{k: v for k, v in x.items() if k != "voice"} for x in items]   # noqa: E731
    assert strip(out["words"]) == strip(before["words"]) and strip(out["utterances"]) == strip(before["utterances"])
    assert before["voices"] == [] and all(w["voice"] is None for w in before["words"])   # the input is not changed


def test_no_turns_leave_every_voice_empty():
    out = diarize(transcript([word("a", 0.0, 0.5)], [(0, 0)]), [])
    assert out["voices"] == [] and out["words"][0]["voice"] is None and out["utterances"][0]["voice"] is None


def test_an_utterance_spanning_words_the_transcript_lacks_is_refused():
    bad = transcript([word("a", 0.0, 0.5), word("b", 0.5, 1.0)], [(0, 1)])
    bad["utterances"][0]["words"] = [1, 2]
    with pytest.raises(NodeError, match="utt:0 spans words 1-2, but the transcript has 2"):
        diarize(bad, [turn(0.0, 1.0, "A")])
