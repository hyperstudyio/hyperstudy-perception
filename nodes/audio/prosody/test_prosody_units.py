"""Which utterances prosody measures, with which words and speaker (prosody_units.py)."""

import pytest

from prosody_units import WORD_MAX_S, speaker_of, units, word_end


def transcript(utts, words):
    return {"producer": {"stage": "transcribe+diarize"}, "words": words, "utterances": utts}


WORDS = [{"w": "hi", "start": 0.0, "end": 0.3, "voice": "voice:SPEAKER_00"},
         {"w": "there.", "start": 0.4, "end": 3.0, "voice": "voice:SPEAKER_00"},      # stretched over a pause
         {"w": "yes", "start": 3.5, "end": 3.8, "voice": "voice:SPEAKER_01"},
         {"w": "no", "start": 4.0, "end": 4.2, "voice": None}]
UTTS = [{"id": "utt:0", "start": 0.0, "end": 3.0, "text": "hi there.", "shot": "shot:0", "words": [0, 1], "voice": "voice:SPEAKER_00"},
        {"id": "utt:1", "start": 3.5, "end": 3.8, "text": "yes", "shot": "shot:0", "words": [2, 2], "voice": "voice:SPEAKER_01"},
        {"id": "utt:2", "start": 4.0, "end": 4.2, "text": "no", "shot": "shot:1", "words": [3, 3], "voice": None}]


def rows(*predicted):
    return {"utterances": [{"utt": f"utt:{i}", "predicted": p} for i, p in enumerate(predicted)]}


def test_a_word_is_analysed_at_most_its_cap_after_its_onset():
    assert WORD_MAX_S == 1.2   # the bake-off's, and the prosody-captions node's (style_captions.WORD_MAX_S)
    assert word_end({"start": 0.4, "end": 3.0}) == pytest.approx(1.6)
    assert word_end({"start": 0.4, "end": 0.9}) == 0.9


def test_the_speaker_is_the_speakers_layers_person_else_the_diarized_voice():
    assert speaker_of({"kind": "face", "face": "face:6", "via": "asd", "person": "person:Person G"}, "voice:SPEAKER_01") == ("Person G", "asd_face")
    assert speaker_of({"kind": "person", "person": "person:Person F", "voice": "voice:SPEAKER_02", "via": "voice_link"},
                      "voice:SPEAKER_02") == ("Person F", "asd_voice_link")
    # an off-screen voice is named as the bake-off named one (the app draws "off-screen ..." neutral)
    assert speaker_of({"kind": "offscreen", "voice": "voice:SPEAKER_00", "via": "none"}, "voice:SPEAKER_00") == ("off-screen SPEAKER_00", "offscreen_voice")
    # a speaking face with no person: the voice, as the bake-off's fallback
    assert speaker_of({"kind": "face", "face": "face:9", "via": "asd"}, "voice:SPEAKER_01") == ("SPEAKER_01", "diarized_voice")
    assert speaker_of({"kind": "offscreen", "voice": None, "via": "none"}, None) == ("off-screen unknown voice", "offscreen_voice")
    with pytest.raises(ValueError, match="neither asd nor voice_link"):
        speaker_of({"kind": "face", "person": "person:X", "via": "hand"}, None)


def test_units_are_the_transcripts_utterances_with_their_words_and_analysed_end():
    got = units(transcript(UTTS, WORDS), rows({"kind": "face", "via": "asd", "person": "person:Person A"},
                                              {"kind": "person", "via": "voice_link", "person": "person:Person B"},
                                              {"kind": "offscreen", "voice": None, "via": "none"}))
    assert [u["id"] for u in got] == ["utt:0", "utt:1", "utt:2"] and [u["n"] for u in got] == [0, 1, 2]
    assert got[0]["words"] == [0, 1] and got[0]["analysed_end"] == 1.6 and got[0]["end"] == 3.0
    assert [(u["speaker"], u["speaker_source"], u["voice"]) for u in got] == [
        ("Person A", "asd_face", "SPEAKER_00"), ("Person B", "asd_voice_link", "SPEAKER_01"),
        ("off-screen unknown voice", "offscreen_voice", None)]


def test_layers_that_disagree_are_refused():
    ok = rows({"kind": "offscreen", "via": "none"}, {"kind": "offscreen", "via": "none"}, {"kind": "offscreen", "via": "none"})
    with pytest.raises(ValueError, match="does not label the transcript's utterances"):
        units(transcript(UTTS, WORDS), {"utterances": ok["utterances"][:2]})
    with pytest.raises(ValueError, match="partition the words in order"):
        units(transcript([UTTS[0], {**UTTS[1], "words": [1, 2]}, UTTS[2]], WORDS), ok)
    with pytest.raises(ValueError, match="but its words run"):
        units(transcript([{**UTTS[0], "start": 0.1}, UTTS[1], UTTS[2]], WORDS), ok)
    with pytest.raises(ValueError, match="not an utterance id"):
        units(transcript([{**UTTS[0], "id": "u0"}], WORDS[:2]), {"utterances": [{"utt": "u0", "predicted": {"kind": "offscreen"}}]})
