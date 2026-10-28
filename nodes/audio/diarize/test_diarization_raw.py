"""The diarization's side files (diarization_raw.py): the stored turns give exactly the layer's voices, the embeddings
are kept per voice as pyannote returned them, and both files are the ones the bundle schema describes."""

import json
import math
from pathlib import Path

import pytest

import diarization_raw
from diarization_raw import DIMS, diarization, pointers, voice_embeddings
from perception_node import NodeError
from voices import diarize

SCHEMA = Path(__file__).resolve().parents[3] / "schema" / "bundle-0.1.schema.json"
MODEL = "pyannote/speaker-diarization-community-1"


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def transcript(words, spans):
    """asr's layer: no voices, every voice null."""
    words = [{"w": w, "start": s, "end": e, "voice": None} for w, s, e in words]
    return {"layer": "transcript", "schema_version": "0.1.0", "producer": {"stage": "transcribe"}, "voices": [],
            "words": words, "utterances": [{"id": f"utt:{i}", "start": words[a]["start"], "end": words[b]["end"], "text": "",
                                            "shot": "shot:0", "words": [a, b], "voice": None} for i, (a, b) in enumerate(spans)]}


def as_pyannote(stored: list[dict]) -> list[dict]:
    """Stored turns back as the turns voices.diarize takes: voice:SPEAKER_00 -> SPEAKER_00."""
    return [turn(t["start"], t["end"], t["voice"].removeprefix("voice:")) for t in stored]


def test_the_stored_turns_give_exactly_the_layers_voices():
    """Crosstalk included, times on pyannote's frame grid (sums no decimal rounds to): reassigning the words from the
    side file's turns gives the layer the node wrote."""
    step = 0.016875
    turns = [turn(0.03096875, 1.0265937500000002, "SPEAKER_01"), turn(0.9, 1.0 + 7 * step, "SPEAKER_00"),
             turn(1.0265937500000002, 2.03909375, "SPEAKER_00"), turn(2.5, 3.0, "SPEAKER_02")]
    before = transcript([("a", 0.1, 0.5), ("b", 0.95, 1.12), ("c", 1.2, 2.0), ("d", 2.6, 2.9), ("e", 4.0, 4.5)],
                        [(0, 1), (2, 4)])
    layer = diarize(before, turns)
    side = diarization(MODEL, ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"], turns, turns[:1] + turns[2:])
    assert diarize(before, as_pyannote(side["turns"])) == layer
    assert [(t["start"], t["end"]) for t in side["turns"]] == [(t["start"], t["end"]) for t in turns]   # unrounded
    assert [w["voice"] for w in layer["words"]] == ["voice:SPEAKER_01", "voice:SPEAKER_00", "voice:SPEAKER_00",
                                                    "voice:SPEAKER_02", None]


def test_every_speaker_is_kept_in_pyannotes_order_with_the_exclusive_turns():
    side = diarization(MODEL, ["SPEAKER_00", "SPEAKER_01"], [turn(0.0, 2.0, "SPEAKER_01"), turn(1.0, 1.5, "SPEAKER_00")],
                       [turn(0.0, 2.0, "SPEAKER_01")])
    assert side == {"model": MODEL, "voices": ["voice:SPEAKER_00", "voice:SPEAKER_01"],
                    "turns": [{"start": 0.0, "end": 2.0, "voice": "voice:SPEAKER_01"},
                              {"start": 1.0, "end": 1.5, "voice": "voice:SPEAKER_00"}],
                    "exclusive_turns": [{"start": 0.0, "end": 2.0, "voice": "voice:SPEAKER_01"}]}


def test_no_speech_is_no_speakers_and_no_turns():
    assert diarization(MODEL, [], [], []) == {"model": MODEL, "voices": [], "turns": [], "exclusive_turns": []}
    assert voice_embeddings([], [])["voices"] == {}


@pytest.mark.parametrize("bad", [turn(0.0, 1.0, "SPEAKER_09"), turn(1.0, 0.5, "SPEAKER_00"), turn(0.0, math.nan, "SPEAKER_00")])
def test_a_turn_pyannote_could_not_have_given_is_an_internal_error(bad):
    with pytest.raises(NodeError) as e:
        diarization(MODEL, ["SPEAKER_00"], [bad], [])
    assert e.value.code == "internal"


def row(seed):
    return [((seed * 7919 + i * 104729) % 10007) / 10007 - 0.5 for i in range(DIMS)]


def test_embeddings_are_kept_per_voice_in_order_to_four_places():
    rows = [row(1), row(2)]
    doc = voice_embeddings(["SPEAKER_00", "SPEAKER_01"], rows)
    assert doc["dims"] == DIMS and doc["normalised"] is False
    assert list(doc["voices"]) == ["voice:SPEAKER_00", "voice:SPEAKER_01"]
    for v, r in zip(doc["voices"].values(), rows):
        assert v == [round(x, 4) for x in r] and max(abs(a - b) for a, b in zip(v, r)) <= 5e-5


def test_a_speaker_pyannote_padded_with_zeros_has_no_embedding():
    """pyannote pads the centroids with zero rows when it labels more speakers than it has centroids."""
    doc = voice_embeddings(["SPEAKER_00", "SPEAKER_01"], [row(1), [0.0] * DIMS])
    assert doc["voices"]["voice:SPEAKER_01"] is None and doc["voices"]["voice:SPEAKER_00"] is not None


@pytest.mark.parametrize("embeddings, message", [
    (None, "no speaker embeddings"), ([row(1)], "1 speaker embeddings for 2 speakers"),
    ([row(1), row(2)[:128]], "128 values, not 256"), ([row(1), row(2)[:-1] + [math.inf]], "not finite")])
def test_embeddings_pyannote_could_not_have_given_are_an_internal_error(embeddings, message):
    with pytest.raises(NodeError, match=message) as e:
        voice_embeddings(["SPEAKER_00", "SPEAKER_01"], embeddings)
    assert e.value.code == "internal"


def test_the_files_are_the_side_files_the_schema_describes():
    """The pointers name #/$defs/side/<name> as layers/<name>.json, and each file has the fields that schema requires
    (the orchestrator validates the files against it when the node writes them)."""
    side = json.loads(SCHEMA.read_text())["$defs"]["side"]
    docs = {"transcript_diarization": diarization(MODEL, ["SPEAKER_00"], [turn(0.0, 1.0, "SPEAKER_00")], []),
            "transcript_voice_embeddings": voice_embeddings(["SPEAKER_00"], [row(1)])}
    assert {p["file"] for p in pointers().values()} == {f"layers/{name}.json" for name in docs}
    assert {diarization_raw.DIARIZATION_FILE, diarization_raw.EMBEDDINGS_FILE} == {f"layers/{name}.json" for name in docs}
    for name, doc in docs.items():
        assert set(doc) == set(side[name]["required"]) == set(side[name]["properties"])
    assert side["transcript_voice_embeddings"]["properties"]["dims"]["const"] == DIMS
