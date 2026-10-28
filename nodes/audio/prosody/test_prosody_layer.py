"""The prosody layer and its JSON side files (prosody_layer.py): the shape convert_bakeoff.py's prosody_layer gave the
annotator, and the emotion the layer keeps from the raw outputs."""

import json
from pathlib import Path

import pytest

from prosody_layer import EMOTION_FILE, EMOTION_KEY, SPEAKERS_FILE, emotion_row, emotion_side, layer, pointers, speakers_side
from prosody_measure import SETTINGS, measure
from test_prosody_measure import RATE, FakePraat, two_speakers

SCHEMA = json.loads((Path(__file__).resolve().parents[3] / "schema" / "bundle-0.1.schema.json").read_text())
LABELS = ["arousal", "dominance", "valence"]


def test_the_layer_keeps_the_raw_emotion_to_six_places_and_none_for_a_clip_too_short():
    assert emotion_row(LABELS, [0.37785184383392334, 0.3467254340648651, 0.4695574641227722]) == {
        EMOTION_KEY: {"arousal": 0.377852, "dominance": 0.346725, "valence": 0.469557}}   # OOO utt:3's, as the bake-off kept it
    assert emotion_row(LABELS, None) == {}
    with pytest.raises(ValueError):
        emotion_row(LABELS, [0.1, 0.2])


def test_the_layer_is_the_bakeoffs_shape_with_the_pipelines_models():
    f0, words, units = two_speakers()
    m = measure(units, words, int(4.0 * RATE), FakePraat(f0))
    raw = [[0.5, 0.4, 0.3], None]
    doc = layer(units, m, LABELS, raw, {"model": "odyssey@x", "licence_mode": "needs-review"}, {"stage": "prosody"},
                pointers("layers/prosody_tracks.parquet", "frames"))
    assert set(doc) == {"layer", "schema_version", "producer", "contour", "default_dimensional", "emotion_models",
                        "caption_models", "utterances", "raw"}
    assert set(doc["contour"]) == {"step", "t0", "speakers", "f0_st", "speaker"}   # Hz goes to the speakers file
    assert doc["default_dimensional"] == EMOTION_KEY and doc["caption_models"] == {}
    assert doc["emotion_models"] == {EMOTION_KEY: {"model": "odyssey@x", "licence_mode": "needs-review", "kind": "dimensional"}}
    u0, u1 = doc["utterances"]
    assert set(u0) == {"utt", "speaker", "speaker_source", "measures", "words", "emotion", "captions"}
    assert u0["utt"] == "utt:0" and u0["speaker"] == "A" and u0["emotion"] == {EMOTION_KEY: {"arousal": 0.5, "dominance": 0.4, "valence": 0.3}}
    assert u1["emotion"] == {} and u1["captions"] == {}
    # every measure the app reads (types.ts ProsodyMeasures), and each word's word and prominence
    for key in ("final_movement", "final_delta_st", "f0_median_hz", "f0_range_st", "speech_rate_syl_s", "internal_pauses",
                "pause_before_s", "intensity_db", "hnr_db", "jitter_local", "shimmer_local"):
        assert key in u0["measures"]
    assert {"word", "prominence"} <= set(u0["words"][0])
    assert {r: p["file"] for r, p in doc["raw"].items()} == {
        "tracks": "layers/prosody_tracks.parquet", "speakers": SPEAKERS_FILE, "emotion": EMOTION_FILE}
    with pytest.raises(ValueError):
        layer(units, m, LABELS, raw[:1], {}, {}, {})


def test_the_side_files_have_the_schemas_required_fields():
    f0, words, units = two_speakers()
    m = measure(units, words, int(4.0 * RATE), FakePraat(f0))
    sp = speakers_side(m, SETTINGS, "0.4.7", "6.1.38")
    spec = SCHEMA["$defs"]["side"]["prosody_speakers"]
    assert set(spec["required"]) <= set(sp) and sp["model"] == "Parselmouth 0.4.7 (Praat 6.1.38)"
    assert set(spec["properties"]["speakers"]["additionalProperties"]["required"]) <= set(sp["speakers"]["A"])
    assert sp["contour"]["f0_hz"] == m["contour"]["f0_hz"]
    em = emotion_side(units, [(1600, 24000), (32000, 54400)], LABELS, [[0.5, 0.4, 0.3], None], "odyssey@x",
                      {"mean": -8e-05, "std": 0.08}, 1e-6, 400, 16000)
    assert set(SCHEMA["$defs"]["side"]["prosody_emotion"]["required"]) <= set(em)
    assert em["utterances"] == [{"utt": "utt:0", "samples": [1600, 24000], "values": [0.5, 0.4, 0.3]},
                                {"utt": "utt:1", "samples": [32000, 54400], "values": None}]
