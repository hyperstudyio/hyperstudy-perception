"""The faces layer's side file (face_detail.py). Run: uv run --no-project --with pytest pytest nodes/vision/face-track"""

import json
import math

import pytest

from face_detail import ANGLES, EMOTIONS, FIELDS, FILE, LANDMARKS, OTHER, au_names, missing_columns, reference, side_file

AUS = ["AU12", "AU01", "AU25"]


def row(seed: float) -> dict:
    """A py-feat row: every column the side file reads, with values that need rounding, and columns it ignores."""
    r = {c: 100 + seed + i / 7 for i, c in enumerate(LANDMARKS)}
    r |= {c: 0.1 + seed / 10 + i / 300 for i, c in enumerate(AUS + EMOTIONS)}
    r |= {c: -0.2 + seed / 100 + i / 3000 for i, c in enumerate(OTHER)}
    r |= {"FaceRectX": 1.0, "mesh_x_0": 5.0, "Identity_1": 0.3}
    return r


def test_samples_hold_each_field_rounded_as_the_bake_off_rounded_it():
    r = row(1)
    doc = side_file([[(0.125, r)]], au_names(r))
    s = doc["tracks"]["face:0"][0]
    assert list(s) == ["t"] + FIELDS
    assert s["t"] == 0.125
    assert len(s["lmk"]) == 136 and s["lmk"][0] == round(r["x_0"], 1) and s["lmk"][68] == round(r["y_0"], 1)
    assert s["au"] == [round(r[c], 2) for c in ["AU01", "AU12", "AU25"]]      # in au_names' order, not the row's
    assert s["emo"] == [round(r[c], 3) for c in EMOTIONS]
    assert s["va"] == [round(r["valence"], 3), round(r["arousal"], 3)]
    assert s["head"] == [round(r["Yaw"], 3), round(r["Pitch"], 3), round(r["Roll"], 3)]
    assert s["gaze"] == [round(r["gaze_yaw"], 3), round(r["gaze_pitch"], 3)]


def test_tracks_are_keyed_by_the_faces_layer_ids_in_its_order():
    tracks = [[(0.0, row(0)), (0.125, row(1))], [(2.0, row(2))], [(3.5, row(3)), (3.625, row(4)), (3.75, row(5))]]
    doc = side_file(tracks, au_names(row(0)))
    assert list(doc["tracks"]) == ["face:0", "face:1", "face:2"]
    assert [[s["t"] for s in samples] for samples in doc["tracks"].values()] == [[0.0, 0.125], [2.0], [3.5, 3.625, 3.75]]
    assert doc["au_names"] == ["AU01", "AU12", "AU25"] and doc["emotion_names"] == EMOTIONS and doc["angles"] == ANGLES


def test_the_document_is_what_the_bake_off_bundle_held():
    """Key order and shape of layers/faces_detail.json in the bake-off's converted bundle, which the annotator reads."""
    doc = side_file([[(0.0, row(0))]], au_names(row(0)))
    assert list(doc) == ["au_names", "emotion_names", "angles", "tracks"]
    assert reference() == {"file": FILE, "fields": ["lmk", "au", "emo", "va", "head", "gaze"]}
    assert FILE == "layers/faces_detail.json"
    json.loads(json.dumps(doc, allow_nan=False))


def test_no_tracks_is_an_empty_side_file():
    assert side_file([], AUS)["tracks"] == {}


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_value_that_is_not_a_number_is_an_error_not_invalid_json(bad):
    r = row(0) | {"AU12": bad}
    with pytest.raises(ValueError, match="AU12"):
        side_file([[(1.5, r)]], au_names(r))


def test_missing_columns_names_each_one():
    r = row(0)
    assert missing_columns(r) == []
    del r["x_3"], r["Roll"]
    assert missing_columns(r) == ["x_3", "Roll"]
    assert missing_columns([c for c in row(0) if not c.startswith("AU")]) == ["AU*"]
