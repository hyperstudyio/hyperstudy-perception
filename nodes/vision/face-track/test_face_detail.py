"""The faces layer's side files (face_detail.py). Run: uv run --no-project --with pytest pytest nodes/vision/face-track"""

import json
import math

import pytest

from face_detail import (ANGLES, EMOTIONS, FIELDS, FILE, LANDMARKS, MESH, MESH_FILE, MESH_LAYOUT, OTHER, au_names,
                         mesh_file, mesh_reference, mesh_sample, missing_columns, reference, side_file)

AUS = ["AU12", "AU01", "AU25"]


def row(seed: float) -> dict:
    """A py-feat row: every column the side files read, with values that need rounding, and columns they ignore."""
    r = {c: 100 + seed + i / 7 for i, c in enumerate(LANDMARKS)}
    r |= {c: 0.1 + seed / 10 + i / 300 for i, c in enumerate(AUS + EMOTIONS)}
    r |= {c: -0.2 + seed / 100 + i / 3000 for i, c in enumerate(OTHER)}
    r |= {c: 300 + seed + i / 3 for i, c in enumerate(MESH)}
    r |= {"FaceRectX": 1.0, "mesh_z_0": 5.0, "Identity_1": 0.3}
    return r


def mesh_values(r: dict) -> list[float]:
    """What the node passes for a sample's mesh: the row's MESH columns in that order."""
    return [r[c] for c in MESH]


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
    del r["x_3"], r["Roll"], r["mesh_y_477"]
    assert missing_columns(r) == ["x_3", "Roll", "mesh_y_477"]
    assert missing_columns([c for c in row(0) if not c.startswith("AU")]) == ["AU*"]


def test_the_mesh_columns_are_all_478_x_then_all_478_y():
    assert len(MESH) == 956 and MESH[0] == "mesh_x_0" and MESH[477] == "mesh_x_477"
    assert MESH[478] == "mesh_y_0" and MESH[-1] == "mesh_y_477"


def test_a_mesh_sample_rounds_each_value_to_a_whole_pixel_as_the_bake_off_did():
    """The bake-off wrote round(float(v)): an int, with halves to even (100.5 -> 100, 101.5 -> 102)."""
    values = mesh_values(row(1))
    values[:5] = [100.5, 101.5, 2.4999, -0.5, 7.5001]
    s = mesh_sample(0.25, values)
    assert list(s) == ["t", "xy"] and s["t"] == 0.25
    assert s["xy"][:5] == [100, 102, 2, 0, 8]
    assert s["xy"] == [round(float(v)) for v in values]
    assert all(type(v) is int for v in s["xy"])


def test_mesh_tracks_are_keyed_by_the_faces_layer_ids_in_its_order():
    tracks = [[(0.0, mesh_values(row(0))), (0.125, mesh_values(row(1)))], [(2.0, mesh_values(row(2)))]]
    doc = mesh_file(tracks)
    assert list(doc["tracks"]) == ["face:0", "face:1"]
    assert [[s["t"] for s in samples] for samples in doc["tracks"].values()] == [[0.0, 0.125], [2.0]]
    r = row(2)
    xy = doc["tracks"]["face:1"][0]["xy"]
    assert xy[0] == round(r["mesh_x_0"]) and xy[478] == round(r["mesh_y_0"]) and xy[-1] == round(r["mesh_y_477"])


def test_the_mesh_document_holds_its_layout_and_the_layer_points_at_it():
    doc = mesh_file([[(0.0, mesh_values(row(0)))]])
    assert list(doc) == ["points", "layout", "tracks"]
    assert doc["points"] == 478 and doc["layout"] == MESH_LAYOUT and "478 x, then all 478 y" in MESH_LAYOUT
    assert mesh_reference() == {"file": MESH_FILE} and MESH_FILE == "layers/faces_mesh.json"
    json.loads(json.dumps(doc, allow_nan=False))
    assert mesh_file([])["tracks"] == {}


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_mesh_value_that_is_not_a_number_is_an_error_not_invalid_json(bad):
    values = mesh_values(row(0))
    values[478 + 3] = bad
    with pytest.raises(ValueError, match="mesh_y_3"):
        mesh_file([[(1.5, values)]])


def test_a_mesh_sample_needs_every_value():
    with pytest.raises(ValueError, match="955 mesh values"):
        mesh_sample(1.5, mesh_values(row(0))[:-1])
