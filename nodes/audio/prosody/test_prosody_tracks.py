"""The prosody tracks side file (prosody_tracks.py): the columns the bundle schema documents, the same bytes for the
same frames, and the layer's measures computed again from the file alone."""

import json
from pathlib import Path

import numpy as np
import pytest

from prosody_measure import measure, semitones
from prosody_tracks import DESCRIPTION, FILE, TRACKS, blocks, read, write
from prosody_units import word_end
from test_prosody_measure import RATE, FakePraat, two_speakers

SCHEMA = Path(__file__).resolve().parents[3] / "schema" / "bundle-0.1.schema.json"


def measured():
    f0, words, units = two_speakers(rise=True)
    return measure(units, words, int(4.0 * RATE), FakePraat(f0)), words, units


def test_rows_come_in_file_order_and_a_value_that_is_not_finite_is_refused():
    m, _, units = measured()
    order = [(name, utt) for name, utt, *_ in blocks(m["tracks"], units)]
    assert order == [("pitch_first_pass", -1), ("intensity", -1), ("pitch", 0), ("harmonicity", 0), ("pulses", 0),
                     ("pitch", 1), ("harmonicity", 1), ("pulses", 1)]
    assert {name for name, _ in order} == set(TRACKS)
    m["tracks"]["intensity"] = (m["tracks"]["intensity"][0], np.full(len(m["tracks"]["intensity"][0]), -np.inf))
    with pytest.raises(ValueError, match="intensity of utterance -1: a value that is not finite"):
        list(blocks(m["tracks"], units))


def test_the_file_has_the_schemas_columns_and_is_the_same_bytes_twice(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/prosody); CI installs it")
    m, _, units = measured()
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    rows = write(a, m["tracks"], units)
    write(b, m["tracks"], units)
    assert a.read_bytes() == b.read_bytes()
    spec = json.loads(SCHEMA.read_text())["$defs"]["parquet"]["prosody_tracks"]
    schema = pq.read_schema(a)
    assert [(f.name, str(f.type)) for f in schema] == [(c["name"], c["arrow"]) for c in spec["columns"]]
    assert not any(f.nullable for f in schema)
    assert pq.read_metadata(a).num_rows == rows
    assert FILE == "layers/prosody_tracks.parquet" and DESCRIPTION


def test_the_layers_pitch_measures_come_back_from_the_file(tmp_path):
    """Every word's F0 summary, each utterance's pitch measures and the contour bins, from the stored frames and the
    speakers' medians (the speakers side file)."""
    pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/prosody); CI installs it")
    m, words, units = measured()
    path = tmp_path / "tracks.parquet"
    write(path, m["tracks"], units)
    got = read(path)
    names = m["contour"]["speakers"]
    for u, row in zip(units, m["utterances"]):
        speaker, t, f, _ = got["pitch"][u["n"]]
        assert names[speaker] == u["speaker"]
        median = m["speakers"][u["speaker"]]["median_hz"]
        for w in row["words"]:
            seg = f[(t >= w["start"]) & (t <= word_end(words[w["i"]])) & (f > 0)]
            assert w["f0_max_st"] == round(float(semitones(seg.max(), median)), 2)
            assert w["f0_mean_st"] == round(float(semitones(seg.mean(), median)), 2)
        voiced = np.concatenate([f[(t >= w["start"]) & (t <= word_end(words[w["i"]])) & (f > 0)] for w in row["words"]])
        assert row["measures"]["f0_median_hz"] == round(float(np.median(voiced)), 1)
    c = m["contour"]
    for b, hz in enumerate(c["f0_hz"]):
        if hz is None:
            continue
        u = next(u for u in units if u["start"] <= b * c["step"] + c["step"] and b * c["step"] < u["analysed_end"])
        _, t, f, _ = got["pitch"][u["n"]]
        a = b * c["step"]
        seg = f[(t >= max(a, u["start"])) & (t < min(a + c["step"], u["analysed_end"])) & (f > 0)]
        assert hz == round(float(np.median(seg)), 1)
    _, ti, db, _ = got["intensity"][-1]
    assert np.array_equal(ti, m["tracks"]["intensity"][0]) and np.array_equal(db, m["tracks"]["intensity"][1])
