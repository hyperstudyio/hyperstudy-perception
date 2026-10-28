"""The raw side files (pyfeat_raw.py) and the faces they hold: their columns are the ones the bundle schema documents,
the same rows are the same bytes, an untracked face keeps everything a tracked one has, and tracking.untracked finds
exactly the faces detection_rows leaves out."""

import json
from pathlib import Path

import numpy as np
import pytest

import face_detail
import pyfeat_raw
from tracking import MIN_FACE_PX, MIN_FACE_SCORE, detection_rows, dropped, untracked

SCHEMA = Path(__file__).resolve().parents[3] / "schema" / "bundle-0.1.schema.json"
AUS = [f"AU{n:02d}" for n in (1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43)]


def values(n, seed):
    """n float32 values (as py-feat gives them), held as float64 like the node's to_numpy."""
    return np.random.default_rng(seed).normal(0, 50, n).astype(np.float32).astype(np.float64)


def write_samples(path, rows):
    with pyfeat_raw.samples_writer(path) as w:
        for row in rows:
            w.add(*row)
    return path


def write_untracked(path, rows):
    with pyfeat_raw.untracked_writer(path, AUS) as w:
        for row in rows:
            w.add(*row)
    return path


SAMPLE_WIDTH = len(pyfeat_raw.flat(pyfeat_raw.sample_columns()))
UNTRACKED_WIDTH = len(pyfeat_raw.flat(pyfeat_raw.untracked_columns(AUS)))
SAMPLES = [("face:0", 0, 0.0, 0, values(SAMPLE_WIDTH, 1)), ("face:2", 0, 0.0, 0, values(SAMPLE_WIDTH, 2)),
           ("face:0", 1, 0.133, 4, values(SAMPLE_WIDTH, 3))]
UNTRACKED = [(0, 0.0, "size", values(UNTRACKED_WIDTH, 4)), (8, 0.267, "size+score", values(UNTRACKED_WIDTH, 5))]


def spec(name):
    return json.loads(SCHEMA.read_text())["$defs"]["parquet"][name]


@pytest.mark.parametrize("name, write, rows", [("faces_pyfeat", write_samples, SAMPLES),
                                               ("faces_untracked", write_untracked, UNTRACKED)])
def test_each_file_has_the_columns_the_bundle_schema_documents_and_keeps_py_feats_values(tmp_path, name, write, rows):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the raw side files (envs/faces); CI installs it")
    table = pq.read_table(write(tmp_path / f"{name}.parquet", rows))
    assert [(f.name, str(f.type)) for f in table.schema] == [(c["name"], c["arrow"]) for c in spec(name)["columns"]]
    assert all(not f.nullable for f in table.schema)
    columns = pyfeat_raw.sample_columns() if name == "faces_pyfeat" else pyfeat_raw.untracked_columns(AUS)
    keys = len(table.schema) - len(columns)
    for i, row in enumerate(rows):
        assert tuple(table[k][i].as_py() for k in table.schema.names[:keys]) == row[:-1]
        stored = np.concatenate([np.atleast_1d(np.asarray(table[c][i].as_py(), dtype=np.float32)) for c, _ in columns])
        assert np.array_equal(stored, row[-1].astype(np.float32))   # py-feat's float32, exactly
    assert spec(name).get("per_sample", False) == (name == "faces_pyfeat")
    meta = table.schema.metadata
    assert meta[b"blendshapes"].decode().split(",") == pyfeat_raw.BLENDSHAPES and len(pyfeat_raw.BLENDSHAPES) == 52
    if name == "faces_untracked":
        assert meta[b"au_names"].decode().split(",") == AUS


def test_the_same_rows_are_the_same_bytes(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the raw side files (envs/faces); CI installs it")
    rows = [(f"face:{k % 3}", k // 3, round(k / 24, 3), 4 * (k // 3), values(SAMPLE_WIDTH, k))
            for k in range(pyfeat_raw.ROW_GROUP_ROWS + 5)]
    first, second = write_samples(tmp_path / "a.parquet", rows), write_samples(tmp_path / "b.parquet", rows)
    assert first.read_bytes() == second.read_bytes()
    meta = pq.ParquetFile(first).metadata
    assert meta.num_rows == len(rows) and meta.num_row_groups == 2   # groups of ROW_GROUP_ROWS as rows come
    assert meta.row_group(0).column(0).compression == "ZSTD"
    encodings = {meta.row_group(0).column(i).path_in_schema: meta.row_group(0).column(i).encodings
                 for i in range(meta.num_columns)}
    assert all("BYTE_STREAM_SPLIT" in e for p, e in encodings.items() if p in ("score", "mesh_z.list.element", "box.list.element"))
    assert "BYTE_STREAM_SPLIT" not in encodings["t"] and "RLE_DICTIONARY" in encodings["face"]
    again = [write_untracked(tmp_path / f"u{i}.parquet", UNTRACKED).read_bytes() for i in range(2)]
    assert again[0] == again[1]


def test_no_detection_in_no_track_is_an_empty_file_with_the_columns(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the raw side files (envs/faces); CI installs it")
    table = pq.read_table(write_untracked(tmp_path / "u.parquet", []))
    assert table.num_rows == 0 and [f.name for f in table.schema] == [c["name"] for c in spec("faces_untracked")["columns"]]


def test_a_value_that_is_not_finite_is_refused_by_its_column(tmp_path):
    pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the raw side files (envs/faces); CI installs it")
    bad = values(SAMPLE_WIDTH, 6)
    bad[pyfeat_raw.flat(pyfeat_raw.sample_columns()).index("mesh_z_7")] = np.nan
    with pytest.raises(ValueError, match="mesh_z_7 = nan for the face in frame 12"):
        write_samples(tmp_path / "x.parquet", [("face:0", 0, 0.4, 12, bad)])
    with pytest.raises(ValueError, match="3 values for a row of"):
        write_samples(tmp_path / "y.parquet", [("face:0", 0, 0.4, 12, [1.0, 2.0, 3.0])])


def test_another_number_of_action_units_is_refused():
    with pytest.raises(ValueError, match="19 action units, not the 20"):
        pyfeat_raw.untracked_columns(AUS[:-1])


def test_an_untracked_face_keeps_everything_a_tracked_face_has():
    """faces_detail, faces_mesh, faces_embeddings and faces_pyfeat hold a tracked face's py-feat columns; an untracked
    face's row holds all of them, each once."""
    tracked = (face_detail.LANDMARKS + AUS + face_detail.EMOTIONS + face_detail.OTHER + face_detail.MESH
               + face_detail.EMBEDDING + pyfeat_raw.flat(pyfeat_raw.sample_columns()))
    held = pyfeat_raw.flat(pyfeat_raw.untracked_columns(AUS))
    assert len(held) == len(set(held)) and set(held) == set(tracked)
    # under faces_detail's names, in its orders
    columns = dict(pyfeat_raw.untracked_columns(AUS))
    assert columns["lmk"] == face_detail.LANDMARKS and columns["emo"] == face_detail.EMOTIONS
    assert columns["va"] + columns["head"] + columns["gaze"] == face_detail.OTHER
    assert columns["embedding"] == face_detail.EMBEDDING and columns["mesh"][:2 * 478] == face_detail.MESH


class Frame:
    """py-feat's output as tracking reads it: iterrows() over (label, row)."""

    def __init__(self, rows):
        self.rows = rows
        self.emotion_columns = face_detail.EMOTIONS

    def iterrows(self):
        return iter(self.rows.items())


def row(frame, x, w, h, score):
    r = {"frame": frame, "FaceRectX": x, "FaceRectY": 10.0, "FaceRectWidth": w, "FaceRectHeight": h, "FaceScore": score,
         "Yaw": 0.0, "Pitch": 0.0, "gaze_yaw": 0.0, "gaze_pitch": 0.0, "AU25": 0.0, "AU26": 0.0}
    return r | {f"Identity_{i + 1}": 0.1 for i in range(512)} | {e: 0.1 for e in face_detail.EMOTIONS}


def test_untracked_is_every_detected_face_detection_rows_leaves_out():
    big, small = 2.0 * MIN_FACE_PX, MIN_FACE_PX - 0.5
    rows = {10: row(0, 5.0, big, big, 0.99), 11: row(0, 300.0, small, big, 0.99), 12: row(0, 600.0, big, big, 0.6),
            13: row(4, 5.0, big, small, MIN_FACE_SCORE - 0.01), 14: row(8, float("nan"), float("nan"), float("nan"), 0.0),
            15: row(8, 9.0, big, big, MIN_FACE_SCORE)}
    fex = Frame(rows)
    kept = {r["index"] for r in detection_rows(fex, 30.0)}
    left = untracked(fex, 30.0)
    assert kept == {10, 15}
    assert left == [(11, 0, 0.0, "size"), (12, 0, 0.0, "score"), (13, 4, 0.133, "size+score")]
    # every row is kept, untracked, or py-feat's placeholder for a frame with no face
    assert kept | {index for index, *_ in left} | {14} == set(rows) and dropped(rows[14]) == "none"
