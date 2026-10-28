"""The side files (detections.py): the Parquet files have the columns the bundle schema documents, hold the rows as
bodies.side_rows gives them (float32), and are the same bytes for the same rows."""

import json
from pathlib import Path

import numpy as np
import pytest

from detections import (DUPLICATES_DESCRIPTION, DUPLICATES_FILE, ROW_GROUP_ROWS, SCORES_DESCRIPTION, SCORES_FILE,
                        write_duplicates, write_scores)

SCHEMA = Path(__file__).resolve().parents[3] / "schema" / "bundle-0.1.schema.json"
SCORES = [("body:0", 0, 0.0, 0.91234567), ("body:3", 0, 0.0, 0.5), ("body:0", 1, 0.125, 0.875)]
KPTS = [float(v) for v in np.linspace(0, 400, 399)]
DUPLICATES = [(0.0, "body:3", 0, 0.45, 0.0123, [302.04, 5.0, 95.0, 190.0], KPTS),
              (69.0, "body:7", 12, 0.31, 0.05, [1.0, 2.0, 3.0, 4.0], KPTS[::-1])]
ARGS = ("YOLOX-m", 0.3)


def columns(name):
    return [(c["name"], c["arrow"]) for c in json.loads(SCHEMA.read_text())["$defs"]["parquet"][name]["columns"]]


def test_the_files_have_the_columns_the_bundle_schema_documents(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side files (envs/vision); CI installs it")
    write_scores(tmp_path / "s.parquet", SCORES, *ARGS)
    write_duplicates(tmp_path / "d.parquet", DUPLICATES, *ARGS, 0.1, 3.0)
    scores, dups = pq.read_table(tmp_path / "s.parquet"), pq.read_table(tmp_path / "d.parquet")
    assert [(f.name, str(f.type)) for f in scores.schema] == columns("bodies_scores")
    assert [(f.name, str(f.type)) for f in dups.schema] == columns("bodies_duplicates")
    assert all(not f.nullable for f in [*scores.schema, *dups.schema])
    assert list(zip(*(scores[c].to_pylist() for c in ("body", "sample", "t")))) == [r[:3] for r in SCORES]
    assert scores["score"].to_pylist() == [float(np.float32(r[3])) for r in SCORES]          # float32, unrounded
    assert dups["body"].to_pylist() == ["body:3", "body:7"] and dups["sample"].to_pylist() == [0, 12]
    assert dups["kpts"].to_pylist()[1] == [float(np.float32(v)) for v in KPTS[::-1]]
    assert dups["box"].to_pylist()[0] == [float(np.float32(v)) for v in DUPLICATES[0][5]]
    meta = dups.schema.metadata
    assert (meta[b"detector"], meta[b"detector_threshold"], meta[b"duplicate_head_dist"], meta[b"score_threshold"]) == (
        b"YOLOX-m", b"0.3", b"0.1", b"3.0")
    assert SCORES_FILE == "layers/bodies_scores.parquet" and DUPLICATES_FILE == "layers/bodies_duplicates.parquet"
    assert SCORES_DESCRIPTION and DUPLICATES_DESCRIPTION


def test_no_duplicates_is_an_empty_file_with_the_columns(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side files (envs/vision); CI installs it")
    write_duplicates(tmp_path / "d.parquet", [], *ARGS, 0.1, 3.0)
    table = pq.read_table(tmp_path / "d.parquet")
    assert table.num_rows == 0 and [(f.name, str(f.type)) for f in table.schema] == columns("bodies_duplicates")


def test_the_same_rows_are_the_same_bytes(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side files (envs/vision); CI installs it")
    rows = [(f"body:{k % 7}", k // 7, round(k / 8, 3), 0.3 + (k % 70) / 100) for k in range(ROW_GROUP_ROWS + 9)]
    for name, write, data in (("s", write_scores, (rows, *ARGS)), ("d", write_duplicates, (DUPLICATES * 50, *ARGS, 0.1, 3.0))):
        first, second = tmp_path / f"{name}1.parquet", tmp_path / f"{name}2.parquet"
        write(first, *data)
        write(second, *data)
        assert first.read_bytes() == second.read_bytes()
    meta = pq.ParquetFile(tmp_path / "s1.parquet").metadata
    assert meta.num_rows == len(rows) and meta.num_row_groups == 2 and meta.row_group(0).column(0).compression == "ZSTD"


@pytest.mark.parametrize("bad", [
    lambda: write_scores("x.parquet", [("body:0", 0, 0.0, float("nan"))], *ARGS),
    lambda: write_duplicates("x.parquet", [DUPLICATES[0][:6] + (KPTS[:-3],)], *ARGS, 0.1, 3.0),   # 132 keypoints
    lambda: write_duplicates("x.parquet", [DUPLICATES[0][:4] + (float("inf"),) + DUPLICATES[0][5:]], *ARGS, 0.1, 3.0),
])
def test_a_value_that_is_not_finite_or_a_short_row_is_refused(tmp_path, monkeypatch, bad):
    pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side files (envs/vision); CI installs it")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        bad()
    assert not (tmp_path / "x.parquet").exists()
