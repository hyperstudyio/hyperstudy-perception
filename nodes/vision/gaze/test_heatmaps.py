"""The heatmaps side file (heatmaps.py): quantisation, and that the Parquet file has the columns the bundle schema
documents and is the same bytes for the same rows."""

import json
from pathlib import Path

import numpy as np
import pytest

from heatmaps import CELLS, DESCRIPTION, FILE, ROW_GROUP_ROWS, Writer, dequantise, quantise

SCHEMA = Path(__file__).resolve().parents[3] / "schema" / "bundle-0.1.schema.json"


def heatmap(seed):
    """A sigmoid-like map with one clear peak and a gentle background."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:64, 0:64]
    cy, cx = rng.integers(0, 64, 2)
    return (0.02 + 0.6 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 30) + rng.uniform(0, 0.01, (64, 64))).astype(np.float32)


def test_a_map_is_its_cells_relative_to_its_maximum_within_half_a_step():
    heat = heatmap(1)
    cells, top = quantise(heat)
    assert len(cells) == CELLS and top == float(heat.max())
    q = np.frombuffer(cells, np.uint8).reshape(64, 64)
    assert np.array_equal(q, np.floor(heat.astype(np.float64) / top * 255))
    assert np.abs(dequantise(cells, top) - heat).max() <= top / 510 + 1e-7   # and a float32 rounding
    # row-major: byte 64 * row + col is cell (row, col), and only the maximum is 255
    row, col = np.unravel_index(int(heat.argmax()), heat.shape)
    assert cells[64 * row + col] == 255 and (q == 255).sum() == 1


def test_the_stored_maps_argmax_is_the_peak_even_a_hair_below_it():
    """Rounded, a cell within half a step of the maximum would be 255 too, and come first in row-major order."""
    heat = heatmap(2)
    near = heat.copy()
    near.reshape(-1)[0] = heat.max() * (1 - 0.4 / 255)
    assert int(np.frombuffer(quantise(near)[0], np.uint8).argmax()) == int(near.argmax()) != 0
    for top in (np.float32(0.9164568), np.float32(0.0553160), np.float32(1.0), np.float32(2.0 ** -20)):
        hair = np.full((64, 64), np.nextafter(top, np.float32(0)), np.float32)   # the float32 just below the maximum
        hair[40, 7] = top
        q = np.frombuffer(quantise(hair)[0], np.uint8)
        assert q[64 * 40 + 7] == 255 and (q == 255).sum() == 1 and q.min() == 254


def test_an_all_zero_map_is_all_zero_cells_and_a_bad_one_is_refused():
    assert quantise(np.zeros((64, 64), np.float32)) == (bytes(CELLS), 0.0)
    for bad in (np.full((64, 64), np.nan, np.float32), -heatmap(3), np.zeros((32, 32), np.float32)):
        with pytest.raises(ValueError):
            quantise(bad)


def write(path, rows):
    with Writer(path) as w:
        for row in rows:
            w.add(*row)
    return path


ROWS = [("face:0", 0, 0.0, heatmap(4), 0.91), ("face:3", 0, 0.0, heatmap(5), 0.12), ("face:0", 1, 0.133, heatmap(6), 0.875)]


def test_the_file_has_the_columns_the_bundle_schema_documents(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/gaze); CI installs it")
    table = pq.read_table(write(tmp_path / "gaze_heatmaps.parquet", ROWS))
    spec = json.loads(SCHEMA.read_text())["$defs"]["parquet"]["gaze_heatmaps"]
    assert [(f.name, str(f.type)) for f in table.schema] == [(c["name"], c["arrow"]) for c in spec["columns"]]
    assert all(not f.nullable for f in table.schema)
    assert table["face"].to_pylist() == ["face:0", "face:3", "face:0"] and table["sample"].to_pylist() == [0, 0, 1]
    assert table["t"].to_pylist() == [0.0, 0.0, 0.133]
    assert table["inout"].to_pylist() == pytest.approx([0.91, 0.12, 0.875], abs=1e-7)
    for (_, _, _, heat, _), cells, top in zip(ROWS, table["heat"].to_pylist(), table["max"].to_pylist()):
        assert (cells, top) == quantise(heat)
    assert FILE == "layers/gaze_heatmaps.parquet" and DESCRIPTION


def test_the_same_rows_are_the_same_bytes(tmp_path):
    pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/gaze); CI installs it")
    rows = [(f"face:{k % 5}", k // 5, round(k / 30, 3), heatmap(k), 0.5) for k in range(ROW_GROUP_ROWS + 7)]
    first, second = write(tmp_path / "a.parquet", rows), write(tmp_path / "b.parquet", rows)
    assert first.read_bytes() == second.read_bytes()
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(first).metadata
    assert meta.num_rows == len(rows) and meta.num_row_groups == 2   # groups of ROW_GROUP_ROWS as rows come
    assert meta.row_group(0).column(0).compression == "ZSTD"


def test_a_score_that_is_not_finite_is_refused(tmp_path):
    pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/gaze); CI installs it")
    with pytest.raises(ValueError, match="in-frame score"):
        write(tmp_path / "x.parquet", [("face:0", 0, 0.0, heatmap(7), float("nan"))])
