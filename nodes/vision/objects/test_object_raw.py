"""The masks side file (object_raw.py): columns as the bundle schema documents, the same bytes for the same rows, and
the rows read back.
Run: uv run --no-project --with pytest --with numpy --with av==16.1.0 --with pillow --with pyarrow==26.0.0 --with-editable ./sdk pytest -q nodes/vision/objects"""

import json
from pathlib import Path

import numpy as np
import pytest

import object_raw
from object_raw import COLUMNS, FILE, OUTCOMES, REASONS, read, row, write

HERE = Path(__file__).resolve().parent
SCHEMA = HERE.parents[2] / "schema" / "bundle-0.1.schema.json"
META = {"segmenter": "facebook/sam3@x", "frame_width": 1920, "frame_height": 1080, "fps": 6, "clothing_fps": 3.0}


def rows():
    a = row(1.0, "shot:0", "vocabulary", "chair", np.float32(0.8125), [10, 20, 30, 40], [[10, 20], [40, 20], [40, 60]],
            "trimmed", sam=3, tracker_score=np.float32(0.6875), person_share=np.float32(0.25), enclosed_share=0.0)
    a.update(outcome="tracks", track="obj:0", sample=0)
    b = row(1.167, "shot:0", "vocabulary", "chair", np.float32(0.8125), [10, 20, 30, 40], [], "covers_person", sam=3,
            tracker_score=np.float32(0.5), person_share=np.float32(0.75))
    b.update(outcome="tracks")                       # a dropped sample of a layer track: no sample named
    c = row(1.0, "shot:0", "body", "t-shirt", np.float32(0.96875), [5, 5, 50, 80], [[5, 5], [55, 5], [55, 85]], "item",
            body="body:4")
    c.update(outcome="rejected", rejected="obj:1")
    d = row(1.0, "shot:0", "body", "person", np.float32(0.375), [0, 0, 60, 90], [[0, 0], [60, 0], [60, 90]], "below_keep",
            body="body:4")
    return [a, b, c, d]


def test_the_columns_are_the_schemas(tmp_path):
    import pyarrow.parquet as pq
    spec = json.loads(SCHEMA.read_text())["$defs"]["parquet"]["objects_masks"]
    write(tmp_path / "m.parquet", rows(), META)
    got = [(f.name, str(f.type)) for f in pq.read_schema(tmp_path / "m.parquet")]
    assert got == [(c["name"], c["arrow"]) for c in spec["columns"]]
    assert [c["name"] for c in spec["columns"]] == list(COLUMNS)
    described = {c["name"]: c["description"] for c in spec["columns"]}
    assert all(word in described["reason"] for word in REASONS)
    assert all(word in described["outcome"] for word in OUTCOMES)
    assert FILE == "layers/objects_masks.parquet"


def test_rows_read_back_and_the_bytes_are_the_same(tmp_path):
    write(tmp_path / "a.parquet", rows(), META)
    write(tmp_path / "b.parquet", rows(), META)
    assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()
    back, meta = read(tmp_path / "a.parquet")
    assert meta["side"] == "objects_masks" and meta["frame_width"] == "1920"
    want = rows()
    assert [r["poly"] for r in back] == [w["poly"] for w in want]
    assert back[0] == {**want[0], "score": 0.8125, "tracker_score": 0.6875, "person_share": 0.25}
    assert [(r["reason"], r["outcome"], r["track"], r["rejected"], r["sample"]) for r in back] == [
        ("trimmed", "tracks", "obj:0", "", 0), ("covers_person", "tracks", "", "", -1),
        ("item", "rejected", "", "obj:1", -1), ("below_keep", "none", "", "", -1)]


@pytest.mark.parametrize("change, message", [
    ({"reason": "lost"}, "reason"), ({"outcome": "kept"}, "outcome"), ({"found": "haiku"}, "found"),
    ({"box": [1, 2, 3]}, "box"), ({"poly": [1, 2, 3]}, "outline"), ({"score": float("nan")}, "finite"),
    ({"track": "obj:0", "rejected": "obj:1", "sample": 0}, "both"), ({"sample": 2}, "sample"),
    ({"track": "obj:0"}, "sample")])
def test_a_bad_row_writes_nothing(tmp_path, change, message):
    bad = rows()
    bad[3].update(change)
    with pytest.raises(ValueError, match=message):
        write(tmp_path / "m.parquet", bad, META)
    assert not (tmp_path / "m.parquet").exists()
    assert object_raw.ROW_GROUP_ROWS > 0
