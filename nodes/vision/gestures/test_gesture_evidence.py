"""The evidence side file (gesture_evidence.py): the columns the bundle schema documents, the same bytes for the same
rows, and read back it gives the layer's events again."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import gesture_rules as rules
from gesture_evidence import DESCRIPTION, FILE, ROW_GROUP_ROWS, THRESHOLDS, Writer, read_rows, stored
from gesture_rules import events_from_evidence, sample_evidence
from test_gesture_rules import OPEN, POINTING, TIMES, hand_with_wrist, scene

SCHEMA = Path(__file__).resolve().parents[3] / "schema" / "bundle-0.1.schema.json"


def pq():
    return pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/core); CI installs it")


def columns():
    return [(c["name"], c["arrow"]) for c in json.loads(SCHEMA.read_text())["$defs"]["parquet"]["gestures_evidence"]["columns"]]


def sample_rows():
    """Rows of every outcome: over the mouth, pointing at face:1, raised to the head (touching the face, above the nose),
    and a hand with too few keypoints."""
    out, hands = [], []
    for k, points in enumerate([np.tile([500, 450], (21, 1)), POINTING * 2 + [1000, 310], hand_with_wrist(400, 400), OPEN]):
        hs, ctx = scene(points)
        hand = {**hs[0], "id": f"hand:{k}"}
        hand["samples"] = [{**s, "kpts": [[x, y, 0.1 if k == 3 else c] for x, y, c in s["kpts"]]} for s in hand["samples"]]
        hands.append(hand)
        out += [sample_evidence(hand, j, ctx) for j in range(len(TIMES))]
    return hands, out


def write(path, rows):
    with Writer(path) as w:
        for r in rows:
            w.add(r)
    return w.rows


def test_the_file_has_the_columns_the_bundle_schema_documents_and_names_its_thresholds(tmp_path):
    table = (pq(), write(tmp_path / "e.parquet", sample_rows()[1]))[0].read_table(tmp_path / "e.parquet")
    assert [(f.name, str(f.type)) for f in table.schema] == columns()
    assert all(not f.nullable for f in table.schema) and table.num_rows == 4 * len(TIMES)
    meta = table.schema.metadata
    assert meta[b"hand_model"] == b"RTMPose-m Hand5" and meta[b"side"] == b"gestures_evidence"
    assert {k.decode() for k in meta} >= {name.lower() for name in THRESHOLDS}
    assert meta[b"mouth_radius"] == repr(rules.MOUTH_RADIUS).encode()
    assert FILE == "layers/gestures_evidence.parquet" and DESCRIPTION


def test_read_back_the_rows_give_the_layers_events(tmp_path):
    hands, rows = sample_rows()
    write(tmp_path / "e.parquet", rows)
    events = events_from_evidence(hands, rows)
    assert {e["kind"] for e in events} == set(rules.KINDS)   # the raised hand is on the forehead too: raised above the nose
    pointing = next(e for e in events if e["kind"] == "pointing")
    assert pointing["detail"]["target"] == "face:1"
    back = read_rows(tmp_path / "e.parquet")
    assert events_from_evidence(hands, back) == events
    assert [(r["hand"], r["sample"], r["t"]) for r in back] == [(r["hand"], r["sample"], r["t"]) for r in rows]
    statuses = {(r["hand"], r["face_status"], r["point_status"], r["raise_status"]) for r in back}
    assert ("hand:3", "few_keypoints", "few_keypoints", "few_keypoints") in statuses


def test_the_same_rows_are_the_same_bytes(tmp_path):
    p = pq()
    _, rows = sample_rows()
    many = [{**r, "hand": f"hand:{k}"} for k in range(ROW_GROUP_ROWS // len(rows) + 2) for r in rows]
    first, second = tmp_path / "1.parquet", tmp_path / "2.parquet"
    assert write(first, many) == write(second, many) == len(many)
    assert first.read_bytes() == second.read_bytes()
    meta = p.ParquetFile(first).metadata
    assert meta.num_rows == len(many) and meta.num_row_groups == 2 and meta.row_group(0).num_rows == ROW_GROUP_ROWS
    assert meta.row_group(0).column(0).compression == "ZSTD"


def test_a_measurement_that_is_not_finite_is_refused():
    _, rows = sample_rows()
    with pytest.raises(ValueError, match="not finite"):
        stored({**rows[0], "rise": math.inf})
    with pytest.raises(ValueError, match="not finite"):
        stored({**rows[0], "mouth_dist": [math.nan] * 21})
    assert stored(rows[0])["target"] == "" and stored(rows[0])["direction"] == [0.0, 0.0]
