"""The detections side file (hand_detections.py): a row per detection naming the layer sample it became, columns as
the bundle schema documents, the same bytes for the same rows, and the layer's hands again from the stored rows."""

import json
from pathlib import Path

import numpy as np
import pytest

from hand_detections import (DESCRIPTION, FILE, ROW_GROUP_ROWS, Detections, check, columns, read, replay, to_rows,
                             without_kpts, write)
from hand_tracks import REASONS, body_indices, bodies_by_time, frame_hands, layer_hands, propose, track_hands

HERE = Path(__file__).resolve().parent
SCHEMA = HERE.parents[2] / "schema" / "bundle-0.1.schema.json"
NAMES = json.loads((HERE.parent / "body-pose" / "keypoints.json").read_text())["names"]
IDX = body_indices(NAMES)
W, H, THRESHOLD = 1920, 1080, 3.0


def kpts(wrists=(), hands=()):
    """133 keypoints as the bodies layer has them: invisible but the given wrists and hands, {side: xy} and
    {side: [xy, ...]}."""
    k = np.zeros((133, 3))
    for side, xy in dict(wrists).items():
        k[IDX[side]["wrist"]] = [*xy, 5.0]
    for side, points in dict(hands).items():
        for i, xy in zip(IDX[side]["hand"], points):
            k[i] = [*xy, 5.0]
    return k.tolist()


def hand_at(x, y, r=40):
    return [(x, y), (x - r, y - r), (x + r, y - r), (x - r, y + r), (x + r, y + r)]


# Two bodies over four frames of one shot and one frame of the next. body:0 has its left hand posed and a free right
# wrist, body:1 only a right wrist. Box values are float32 numbers, as the detector's are.
BODIES = {"keypoint_names": NAMES, "score_threshold": THRESHOLD, "fps": 8, "tracks": [
    {"id": "body:0", "face": "face:0", "samples": [
        {"t": t, "kpts": kpts({"left": (500, 600), "right": (900, 600)}, {"left": hand_at(500, 500)})}
        for t in (0.0, 0.125, 0.25, 0.375)]},
    {"id": "body:1", "face": None, "samples": [
        {"t": t, "kpts": kpts({"right": (1300, 300)})} for t in (0.0, 0.125, 1.0)]}]}
SHOTS = [{"index": 0, "start": 0.0, "end": 0.9}, {"index": 1, "start": 0.9, "end": 2.0}]
DETECTIONS = {   # t: [(box, score)], highest score first, as models.Hands.detect gives them
    0.0: [([420.5, 470.5, 168.0, 168.0], 0.875), ([880.0, 580.0, 60.0, 60.0], 0.5), ([1500.0, 800.0, 60.0, 60.0], 0.375),
          ([1500.0, 200.0, 20.0, 60.0], 0.3125), ([1283.25, 280.0, 60.0, 60.0], 0.125)],
    0.125: [([878.0, 582.0, 60.0, 60.0], 0.625), ([882.0, 584.0, 64.0, 64.0], 0.25), ([100.0, 100.0, 600.0, 600.0], 0.5)],
    0.25: [([1504.0, 802.0, 60.0, 60.0], 0.4375)],
    0.375: [([470.0, 580.0, 60.0, 60.0], 0.75), ([1280.0, 280.0, 60.0, 60.0], 0.0625)],
    1.0: [([1280.0, 270.0, 50.0, 50.0], 0.28125)],
}


class Model:
    """The detector's fixed detections per frame (set before each frame); keypoints from the box."""
    t = None

    def detect(self, frame):
        return list(DETECTIONS.get(self.t, []))

    def keypoints(self, frame, box):
        return np.tile([box[0] + 1.0, box[1] + 2.0, 0.5], (21, 1))


def run_node():
    """What node.py does with these frames: (the layer's hands, the side file's rows)."""
    bodies, model, stats = bodies_by_time(BODIES["tracks"]), Model(), {k: 0 for k in (
        "wrist_below_gate", "detector_below_score", "detector_below_gate", "detector_too_large", "detector_duplicate",
        "detector_linked", "detector_unlinked")}
    frame = np.zeros((H, W, 3), np.uint8)
    frames, detections = [], Detections()
    for t in (0.0, 0.125, 0.25, 0.375, 0.5, 1.0):   # the sampling grid: 0.5 has neither a body nor a detection
        model.t = t
        shot = "shot:0" if t < 0.9 else "shot:1"
        record = []
        posed = [propose(b, IDX, THRESHOLD, W, H) for b in bodies.get(t, [])]
        frames.append((t, shot, frame_hands(frame, posed, model, stats, record)))
        detections.add(t, record)
    tracks = track_hands(frames)
    return layer_hands(tracks), to_rows(detections.columns(tracks))


def test_a_row_per_detection_naming_its_reason_wrist_and_sample():
    hands, rows = run_node()
    assert [(h["id"], h["box_source"], h["side"], h["body"], [s["t"] for s in h["samples"]]) for h in hands] == [
        ("hand:0", "wrist", "left", "body:0", [0.0, 0.125, 0.25, 0.375]),
        ("hand:1", "detector", "right", "body:0", [0.0, 0.125]),
        ("hand:2", "detector", None, None, [0.0, 0.25]),
        ("hand:3", "detector", "right", "body:1", [1.0])]
    assert [(t, score, reason, body, side, hand, sample) for t, score, _, reason, body, side, hand, sample in rows] == [
        (0.0, 0.875, "duplicate", "", "", "", -1),                 # the wrist hand again
        (0.0, 0.5, "linked", "body:0", "right", "hand:1", 0),
        (0.0, 0.375, "unlinked", "", "", "hand:2", 0),
        (0.0, 0.3125, "below_gate", "", "", "", -1),
        (0.0, 0.125, "below_score", "", "", "", -1),               # on body:1's wrist, but under the threshold
        (0.125, 0.625, "linked", "body:0", "right", "hand:1", 1),
        (0.125, 0.25, "duplicate", "", "", "", -1),                # a second box on that hand (at the threshold: kept)
        (0.125, 0.5, "too_large", "", "", "", -1),
        (0.25, 0.4375, "unlinked", "", "", "hand:2", 1),
        (0.375, 0.75, "wrist_taken", "body:0", "left", "", -1),    # on the wrist whose hand is posed
        (0.375, 0.0625, "below_score", "", "", "", -1),
        (1.0, 0.28125, "linked", "body:1", "right", "hand:3", 0)]
    assert [row[2] for row in rows] == [box for t in DETECTIONS for box, _ in DETECTIONS[t]]
    assert to_rows(columns(rows)) == rows
    # every detector sample of the layer is named by one row, with the box the layer rounds
    named = {(row[6], row[7]): row[2] for row in rows if row[6]}
    detector = {(h["id"], j): s["box"] for h in hands if h["box_source"] == "detector" for j, s in enumerate(h["samples"])}
    assert detector == {k: [round(v, 1) for v in box] for k, box in named.items()}


def test_the_stored_rows_give_the_layers_hands_through_the_nodes_own_matching(tmp_path):
    hands, rows = run_node()
    again, rows_again = replay(rows, BODIES, SHOTS, W, H)
    assert without_kpts(again) == without_kpts(hands) and rows_again == rows
    pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/vision); CI installs it")
    write(tmp_path / "d.parquet", columns(rows), "test-rtmdet", W, H)
    stored, meta = read(tmp_path / "d.parquet")
    assert stored == rows and (meta["frame_width"], meta["frame_height"]) == ("1920", "1080")
    assert without_kpts(replay(stored, BODIES, SHOTS, int(meta["frame_width"]), int(meta["frame_height"]))[0]) == without_kpts(hands)


@pytest.mark.parametrize("tamper, problem", [
    (lambda rows: rows, None),
    # a detection under the threshold raised above it, still said to be under it
    (lambda rows: [r if i != 4 else (r[0], 0.5, *r[2:]) for i, r in enumerate(rows)], "is a below_score row"),
    # ... and called unlinked: replayed, it is a hand the layer does not have
    (lambda rows: [r if i != 4 else (r[0], 0.5, r[2], "unlinked", *r[4:]) for i, r in enumerate(rows)],
     "do not give themselves"),
    # a detection under the threshold said to be a hand's sample
    (lambda rows: [r if i != 4 else (*r[:6], "hand:2", 0) for i, r in enumerate(rows)], "not below_score rows"),
    (lambda rows: rows[5:] + rows[:5], "not in time order"),
    # a row's sample renamed: the matching still gives the layer, but not this row
    (lambda rows: [r if i != 2 else (*r[:7], 5) for i, r in enumerate(rows)], "do not give themselves"),
    # a kept detection lost: the layer has a hand sample no row gives
    (lambda rows: [r for i, r in enumerate(rows) if i != 8], "hands, not the layer's"),
])
def test_check_says_whether_the_written_file_gives_the_layer(tmp_path, tamper, problem):
    pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/vision); CI installs it")
    hands, rows = run_node()
    write(tmp_path / "d.parquet", columns(tamper(rows)), "test-rtmdet", W, H)
    result = check(tmp_path / "d.parquet", hands, BODIES, SHOTS)
    assert result is None if problem is None else problem in result, result


def test_replay_refuses_rows_outside_the_shots():
    _, rows = run_node()
    with pytest.raises(ValueError, match="in no shot"):
        replay(rows, BODIES, SHOTS[:1], W, H)


def test_the_file_has_the_columns_the_bundle_schema_documents(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/vision); CI installs it")
    _, rows = run_node()
    write(tmp_path / "d.parquet", columns(rows), "test-rtmdet", W, H)
    table = pq.read_table(tmp_path / "d.parquet")
    spec = json.loads(SCHEMA.read_text())["$defs"]["parquet"]["hands_detections"]
    assert [(f.name, str(f.type)) for f in table.schema] == [(c["name"], c["arrow"]) for c in spec["columns"]]
    assert all(not f.nullable for f in table.schema)
    meta = table.schema.metadata
    assert (meta[b"side"], meta[b"detector"], meta[b"detector_threshold"], meta[b"min_hand_px"]) == (
        b"hands_detections", b"test-rtmdet", b"0.25", b"32")
    assert meta[b"reasons"].decode().split() == list(REASONS)
    assert FILE == "layers/hands_detections.parquet" and DESCRIPTION


def test_no_detections_is_an_empty_file_with_the_columns(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/vision); CI installs it")
    write(tmp_path / "d.parquet", columns([]), "test-rtmdet", W, H)
    table = pq.read_table(tmp_path / "d.parquet")
    assert table.num_rows == 0 and table.schema.names == ["t", "score", "box", "reason", "body", "side", "hand", "sample"]


def test_the_same_rows_are_the_same_bytes(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/vision); CI installs it")
    rows = [(round(k // 38 / 8, 3), float(np.float32(0.05 + (k % 90) / 100)), [float(k % 1900), 10.5, 60.0, 61.25],
             REASONS[k % 7], "body:1" if k % 7 in (4, 5) else "", "left" if k % 7 in (4, 5) else "",
             f"hand:{k % 5}" if k % 7 in (5, 6) else "", k // 7 if k % 7 in (5, 6) else -1) for k in range(ROW_GROUP_ROWS + 9)]
    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write(first, columns(rows), "test-rtmdet", W, H)
    write(second, columns(rows), "test-rtmdet", W, H)
    assert first.read_bytes() == second.read_bytes()
    meta = pq.ParquetFile(first).metadata
    assert meta.num_rows == len(rows) and meta.num_row_groups == 2 and meta.row_group(0).column(0).compression == "ZSTD"


GOOD = (0.0, 0.5, [1.0, 2.0, 60.0, 60.0], "linked", "body:0", "left", "hand:0", 0)


@pytest.mark.parametrize("bad", [
    (0.0, float("nan"), *GOOD[2:]),
    (0.0, 0.1, *GOOD[2:]),                                   # 0.1 is no float32: the detector's scores are
    (0.0, 0.5, [1.0, 2.0, 60.0], *GOOD[3:]),
    (0.0, 0.5, [1.0, 2.0, 60.0, float("inf")], *GOOD[3:]),
    (*GOOD[:3], "kept", *GOOD[4:]),
])
def test_a_value_that_is_not_the_detectors_or_an_unknown_reason_is_refused(tmp_path, bad):
    pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/vision); CI installs it")
    with pytest.raises(ValueError):
        write(tmp_path / "x.parquet", columns([GOOD, bad]), "test-rtmdet", W, H)
    assert not (tmp_path / "x.parquet").exists()


def test_detections_keep_the_detectors_order_and_refuse_another():
    """The rows under the threshold are kept apart as float32 arrays and written after the others, which is the
    detector's order only if it returns the highest score first (models.Hands.detect does): another order is refused,
    as is a value that is not a float32."""
    entry = lambda score, reason: {"box": [1.0, 2.0, 60.0, 60.0], "score": score, "reason": reason, "wrist": None, "hand": None}
    d = Detections()
    d.add(0.0, [entry(0.5, "unlinked"), entry(0.125, "below_score")])
    with pytest.raises(ValueError, match="highest score first"):
        d.add(0.125, [entry(0.125, "below_score"), entry(0.5, "unlinked")])
    with pytest.raises(ValueError, match="float32"):
        d.add(0.125, [entry(0.1, "below_score")])
    assert [(r[1], r[3], r[6]) for r in to_rows(d.columns([]))] == [(0.5, "unlinked", ""), (0.125, "below_score", "")]
