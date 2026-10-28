"""The hands layer's raw side file, what it drops of the hand detector's output (schema: #/$defs/parquet/hands_detections):

layers/hands_detections.parquet  every detection RTMDet-nano hand returned on every sampled frame, a row per detection,
                                 in time order then the detector's (highest score first): its score and box, why the
                                 node did or did not make it a hand (hand_tracks.REASONS), the wrist the matching gave
                                 it (body, side), and the layer sample it became (hand, sample).

With the bodies layer, the shots layer and the frame size (the file's metadata), these rows give the layer's hands
again through the node's own code: replay() runs hand_tracks' gates, matching and tracking on them and gives the
layer's hands, keypoints aside (RTMPose-hand runs only on the hands kept, and the layer keeps every hand it posed, so
there are no other keypoints to keep); check() does so on the written file, which the node runs before it finishes.
The box and score are the detector's float32 values unrounded, so a revisit sees what the node saw.

Parquet, not JSON: RTMDet's NMS returns boxes down to a score of 0.05, about 36 a frame on OOO clip 1, of which 1 or 2
score at least the layer's 0.25: 32,222 rows and 0.58 MB for its 119 s, about 18 MB an hour. While the video runs, the
detections under the threshold are kept as float32 arrays (Detections: 20 bytes each, about 20 MB for an hour at 8
frames a second, where frame_hands' entries take about 330 bytes), the few others as frame_hands' entries until
tracking names their samples.

Deterministic: identical rows give identical bytes. Every writer option that shapes the file is fixed below, and the
file carries no time; the pyarrow version (envs/vision's lock, part of the node's cache key) names itself in the
footer. A detection's sample is known only once tracking ends (a hand's id is its track's place in the layer), so the
file is written whole, in row groups of ROW_GROUP_ROWS.
"""

from __future__ import annotations

import numpy as np

from hand_tracks import (DEDUP_IOU, DETECTOR_SCORE, MAX_DETECTOR_BOX, MIN_HAND_PX, REASONS, WRIST_GATE, body_indices,
                         bodies_by_time, frame_hands, layer_hands, ordered, propose, track_hands)

ROW_GROUP_ROWS = 65536
# OOO clip 1: zstd 3 583.6 KB, 9 583.0 KB, 19 582.0 KB (the scores and boxes are most of it; byte-stream-split on them
# saves 2-3%)
ZSTD_LEVEL = 9
FILE = "layers/hands_detections.parquet"
DESCRIPTION = ("Every RTMDet-nano hand detection on every sampled frame, with its score and box, why it is or is not a "
               "hand of the layer, the wrist it matched and the layer sample it became")
COLUMNS = ("t", "score", "box", "reason", "body", "side", "hand", "sample")
NONE = ("", "", "", -1)   # body, side, hand, sample of a row with no wrist and no sample


def _float32(values, what: str) -> np.ndarray:
    """values as float32, which they must be exactly (the detector's are), and finite: else a ValueError."""
    wide = np.asarray(values, dtype=np.float64)
    narrow = wide.astype(np.float32)
    if not np.all(np.isfinite(wide)) or not np.array_equal(narrow.astype(np.float64), wide):
        raise ValueError(f"{what} holds a value that is not a finite float32 (the detector's values are)")
    return narrow


class Detections:
    """Every frame's detections as frame_hands records them, until tracking has named the layer's samples: the ones
    under DETECTOR_SCORE as float32 arrays, the rest as the record's entries (which hold the hand they became)."""

    def __init__(self):
        self.frames: list[tuple[float, list[dict], np.ndarray, np.ndarray]] = []   # (t, entries, scores, boxes)

    def add(self, t: float, record: list[dict]):
        """One frame's record, in the detector's order (highest score first, so the ones under the threshold last).
        Another order, or a score or box that is not a finite float32, is a ValueError."""
        entries = [d for d in record if d["reason"] != "below_score"]
        below = record[len(entries):]
        if any(d["reason"] != "below_score" for d in below):
            raise ValueError(f"the detections at t={t} are not highest score first")
        for d in entries:
            _float32([d["score"], *d["box"]], f"a detection at t={t}")
        scores = _float32([d["score"] for d in below], f"a score at t={t}")
        boxes = _float32(np.reshape([d["box"] for d in below], (-1, 4)), f"a box at t={t}")
        self.frames.append((t, entries, scores, boxes))

    def columns(self, tracks: list[dict]) -> dict:
        """The side file's columns: tracks are track_hands' (ordered() gives the layer's hand ids)."""
        where = {id(s["hand"]): (f"hand:{i}", j) for i, tr in enumerate(ordered(tracks)) for j, s in enumerate(tr["samples"])}
        t, score, box, labels = [np.empty(0)], [np.empty(0, np.float32)], [np.empty((0, 4), np.float32)], []
        below = ("below_score", *NONE)
        for ft, entries, scores, boxes in self.frames:
            for d in entries:
                body, side = d["wrist"] or ("", "")
                hand, sample = where[id(d["hand"])] if d["hand"] is not None else ("", -1)
                labels.append((d["reason"], body, side, hand, sample))
            labels += [below] * len(scores)
            t.append(np.full(len(entries) + len(scores), ft))
            score += [np.array([d["score"] for d in entries], np.float32), scores]
            box += [np.reshape([d["box"] for d in entries], (-1, 4)).astype(np.float32), boxes]
        reason, body, side, hand, sample = (list(c) for c in zip(*labels)) if labels else ([], [], [], [], [])
        return {"t": np.concatenate(t), "score": np.concatenate(score), "box": np.concatenate(box), "reason": reason,
                "body": body, "side": side, "hand": hand, "sample": np.asarray(sample, np.int32)}


def columns(rows: list[tuple]) -> dict:
    """The side file's columns from rows (t, score, box, reason, body, side, hand, sample), as to_rows gives them."""
    t, score, box, reason, body, side, hand, sample = (list(c) for c in zip(*rows)) if rows else ([] for _ in COLUMNS)
    return {"t": np.asarray(t, np.float64), "score": _float32(score, "a score"),
            "box": _float32(np.reshape(box, (-1, 4)), "a box"), "reason": reason, "body": body, "side": side,
            "hand": hand, "sample": np.asarray(sample, np.int32)}


def to_rows(cols: dict) -> list[tuple]:
    """columns' inverse: a row per detection, scores and box values as Python floats (exactly the float32 values)."""
    return list(zip(cols["t"].tolist(), cols["score"].astype(np.float64).tolist(), cols["box"].astype(np.float64).tolist(),
                    cols["reason"], cols["body"], cols["side"], cols["hand"], cols["sample"].tolist()))


def _schema(metadata: dict):
    import pyarrow as pa
    return pa.schema([pa.field("t", pa.float64(), nullable=False), pa.field("score", pa.float32(), nullable=False),
                      pa.field("box", pa.list_(pa.field("element", pa.float32(), nullable=False), 4), nullable=False),
                      pa.field("reason", pa.string(), nullable=False), pa.field("body", pa.string(), nullable=False),
                      pa.field("side", pa.string(), nullable=False), pa.field("hand", pa.string(), nullable=False),
                      pa.field("sample", pa.int32(), nullable=False)],
                     metadata={k: v if isinstance(v, str) else repr(v) for k, v in metadata.items()})


def metadata(detector: str, width: int, height: int) -> dict:
    """The file's metadata: the detector, the frame size the boxes are in, and the gates the reasons apply."""
    return {"side": "hands_detections", "detector": detector, "frame_width": width, "frame_height": height,
            "detector_threshold": DETECTOR_SCORE, "min_hand_px": MIN_HAND_PX, "max_detector_box": MAX_DETECTOR_BOX,
            "dedup_iou": DEDUP_IOU, "wrist_gate": WRIST_GATE, "reasons": " ".join(REASONS)}


def write(path, cols: dict, detector: str, width: int, height: int):
    """cols: Detections.columns' (or columns'). A score or box that is not a finite float32, columns of different
    lengths or an unknown reason is a ValueError, and nothing is written."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = len(cols["t"])
    if cols["box"].shape != (n, 4) or any(len(cols[c]) != n for c in COLUMNS):
        raise ValueError(f"the columns are not {n} rows each (box {cols['box'].shape})")
    _float32(cols["score"], "the scores")
    _float32(cols["box"], "the boxes")
    unknown = set(cols["reason"]) - set(REASONS)
    if unknown:
        raise ValueError(f"reasons {sorted(unknown)} are not among {REASONS}")
    schema = _schema(metadata(detector, width, height))
    box = pa.FixedSizeListArray.from_arrays(pa.array(np.ascontiguousarray(cols["box"], np.float32).ravel()),
                                            type=schema.field("box").type)
    table = pa.table([pa.array(np.asarray(cols["t"], np.float64)), pa.array(np.asarray(cols["score"], np.float32)), box,
                      *(pa.array(cols[c], pa.string()) for c in ("reason", "body", "side", "hand")),
                      pa.array(np.asarray(cols["sample"], np.int32))], schema=schema)
    with pq.ParquetWriter(path, schema, compression="zstd", compression_level=ZSTD_LEVEL,
                          use_dictionary=["reason", "body", "side", "hand"], write_statistics=["t", "hand", "sample"],
                          version="2.6", data_page_version="1.0", store_schema=True, write_page_index=False,
                          use_byte_stream_split=False) as writer:
        writer.write_table(table, row_group_size=ROW_GROUP_ROWS)


def read(path, min_score: float | None = None) -> tuple[list[tuple], dict]:
    """(rows as to_rows gives them, the metadata as text) from a written file; with min_score, only the rows scoring at
    least that."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    if min_score is not None:
        table = table.filter(pc.greater_equal(table["score"], np.float32(min_score)))
    return list(zip(*(table[c].to_pylist() for c in COLUMNS))), meta


class _Stored:
    """The detector's detections, as stored: frame_hands' `detect` gives a frame's rows, `keypoints` none."""

    def __init__(self, rows):
        self.rows = rows

    def detect(self, frame):
        return [(box, score) for _, score, box, *_ in self.rows]

    def keypoints(self, frame, box):
        return np.zeros((21, 3))


def replay(rows: list[tuple], bodies_layer: dict, shots: list[dict], width: int, height: int) -> tuple[list[dict], list[tuple]]:
    """The hands layer's hands (keypoints zero) and the rows (reason, wrist, sample) that hand_tracks makes of stored
    detections: frame_hands on each frame that has a row or a body, as the node runs it. Frames with neither have no
    hand. width and height: the frames' (the file's frame_width and frame_height)."""
    idx = body_indices(bodies_layer["keypoint_names"])
    bodies = bodies_by_time(bodies_layer["tracks"])
    by_t: dict[float, list[tuple]] = {}
    for row in rows:
        by_t.setdefault(row[0], []).append(row)
    frame = np.broadcast_to(np.uint8(0), (height, width, 3))   # frame_hands reads its height; no pixels needed
    stats = dict.fromkeys(("detector_below_score", "wrist_below_gate", "detector_below_gate", "detector_too_large",
                           "detector_duplicate", "detector_linked", "detector_unlinked"), 0)
    frames, records = [], []
    for t in sorted(set(by_t) | set(bodies)):
        shot = next((s for s in shots if s["start"] <= t < s["end"]), None)
        if shot is None:
            raise ValueError(f"t={t} is in no shot: these rows are not this video's")
        posed = [propose(b, idx, bodies_layer["score_threshold"], width, height) for b in bodies.get(t, [])]
        record: list[dict] = []
        frames.append((t, f"shot:{shot['index']}", frame_hands(frame, posed, _Stored(by_t.get(t, [])), stats, record)))
        records.append((t, record))
    tracks = track_hands(frames)
    detections = Detections()
    for t, record in records:
        detections.add(t, record)
    return layer_hands(tracks), to_rows(detections.columns(tracks))


def without_kpts(hands: list[dict]) -> list[dict]:
    return [{**h, "samples": [{k: v for k, v in s.items() if k != "kpts"} for s in h["samples"]]} for h in hands]


def check(path, hands: list[dict], bodies_layer: dict, shots: list[dict]) -> str | None:
    """None if the written file gives the layer's `hands` (keypoints aside) and its own rows again through replay();
    else what differs. The rows under the file's detector_threshold take no part in the gates or the matching, so only
    the others are replayed, and these must all be below_score rows with no wrist and no sample."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    threshold = np.float32(float(meta["detector_threshold"]))
    t = table["t"].to_numpy()
    if np.any(np.diff(t) < 0):
        return "its rows are not in time order"
    below = table.filter(pc.less(table["score"], threshold))
    other = pc.sum(pc.or_(pc.not_equal(below["reason"], "below_score"), pc.or_(
        pc.or_(pc.not_equal(below["body"], ""), pc.not_equal(below["side"], "")),
        pc.or_(pc.not_equal(below["hand"], ""), pc.not_equal(below["sample"], -1))))).as_py() or 0
    if other:
        return f"{other} rows under the threshold are not below_score rows with no wrist and no sample"
    kept, _ = read(path, min_score=float(threshold))
    if any(r[3] == "below_score" for r in kept):
        return "a row at or over the threshold is a below_score row"
    again, kept_again = replay(kept, bodies_layer, shots, int(meta["frame_width"]), int(meta["frame_height"]))
    if kept_again != kept:
        first = next((i for i, (a, b) in enumerate(zip(kept, kept_again)) if a != b), min(len(kept), len(kept_again)))
        return f"its rows at or over the threshold do not give themselves again: row {first} of {len(kept)}"
    if without_kpts(again) != without_kpts(hands):
        return f"its rows give {len(again)} hands, not the layer's {len(hands)} (or not the same)"
    return None
