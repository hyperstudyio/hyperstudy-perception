"""The objects layer's raw side file (schema: #/$defs/parquet/objects_masks):

layers/objects_masks.parquet  every mask SAM 3 gave the node, a row each: every sample of every track of the shots'
                              sessions (the vocabulary's names), and every detection scoring at least
                              garments.STORE_SCORE in every body crop (the fixed nouns and "person"), with its scores,
                              box and outline, what the node made of it (reason) and the layer sample it became.

So the thresholds can be revisited: SAM 3's detection and tracker scores (the layer keeps only the tracker's, rounded),
the masks dropped for holding people with their person and enclosed shares (person_masks.py), the outline before a
trim, the body detections no item took and why (garments.py), and the tracks too short to keep. Scores are SAM 3's
float32 values, unrounded; boxes and outlines are whole frame pixels, as the layer's.

Rows are in the order the node made them: shot by shot, the body crops' detections (by time, then body, then prompt, then
SAM 3's query order), then the session's masks (by frame, then SAM 3's order). Deterministic: identical rows give
identical bytes. Every writer option that shapes the file is fixed below, and the file carries no time; the pyarrow
version (envs/gaze's lock, part of the node's cache key) names itself in the footer.
"""

from __future__ import annotations

import numpy as np

FILE = "layers/objects_masks.parquet"
DESCRIPTION = ("Every mask SAM 3 gave: each sample of each session track and each body-crop detection, with its scores, "
               "box and outline, what became of it (kept, trimmed, dropped for holding people, not this body's...) and "
               "the layer sample it became")
ROW_GROUP_ROWS = 65536
ZSTD_LEVEL = 9
FOUND = ("vocabulary", "body")
# vocabulary rows: person_masks.REASONS and worn (on a person by nature: not checked); body rows: item (part of one of
# the body's items), person (the body's person mask), below_keep (under garments.KEEP_SCORE), not_this_body (its centroid
# is in another body's region for its slot, or in none), not_picked (this body's, but its slot took another)
REASONS = ("kept", "trimmed", "covers_person", "encloses_person", "worn",
           "item", "person", "below_keep", "not_this_body", "not_picked")
# what became of the row's track (vocabulary) or item (body): in the layer's tracks, in its rejected_tracks (the crop
# check), too_short (fewer than the node's minimum samples were kept), or none (a body detection that is no item)
OUTCOMES = ("tracks", "rejected", "too_short", "none")
COLUMNS = ("t", "shot", "found", "label", "sam", "body", "score", "tracker_score", "box", "poly", "person_share",
           "enclosed_share", "reason", "outcome", "track", "rejected", "sample")


def row(t, shot, found, label, score, box, poly, reason, sam=-1, body="", tracker_score=-1.0, person_share=-1.0,
        enclosed_share=-1.0) -> dict:
    """One row before the layer is made; the node fills in outcome, track or rejected, and sample."""
    return {"t": t, "shot": shot, "found": found, "label": label, "sam": sam, "body": body, "score": score,
            "tracker_score": tracker_score, "box": list(box), "poly": [v for p in poly for v in p],
            "person_share": person_share, "enclosed_share": enclosed_share, "reason": reason,
            "outcome": "none", "track": "", "rejected": "", "sample": -1}


def _float32(values, what: str) -> np.ndarray:
    wide = np.asarray(values, dtype=np.float64)
    narrow = wide.astype(np.float32)
    if not np.all(np.isfinite(wide)):
        raise ValueError(f"{what} holds a value that is not finite")
    return narrow


def _schema(metadata: dict):
    import pyarrow as pa
    ints = pa.field("element", pa.int32(), nullable=False)
    fields = [pa.field("t", pa.float64(), nullable=False), pa.field("shot", pa.string(), nullable=False),
              pa.field("found", pa.string(), nullable=False), pa.field("label", pa.string(), nullable=False),
              pa.field("sam", pa.int32(), nullable=False), pa.field("body", pa.string(), nullable=False),
              pa.field("score", pa.float32(), nullable=False), pa.field("tracker_score", pa.float32(), nullable=False),
              pa.field("box", pa.list_(ints, 4), nullable=False), pa.field("poly", pa.list_(ints), nullable=False),
              pa.field("person_share", pa.float32(), nullable=False),
              pa.field("enclosed_share", pa.float32(), nullable=False), pa.field("reason", pa.string(), nullable=False),
              pa.field("outcome", pa.string(), nullable=False), pa.field("track", pa.string(), nullable=False),
              pa.field("rejected", pa.string(), nullable=False), pa.field("sample", pa.int32(), nullable=False)]
    return pa.schema(fields, metadata={k: v if isinstance(v, str) else repr(v) for k, v in metadata.items()})


def write(path, rows: list[dict], metadata: dict):
    """rows: row()'s, finished. An unknown found, reason or outcome, a score that is not finite, a box that is not four
    values, an outline of an odd number of values, or a row naming both a track and a rejected track, or a sample
    without a track, is a ValueError, and nothing is written."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    for name, allowed in (("found", FOUND), ("reason", REASONS), ("outcome", OUTCOMES)):
        unknown = {r[name] for r in rows} - set(allowed)
        if unknown:
            raise ValueError(f"{name} {sorted(unknown)} not among {allowed}")
    if any(len(r["box"]) != 4 or len(r["poly"]) % 2 for r in rows):
        raise ValueError("a box is not 4 values, or an outline is not x, y pairs")
    if any((r["track"] and r["rejected"]) or ((r["sample"] >= 0) != bool(r["track"])) for r in rows):
        raise ValueError("a row names both a track and a rejected track, or a sample without its track (or the reverse)")
    schema = _schema({"side": "objects_masks", **metadata})
    floats = {c: _float32([r[c] for r in rows], c) for c in ("score", "tracker_score", "person_share", "enclosed_share")}
    box = pa.FixedSizeListArray.from_arrays(pa.array(np.asarray([v for r in rows for v in r["box"]], np.int32)),
                                            type=schema.field("box").type)
    offsets = np.concatenate([[0], np.cumsum([len(r["poly"]) for r in rows])]).astype(np.int32)
    poly = pa.ListArray.from_arrays(pa.array(offsets), pa.array(np.asarray([v for r in rows for v in r["poly"]], np.int32)),
                                    type=schema.field("poly").type)
    table = pa.table([pa.array(np.asarray([r["t"] for r in rows], np.float64)),
                      *(pa.array([r[c] for r in rows], pa.string()) for c in ("shot", "found", "label")),
                      pa.array(np.asarray([r["sam"] for r in rows], np.int32)), pa.array([r["body"] for r in rows], pa.string()),
                      pa.array(floats["score"]), pa.array(floats["tracker_score"]), box, poly,
                      pa.array(floats["person_share"]), pa.array(floats["enclosed_share"]),
                      *(pa.array([r[c] for r in rows], pa.string()) for c in ("reason", "outcome", "track", "rejected")),
                      pa.array(np.asarray([r["sample"] for r in rows], np.int32))], schema=schema)
    with pq.ParquetWriter(path, schema, compression="zstd", compression_level=ZSTD_LEVEL,
                          use_dictionary=["shot", "found", "label", "body", "reason", "outcome", "track", "rejected"],
                          write_statistics=["t", "shot", "track", "sample"], version="2.6", data_page_version="1.0",
                          store_schema=True, write_page_index=False, use_byte_stream_split=False) as writer:
        writer.write_table(table, row_group_size=ROW_GROUP_ROWS)


def read(path) -> tuple[list[dict], dict]:
    """(rows as row() gives them, finished; the metadata as text) from a written file."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    cols = {c: table[c].to_pylist() for c in COLUMNS}
    return [dict(zip(COLUMNS, values)) for values in zip(*(cols[c] for c in COLUMNS))], meta
