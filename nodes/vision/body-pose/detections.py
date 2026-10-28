"""The bodies layer's raw side files, what it drops of the models' output (schema: #/$defs/parquet/<name>):

layers/bodies_scores.parquet      YOLOX-m's detection score for every sample of the layer's tracks, a row per sample
                                  (body, sample, t, score), in time order, the bodies of one time in track order.
layers/bodies_duplicates.parquet  every detection dropped as a second detection of a person (bodies.drop_duplicate_bodies),
                                  in time order then by score: the layer sample it duplicated (body, sample, t), its score,
                                  the head/shoulder distance that made it a duplicate (dist), and its box and RTMW keypoints.

Both are the models' numbers unrounded (float32), so a re-run of the de-duplication from these files and the layer
sees what the node saw. Parquet, not JSON: on OOO clip 1 (3136 samples, 28 duplicates in 119 s) these files are 17.7
and 34.4 KB, about 1.6 MB an hour together, where compact JSON with values rounded was 3.4 MB an hour (2.7 MB of it
the duplicates' keypoints).

Deterministic: identical rows give identical bytes. Every writer option that shapes the file is fixed below, and the
file carries no time; the pyarrow version (envs/vision's lock, part of the node's cache key) names itself in the
footer. The rows are known only once tracking ends (a sample's body id is its track's place in the layer), so each
file is written whole, in row groups of ROW_GROUP_ROWS: about 25 bytes a sample in memory before it is written.
"""

from __future__ import annotations

import math

KEYPOINTS = 133
ROW_GROUP_ROWS = 65536
# OOO clip 1 (scores, duplicates): zstd 3 19.0 + 35.8 KB, 9 17.7 + 34.4 KB, 19 17.3 + 34.3 KB (22 no smaller)
ZSTD_LEVEL = 9
SCORES_FILE = "layers/bodies_scores.parquet"
DUPLICATES_FILE = "layers/bodies_duplicates.parquet"
SCORES_DESCRIPTION = "YOLOX-m's detection score for every body sample (the layer keeps its box), a row per sample"
DUPLICATES_DESCRIPTION = ("The detections dropped as second detections of a person: the body sample each duplicated, "
                          "its score, head/shoulder distance, box and RTMW keypoints")


def _floats(n: int):
    import pyarrow as pa
    return pa.list_(pa.field("element", pa.float32(), nullable=False), n)


def scores_schema(detector: str, detector_threshold: float):
    import pyarrow as pa
    return pa.schema([pa.field("body", pa.string(), nullable=False), pa.field("sample", pa.int32(), nullable=False),
                      pa.field("t", pa.float64(), nullable=False), pa.field("score", pa.float32(), nullable=False)],
                     metadata={"side": "bodies_scores", "detector": detector, "detector_threshold": repr(detector_threshold)})


def duplicates_schema(detector: str, detector_threshold: float, duplicate_head_dist: float, score_threshold: float):
    import pyarrow as pa
    return pa.schema([pa.field("t", pa.float64(), nullable=False), pa.field("body", pa.string(), nullable=False),
                      pa.field("sample", pa.int32(), nullable=False), pa.field("score", pa.float32(), nullable=False),
                      pa.field("dist", pa.float32(), nullable=False), pa.field("box", _floats(4), nullable=False),
                      pa.field("kpts", _floats(3 * KEYPOINTS), nullable=False)],
                     metadata={"side": "bodies_duplicates", "detector": detector,
                               "detector_threshold": repr(detector_threshold),
                               "duplicate_head_dist": repr(duplicate_head_dist), "score_threshold": repr(score_threshold),
                               "kpts": f"{KEYPOINTS} keypoints in the bodies layer's keypoint_names order, each x, y, score"})


def _write(path, schema, rows: list[tuple], dictionary: list[str], statistics: list[str]):
    import pyarrow as pa
    import pyarrow.parquet as pq
    columns = list(zip(*rows)) if rows else [[] for _ in schema.names]
    table = pa.table({name: list(values) for name, values in zip(schema.names, columns)}, schema=schema)
    with pq.ParquetWriter(path, schema, compression="zstd", compression_level=ZSTD_LEVEL, use_dictionary=dictionary,
                          write_statistics=statistics, version="2.6", data_page_version="1.0", store_schema=True,
                          write_page_index=False, use_byte_stream_split=False) as writer:
        writer.write_table(table, row_group_size=ROW_GROUP_ROWS)


def _finite(values, what: str):
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{what} holds a value that is not finite")


def write_scores(path, rows: list[tuple], detector: str, detector_threshold: float):
    """rows: (body, sample, t, score), as bodies.side_rows gives them. A score that is not finite is a ValueError."""
    _finite((r[3] for r in rows), "a detector score")
    _write(path, scores_schema(detector, detector_threshold), rows, ["body"], ["body", "sample", "t"])


def write_duplicates(path, rows: list[tuple], detector: str, detector_threshold: float, duplicate_head_dist: float,
                     score_threshold: float):
    """rows: (t, body, sample, score, dist, box [4], kpts [399]), as bodies.side_rows gives them. A value that is not
    finite, or a box or keypoints of another length, is a ValueError."""
    for t, body, sample, score, dist, box, kpts in rows:
        if len(box) != 4 or len(kpts) != 3 * KEYPOINTS:
            raise ValueError(f"the duplicate of {body} at t={t} has {len(box)} box values and {len(kpts)} keypoint values")
        _finite([score, dist, *box, *kpts], f"the duplicate of {body} at t={t}")
    _write(path, duplicates_schema(detector, detector_threshold, duplicate_head_dist, score_threshold), rows, ["body"],
           ["t", "body", "sample"])
