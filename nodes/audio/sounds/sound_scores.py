"""The sounds layer's raw side files: every score the three models computed, of which the layer keeps only the events
above their thresholds (schema: #/$defs/parquet/sounds_tags and #/$defs/parquet/sounds_queries).

layers/sounds_tags.parquet: the AudioSet taggers' scores for all 527 classes, a row per model output. Columns: model
("panns" or "mn40"), start and end (seconds of the soundtrack) and scores (527 float32, in audioset_labels.csv's order,
which the file's metadata also lists under "labels"). PANNs rows first, then mn40's, each in time order.
- PANNs Cnn14_DecisionLevelMax: one row per segment of 0.32 s. Its CNN pools time by 32 frames, and its frame-wise
  output (100 frames/s) is each segment's scores repeated 32 times, the last segment's repeated to the end of the
  soundtrack (panns/pytorch_utils.py: interpolate, pad_framewise_output). So the segments are all it computed, and
  panns_framewise gives back the frame-wise scores the layer pools into 1 s steps, exactly (panns_segments checks it).
  Row i covers frames [32 i, 32 i + 32): start 0.32 i, end 0.32 (i + 1); the last row's end is the soundtrack's frame
  count / 100 (its frames, centred every 10 ms, run past its last segment by up to 31).
- EfficientAT mn40: one row per 1 s step [i, i + 1), the scores of the 5 s window centred on it (sound_events.windows).
Every score is the model's float32 sigmoid, unrounded: tag_rows on these rows (after panns_framewise and step_max for
PANNs) is the layer's tagger steps.

layers/sounds_queries.parquet: PE-A-Frame's probability for each query, a row per 40 ms frame. Columns: start, end
(frame k: k * 0.04, (k + 1) * 0.04) and probs (one float32 per query, in sound_events.QUERIES' order, which the file's
metadata lists under "queries"). spans() on each query's column is the layer's query events.

Why rows of arrays, not a column per class: 527 AudioSet classes (and a query list that may grow) would make the bundle
schema list hundreds of columns, and a reader takes a model's matrix in one read (numpy: the scores column's flat
values, reshaped). Why two files: the taggers score AudioSet's 527 classes and PE-A-Frame the node's own queries, on
different grids; one table would need a column whose width depends on the row.

Lossless: the scores are float32 as the models returned them, so re-thresholding them gives the layer's events exactly.
Sizes: about 26 MB an hour of soundtrack for the tags (PANNs 3.125 rows/s, mn40 1 row/s, 2.1 KB a row before
compression; 867 KB for OOO clip 1's 119 s) and 6 MB for the queries (25 rows/s; 206 KB).

Deterministic: identical scores give identical bytes. Every writer option that shapes the file is fixed below, and the
file carries no time; the pyarrow version (envs/audio-events' lock, part of the node's cache key) names itself in the
footer. The scores are all in memory already (the node holds them to make the events): a row group of ROW_GROUP_ROWS
is written at a time.
"""

from __future__ import annotations

import json

import numpy as np

from sound_events import PANNS_FPS

PANNS_SEGMENT = 32          # Cnn14's frames per segment: five 2x poolings of time
CHECK_SEGMENTS = 1024       # segments compared at once in panns_segments: 17 MB of comparisons
PE_FPS = 25                 # PE-A-Frame's 40 ms frames (sound_events.PE_HOP at 48 kHz)
ROW_GROUP_ROWS = 4096       # 8.6 MB of tagger scores a group before compression (about 22 min of PANNs)
# OOO clip 1 (tags, queries; byte stream split on the scores): zstd 3 881 + 215 KB, 9 873 + 213, 19 867 + 206 (0.1 s
# for both), 22 the same as 19; without byte stream split zstd 19 is 980 + 237. Float32 sigmoids keep most of their bits.
ZSTD_LEVEL = 19
TAGS_FILE, QUERIES_FILE = "layers/sounds_tags.parquet", "layers/sounds_queries.parquet"
TAGS_DESCRIPTION = ("PANNs Cnn14's scores for the 527 AudioSet classes every 0.32 s segment and EfficientAT mn40's every "
                    "1 s step, unrounded float32, a row per model output (model, start, end, scores)")
QUERIES_DESCRIPTION = "PE-A-Frame's probability for each query every 40 ms frame, unrounded float32 (start, end, probs)"


# --- PANNs: its segments, and the frame-wise scores they repeat -------------------------------------------------------

def panns_framewise(segments: np.ndarray, frames: int) -> np.ndarray:
    """Cnn14's frame-wise output from its segments: each repeated PANNS_SEGMENT times, the last to `frames`."""
    if len(segments) == 0 or not len(segments) * PANNS_SEGMENT <= frames < (len(segments) + 1) * PANNS_SEGMENT:
        raise ValueError(f"{len(segments)} segments of {PANNS_SEGMENT} frames do not make {frames} frames")
    out = np.repeat(segments, PANNS_SEGMENT, axis=0)
    return np.concatenate([out, np.repeat(segments[-1:], frames - len(out), axis=0)])


def panns_segments(framewise: np.ndarray) -> np.ndarray:
    """The segments Cnn14 computed, from its frame-wise output (stitched from chunks that start on its 32-frame grid,
    sound_events.PANNS_GRID): every PANNS_SEGMENT-th frame of the whole segments. A frame-wise output that is not those
    segments repeated is a ValueError: the side file would not give back what the layer pooled."""
    k = len(framewise) // PANNS_SEGMENT
    if k == 0:
        raise ValueError(f"PANNs' frame-wise scores ({len(framewise)} frames) hold no whole {PANNS_SEGMENT}-frame segment")
    # a copy, 32 times smaller, so that the frames can be freed; compared a block at a time, in place of building the
    # repeated frames (an hour's frames are 760 MB)
    segments = framewise[:k * PANNS_SEGMENT:PANNS_SEGMENT].copy()
    whole = framewise[:k * PANNS_SEGMENT].reshape(k, PANNS_SEGMENT, -1)
    repeated = all((whole[a:a + CHECK_SEGMENTS] == segments[a:a + CHECK_SEGMENTS, None]).all()
                   for a in range(0, k, CHECK_SEGMENTS)) and (framewise[k * PANNS_SEGMENT:] == segments[-1]).all()
    if not repeated:
        raise ValueError(f"PANNs' frame-wise scores ({len(framewise)} frames) are not its {PANNS_SEGMENT}-frame "
                         "segments repeated")
    return segments


def panns_times(n_segments: int, frames: int) -> tuple[list[float], list[float]]:
    starts = [i * PANNS_SEGMENT / PANNS_FPS for i in range(n_segments)]
    return starts, starts[1:] + [frames / PANNS_FPS]


def panns_frames(ends: list[float]) -> int:
    """The frame count a PANNs row's end gives back: the last row's end is frames / 100."""
    return round(ends[-1] * PANNS_FPS)


# --- the files --------------------------------------------------------------------------------------------------------

def _scores(n: int):
    import pyarrow as pa
    return pa.list_(pa.field("element", pa.float32(), nullable=False), n)


def tags_schema(labels: list[str]):
    import pyarrow as pa
    return pa.schema([pa.field("model", pa.string(), nullable=False), pa.field("start", pa.float64(), nullable=False),
                      pa.field("end", pa.float64(), nullable=False),
                      pa.field("scores", _scores(len(labels)), nullable=False)],
                     metadata={"side": "sounds_tags", "labels": json.dumps(labels)})


def queries_schema(queries: list[str]):
    import pyarrow as pa
    return pa.schema([pa.field("start", pa.float64(), nullable=False), pa.field("end", pa.float64(), nullable=False),
                      pa.field("probs", _scores(len(queries)), nullable=False)],
                     metadata={"side": "sounds_queries", "queries": json.dumps(queries)})


def _matrix(values, width: int, what: str) -> np.ndarray:
    m = np.asarray(values)
    if m.dtype != np.float32 or m.ndim != 2 or m.shape[1] != width:
        raise ValueError(f"{what} are {m.dtype} {m.shape}, not float32 (rows, {width})")
    if not np.isfinite(m).all() or m.min(initial=0) < 0 or m.max(initial=0) > 1:
        raise ValueError(f"{what} hold a value outside [0, 1] or not finite")
    return m


def _write(path, schema, columns: dict, list_column: str):
    """Rows in groups of ROW_GROUP_ROWS, every option that shapes the bytes fixed."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = len(columns[list_column])
    with pq.ParquetWriter(path, schema, compression="zstd", compression_level=ZSTD_LEVEL,
                          use_dictionary=[c for c in ("model",) if c in schema.names],
                          write_statistics=[c for c in schema.names if c != list_column], version="2.6",
                          data_page_version="1.0", store_schema=True, write_page_index=False,
                          use_byte_stream_split=[f"{list_column}.list.element"]) as writer:
        for a in range(0, max(n, 1), ROW_GROUP_ROWS):
            b = min(a + ROW_GROUP_ROWS, n)
            arrays = [pa.array(columns[c][a:b], type=schema.field(c).type) if c != list_column else
                      pa.FixedSizeListArray.from_arrays(pa.array(columns[c][a:b].reshape(-1), pa.float32()),
                                                        type=schema.field(c).type)
                      for c in schema.names]
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema), row_group_size=ROW_GROUP_ROWS)
    return path


def write_tags(path, panns: np.ndarray, frames: int, mn40: np.ndarray, labels: list[str]):
    """PANNs' segments (panns_segments) for a soundtrack of `frames` frame-wise frames, and mn40's step scores."""
    panns, mn40 = _matrix(panns, len(labels), "PANNs' scores"), _matrix(mn40, len(labels), "mn40's scores")
    p_start, p_end = panns_times(len(panns), frames)
    panns_framewise(panns, frames)   # the frame count fits the segments
    columns = {"model": ["panns"] * len(panns) + ["mn40"] * len(mn40),
               "start": p_start + [float(i) for i in range(len(mn40))],
               "end": p_end + [float(i + 1) for i in range(len(mn40))],
               "scores": np.concatenate([panns, mn40])}
    return _write(path, tags_schema(labels), columns, "scores")


def write_queries(path, probs: np.ndarray, queries: list[str]):
    """PE-A-Frame's probabilities as the node has them: (queries, frames)."""
    probs = _matrix(np.asarray(probs).T, len(queries), "PE-A-Frame's probabilities")
    columns = {"start": [k / PE_FPS for k in range(len(probs))], "end": [(k + 1) / PE_FPS for k in range(len(probs))],
               "probs": np.ascontiguousarray(probs)}
    return _write(path, queries_schema(queries), columns, "probs")


def read_tags(path) -> dict:
    """{"labels": [...], "panns": (segments, 527), "panns_frames": int, "mn40": (steps, 527)}, float32 as written."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    labels = json.loads(table.schema.metadata[b"labels"])
    scores = table["scores"].combine_chunks().flatten().to_numpy().reshape(-1, len(labels))
    model = np.array(table["model"].to_pylist())
    ends = table["end"].to_pylist()
    panns_ends = [e for e, m in zip(ends, model) if m == "panns"]
    return {"labels": labels, "panns": scores[model == "panns"], "panns_frames": panns_frames(panns_ends),
            "mn40": scores[model == "mn40"]}


def read_queries(path) -> dict:
    """{"queries": [...], "probs": (queries, frames)}, float32 as written: the node's shape."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    queries = json.loads(table.schema.metadata[b"queries"])
    return {"queries": queries,
            "probs": table["probs"].combine_chunks().flatten().to_numpy().reshape(-1, len(queries)).T.copy()}
