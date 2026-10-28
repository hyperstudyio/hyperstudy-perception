"""The gaze layer's raw side file, layers/gaze_heatmaps.parquet: Gaze-LLE's whole output for every head, of which the
layer keeps only the peak, the in-frame score and the target (schema: #/$defs/parquet/gaze_heatmaps).

One row per head (a face sample), in the order the node scores them: frames in time order, the heads of one frame in
the faces layer's track order. Columns: face (face:N), sample (its index in the track's samples), t, inout (the
in-frame score, unrounded), max (the heatmap's largest value, float32) and heat (the 64x64 heatmap as uint8 relative to
max, row-major). About 1 KB a head.

Quantisation: cell = floor(255 * heat / max), so heat lies in [cell, cell + 1) / 255 * max, and (cell + 0.5) / 255 * max
is within max / 510 of it (the maximum itself is 255, exactly max, read back as 255.5 / 255 * max). Floor rather than
round: only the maximum reaches 255, so the stored map's argmax is the heatmap's, the cell of the layer's peak. Rounded,
every cell within max / 510 of the maximum would be 255 too, and on OOO clip 1 the first such cell was not the peak for
75 of 2690 heads.

Deterministic: identical rows give identical bytes. Every writer option that shapes the file is fixed below, and the
file carries no time; the pyarrow version (envs/gaze's lock, part of the node's cache key) names itself in the footer.
Rows are written in groups of ROW_GROUP_ROWS as they come, so a long film's maps are never all in memory.
"""

from __future__ import annotations

import numpy as np

HEATMAP_SIZE = 64
CELLS = HEATMAP_SIZE * HEATMAP_SIZE
ROW_GROUP_ROWS = 4096      # 16 MB of maps a group before compression
# OOO clip 1's 2690 maps: zstd 3 2.91 MB (0.02 s), 9 2.76 MB (0.1 s), 15 2.68 MB (0.8 s), 19 2.52 MB (2.0 s, 2% of the
# node's time; 22 saves 3 KB more)
ZSTD_LEVEL = 19
FILE = "layers/gaze_heatmaps.parquet"
DESCRIPTION = ("Gaze-LLE's 64x64 heatmap per head (uint8, floor(255 * heat / max), with its max as float32) and its "
               "unrounded in-frame score, a row per sample")


def quantise(heat) -> tuple[bytes, float]:
    """A 64x64 heatmap -> (4096 bytes, row-major: floor(255 * heat / max); its max, a float32). Gaze-LLE's heatmap is a
    sigmoid: every value is finite and non-negative; anything else is a ValueError. An all-zero map is all zeros."""
    heat = np.asarray(heat, dtype=np.float32)
    if heat.shape != (HEATMAP_SIZE, HEATMAP_SIZE):
        raise ValueError(f"a heatmap is {HEATMAP_SIZE}x{HEATMAP_SIZE}, not {'x'.join(map(str, heat.shape))}")
    if not np.isfinite(heat).all() or heat.min() < 0:
        raise ValueError("a heatmap holds a value that is negative or not finite")
    top = heat.max()
    if top == 0:
        return bytes(CELLS), 0.0
    # in float64, where 255 * heat / max of a float32 value below the maximum is exactly below 255: only the maximum is 255
    cells = np.floor(heat.astype(np.float64) / float(top) * 255).astype(np.uint8)
    return cells.tobytes(), float(top)


def dequantise(cells: bytes, top: float) -> np.ndarray:
    """The stored map back as float32 64x64, each cell at the middle of its step: (cell + 0.5) / 255 * max."""
    q = np.frombuffer(cells, dtype=np.uint8).reshape(HEATMAP_SIZE, HEATMAP_SIZE)
    return ((q.astype(np.float64) + 0.5) / 255 * top).astype(np.float32)


def schema():
    import pyarrow as pa
    return pa.schema([pa.field("face", pa.string(), nullable=False), pa.field("sample", pa.int32(), nullable=False),
                      pa.field("t", pa.float64(), nullable=False), pa.field("inout", pa.float32(), nullable=False),
                      pa.field("max", pa.float32(), nullable=False),
                      pa.field("heat", pa.binary(CELLS), nullable=False)],
                     metadata={"side": "gaze_heatmaps",
                               "heat": "64x64 uint8, row-major: floor(255 * heat / max); heat ~ (cell + 0.5) / 255 * max"})


class Writer:
    """add(face id, sample index, t, heatmap, in-frame score) per head, then close(); a with block closes it."""

    def __init__(self, path):
        import pyarrow.parquet as pq
        self.schema = schema()
        self._writer = pq.ParquetWriter(
            path, self.schema, compression="zstd", compression_level=ZSTD_LEVEL, use_dictionary=["face"],
            write_statistics=["face", "sample", "t"], version="2.6", data_page_version="1.0", store_schema=True,
            write_page_index=False, use_byte_stream_split=False)
        self._rows: dict[str, list] = {name: [] for name in self.schema.names}
        self.rows = 0

    def add(self, face: str, sample: int, t: float, heat, inout: float):
        if not np.isfinite(inout):
            raise ValueError(f"{face} sample {sample}: the in-frame score {inout} is not finite")
        cells, top = quantise(heat)
        for name, value in zip(self.schema.names, (face, sample, t, inout, top, cells)):
            self._rows[name].append(value)
        self.rows += 1
        if len(self._rows["face"]) == ROW_GROUP_ROWS:
            self._flush()

    def _flush(self):
        import pyarrow as pa
        if self._rows["face"]:
            self._writer.write_table(pa.table(self._rows, schema=self.schema), row_group_size=ROW_GROUP_ROWS)
            self._rows = {name: [] for name in self.schema.names}

    def close(self):
        self._flush()
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
