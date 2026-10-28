"""The shots layer's raw side file, layers/shots_scores.parquet: what PySceneDetect's AdaptiveDetector computes for
every frame, of which the layer keeps only the cuts (schema: #/$defs/parquet/shots_scores).

One row per frame the detector scored, in order. Columns: frame (its index among them), t (its time, as scenedetect
places it: a shot's start is a cut's t to 3 places), content_val (the frame's score: the weighted mean of the deltas
below, against the frame before), delta_hue, delta_sat, delta_lum, delta_edges (mean absolute differences of the HSV
channels and of the dilated Canny edge maps, on the frame as the detector sees it, downscaled), and adaptive_ratio
(content_val over the mean content_val of the window_width frames on each side, capped at 255).

What the detector does not compute is 0: the first frame has no frame before it (the detector scores it 0, so its
content_val and deltas are 0), and the first and last window_width frames have no full window around them, so no
adaptive_ratio (0: none of them can be a cut). A cut is a frame with adaptive_ratio >= adaptive_threshold and
content_val >= min_content_val, at least min_scene_len frames after the previous cut (or the first frame): cuts() does
that from the file, at the detector's threshold or another, and the node checks that it gives the layer's cuts exactly.
The detector's parameters are in the file's metadata (key "detector", JSON).

Deterministic: identical rows give identical bytes. Every writer option that shapes the file is fixed below, and the
file carries no time; the pyarrow version (envs/core's lock, part of the node's cache key) names itself in the footer.
Rows are written in groups of ROW_GROUP_ROWS as they come, so a long film's scores are never all in memory.
"""

from __future__ import annotations

import json
import math
from collections import deque
from fractions import Fraction

FILE = "layers/shots_scores.parquet"
DESCRIPTION = ("PySceneDetect AdaptiveDetector's scores for every frame (content_val, its hue, saturation, luma and edge "
               "deltas, adaptive_ratio) that the cuts are thresholded from, a row per frame")
COLUMNS = ("frame", "t", "content_val", "delta_hue", "delta_sat", "delta_lum", "delta_edges", "adaptive_ratio")
DELTAS = ("delta_hue", "delta_sat", "delta_lum", "delta_edges")   # ContentDetector.Components' fields, in order
ROW_GROUP_ROWS = 65536     # 36 min of 30 fps video, 4 MB of rows a group before compression
# OOO clip 1's 3567 frames: zstd 3 130 KB, 9 125 KB, 19 121 KB (0.03 s; 22 no smaller); byte stream split on the
# doubles makes it larger (134 KB); frame delta-encoded 113 KB
ZSTD_LEVEL = 19
# the detector's parameters the metadata records (cuts() reads adaptive_threshold, min_content_val, min_scene_len,
# window_width and frame_rate)
PARAMETERS = ("adaptive_threshold", "min_content_val", "min_scene_len", "window_width", "weights", "luma_only",
              "kernel_size", "frame_size", "frame_rate")


def schema(parameters: dict):
    """The file's columns, none nullable, with the detector's `parameters` (PARAMETERS) in its metadata as JSON."""
    import pyarrow as pa
    missing = [k for k in PARAMETERS if k not in parameters]
    if missing:
        raise ValueError(f"the detector's parameters lack {missing}")
    fields = [pa.field("frame", pa.int32(), nullable=False)]
    fields += [pa.field(name, pa.float64(), nullable=False) for name in COLUMNS[1:]]
    return pa.schema(fields, metadata={"side": "shots_scores",
                                       "detector": json.dumps({k: parameters[k] for k in PARAMETERS}, sort_keys=True)})


class Writer:
    """add(frame, t, content_val, deltas, adaptive_ratio) per frame in order, then close(); a with block closes it."""

    def __init__(self, path, parameters: dict):
        import pyarrow.parquet as pq
        self.schema = schema(parameters)
        self._writer = pq.ParquetWriter(
            path, self.schema, compression="zstd", compression_level=ZSTD_LEVEL, use_dictionary=False,
            column_encoding={"frame": "DELTA_BINARY_PACKED"}, write_statistics=["frame", "t"], version="2.6",
            data_page_version="1.0", store_schema=True, write_page_index=False, use_byte_stream_split=False)
        self._rows: dict[str, list] = {name: [] for name in COLUMNS}
        self.rows = 0

    def add(self, frame: int, t: float, content_val: float, deltas, adaptive_ratio: float):
        values = (frame, t, content_val, *deltas, adaptive_ratio)
        if len(values) != len(COLUMNS):
            raise ValueError(f"frame {frame}: {len(deltas)} deltas, not {len(DELTAS)}")
        if frame != self.rows:
            raise ValueError(f"frame {frame} where frame {self.rows} comes next: every frame is written, in order")
        if not all(math.isfinite(v) for v in values[1:]):
            raise ValueError(f"frame {frame}: a score is not finite: {dict(zip(COLUMNS, values))}")
        for name, value in zip(COLUMNS, values):
            self._rows[name].append(value)
        self.rows += 1
        if len(self._rows["frame"]) == ROW_GROUP_ROWS:
            self._flush()

    def _flush(self):
        import pyarrow as pa
        if self._rows["frame"]:
            self._writer.write_table(pa.table(self._rows, schema=self.schema), row_group_size=ROW_GROUP_ROWS)
            self._rows = {name: [] for name in COLUMNS}

    def close(self):
        self._flush()
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class Recorder:
    """Collects what the detector computes for each frame and writes it a row per frame, in order, once it is complete.

    It stands in for scenedetect's StatsManager, which keeps every frame's metrics in memory until the end: the
    SceneManager hands it to the detector, which calls set_metrics(timecode, {...}) with a frame's content_val and
    deltas when it scores the frame, and with its adaptive_ratio window_width frames later, when the frame is centred in
    its window. frame(timecode, score) is called after the detector has processed each frame; a frame is written once
    window_width more frames have come (its ratio, if it has one, is set by then), and finish() writes the rest.
    open_writer() makes the Writer when the first row is written: the detector's kernel size and the processed frame
    size, which the file's metadata records, are known only once it has seen a frame."""

    def __init__(self, open_writer, window_width: int, ratio_key: str):
        self.open_writer, self.window_width, self.ratio_key = open_writer, window_width, ratio_key
        self.writer: Writer | None = None
        self._metrics: dict = {}
        self._pending: deque = deque()
        self._base_timecode = None     # the SceneManager sets it, as on a StatsManager

    def register_metrics(self, keys) -> None:
        wanted = {"content_val", *DELTAS, self.ratio_key}
        if not wanted <= set(keys):
            raise ValueError(f"the detector reports {sorted(keys)}, not {sorted(wanted)}")

    def set_metrics(self, timecode, metrics: dict) -> None:
        self._metrics.setdefault(timecode, {}).update(metrics)

    def frame(self, timecode, score: float) -> None:
        self._pending.append((timecode, score))
        if len(self._pending) > self.window_width:
            self._write(*self._pending.popleft())

    def finish(self) -> int:
        """Writes the last frames and closes the file. -> how many rows it holds; none is a ValueError (no file)."""
        while self._pending:
            self._write(*self._pending.popleft())
        if self.writer is None:
            raise ValueError("the detector scored no frames")
        self.writer.close()
        return self.writer.rows

    def _write(self, timecode, score: float) -> None:
        if self.writer is None:
            self.writer = self.open_writer()
        m = self._metrics.pop(timecode, {})
        frame = self.writer.rows
        if frame and m.get("content_val") != score:
            raise ValueError(f"frame {frame}: the detector scored {score}, but reported content_val {m.get('content_val')}")
        self.writer.add(frame, timecode.seconds, score, [m.get(k, 0.0) for k in DELTAS], m.get(self.ratio_key, 0.0))


def detect(video, threshold: float, path) -> tuple[list, dict]:
    """PySceneDetect's AdaptiveDetector at adaptive_threshold=`threshold` (its other parameters at their defaults), as
    scenedetect.detect(video, AdaptiveDetector(...)) runs it, writing every frame's scores to `path`.
    -> (the scene list, the detector's parameters as the file's metadata records them)."""
    from scenedetect import AdaptiveDetector, SceneManager, open_video

    class Recording(AdaptiveDetector):
        recorder: Recorder
        frame_size: list | None = None

        def process_frame(self, timecode, frame_img):
            cuts = super().process_frame(timecode, frame_img)
            if self.frame_size is None:
                self.frame_size = [int(frame_img.shape[1]), int(frame_img.shape[0])]
            if self._frame_score is not None:   # the detector buffers a frame only when it scored it
                self.recorder.frame(timecode, self._frame_score)
            return cuts

    stream = open_video(str(video), backend="opencv")
    detector = Recording(adaptive_threshold=threshold)
    if not isinstance(detector.min_scene_len, int):
        raise ValueError(f"min_scene_len is {detector.min_scene_len!r}, not a number of frames")
    rate = Fraction(stream.frame_rate)

    def parameters() -> dict:
        return {"adaptive_threshold": detector.adaptive_threshold, "min_content_val": detector.min_content_val,
                "min_scene_len": detector.min_scene_len, "window_width": detector.window_width,
                "weights": dict(detector._weights._asdict()),
                "luma_only": detector._weights == AdaptiveDetector.LUMA_ONLY_WEIGHTS,
                "kernel_size": int(detector._kernel.shape[0]), "frame_size": detector.frame_size,
                "frame_rate": f"{rate.numerator}/{rate.denominator}"}

    detector.recorder = Recorder(lambda: Writer(path, parameters()), detector.window_width, detector._adaptive_ratio_key)
    manager = SceneManager(detector.recorder)
    manager.add_detector(detector)
    manager.detect_scenes(video=stream)
    detector.recorder.finish()
    return manager.get_scene_list(), parameters()


def frames_between(t0: float, t1: float, frame_rate: Fraction) -> int:
    """How many frames t1 is after t0, as scenedetect counts it between two positions: their difference in microseconds
    (the OpenCV backend's time base) as seconds, times the frame rate, rounded."""
    return round(float(Fraction(round(t1 * 1_000_000) - round(t0 * 1_000_000), 1_000_000)) * float(frame_rate))


def cuts(t, content_val, adaptive_ratio, parameters: dict, threshold: float | None = None) -> list[int]:
    """The frames (row indices) AdaptiveDetector cuts at, from its stored scores: at `threshold`, or the file's
    adaptive_threshold. A frame centred in a full window (not among the first or last window_width) is a cut when its
    adaptive_ratio >= threshold and its content_val >= min_content_val, min_scene_len frames or more after the last
    cut (at first, the first frame)."""
    threshold = parameters["adaptive_threshold"] if threshold is None else threshold
    w, rate = parameters["window_width"], Fraction(parameters["frame_rate"])
    found, last = [], 0
    for k in range(w, len(t) - w):
        if (adaptive_ratio[k] >= threshold and content_val[k] >= parameters["min_content_val"]
                and frames_between(t[last], t[k], rate) >= parameters["min_scene_len"]):
            found.append(k)
            last = k
    return found


def read(path) -> tuple[dict, dict]:
    """A shots_scores file -> ({column: list}, the detector's parameters)."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    return table.to_pydict(), json.loads(table.schema.metadata[b"detector"])
