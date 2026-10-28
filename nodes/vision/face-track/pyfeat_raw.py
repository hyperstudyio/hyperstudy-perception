"""The faces layer's raw side files: the rest of py-feat's output, which neither the layer nor faces_detail, faces_mesh
and faces_embeddings keep (schema: #/$defs/parquet/faces_pyfeat and #/$defs/parquet/faces_untracked).

layers/faces_pyfeat.parquet     per track sample (raw.pyfeat): RetinaFace's detection score and its unrounded box, the
                                multitask model's head translation, 52 blendshapes and the mesh's depth (z) per point.
                                One row per sample of the layer's tracks, in time order, the samples of one time in
                                track order (the order validate.py checks).
layers/faces_untracked.parquet  per detection that is in no track (raw.untracked): every face py-feat returned that
                                tracking.py left out (smaller than MIN_FACE_PX or scored below MIN_FACE_SCORE), with all
                                of py-feat's values for it, so tracking can be revisited with other thresholds. In
                                py-feat's row order (frames in time order, RetinaFace's order within a frame).

What py-feat's frame holds and where each column is kept (Detectorv2, py-feat 2.1.5), for a tracked face:
  FaceRectX/Y/Width/Height   the layer's samples[].box (to 0.1 px); unrounded here (box)
  FaceScore                  here (score); tracking.py drops faces below MIN_FACE_SCORE
  x_0..67, y_0..67           faces_detail lmk (0.1 px)
  Pitch, Roll, Yaw           faces_detail head (3 places)
  X, Y, Z                    here (translation): the multitask pose head's translation outputs
  AU*                        faces_detail au (2 places)
  Neutral ... Anger          faces_detail emo (3 places); the layer's summary.emotions are their track means
  valence, arousal           faces_detail va (3 places)
  gaze_pitch, gaze_yaw       faces_detail gaze (3 places); the layer's samples[].gaze
  gaze_angle                 not kept: arccos(cos(gaze_pitch) * cos(gaze_yaw)), py-feat's own formula
  Identity_1..512            faces_embeddings v (4 places)
  mesh_x_*, mesh_y_*         faces_mesh xy (whole pixels)
  mesh_z_*                   here (mesh_z)
  52 blendshapes             here (blendshapes), in BLENDSHAPES order
  FrameHeight, FrameWidth    not kept: the video's height and width (media)
  frame                      here (frame); the samples' t is frame / fps, to 3 places
  approx_time                not kept: py-feat's clock string for frame, from the same frame rate
  input                      not kept: the path of the video file on the machine that ran it
  Identity                   not kept: py-feat's own clustering of the embeddings (compute_identities, cosine 0.8),
                             which tracking.py replaces with its own; it can be recomputed from the embeddings
An untracked face keeps all of the kept columns in its row, at full precision.

A frame with no face is one row of NaN in py-feat's output (its placeholder): no detection, in neither file.

Values are float32, as py-feat computes them (torch float32), unrounded. A value that is not finite is a ValueError.
Deterministic: the same rows give the same bytes. Every writer option that shapes the file is fixed below and the file
carries no time; the pyarrow version (envs/faces' lock, part of the node's cache key) names itself in the footer. Rows
are written in groups of ROW_GROUP_ROWS as they come.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PYFEAT_FILE = "layers/faces_pyfeat.parquet"
UNTRACKED_FILE = "layers/faces_untracked.parquet"
PYFEAT_DESCRIPTION = ("py-feat's output per sample that the faces layer and its other side files do not keep: the "
                      "detection score, the unrounded box, head translation, 52 blendshapes and the mesh's depth, a "
                      "row per sample")
UNTRACKED_DESCRIPTION = ("every face py-feat detected that is in no track (too small or scored too low for "
                         "tracking), with all of py-feat's values for it, a row per detection")

BOX = ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]   # x, y, w, h in source pixels: the layer's box
SCORE = ["FaceScore"]
TRANSLATION = ["X", "Y", "Z"]
BLENDSHAPES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight"]   # py-feat's MP_BLENDSHAPE_NAMES (MediaPipe/ARKit), each in [0, 1]
MESH_POINTS = 478
MESH_Z = [f"mesh_z_{i}" for i in range(MESH_POINTS)]
MESH_XYZ = ([f"mesh_x_{i}" for i in range(MESH_POINTS)] + [f"mesh_y_{i}" for i in range(MESH_POINTS)] + MESH_Z)
# faces_detail's per-sample fields, in its order (face_detail.py)
LANDMARKS = [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]
EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
VA = ["valence", "arousal"]
HEAD = ["Yaw", "Pitch", "Roll"]
GAZE = ["gaze_yaw", "gaze_pitch"]
EMBEDDING = [f"Identity_{i + 1}" for i in range(512)]
AUS = 20   # Detectorv2's multitask checkpoint: AU01 ... AU43, faces_detail's au_names

ROW_GROUP_ROWS = 1024   # about 2 MB of faces_pyfeat, 9 MB of faces_untracked a group before compression
# Float values are stored byte-stream-split (each value's 4 bytes in 4 streams), which zstd compresses better than
# plain floats: on OOO clip 1's 2690 samples faces_pyfeat is 4.62 MB at zstd 9 (plain: 5.36 MB; 5.79 MB raw) and its
# 269 untracked faces 1.66 MB (plain: 2.11 MB). zstd 3 is 0.6% larger, 19 1.4% smaller and ten times slower
ZSTD_LEVEL = 9


def sample_columns() -> list[tuple[str, list[str]]]:
    """faces_pyfeat's value columns: (column, the py-feat columns it holds in order)."""
    return [("box", BOX), ("score", SCORE), ("translation", TRANSLATION), ("blendshapes", BLENDSHAPES),
            ("mesh_z", MESH_Z)]


def untracked_columns(aus: Sequence[str]) -> list[tuple[str, list[str]]]:
    """faces_untracked's value columns, faces_detail's fields first under its names (lmk, au, emo, va, head, gaze)."""
    if len(aus) != AUS:
        raise ValueError(f"py-feat gave {len(aus)} action units, not the {AUS} this file holds")
    return [("box", BOX), ("score", SCORE), ("lmk", LANDMARKS), ("au", list(aus)), ("emo", EMOTIONS), ("va", VA),
            ("head", HEAD), ("translation", TRANSLATION), ("gaze", GAZE), ("embedding", EMBEDDING),
            ("mesh", MESH_XYZ), ("blendshapes", BLENDSHAPES)]


def flat(columns: list[tuple[str, list[str]]]) -> list[str]:
    """Every py-feat column the value columns hold, in order: what the node reads from py-feat's frame for a row."""
    return [c for _, cs in columns for c in cs]


def _values_type(pa, n: int):
    """A scalar float32, or n of them (a fixed-size list); never null."""
    return pa.float32() if n == 1 else pa.list_(pa.field("element", pa.float32(), nullable=False), n)


def sample_schema():
    import pyarrow as pa
    keys = [pa.field("face", pa.string(), nullable=False), pa.field("sample", pa.int32(), nullable=False),
            pa.field("t", pa.float64(), nullable=False), pa.field("frame", pa.int32(), nullable=False)]
    return pa.schema(keys + [pa.field(name, _values_type(pa, len(cs)), nullable=False) for name, cs in sample_columns()],
                     metadata={"side": "faces_pyfeat", "translation": ",".join(TRANSLATION),
                               "blendshapes": ",".join(BLENDSHAPES)})


def untracked_schema(aus: Sequence[str]):
    import pyarrow as pa
    keys = [pa.field("frame", pa.int32(), nullable=False), pa.field("t", pa.float64(), nullable=False),
            pa.field("dropped", pa.string(), nullable=False)]
    return pa.schema(keys + [pa.field(name, _values_type(pa, len(cs)), nullable=False)
                             for name, cs in untracked_columns(aus)],
                     metadata={"side": "faces_untracked", "au_names": ",".join(aus),
                               "emotion_names": ",".join(EMOTIONS), "translation": ",".join(TRANSLATION),
                               "blendshapes": ",".join(BLENDSHAPES)})


def _float_paths(schema) -> list[str]:
    """The Parquet column paths of the float32 values: a scalar's name, a list's <name>.list.element."""
    import pyarrow as pa
    return [f.name if pa.types.is_float32(f.type) else f"{f.name}.list.element" for f in schema
            if pa.types.is_float32(f.type) or (pa.types.is_fixed_size_list(f.type) and pa.types.is_float32(f.type.value_type))]


class _Writer:
    """add(*keys, values) per row, values being the row's py-feat columns in flat(columns) order; then close()."""

    def __init__(self, path, schema, columns, dictionary: list[str], statistics: list[str]):
        import pyarrow.parquet as pq
        self.schema, self.columns = schema, columns
        self.keys = schema.names[:len(schema.names) - len(columns)]
        self.width = len(flat(columns))
        self._writer = pq.ParquetWriter(
            path, schema, compression="zstd", compression_level=ZSTD_LEVEL, use_dictionary=dictionary,
            write_statistics=statistics, version="2.6", data_page_version="1.0", store_schema=True,
            write_page_index=False, use_byte_stream_split=_float_paths(schema))
        self._keys: dict[str, list] = {k: [] for k in self.keys}
        self._values: list[np.ndarray] = []
        self.rows = 0

    def add(self, *row):
        *keys, values = row
        v = np.asarray(values, dtype=np.float64)
        if v.shape != (self.width,):
            raise ValueError(f"{len(v)} values for a row of {self.width}")
        bad = np.flatnonzero(~np.isfinite(v))
        if len(bad):
            column = flat(self.columns)[bad[0]]
            raise ValueError(f"py-feat gave {column} = {v[bad[0]]} for the face in frame {keys[self.keys.index('frame')]}")
        for k, value in zip(self.keys, keys):
            self._keys[k].append(value)
        self._values.append(v.astype(np.float32))
        self.rows += 1
        if len(self._values) == ROW_GROUP_ROWS:
            self._flush()

    def _flush(self):
        import pyarrow as pa
        if not self._values:
            return
        values, arrays, at = np.stack(self._values), [], 0
        for k in self.keys:
            arrays.append(pa.array(self._keys[k], type=self.schema.field(k).type))
        for name, cs in self.columns:
            block = values[:, at:at + len(cs)]
            at += len(cs)
            kind = self.schema.field(name).type
            arrays.append(pa.array(block[:, 0], type=kind) if len(cs) == 1 else
                          pa.FixedSizeListArray.from_arrays(pa.array(block.reshape(-1), type=pa.float32()),
                                                            type=kind))
        self._writer.write_table(pa.Table.from_arrays(arrays, schema=self.schema), row_group_size=ROW_GROUP_ROWS)
        self._keys, self._values = {k: [] for k in self.keys}, []

    def close(self):
        self._flush()
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def samples_writer(path) -> _Writer:
    """faces_pyfeat: add(face id, sample index, t, frame, values in flat(sample_columns()) order) per track sample."""
    return _Writer(path, sample_schema(), sample_columns(), dictionary=["face"], statistics=["face", "sample", "t", "frame"])


def untracked_writer(path, aus: Sequence[str]) -> _Writer:
    """faces_untracked: add(frame, t, why tracking dropped it, values in flat(untracked_columns(aus)) order) per face."""
    return _Writer(path, untracked_schema(aus), untracked_columns(aus), dictionary=["dropped"],
                   statistics=["frame", "t", "dropped"])
