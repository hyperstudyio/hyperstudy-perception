"""The facing layer's raw side files, what it drops of MediaPipe's output and of the geometry its events are thresholded
from (schema: #/$defs/parquet/<name>):

layers/facing_poses.parquet   every pose MediaPipe Pose Landmarker returned for a face-linked body sample's crop, a row
                              per pose (up to facing.POSES a crop): the body sample (body, sample, t, face), the pose's
                              index in the landmarker's result and whether it is the one the layer uses (kept: the
                              nearest the box centre), the crop (window: x0, y0, side in frame pixels), all 33 landmarks
                              as the model returned them (x, y in [0, 1] of the crop, z, visibility, presence), the 33
                              world landmarks (metres, origin mid-hip), the torso direction from them (source, yaw, pitch,
                              unrounded) and the torso centre the facing is measured from. A face-linked body sample with
                              no row is one whose crop held no pose.
layers/facing_angles.parquet  every angle the events are thresholded from (facing.pair_angles): at each sample time, for
                              each kept pose A with a facing and each other kept pose B, the angle between A's facing
                              vector and the image direction to B, and their distance in the image.

Re-thresholding: facing.events_from_angles over the angles file's rows (read_angles) at the layer's cone_deg,
min_facing_s and max_gap_s gives the layer's events exactly, and the node checks it does before it finishes; at other
values it gives the events those would have given. A wider cone than any angle stored changes nothing the file lacks:
every pair is stored, whatever its angle.

Exact: the landmarks are MediaPipe's float32 values as it returned them, so in_frame's frame pixels (x * side + x0)
and nearest_index's choice can be recomputed from the file; yaw, pitch, centre, angle and dist are the float64 values the
node computed (the events use yaw and pitch rounded to 0.1 degree, as the bake-off's pose file stored them).

Deterministic: identical rows give identical bytes. Every writer option that shapes the files is fixed below, and they
carry no time; the pyarrow version (envs/vision's lock, part of the node's cache key) names itself in the footer. Poses
are written in groups of ROW_GROUP_ROWS as they come; the angles are known once every sample is posed and are written
whole (a few rows a sample time).
"""

from __future__ import annotations

import math

from facing import CONE_DEG, MARGIN, MAX_GAP_S, MIN_FACING_S, NAMES, POSES, SCORE_THRESHOLD, face_number

LANDMARK_VALUES = 5          # x, y, z, visibility, presence
WORLD_VALUES = 3             # x, y, z
ROW_GROUP_ROWS = 4096        # about 4.5 MB of poses a group before compression
ANGLES_GROUP_ROWS = 65536    # about 2.6 MB of angles a group
# OOO clip 1 (2577 poses, 7747 angles): poses zstd 3 2.555 MB, 9 2.550, 15 2.548, 19 2.551; byte_stream_split no smaller.
# The landmarks are float32 values whose low mantissa bits barely compress (264 a pose: 1.06 KB raw, 0.96 KB stored)
ZSTD_LEVEL = 9
POSES_FILE = "layers/facing_poses.parquet"
ANGLES_FILE = "layers/facing_angles.parquet"
POSES_DESCRIPTION = ("Every pose MediaPipe Pose Landmarker returned for each face-linked body sample: its 33 landmarks "
                     "(x, y, z, visibility, presence) and 33 world landmarks, the torso direction and centre, and which "
                     "pose the layer kept, a row per pose")
ANGLES_DESCRIPTION = ("The angle between each posed person's torso direction and the image direction to each other "
                      "posed person at every sample time, and their distance: what the events are thresholded from")


class BadRow(ValueError):
    """A row the side files cannot hold (a value that is not finite, landmarks of another shape): the node fails."""


def _list(arrow_type, n: int):
    import pyarrow as pa
    return pa.list_(pa.field("element", arrow_type, nullable=False), n)


def poses_schema(model: str):
    import pyarrow as pa
    return pa.schema([
        pa.field("body", pa.string(), nullable=False), pa.field("sample", pa.int32(), nullable=False),
        pa.field("t", pa.float64(), nullable=False), pa.field("face", pa.string(), nullable=False),
        pa.field("pose", pa.int32(), nullable=False), pa.field("kept", pa.bool_(), nullable=False),
        pa.field("window", _list(pa.int32(), 3), nullable=False),
        pa.field("landmarks", _list(pa.float32(), LANDMARK_VALUES * len(NAMES)), nullable=False),
        pa.field("world", _list(pa.float32(), WORLD_VALUES * len(NAMES)), nullable=False),
        pa.field("source", pa.string(), nullable=False), pa.field("yaw", pa.float64(), nullable=False),
        pa.field("pitch", pa.float64(), nullable=False), pa.field("centre", _list(pa.float64(), 2), nullable=False)],
        metadata={"side": "facing_poses", "model": model, "names": ",".join(NAMES),
                  "landmarks": "33 landmarks in names order, each x, y (in [0, 1] of the crop), z, visibility, presence",
                  "world": "33 world landmarks in names order, each x, y, z (metres, camera-aligned, origin mid-hip)",
                  "margin": repr(MARGIN), "poses": repr(POSES), "score_threshold": repr(SCORE_THRESHOLD)})


def angles_schema(cone_deg: float, min_facing_s: float, max_gap_s: float):
    import pyarrow as pa
    return pa.schema([
        pa.field("t", pa.float64(), nullable=False), pa.field("from", pa.string(), nullable=False),
        pa.field("to", pa.string(), nullable=False), pa.field("angle", pa.float64(), nullable=False),
        pa.field("dist", pa.float64(), nullable=False)],
        metadata={"side": "facing_angles", "cone_deg": repr(cone_deg), "min_facing_s": repr(min_facing_s),
                  "max_gap_s": repr(max_gap_s)})


def _writer(path, schema, dictionary: list[str], statistics: list[str]):
    import pyarrow.parquet as pq
    return pq.ParquetWriter(path, schema, compression="zstd", compression_level=ZSTD_LEVEL, use_dictionary=dictionary,
                            write_statistics=statistics, version="2.6", data_page_version="1.0", store_schema=True,
                            write_page_index=False, use_byte_stream_split=False)


def _finite(values, what: str):
    if not all(math.isfinite(v) for v in values):
        raise BadRow(f"{what} holds a value that is not finite")


class PosesWriter:
    """add(body id, sample index, t, face number, window, pose) for each of facing.body_poses' poses, in the order the
    node poses them (time order, the bodies of one time in the bodies layer's track order), then close(); a with block
    closes it. A value that is not finite, or landmarks of another shape, is a BadRow (a ValueError)."""

    def __init__(self, path, model: str):
        self.schema = poses_schema(model)
        self._writer = _writer(path, self.schema, ["body", "face", "source"], ["body", "sample", "t", "face"])
        self._rows: dict[str, list] = {name: [] for name in self.schema.names}
        self.rows = 0

    def add(self, body: str, sample: int, t: float, face: int, window, pose: dict):
        what = f"{body} sample {sample}, pose {pose['pose']}"
        landmarks = [float(v) for v in pose["landmarks"].reshape(-1)]
        world = [float(v) for v in pose["world"].reshape(-1)]
        if len(landmarks) != LANDMARK_VALUES * len(NAMES) or len(world) != WORLD_VALUES * len(NAMES):
            raise BadRow(f"{what}: {len(landmarks)} landmark values and {len(world)} world values")
        if pose["source"] not in (None, "torso", "shoulders"):
            raise BadRow(f"{what}: unknown facing source {pose['source']!r}")
        # no nulls in a side file: a facing is "none" with yaw 0, and a pitch only from the torso (else 0, as the events
        # take a missing pitch)
        yaw = 0.0 if pose["yaw"] is None else float(pose["yaw"])
        pitch = 0.0 if pose["pitch"] is None else float(pose["pitch"])
        centre = [float(v) for v in pose["centre"]]
        _finite([*landmarks, *world, yaw, pitch, *centre], what)
        values = (body, sample, t, f"face:{face}", pose["pose"], bool(pose["kept"]), [int(v) for v in window],
                  landmarks, world, pose["source"] or "none", yaw, pitch, centre)
        for name, value in zip(self.schema.names, values):
            self._rows[name].append(value)
        self.rows += 1
        if len(self._rows["body"]) == ROW_GROUP_ROWS:
            self._flush()

    def _flush(self):
        import pyarrow as pa
        if self._rows["body"]:
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


def write_angles(path, rows: list[tuple], cone_deg: float = CONE_DEG, min_facing_s: float = MIN_FACING_S,
                 max_gap_s: float = MAX_GAP_S):
    """rows: facing.pair_angles' (t, A, B, angle, dist), faces as numbers; the thresholds go in the file's metadata. A
    value that is not finite is a BadRow (a ValueError)."""
    import pyarrow as pa
    schema = angles_schema(cone_deg, min_facing_s, max_gap_s)
    for t, a, b, angle, dist in rows:
        _finite([t, angle, dist], f"the angle from face:{a} to face:{b} at t={t}")
    columns = list(zip(*rows)) if rows else [[] for _ in schema.names]
    data = {"t": list(columns[0]), "from": [f"face:{a}" for a in columns[1]], "to": [f"face:{b}" for b in columns[2]],
            "angle": list(columns[3]), "dist": list(columns[4])}
    with _writer(path, schema, ["from", "to"], ["t", "from", "to"]) as writer:
        writer.write_table(pa.table(data, schema=schema), row_group_size=ANGLES_GROUP_ROWS)


def read_angles(path) -> list[tuple]:
    """The angles file's rows as facing.pair_angles gives them: (t, A, B, angle, dist), faces as numbers, in file
    order; facing.events_from_angles thresholds them."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    return list(zip(table["t"].to_pylist(), map(face_number, table["from"].to_pylist()),
                    map(face_number, table["to"].to_pylist()), table["angle"].to_pylist(), table["dist"].to_pylist()))
