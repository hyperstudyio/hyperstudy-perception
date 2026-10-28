"""The gestures layer's raw side file, layers/gestures_evidence.parquet: every hand sample's measurements for every rule
and each rule's outcome, of which the layer keeps only the merged events (schema: #/$defs/parquet/gestures_evidence).
The events are made from these rows alone (gesture_rules.hand_events), so read back (read_rows) they give the layer
again, and a threshold can be revisited without re-running the node.

One row per sample of the hands layer, in its order (hands in layer order, each hand's samples in order). Columns:
  hand, sample, t        the hands-layer sample (its index in the hand's samples)
  valid, valid_mask      how many keypoints count (score >= kpt_score_threshold and inside the frame), and which
                         (bit i: keypoint i, in the hands layer's keypoint_names order)
  clasp_iou              IoU of the hand box with the same body's other hand at t (0 if none)
  face_status            ok | few_keypoints (fewer than min_valid_kpts valid: no rule is evaluated) | no_face (the hand
                         has no face track) | no_landmarks (no faces_detail landmarks or face box within max_face_gap_s)
  face_size              the face box side (px) the face rules scale by
  mouth_dist             each of the 21 keypoints' distance to the mouth centre, in face sizes (valid or not)
  mouth_near             valid keypoints within mouth_radius face sizes of the mouth centre
  mouth_in_hull          the mouth centre inside the valid keypoints' hull
  face_inside            valid keypoints inside the face region (landmark hull, forehead band, margin)
  hand_over_mouth, self_touch_face   the face rules' outcomes; score = mouth_near / valid, face_inside / valid
  point_status           ok | few_keypoints | clasped (clasp_iou >= clasped_iou) | missing_keypoints (a joint the
                         pointing rule needs is not valid)
  index_extension, index_reach, index_straightness, curl   the pointing rule's ratios (gesture_rules.finger_measures)
  pointing, direction, target   its outcome, the unit direction MCP -> tip and the face:N or obj:N pointed at ("" none)
  raise_status           ok | few_keypoints | no_body | no_wrist | no_shoulders_or_nose
  rise, shoulder_width   the wrist's rise over its same-side shoulder (px, the lesser of image up and body up) and the
                         shoulder width (px); raised needs rise > raise_margin x shoulder_width
  above_nose, raised_hand, level   wrist above the nose both ways, the outcome, and head | shoulder ("" not raised)
Values not measured (status not ok) are 0, false or "". The file metadata holds every threshold.

Parquet, not JSON: on OOO clip 1 (3657 hand samples in 119 s) the file is 134 KB, 37 bytes a sample and 4.0 MB an
hour, where compact JSON with values to 3 places was 2.3 MB (622 bytes a sample, 69 MB an hour).
Deterministic: identical rows give identical bytes. Every writer option that shapes the file is fixed below, and the
file carries no time; the pyarrow version (envs/core's lock, part of the node's cache key) names itself in the footer.
Rows are written in groups of ROW_GROUP_ROWS as they come.
"""

from __future__ import annotations

import math

import gesture_rules as rules

ROW_GROUP_ROWS = 16384
# OOO clip 1: zstd 3 135.8 KB, 9 133.8 KB, 19 132.8 KB
ZSTD_LEVEL = 9
FILE = "layers/gestures_evidence.parquet"
DESCRIPTION = ("Every hand sample's measurements and outcome for each gesture rule (mouth and face contact, finger "
               "ratios and pointing target, wrist rise over the shoulder), a row per hands-layer sample")
STATUS_COLUMNS = ("face_status", "point_status", "raise_status", "level", "hand", "target")
THRESHOLDS = ("KPT_SCORE_THRESHOLD", "MIN_VALID_KPTS", "MAX_FACE_GAP_S", "MIN_REGION_KPTS", "MOUTH_RADIUS", "FOREHEAD",
              "FACE_MARGIN", "EXTENDED_RATIO", "CURLED_RATIO", "STRAIGHT_COS", "MIN_INDEX_TO_PALM", "CLASPED_IOU",
              "RAISE_MARGIN", "MAX_EVENT_GAP_S", "MIN_EVENT_S")


def schema():
    import pyarrow as pa
    f32 = lambda n: pa.list_(pa.field("element", pa.float32(), nullable=False), n)
    columns = [("hand", pa.string()), ("sample", pa.int32()), ("t", pa.float64()), ("valid", pa.int32()),
               ("valid_mask", pa.int32()), ("clasp_iou", pa.float32()),
               ("face_status", pa.string()), ("face_size", pa.float32()), ("mouth_dist", f32(21)), ("mouth_near", pa.int32()),
               ("mouth_in_hull", pa.bool_()), ("face_inside", pa.int32()), ("hand_over_mouth", pa.bool_()),
               ("self_touch_face", pa.bool_()),
               ("point_status", pa.string()), ("index_extension", pa.float32()), ("index_reach", pa.float32()),
               ("index_straightness", pa.float32()), ("curl", f32(3)), ("pointing", pa.bool_()),
               ("direction", pa.list_(pa.field("element", pa.float64(), nullable=False), 2)), ("target", pa.string()),
               ("raise_status", pa.string()), ("rise", pa.float32()), ("shoulder_width", pa.float32()),
               ("above_nose", pa.bool_()), ("raised_hand", pa.bool_()), ("level", pa.string())]
    meta = {"side": "gestures_evidence", "hand_model": rules.HAND_MODEL,
            **{name.lower(): repr(getattr(rules, name)) for name in THRESHOLDS}}
    return pa.schema([pa.field(name, kind, nullable=False) for name, kind in columns], metadata=meta)


def stored(row: dict) -> dict:
    """A gesture_rules.sample_evidence row as its columns hold it: no None (a missing direction is [0, 0], a missing
    target ""), and every float finite (a ValueError otherwise: the row would not be read back as it was)."""
    out = {**row, "mouth_dist": [float(v) for v in row["mouth_dist"]], "curl": [float(v) for v in row["curl"]],
           "direction": [float(v) for v in row["direction"]] if row["direction"] is not None else [0.0, 0.0],
           "target": row["target"] or ""}
    floats = [out[k] for k in ("t", "clasp_iou", "face_size", "index_extension", "index_reach", "index_straightness",
                               "rise", "shoulder_width")] + out["mouth_dist"] + out["curl"] + out["direction"]
    if not all(math.isfinite(v) for v in floats):
        raise ValueError(f"{row['hand']} sample {row['sample']}: a measurement is not finite")
    return out


class Writer:
    """add(row) per hand sample (gesture_rules.sample_evidence), then close(); a with block closes it."""

    def __init__(self, path):
        import pyarrow.parquet as pq
        self.schema = schema()
        self._writer = pq.ParquetWriter(
            path, self.schema, compression="zstd", compression_level=ZSTD_LEVEL, use_dictionary=list(STATUS_COLUMNS),
            write_statistics=["hand", "sample", "t"], version="2.6", data_page_version="1.0", store_schema=True,
            write_page_index=False, use_byte_stream_split=False)
        self._rows: dict[str, list] = {name: [] for name in self.schema.names}
        self.rows = 0

    def add(self, row: dict):
        row = stored(row)
        for name in self.schema.names:
            self._rows[name].append(row[name])
        self.rows += 1
        if len(self._rows["hand"]) == ROW_GROUP_ROWS:
            self._flush()

    def _flush(self):
        import pyarrow as pa
        if self._rows["hand"]:
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


def read_rows(path) -> list[dict]:
    """The file's rows as gesture_rules rows (target None for "", direction None when not pointing): what
    gesture_rules.hand_events makes the layer's events from."""
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["target"] = row["target"] or None
        row["direction"] = row["direction"] if row["pointing"] else None
    return rows
