"""The side files (facing_raw.py): the Parquet files have the columns the bundle schema documents, hold the poses and
angles as the node computed them, give back the layer's events when thresholded again, and are the same bytes for the
same rows."""

import json
from pathlib import Path

import numpy as np
import pytest

from facing import NAMES, TORSO, body_poses, events_from_angles, facing_events, pair_angles
from facing_raw import (ANGLES_DESCRIPTION, ANGLES_FILE, POSES_DESCRIPTION, POSES_FILE, ROW_GROUP_ROWS, BadRow,
                        PosesWriter, read_angles, write_angles)
from test_facing import random_frames, world

SCHEMA = Path(__file__).resolve().parents[3] / "schema" / "bundle-0.1.schema.json"
MODEL = "test-mediapipe"


def columns(name):
    return [(c["name"], c["arrow"]) for c in json.loads(SCHEMA.read_text())["$defs"]["parquet"][name]["columns"]]


def pq():
    return pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side files (envs/vision); CI installs it")


def crop_poses(seed=0):
    """body_poses' output for a crop holding two poses, with float32 landmarks as MediaPipe returns them."""
    rng = np.random.default_rng(seed)
    raw = []
    for yaw in (-25.0, 140.0):
        lms = rng.uniform(0, 1, (33, 5))
        lms[list(TORSO), 3] = 0.9
        lms = lms.astype(np.float32).astype(float)
        raw.append((lms, (world(yaw) + rng.normal(0, 0.01, (33, 3))).astype(np.float32).astype(float)))
    return body_poses(raw, (10, 20, 300), [40, 40, 240, 260])


def write_poses(path, rows=1):
    poses, _ = crop_poses()
    with PosesWriter(path, MODEL) as writer:
        for k in range(rows):
            for pose in poses:
                writer.add(f"body:{k % 3}", k // 3, round(k / 8, 3), 7 + k % 3, (10, 20, 300), pose)
    return poses


def test_the_files_have_the_columns_the_bundle_schema_documents(tmp_path):
    parquet = pq()
    poses = write_poses(tmp_path / "p.parquet")
    write_angles(tmp_path / "a.parquet", [(0.0, 3, 4, 12.5, 200.0)], 60.0, 1.0, 0.3)
    p, a = parquet.read_table(tmp_path / "p.parquet"), parquet.read_table(tmp_path / "a.parquet")
    assert [(f.name, str(f.type)) for f in p.schema] == columns("facing_poses")
    assert [(f.name, str(f.type)) for f in a.schema] == columns("facing_angles")
    assert all(not f.nullable for f in [*p.schema, *a.schema])
    assert p.num_rows == 2 and p["body"].to_pylist() == ["body:0"] * 2 and p["face"].to_pylist() == ["face:7"] * 2
    assert p["pose"].to_pylist() == [0, 1] and p["kept"].to_pylist() == [pose["kept"] for pose in poses]
    assert p["window"].to_pylist() == [[10, 20, 300]] * 2
    # the landmarks and world landmarks exactly as the landmarker returned them (float32), 33 x 5 and 33 x 3
    assert np.array_equal(np.array(p["landmarks"].to_pylist()[1]).reshape(33, 5), poses[1]["landmarks"])
    assert np.array_equal(np.array(p["world"].to_pylist()[0]).reshape(33, 3), poses[0]["world"])
    # the torso direction and centre unrounded (float64)
    assert p["yaw"].to_pylist() == [pose["yaw"] for pose in poses] and p["source"].to_pylist() == ["torso"] * 2
    assert p["centre"].to_pylist() == [pose["centre"].tolist() for pose in poses]
    meta = p.schema.metadata
    assert meta[b"model"] == MODEL.encode() and meta[b"names"].decode().split(",") == NAMES
    assert a.to_pylist() == [{"t": 0.0, "from": "face:3", "to": "face:4", "angle": 12.5, "dist": 200.0}]
    assert (a.schema.metadata[b"cone_deg"], a.schema.metadata[b"min_facing_s"], a.schema.metadata[b"max_gap_s"]) == (
        b"60.0", b"1.0", b"0.3")
    assert POSES_FILE == "layers/facing_poses.parquet" and ANGLES_FILE == "layers/facing_angles.parquet"
    assert POSES_DESCRIPTION and ANGLES_DESCRIPTION


def test_a_pose_with_no_facing_is_stored_as_none_with_zero_angles(tmp_path):
    parquet = pq()
    poses, _ = crop_poses()
    shoulders = {**poses[0], "pitch": None, "source": "shoulders"}
    none = {**poses[1], "yaw": None, "pitch": None, "source": None}
    with PosesWriter(tmp_path / "p.parquet", MODEL) as writer:
        writer.add("body:0", 0, 0.0, 1, (0, 0, 10), shoulders)
        writer.add("body:0", 0, 0.0, 1, (0, 0, 10), none)
    table = parquet.read_table(tmp_path / "p.parquet")
    assert table["source"].to_pylist() == ["shoulders", "none"]
    assert table["yaw"].to_pylist() == [poses[0]["yaw"], 0.0] and table["pitch"].to_pylist() == [0.0, 0.0]


@pytest.mark.parametrize("seed", range(4))
def test_thresholding_the_angles_file_gives_the_events_again(tmp_path, seed):
    pq()
    frames = random_frames(seed)
    rows = pair_angles(frames)
    write_angles(tmp_path / "a.parquet", rows)
    back = read_angles(tmp_path / "a.parquet")
    assert back == rows   # float64, exactly
    assert events_from_angles(back) == facing_events(frames)
    assert events_from_angles(back, 45.0, 0.5, 0.5) == facing_events(frames, 45.0, 0.5, 0.5)


def test_no_poses_or_angles_are_empty_files_with_the_columns(tmp_path):
    parquet = pq()
    with PosesWriter(tmp_path / "p.parquet", MODEL):
        pass
    write_angles(tmp_path / "a.parquet", [])
    for name, file in (("facing_poses", "p.parquet"), ("facing_angles", "a.parquet")):
        table = parquet.read_table(tmp_path / file)
        assert table.num_rows == 0 and [(f.name, str(f.type)) for f in table.schema] == columns(name)
    assert read_angles(tmp_path / "a.parquet") == []


def test_the_same_rows_are_the_same_bytes(tmp_path):
    parquet = pq()
    rows = ROW_GROUP_ROWS // 2 + 5   # two poses a row: more than one row group
    for name in ("1", "2"):
        write_poses(tmp_path / f"p{name}.parquet", rows)
        write_angles(tmp_path / f"a{name}.parquet", pair_angles(random_frames(3)))
    assert (tmp_path / "p1.parquet").read_bytes() == (tmp_path / "p2.parquet").read_bytes()
    assert (tmp_path / "a1.parquet").read_bytes() == (tmp_path / "a2.parquet").read_bytes()
    meta = parquet.ParquetFile(tmp_path / "p1.parquet").metadata
    assert meta.num_rows == 2 * rows and meta.num_row_groups == 2 and meta.row_group(0).column(0).compression == "ZSTD"


@pytest.mark.parametrize("damage", [
    lambda p: {**p, "yaw": float("nan")},
    lambda p: {**p, "centre": np.array([np.inf, 0.0])},
    lambda p: {**p, "landmarks": np.vstack([p["landmarks"][:32], [[0, 0, 0, np.nan, 1]]])},
    lambda p: {**p, "landmarks": p["landmarks"][:32]},   # 32 landmarks
    lambda p: {**p, "world": p["world"][:, :2]},
    lambda p: {**p, "source": "elbows"},
])
def test_a_pose_the_file_cannot_hold_is_refused(tmp_path, damage):
    pq()
    poses, _ = crop_poses()
    with pytest.raises(BadRow), PosesWriter(tmp_path / "p.parquet", MODEL) as writer:
        writer.add("body:0", 0, 0.0, 1, (0, 0, 10), damage(poses[0]))


def test_an_angle_that_is_not_finite_is_refused(tmp_path, monkeypatch):
    pq()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(BadRow):
        write_angles("a.parquet", [(0.0, 1, 2, float("nan"), 3.0)])
    assert not (tmp_path / "a.parquet").exists()
