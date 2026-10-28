"""Writes facing_poses.parquet and facing_angles.parquet beside this file with the facing node's own writers: two
poses of one crop (body:0's sample 0 at t = 0.0, linked to face:0; the second kept) and the angle between two people at
t = 0.0. The orchestrator has no pyarrow, so its tests copy these bytes. From the repo root:

    uv run --no-project --with numpy --with pyarrow==26.0.0 python orchestrator/tests/nodes/facing/make_fixture.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "nodes" / "vision" / "facing"))
from facing import body_poses  # noqa: E402
from facing_raw import PosesWriter, write_angles  # noqa: E402

landmarks = np.tile([0.5, 0.5, -0.25, 0.75, 0.875], (33, 1))
world = np.zeros((33, 3))
world[[11, 12, 23, 24]] = [[0.25, -0.5, 0.0], [-0.25, -0.5, 0.0], [0.125, 0.0, 0.0], [-0.125, 0.0, 0.0]]
poses, _ = body_poses([(landmarks * [0.25, 0.25, 1, 1, 1], world), (landmarks, world)], (0, 0, 8), [2, 2, 4, 4])
with PosesWriter(HERE / "facing_poses.parquet", "test-mediapipe") as writer:
    for pose in poses:
        writer.add("body:0", 0, 0.0, 0, (0, 0, 8), pose)
write_angles(HERE / "facing_angles.parquet", [(0.0, 0, 1, 30.5, 120.0), (0.0, 1, 0, 150.25, 120.0)])
