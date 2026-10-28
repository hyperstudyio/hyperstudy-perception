"""Writes gaze_heatmaps.parquet beside this file with the gaze node's own writer: two heads of face:0 at t = 0.0 and
0.125, the samples test.faces writes. The orchestrator has no pyarrow, so its tests copy these bytes. From the repo root:

    uv run --no-project --with numpy --with pyarrow==26.0.0 python orchestrator/tests/nodes/gaze/make_fixture.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "nodes" / "vision" / "gaze"))
from heatmaps import Writer  # noqa: E402

y, x = np.mgrid[0:64, 0:64]
with Writer(HERE / "gaze_heatmaps.parquet") as w:
    for j, (t, cy, cx) in enumerate([(0.0, 20, 30), (0.125, 22, 31)]):
        heat = (0.01 + 0.5 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / 20)).astype(np.float32)
        w.add("face:0", j, t, heat, 0.9)
