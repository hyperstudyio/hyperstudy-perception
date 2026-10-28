"""Writes faces_pyfeat.parquet and faces_untracked.parquet beside this file with vision.face-track's own writers
(pyfeat_raw.py): face:0's two samples at t = 0.0 and 0.125, the samples test.faces writes, and one face in no track.
The orchestrator has no pyarrow, so its tests copy these bytes. From the repo root:

    uv run --no-project --with numpy --with pyarrow==26.0.0 python orchestrator/tests/nodes/faces/make_fixture.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "nodes" / "vision" / "face-track"))
import pyfeat_raw  # noqa: E402

AUS = [f"AU{n:02d}" for n in (1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43)]
with pyfeat_raw.samples_writer(HERE / "faces_pyfeat.parquet") as w:
    width = len(pyfeat_raw.flat(pyfeat_raw.sample_columns()))
    for j, (t, frame) in enumerate([(0.0, 0), (0.125, 4)]):
        w.add("face:0", j, t, frame, np.linspace(0, 1, width) + j)
with pyfeat_raw.untracked_writer(HERE / "faces_untracked.parquet", AUS) as w:
    w.add(4, 0.125, "size", np.linspace(0, 1, len(pyfeat_raw.flat(pyfeat_raw.untracked_columns(AUS)))))
