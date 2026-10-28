"""Writes the sounds side files beside this file with the sounds node's own writer, for 1.5 s of soundtrack: PANNs'
4 segments of 151 frames and mn40's 2 steps (sounds_tags.parquet), PE-A-Frame's 38 frames (sounds_queries.parquet),
and the same queries with one probability not a number (sounds_queries_nan.parquet, written past the writer's check,
for validate.py to refuse). The orchestrator has no pyarrow, so its tests copy these bytes. From the repo root:

    uv run --no-project --with numpy --with pyarrow==26.0.0 python orchestrator/tests/nodes/sounds/make_fixture.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "nodes" / "audio" / "sounds"))
from sound_events import QUERIES, audioset_labels  # noqa: E402
from sound_scores import _write, queries_schema, write_queries, write_tags  # noqa: E402

labels = audioset_labels()
panns, mn40 = np.zeros((4, len(labels)), np.float32), np.zeros((2, len(labels)), np.float32)
panns[:, 0], mn40[:, 0], panns[1:3, 16] = 0.8, 0.75, 0.5   # Speech throughout, Laughter in 0.32-0.96 s
write_tags(HERE / "sounds_tags.parquet", panns, 151, mn40, labels)
probs = np.zeros((len(QUERIES), 38), np.float32)
probs[0, 8:24] = 0.6                                          # laughter, 0.32-0.96 s
write_queries(HERE / "sounds_queries.parquet", probs, QUERIES)
bad = probs.T.copy()
bad[3, 1] = np.nan
_write(HERE / "sounds_queries_nan.parquet", queries_schema(QUERIES),
       {"start": [k / 25 for k in range(38)], "end": [(k + 1) / 25 for k in range(38)], "probs": bad}, "probs")
