"""Writes hands_detections.parquet beside this file with the hands node's own writer: node.py's detector hand
(hand:1, unlinked, at t = 0.0 and 0.125) from two of four detections, one under the score threshold and one a
duplicate. The orchestrator has no pyarrow, so its tests copy these bytes. From the repo root:

    uv run --no-project --with numpy --with pyarrow==26.0.0 python orchestrator/tests/nodes/hands/make_fixture.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "nodes" / "vision" / "hands"))
from hand_detections import columns, write  # noqa: E402

write(HERE / "hands_detections.parquet", columns([
    (0.0, 0.5, [1.0, 1.0, 2.0, 2.0], "unlinked", "", "", "hand:1", 0),
    (0.0, 0.125, [3.0, 3.0, 1.0, 1.0], "below_score", "", "", "", -1),
    (0.125, 0.75, [1.0, 1.0, 2.0, 2.0], "unlinked", "", "", "hand:1", 1),
    (0.125, 0.375, [1.5, 1.0, 2.0, 2.0], "duplicate", "", "", "", -1)]), "test-rtmdet", 4, 4)
