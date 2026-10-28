"""Writes bodies_scores.parquet and bodies_duplicates.parquet beside this file with the body-pose node's own writers:
the scores of node.py's three samples (body:0 at t = 0.0 and 0.125, body:1 at 0.0) and one duplicate of body:1's
sample. The orchestrator has no pyarrow, so its tests copy these bytes. From the repo root:

    uv run --no-project --with numpy --with pyarrow==26.0.0 python orchestrator/tests/nodes/bodies/make_fixture.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "nodes" / "vision" / "body-pose"))
from detections import write_duplicates, write_scores  # noqa: E402

write_scores(HERE / "bodies_scores.parquet", [("body:0", 0, 0.0, 0.9), ("body:1", 0, 0.0, 0.8), ("body:0", 1, 0.125, 0.85)],
             "test-yolox", 0.3)
write_duplicates(HERE / "bodies_duplicates.parquet", [(0.0, "body:1", 0, 0.4, 0.02, [1.0, 1.0, 2.0, 4.0], [1.0] * 399)],
                 "test-yolox", 0.3, 0.1, 3.0)
