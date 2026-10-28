"""Writes transcript_tokens.parquet beside this file with the asr node's own merge and writer: the tokens of the three
words test.transcript writes (WORDS in node.py). The orchestrator has no pyarrow, so its tests copy these bytes. From
the repo root:

    uv run --no-project --with pyarrow==26.0.0 python orchestrator/tests/nodes/transcript/make_fixture.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "nodes" / "audio" / "asr"))
from raw_tokens import to_words, write  # noqa: E402


def token(id, text, frame, frames, confidence):
    return SimpleNamespace(id=id, text=text, start=frame * 0.08, end=(frame + frames) * 0.08, confidence=confidence)


sentences = [SimpleNamespace(tokens=[token(11, " Hel", 2, 2, 0.97), token(12, "lo", 4, 1, 0.99),
                                     token(13, " world", 6, 3, 0.9), token(14, ".", 9, 0, 0.8)]),
             SimpleNamespace(tokens=[token(15, " Yes", 12, 4, 0.75)])]
words, rows = to_words(sentences)
print(words)   # node.py's WORDS
write(HERE / "transcript_tokens.parquet", rows)
