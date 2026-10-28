"""Parakeet's tokens into the layer's words and the tokens side file (raw_tokens.py): the words are the ones the layer
always had, each token knows its word, and the Parquet file has the columns the bundle schema documents and is the
same bytes for the same tokens. Run: uv run --no-project --with pytest --with numpy --with pyarrow==26.0.0
--with-editable ./sdk pytest nodes/audio/asr"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from raw_tokens import COLUMNS, DESCRIPTION, FILE, ROW_GROUP_ROWS, to_words, write

SCHEMA = Path(__file__).resolve().parents[3] / "schema" / "bundle-0.1.schema.json"


@dataclass
class Token:   # the fields of parakeet-mlx's alignment.AlignedToken that the node reads
    id: int
    text: str
    start: float
    duration: float
    confidence: float = 1.0
    end: float = field(init=False)

    def __post_init__(self):
        self.end = self.start + self.duration


@dataclass
class Sentence:
    tokens: list


def frames(*tokens):
    """(text, start frame, duration frames, confidence) -> Tokens at Parakeet's 0.08 s a frame, as it computes times."""
    return [Token(100 + k, text, f * 0.08, d * 0.08, c) for k, (text, f, d, c) in enumerate(tokens)]


def old_to_words(sentence):
    """asr 1.0.0's word merge, which made the layer's words: the new one must give the same words."""
    words = []
    for token in sentence.tokens:
        if not words or token.text.startswith(" "):
            words.append({"w": token.text.strip(), "start": token.start, "end": token.end})
        else:
            words[-1]["w"] += token.text
            words[-1]["end"] = token.end
    return [{**w, "start": round(w["start"], 3), "end": round(w["end"], 3)} for w in words if w["w"]]


SENTENCES = [
    Sentence(frames((" As", 2, 2, 0.97), (" rec", 8, 2, 0.9999), ("ent", 10, 2, 0.98), (" tr", 12, 1, 0.9),
                    ("end", 13, 1, 0.99), ("s", 14, 2, 0.95), (".", 16, 0, 0.6))),
    # a lone space token starts a word that the next token, with no space, fills; a trailing lone space is no word
    Sentence(frames((" ", 20, 1, 0.7), ("Yeah", 21, 3, 0.8), ("?", 24, 1, 0.75), (" ", 25, 4, 0.65))),
    Sentence(frames(("Okay", 30, 4, 1.0))),   # a sentence starting with no space still starts a word
]


def test_the_words_are_the_ones_the_layer_always_had_and_each_token_knows_its_word():
    words, rows = to_words(SENTENCES)
    assert words == [w for s in SENTENCES for w in old_to_words(s)]
    assert [w["w"] for w in words] == ["As", "recent", "trends.", "Yeah?", "Okay"]
    assert [r["word"] for r in rows] == [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, -1, 4]
    assert [r["sentence"] for r in rows] == [0] * 7 + [1] * 4 + [2]
    tokens = [t for s in SENTENCES for t in s.tokens]
    assert [r["id"] for r in rows] == [t.id for t in tokens] and [r["text"] for r in rows] == [t.text for t in tokens]
    assert [r["confidence"] for r in rows] == [t.confidence for t in tokens]
    # times to 3 places, as the layer's: 0.08 s frames, so 0.24000000000000002 is 0.24
    assert rows[3]["start"] == 0.96 and rows[0]["end"] == 0.32 and rows[6]["start"] == rows[6]["end"] == 1.28
    # a layer word's text, start and end come back from its tokens
    for i, word in enumerate(words):
        mine = [r for r in rows if r["word"] == i]
        assert mine[0]["text"].strip() + "".join(r["text"] for r in mine[1:]) == word["w"]
        assert (mine[0]["start"], mine[-1]["end"]) == (word["start"], word["end"])


def test_no_tokens_is_no_words_and_a_confidence_that_is_not_finite_is_refused():
    assert to_words([]) == ([], [])
    with pytest.raises(ValueError, match=r"token 1 \('b'\): its confidence nan is not finite"):
        to_words([Sentence(frames((" a", 0, 1, 0.9), ("b", 1, 1, float("nan"))))])


def test_the_file_has_the_columns_the_bundle_schema_documents(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/audio); CI installs it")
    words, rows = to_words(SENTENCES)
    assert write(tmp_path / "transcript_tokens.parquet", rows) == len(rows)
    table = pq.read_table(tmp_path / "transcript_tokens.parquet")
    spec = json.loads(SCHEMA.read_text())["$defs"]["parquet"]["transcript_tokens"]
    assert [(f.name, str(f.type)) for f in table.schema] == [(c["name"], c["arrow"]) for c in spec["columns"]]
    assert tuple(table.schema.names) == COLUMNS and all(not f.nullable for f in table.schema)
    for c in ("id", "text", "start", "end", "sentence", "word"):
        assert table[c].to_pylist() == [r[c] for r in rows], c
    # confidences as float32, as parakeet-mlx computes them (in float32, returned as Python floats)
    assert table["confidence"].to_pylist() == [float(np.float32(r["confidence"])) for r in rows]
    assert FILE == "layers/transcript_tokens.parquet" and DESCRIPTION


def test_the_same_tokens_are_the_same_bytes(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/audio); CI installs it")
    rng = np.random.default_rng(0)
    rows = [{"id": int(rng.integers(0, 8192)), "text": f" w{k % 300}", "start": round(k * 0.08, 3),
             "end": round((k + 2) * 0.08, 3), "confidence": float(np.float32(rng.uniform(0.6, 1))), "sentence": k // 12,
             "word": k} for k in range(ROW_GROUP_ROWS + 7)]
    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write(first, rows)
    write(second, rows)
    assert first.read_bytes() == second.read_bytes()
    meta = pq.ParquetFile(first).metadata
    assert meta.num_rows == len(rows) and meta.num_row_groups == 2   # groups of ROW_GROUP_ROWS
    assert meta.row_group(0).column(0).compression == "ZSTD"


def test_no_tokens_is_a_file_with_no_rows(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/audio); CI installs it")
    path = tmp_path / "transcript_tokens.parquet"
    assert write(path, []) == 0
    data = path.read_bytes()
    assert data[:4] == data[-4:] == b"PAR1"
    table = pq.read_table(path)
    assert table.num_rows == 0 and tuple(table.schema.names) == COLUMNS
