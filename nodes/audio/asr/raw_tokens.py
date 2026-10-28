"""The transcript layer's words from Parakeet's tokens, and its raw side file, layers/transcript_tokens.parquet: every
token Parakeet returned (parakeet-mlx on the Apple GPU, NeMo on a CUDA GPU or a CPU: node_nemo.py, nemo_tokens.py), of
which the layer keeps only the merged word text and times (schema: #/$defs/parquet/transcript_tokens).

parakeet-mlx 0.5.3 returns, per token (alignment.AlignedToken): its vocabulary id, its text, its start (the encoder
frame it was emitted at, 0.08 s a frame), its duration (TDT's predicted duration class, 0-4 frames) and a confidence,
1 - H / ln(V + 1) of the softmax over the 8192-token vocabulary and the blank at the step that emitted it
(parakeet.ParakeetTDT.decode_greedy). It groups the tokens into sentences at . ? ! (alignment.tokens_to_sentences);
a sentence's confidence is the geometric mean of its tokens'. Greedy decoding has no alternative hypotheses, and
the blank steps' scores are not returned.

One row per token, in order: id, text, start and end (the layer's times: seconds to 3 places), confidence (float32 as
computed), sentence (Parakeet's sentence index) and word (the layer's word index, -1 for a token in no word). to_words
merges the tokens into the layer's words, recording each token's word, so the file reproduces the layer's words.

NeMo 3.0.0 returns the same per token from its own greedy TDT decoding (nemo_tokens.py turns them into parakeet-mlx's
tokens and sentences), with the same confidence measure configured: its entropy method, Gibbs entropy, alpha 1, linear
normalisation, 1 + sum(p ln p) / ln(V + 1), over the vocabulary and the blank at the emitting step (exclude_blank).
The file's metadata says which computed it (MLX_CONFIDENCE, NEMO_CONFIDENCE).

Deterministic: identical tokens give identical bytes. Every writer option that shapes the file is fixed below, and
the file carries no time; the pyarrow version (envs/audio's and envs/asr-cuda's locks, part of the node's cache key)
names itself in the footer. About 15 bytes a token after compression.
"""

from __future__ import annotations

import math

ROW_GROUP_ROWS = 65536     # tokens a row group: about three hours of dense speech
ZSTD_LEVEL = 19
FILE = "layers/transcript_tokens.parquet"
DESCRIPTION = ("Parakeet's tokens (vocabulary id, text, start, end, entropy confidence, sentence) and the layer word "
               "each makes up, a row per token")
COLUMNS = ("id", "text", "start", "end", "confidence", "sentence", "word")
# the file metadata's "confidence": how the confidences were computed, by which runner
MLX_CONFIDENCE = ("1 - H / ln(V + 1) of the token's softmax over the vocabulary and the blank (parakeet-mlx 0.5.3, "
                  "greedy TDT decoding; the model in bfloat16 on the Apple GPU)")
NEMO_CONFIDENCE = ("1 - H / ln(V + 1) of the token's softmax over the vocabulary and the blank (NeMo 3.0.0, "
                   "greedy_batch TDT decoding; confidence method entropy, gibbs, alpha 1, lin, exclude_blank; the "
                   "model in float32 on {device})")


def to_words(sentences) -> tuple[list[dict], list[dict]]:
    """Parakeet's sentences -> (the layer's words, a row per token). A token with a leading space starts a new word,
    the others extend the current one, never across a sentence; a word whose text is empty (only spaces) is dropped
    and its tokens' word is -1."""
    words, rows = [], []
    for s, sentence in enumerate(sentences):
        groups: list[tuple[dict, list[dict]]] = []
        for token in sentence.tokens:
            if not math.isfinite(token.confidence):
                raise ValueError(f"token {len(rows)} ({token.text!r}): its confidence {token.confidence} is not finite")
            row = {"id": token.id, "text": token.text, "start": round(token.start, 3), "end": round(token.end, 3),
                   "confidence": token.confidence, "sentence": s, "word": -1}
            rows.append(row)
            if not groups or token.text.startswith(" "):
                groups.append(({"w": token.text.strip(), "start": token.start, "end": token.end}, [row]))
            else:
                word, members = groups[-1]
                word["w"] += token.text
                word["end"] = token.end
                members.append(row)
        for word, members in groups:
            if word["w"]:
                for row in members:
                    row["word"] = len(words)
                words.append({"w": word["w"], "start": round(word["start"], 3), "end": round(word["end"], 3)})
    return words, rows


def schema(confidence: str):
    import pyarrow as pa
    return pa.schema([pa.field("id", pa.int32(), nullable=False), pa.field("text", pa.string(), nullable=False),
                      pa.field("start", pa.float64(), nullable=False), pa.field("end", pa.float64(), nullable=False),
                      pa.field("confidence", pa.float32(), nullable=False),
                      pa.field("sentence", pa.int32(), nullable=False), pa.field("word", pa.int32(), nullable=False)],
                     metadata={"side": "transcript_tokens", "confidence": confidence})


def write(path, rows: list[dict], confidence: str) -> int:
    """Write the token rows (to_words') to `path` as Parquet, `confidence` saying how their confidences were computed
    (MLX_CONFIDENCE, NEMO_CONFIDENCE); returns the number of rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({c: [r[c] for r in rows] for c in COLUMNS}, schema=schema(confidence))
    with pq.ParquetWriter(path, table.schema, compression="zstd", compression_level=ZSTD_LEVEL, use_dictionary=["text"],
                          write_statistics=["start", "end", "sentence", "word"], version="2.6", data_page_version="1.0",
                          store_schema=True, write_page_index=False, use_byte_stream_split=False) as writer:
        writer.write_table(table, row_group_size=ROW_GROUP_ROWS)
    return table.num_rows
