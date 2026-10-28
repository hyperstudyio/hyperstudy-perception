"""The delivery captions' raw side file, layers/prosody_captions.parquet: per utterance, what was sent to the OpenAI API
and what came back, of which the layer keeps the parsed description, tags, emphasised words, tagger tags and costs
(schema: #/$defs/parquet/prosody_captions).

One row per utterance, in the layer's order: the snippet's times (its audio is not kept), the user text sent with it,
and for each of the two calls (gpt-audio-1.5's caption, gpt-5.4-mini's tags) the model that answered, the finish
reason, the reply as returned (the function call's arguments, the tagger's content: JSON strings) and the API's usage
object (as JSON). The file's metadata holds what every row shares, as JSON under `request`: the models, the
instructions (system prompts), the tag vocabulary and notes, the context and the prices. Parquet rather than JSON: about
4 MB of text an hour of conversation, which zstd shrinks to about a quarter.

Captions are sampled, so a rerun's file differs; the writer's options are fixed all the same (a stored artifact is the
same bytes for the same rows), and the file carries no time.
"""

from __future__ import annotations

import json

FILE = "layers/prosody_captions.parquet"
DESCRIPTION = ("per utterance, the prompt sent to gpt-audio-1.5 and the raw replies of it and of the gpt-5.4-mini tagger "
               "(arguments, finish reason, token usage); the instructions and prices in the metadata")
COLUMNS = ("utt", "snippet_start", "snippet_end", "prompt", "caption_model", "caption_finish_reason", "caption_arguments",
           "caption_usage", "tagger_model", "tagger_finish_reason", "tagger_content", "tagger_usage")
ZSTD_LEVEL = 19
ROW_GROUP_ROWS = 65536


def schema(request: dict):
    import pyarrow as pa
    floats = {"snippet_start", "snippet_end"}
    return pa.schema([pa.field(c, pa.float64() if c in floats else pa.string(), nullable=False) for c in COLUMNS],
                     metadata={"side": "prosody_captions", "request": json.dumps(request, sort_keys=True)})


def row(utt: str, snippet: tuple[float, float], prompt: str, caption: dict, tagger: dict) -> dict:
    """One utterance's row. caption: {model, finish_reason, arguments, usage}; tagger: {model, finish_reason, content,
    usage}; the usages as the SDK dumps them (dicts), stored as JSON with sorted keys."""
    return {"utt": utt, "snippet_start": float(snippet[0]), "snippet_end": float(snippet[1]), "prompt": prompt,
            "caption_model": caption["model"], "caption_finish_reason": caption["finish_reason"] or "",
            "caption_arguments": caption["arguments"], "caption_usage": json.dumps(caption["usage"], sort_keys=True),
            "tagger_model": tagger["model"], "tagger_finish_reason": tagger["finish_reason"] or "",
            "tagger_content": tagger["content"], "tagger_usage": json.dumps(tagger["usage"], sort_keys=True)}


def write(path, rows: list[dict], request: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    s = schema(request)
    table = pa.table({c: pa.array([r[c] for r in rows], s.field(c).type) for c in COLUMNS}, schema=s)
    with pq.ParquetWriter(path, s, compression="zstd", compression_level=ZSTD_LEVEL, use_dictionary=["caption_model", "tagger_model"],
                          write_statistics=["utt"], version="2.6", data_page_version="1.0", store_schema=True,
                          write_page_index=False, use_byte_stream_split=False) as w:
        w.write_table(table, row_group_size=ROW_GROUP_ROWS)
