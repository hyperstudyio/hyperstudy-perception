"""The delivery captions' logic (style_captions.py): the snippets and prompts the bake-off sent, its prices, the replies
checked against the tag list, and the prosody layer with the captions in."""

import json
from pathlib import Path

import pytest

import captions_raw
import style_captions as sc

WORDS = [{"w": "Blue", "start": 6.0, "end": 6.3}, {"w": "kettle.", "start": 6.4, "end": 7.0},
         {"w": "orange", "start": 7.36, "end": 7.68}, {"w": "lamp.", "start": 7.84, "end": 8.48},
         {"w": "done.", "start": 13.92, "end": 63.0}]
TRANSCRIPT = {"words": WORDS, "utterances": [
    {"id": "utt:2", "start": 6.0, "end": 7.0, "text": "Blue kettle.", "words": [0, 1]},
    {"id": "utt:3", "start": 7.36, "end": 8.48, "text": "orange lamp.", "words": [2, 3]},
    {"id": "utt:4", "start": 13.92, "end": 63.0, "text": "done.", "words": [4, 4]}]}
PROSODY = {"layer": "prosody", "producer": {"stage": "prosody", "models": []}, "caption_models": {},
           "raw": {"tracks": {"file": "layers/prosody_tracks.parquet", "format": "parquet", "description": "frames"}},
           "utterances": [{"utt": "utt:2", "measures": {"duration": 1.0}, "captions": {}},
                          {"utt": "utt:3", "measures": {"duration": 1.12}, "captions": {}},
                          {"utt": "utt:4", "measures": {"duration": 1.2}, "captions": {}}]}
N = 16000 * 119


def test_word_cap_is_the_prosody_nodes():
    assert sc.WORD_MAX_S == 1.2   # prosody_units.WORD_MAX_S: the analysed end both nodes compute


def test_each_target_is_heard_with_a_second_each_side_and_its_end_is_the_prosody_layers():
    units = sc.targets(TRANSCRIPT, PROSODY, N)
    u = units[1]
    assert u["analysed_end"] == 8.48 and u["snippet"] == pytest.approx((6.36, 9.48))   # the bake-off's utt:3 snippet
    assert u["samples"] == (int(6.36 * 16000), int(9.48 * 16000))
    assert units[2]["analysed_end"] == 15.12   # "done." stretched over 49 s: analysed for 1.2 s
    near_start = sc.targets({"words": [{"w": "hi", "start": 0.4, "end": 0.6}],
                             "utterances": [{"id": "utt:0", "start": 0.4, "end": 0.6, "text": "hi", "words": [0, 0]}]},
                            {"utterances": [{"utt": "utt:0", "measures": {"duration": 0.2}}]}, int(0.9 * 16000))
    assert near_start[0]["snippet"] == (0.0, 0.9)   # cut at the clip's ends


def test_a_prosody_layer_from_another_transcript_is_refused():
    with pytest.raises(ValueError, match="not the transcript's"):
        sc.targets(TRANSCRIPT, {**PROSODY, "utterances": PROSODY["utterances"][:2]}, N)
    off = json.loads(json.dumps(PROSODY))
    off["utterances"][1]["measures"]["duration"] = 1.0
    with pytest.raises(ValueError, match="utt:3: the prosody layer measured 1.0 s"):
        sc.targets(TRANSCRIPT, off, N)


def test_the_prompt_is_the_bakeoffs():
    units = sc.targets(TRANSCRIPT, PROSODY, N)
    assert sc.prompt(units, 1) == (
        "Snippet length 3.12 s. The target utterance runs from 1.00 s to 2.12 s of the snippet.\n"
        "Target transcript (automatic, may contain errors): \"orange lamp.\"\n"
        "Previous line in the conversation: \"Blue kettle.\"\nNext line: \"done.\"")
    assert "Previous line in the conversation: \"(none)\"" in sc.prompt(units, 0)
    assert "Next line: \"(none)\"" in sc.prompt(units, 2)


def test_cost_is_the_bakeoffs_at_its_prices():
    assert sc.cost(sc.GPT_MODEL, 740, 31, 86) == {"text_in": 709, "audio_in": 31, "out": 86, "usd": pytest.approx(0.0036245)}
    assert sc.cost(sc.TAGGER, 440, None, 18) == {"text_in": 440, "audio_in": 0, "out": 18, "usd": pytest.approx(0.000411)}


def test_replies_must_be_the_schemas_with_tags_from_the_list():
    ok = {"description": "Calm.", "tags": ["calm", "falling-end"], "emphasized_words": []}
    assert sc.parse_caption(json.dumps(ok)) == ok
    with pytest.raises(ValueError, match="outside the vocabulary"):
        sc.parse_caption(json.dumps({**ok, "tags": ["calm", "grumpy"]}))
    with pytest.raises(ValueError, match="not {description, tags, emphasized_words}"):
        sc.parse_caption(json.dumps({"description": "x"}))
    assert sc.parse_tags('{"tags": ["neutral"]}') == ["neutral"]
    with pytest.raises(ValueError):
        sc.parse_tags('{"tags": "neutral"}')
    assert len(sc.ALL_TAGS) == len(set(sc.ALL_TAGS)) == 57   # the bake-off's 10 groups


def test_the_layer_gets_the_captions_and_keeps_the_prosody_nodes_pointers():
    entry = sc.caption_entry({"description": "Calm.", "tags": ["calm"], "emphasized_words": ["lamp"]}, (6.36, 9.48),
                             {"usd": 0.003}, ["calm", "neutral"], {"usd": 0.0004})
    assert entry == {"description": "Calm.", "tags": ["calm"], "emphasized_words": ["lamp"], "snippet": [6.36, 9.48],
                     "usage": {"usd": 0.003}, "tags_from_text": ["calm", "neutral"], "tagger_usage": {"usd": 0.0004}}
    pointer = {"file": captions_raw.FILE, "format": "parquet", "description": captions_raw.DESCRIPTION}
    doc = sc.refine(PROSODY, [entry] * 3, {"stage": "prosody+captions"}, pointer)
    assert doc["producer"] == {"stage": "prosody+captions"}
    assert doc["caption_models"] == {"gpt": {"model": "gpt-audio-1.5", "licence_mode": "api-terms"}}
    assert all(u["captions"] == {"gpt": entry} for u in doc["utterances"])
    assert doc["raw"]["tracks"] == PROSODY["raw"]["tracks"]
    assert doc["raw"]["captions"] == pointer and pointer["file"] == "layers/prosody_captions.parquet"
    assert PROSODY["utterances"][0]["captions"] == {}   # the input is not changed
    with pytest.raises(ValueError):
        sc.refine(PROSODY, [entry], {}, pointer)


def raw_rows():
    usage = {"completion_tokens": 95, "prompt_tokens": 740, "prompt_tokens_details": {"audio_tokens": 31, "cached_tokens": 0}}
    return [captions_raw.row(f"utt:{k}", (6.36 + k, 9.48 + k), f"prompt {k}",
                             {"model": "gpt-audio-1.5-2026-01-01", "finish_reason": "stop",
                              "arguments": '{"description": "Calm.", "tags": ["calm"], "emphasized_words": []}', "usage": usage},
                             {"model": "gpt-5.4-mini-2026-01-01", "finish_reason": None, "content": '{"tags": ["calm"]}', "usage": usage})
            for k in range(3)]


def test_the_raw_file_has_the_schemas_columns_the_request_and_is_the_same_bytes_twice(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/prosody); CI installs it")
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    captions_raw.write(a, raw_rows(), sc.request())
    captions_raw.write(b, raw_rows(), sc.request())
    assert a.read_bytes() == b.read_bytes()
    spec = json.loads((Path(__file__).resolve().parents[3] / "schema" / "bundle-0.1.schema.json").read_text())["$defs"]["parquet"]["prosody_captions"]
    schema = pq.read_schema(a)
    assert [(f.name, str(f.type)) for f in schema] == [(c["name"], c["arrow"]) for c in spec["columns"]]
    assert not any(f.nullable for f in schema)
    request = json.loads(schema.metadata[b"request"])
    assert request["instructions"] == sc.GPT_INSTRUCTIONS and request["tags"] == sc.STYLE_TAGS and request["prices_usd_per_mtok"] == sc.PRICES
    t = pq.read_table(a).to_pylist()
    assert t[1]["utt"] == "utt:1" and t[1]["snippet_start"] == 7.36 and t[1]["tagger_finish_reason"] == ""
    assert sc.parse_caption(t[0]["caption_arguments"])["tags"] == ["calm"]   # the layer's caption, from the file
    assert json.loads(t[0]["caption_usage"])["prompt_tokens_details"]["audio_tokens"] == 31
