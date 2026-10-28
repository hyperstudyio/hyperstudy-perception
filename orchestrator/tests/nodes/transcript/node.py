from pathlib import Path

from perception_node import Node

# transcript_tokens.parquet: written by the asr node's own merge and writer (make_fixture.py), for these words
FIXTURE = Path(__file__).parent / "transcript_tokens.parquet"
WORDS = [{"w": "Hello", "start": 0.16, "end": 0.4}, {"w": "world.", "start": 0.48, "end": 0.72},
         {"w": "Yes", "start": 0.96, "end": 1.28}]

with Node() as node:
    node.model("test-asr", licence_mode="commercial-ok")
    node.path("tokens", "transcript_tokens.parquet").write_bytes(FIXTURE.read_bytes())
    words = [{**w, "voice": None} for w in WORDS]
    node.write_json("transcript", "transcript.json", {
        "layer": "transcript", "schema_version": "0.1.0", "producer": node.producer(), "voices": [], "words": words,
        "utterances": [{"id": "utt:0", "start": 0.16, "end": 0.72, "text": "Hello world.", "shot": "shot:0",
                        "words": [0, 1], "voice": None},
                       {"id": "utt:1", "start": 0.96, "end": 1.28, "text": "Yes", "shot": "shot:0", "words": [2, 2],
                        "voice": None}],
        "raw": {"tokens": {"file": "layers/transcript_tokens.parquet", "format": "parquet",
                           "description": "Parakeet's tokens, a row per token"}}})
