"""audio.asr: words with times, grouped into utterances, as the bundle's transcript layer.

Parakeet TDT 0.6B v3 (MLX, Apple silicon). Utterances split at a pause over 0.25 s, a sentence end, or a shot cut,
as the annotator prototype did (pipeline/bakeoff/label_page.py). Voices come from the diarization node, not here:
every word's voice is null.
"""

import time

from perception_node import Node, NodeError
from wav import read_wav

PAUSE_SPLIT = 0.25            # seconds of silence between words that starts a new utterance
SENTENCE_END = (".", "?", "!")


def to_words(sentence) -> list[dict]:
    """Merge Parakeet's subword tokens into words; a leading space starts a new word."""
    words = []
    for token in sentence.tokens:
        if not words or token.text.startswith(" "):
            words.append({"w": token.text.strip(), "start": token.start, "end": token.end})
        else:
            words[-1]["w"] += token.text
            words[-1]["end"] = token.end
    return [{**w, "start": round(w["start"], 3), "end": round(w["end"], 3)} for w in words if w["w"]]


def utterances(words: list[dict], shots: list[dict]) -> list[dict]:
    cuts = [s["start"] for s in shots[1:]]
    units, current = [], []
    for i, word in enumerate(words):
        if current:
            prev = words[current[-1]]
            crosses_cut = any(prev["end"] <= c <= word["start"] or prev["start"] < c <= word["start"] for c in cuts)
            if word["start"] - prev["end"] > PAUSE_SPLIT or prev["w"].endswith(SENTENCE_END) or crosses_cut:
                units.append(current)
                current = []
        current.append(i)
    if current:
        units.append(current)

    def shot_at(t):
        return next(s["id"] for s in reversed(shots) if s["start"] <= t)

    return [{"id": f"utt:{n}", "start": words[u[0]]["start"], "end": words[u[-1]]["end"],
             "text": " ".join(words[i]["w"] for i in u), "shot": shot_at((words[u[0]]["start"] + words[u[-1]]["end"]) / 2),
             "words": [u[0], u[-1]], "voice": None} for n, u in enumerate(units)]


with Node() as node:
    import mlx.core as mx
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.audio import get_logmel

    model_id = node.config["model"]
    t0 = time.time()
    model = from_pretrained(model_id)
    # what model.transcribe(path) does with no chunking, from the samples read here (wav.py)
    samples = read_wav(node.input("audio"), model.preprocessor_config.sample_rate)
    if len(samples) < model.preprocessor_config.hop_length:
        raise NodeError("input", f"the audio has {len(samples)} samples: too short to transcribe")
    audio = mx.array(samples).astype(mx.float32) / 32768.0
    result = model.generate(get_logmel(audio, model.preprocessor_config))[0]
    seconds = round(time.time() - t0, 1)
    node.model(model_id, licence_mode="commercial-ok", source=f"hf://{model_id}", licence="CC-BY-4.0")

    words = [{**w, "voice": None} for sentence in result.sentences for w in to_words(sentence)]
    shots = node.input_json("shots")["shots"]
    node.write_json("transcript", "transcript.json", {
        "layer": "transcript", "schema_version": "0.1.0",
        "producer": node.producer(timings={"seconds": seconds}, device="mlx"),
        "voices": [], "words": words, "utterances": utterances(words, shots)})
    node.usage(audio_seconds=round(words[-1]["end"], 1) if words else 0, words=len(words))
