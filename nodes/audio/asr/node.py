"""audio.asr: words with times, grouped into utterances, as the bundle's transcript layer.

Parakeet TDT 0.6B v3 (MLX, Apple silicon; node_nemo.py runs NVIDIA's NeMo on a CUDA GPU or a CPU). Utterances split
at a pause over 0.25 s, a sentence end, or a shot cut, as the annotator prototype did (utterances.py). Voices come from
the diarization node, not here: every word's voice is null.

Beside the layer, the side file layers/transcript_tokens.parquet keeps every token Parakeet returned: its vocabulary
id, text, times and confidence, its sentence, and the layer word it is part of (raw_tokens.py, which also merges the
tokens into the layer's words). The layer points at it with raw.tokens.
"""

import time

import raw_tokens
from perception_node import Node, NodeError
from utterances import utterances
from wav import read_wav


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

    try:
        words, tokens = raw_tokens.to_words(result.sentences)
    except ValueError as e:   # a confidence that is not finite
        raise NodeError("internal", f"Parakeet's output: {e}") from None
    words = [{**w, "voice": None} for w in words]
    raw_tokens.write(node.path("tokens", "transcript_tokens.parquet"), tokens, raw_tokens.MLX_CONFIDENCE)
    shots = node.input_json("shots")["shots"]
    node.write_json("transcript", "transcript.json", {
        "layer": "transcript", "schema_version": "0.1.0",
        "producer": node.producer(timings={"seconds": seconds}, device="mlx"),
        "voices": [], "words": words, "utterances": utterances(words, shots),
        "raw": {"tokens": {"file": raw_tokens.FILE, "format": "parquet", "description": raw_tokens.DESCRIPTION}}})
    node.usage(audio_seconds=round(words[-1]["end"], 1) if words else 0, words=len(words))
