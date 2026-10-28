"""audio.asr on an NVIDIA GPU (or a CPU): the same transcript layer and tokens side file as node.py, from NVIDIA NeMo's
Parakeet TDT 0.6B v3 (nemo_model.py) instead of parakeet-mlx's conversion of it.

NeMo's tokens become parakeet-mlx's tokens and sentences (nemo_tokens.py), and from there everything is node.py's: the
words (raw_tokens.to_words), the utterances (utterances.py), the side file (raw_tokens.write), whose metadata says
NeMo computed the confidences (raw_tokens.NEMO_CONFIDENCE).

The two runners' transcripts differ where their log-mel features do: parakeet-mlx 0.5.3 takes each STFT bin's
magnitude as |re| + |im| (audio.get_logmel), NeMo, which the model was trained with, as sqrt(re^2 + im^2). On OOO
(NeMo on a CPU) 311 of the MLX runner's 330 words come out the same, in 6 places they differ; given NeMo's features,
parakeet-mlx decodes NeMo's 669 tokens exactly, with confidences within 6e-4 for 95% of them.
"""

import os
import time

import nemo_tokens
import raw_tokens
from perception_node import Node, NodeError
from utterances import utterances
from wav import read_wav


with Node() as node:
    # cuBLAS is deterministic only with a fixed workspace, which it reads when CUDA starts: before torch is imported
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import nemo_model
    import torch

    model_id = node.config["model"]
    device = nemo_model.device_for(node.accelerator)
    t0 = time.time()
    model = nemo_model.load(model_id, device)
    rate = int(model.cfg.preprocessor.sample_rate)
    samples = read_wav(node.input("audio"), rate)
    if len(samples) < int(model.cfg.preprocessor.window_stride * rate):
        raise NodeError("input", f"the audio has {len(samples)} samples: too short to transcribe")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    out = nemo_model.transcribe(model, samples, device)
    seconds = round(time.time() - t0, 1)
    node.model(nemo_model.REPO, licence_mode="commercial-ok", source=nemo_model.SOURCE, revision=nemo_model.REVISION,
               licence="CC-BY-4.0")

    try:
        ratio = nemo_tokens.time_ratio(out["subsampling_factor"], out["sample_rate"], out["window_stride"])
        tokens = nemo_tokens.tokens(out["ids"], out["frames"], out["durations"], out["confidences"], out["vocabulary"],
                                    ratio, out["durations_allowed"])
        words, rows = raw_tokens.to_words(nemo_tokens.sentences(tokens))
    except ValueError as e:   # a hypothesis that is not one entry per token, or a confidence that is not finite
        raise NodeError("internal", f"NeMo's output: {e}") from None
    words = [{**w, "voice": None} for w in words]
    raw_tokens.write(node.path("tokens", "transcript_tokens.parquet"), rows,
                     raw_tokens.NEMO_CONFIDENCE.format(device=device.type))
    shots = node.input_json("shots")["shots"]
    node.write_json("transcript", "transcript.json", {
        "layer": "transcript", "schema_version": "0.1.0",
        "producer": node.producer(timings={"seconds": seconds}, device=device.type),
        "voices": [], "words": words, "utterances": utterances(words, shots),
        "raw": {"tokens": {"file": raw_tokens.FILE, "format": "parquet", "description": raw_tokens.DESCRIPTION}}})
    usage = {"audio_seconds": round(words[-1]["end"], 1) if words else 0, "words": len(words), "tokens": len(rows)}
    if device.type == "cuda":
        usage["gpu"] = torch.cuda.get_device_name(device)
        usage["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    node.usage(**usage)
