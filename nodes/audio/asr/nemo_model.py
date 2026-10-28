"""Parakeet TDT 0.6B v3 through NVIDIA NeMo (envs/asr-cuda): the CUDA runner's model (node_nemo.py, prepare.py).

The weights are NVIDIA's own checkpoint, parakeet-tdt-0.6b-v3.nemo, at a pinned commit of nvidia/parakeet-tdt-0.6b-v3:
the file mlx-community/parakeet-tdt-0.6b-v3 (the MLX runner's) was converted from (its model card's base_model; the
.nemo file is unchanged since 2025-08-08, before that conversion, through this commit). The node's `model` setting
names the model by its MLX id for both runners; CHECKPOINTS maps it to the checkpoint NeMo reads.

Decoding as parakeet-mlx decodes (parakeet.ParakeetTDT.decode_greedy): greedy TDT over the whole recording in one pass
(no chunking), up to 10 symbols a frame (the model's max_symbols), with each token's confidence at the step that
emitted it measured as parakeet-mlx measures it (raw_tokens.NEMO_CONFIDENCE). NeMo's greedy_batch strategy (its
label-looping decoder, the model's default), not greedy: NeMo 3.0.0's per-frame greedy decoder can only rebuild token
confidences from alignments, and fails doing it (IndexError in compute_confidence, tried on OOO). CUDA graphs off:
they only make the same decoding faster, through cuda-bindings.

Float32 throughout, TF32 off (PyTorch 2.12 lets cuDNN's convolutions and LSTMs use TF32 by default), deterministic
algorithms on, so the same audio gives the same tokens and confidences, bit for bit, on the same device: measured on a
CPU (two runs on OOO, the same bytes); not yet on a GPU. The whole recording is one attention window, as on MLX: its
memory grows with the square of the length (not yet measured on a GPU; an hour is about 45,000 encoder frames).
"""

from __future__ import annotations

from perception_node import NodeError

REPO = "nvidia/parakeet-tdt-0.6b-v3"
REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
FILE = "parakeet-tdt-0.6b-v3.nemo"
CHECKPOINTS = {"mlx-community/parakeet-tdt-0.6b-v3": (REPO, REVISION, FILE)}
SOURCE = f"hf://{REPO}@{REVISION}/{FILE}"   # as the manifest declares it


def checkpoint(model_id: str):
    """The .nemo file for the node's `model` setting, from the Hugging Face cache (downloaded there when missing)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError, RemoteEntryNotFoundError,
                                        RepositoryNotFoundError, RevisionNotFoundError)
    if model_id not in CHECKPOINTS:
        raise NodeError("input", f"no NeMo checkpoint for model {model_id!r} (known: {', '.join(CHECKPOINTS)})")
    repo, revision, file = CHECKPOINTS[model_id]
    where = f"{repo}@{revision[:8]}/{file}"
    # HfHubHTTPError is the base of the permanent answers too (a missing repo, revision or file, a gated repo): only the
    # network or the hub failing is worth retrying (transient), the same request gets the same refusal
    try:
        return hf_hub_download(repo, file, revision=revision)
    except GatedRepoError:   # before RepositoryNotFoundError, which it is a kind of
        raise NodeError("licence", f"{repo} is gated, and Hugging Face refused this machine's HF_TOKEN (or none) for it: "
                                   f"check https://hf.co/{repo}") from None
    except RepositoryNotFoundError:   # deleted, renamed, or not visible to this token (an invalid token gets it too)
        raise NodeError("input", f"Hugging Face finds no repository {repo} for this machine's HF_TOKEN (or none)") from None
    except (RevisionNotFoundError, RemoteEntryNotFoundError) as e:   # the pinned commit or the file is gone upstream
        raise NodeError("internal", f"{where} is not on Hugging Face (the pinned commit or file is gone, so the pin in "
                                    f"nemo_model.py needs changing): {e}") from None
    except (HfHubHTTPError, LocalEntryNotFoundError) as e:
        # a HEAD answer hf_hub_download has no class for (a plain 403, a 5xx) comes wrapped in LocalEntryNotFoundError,
        # as its cause, as does a failed connection (offline, and not in the cache)
        answer = e if isinstance(e, HfHubHTTPError) else e.__cause__
        status = answer.response.status_code if isinstance(answer, HfHubHTTPError) else None
        if status is not None and 400 <= status < 500 and status not in (408, 429):   # refused: 401, 403, 404, ...
            raise NodeError("input", f"Hugging Face refused {where} ({status}): {e}") from None
        raise NodeError("transient", f"could not download {where} from Hugging Face: {e}") from None


def device_for(accelerator: str):
    """The torch device for the request's accelerator (cuda or cpu), with float32 kept exact and algorithms
    deterministic; NodeError when PyTorch cannot use it here."""
    import torch
    if accelerator == "cuda" and not torch.cuda.is_available():
        raise NodeError("resources", f"PyTorch {torch.__version__} (CUDA {torch.version.cuda}) finds no usable CUDA GPU here: "
                                     "run the transcript on another device")
    if accelerator not in ("cuda", "cpu"):
        raise NodeError("resources", f"NeMo's Parakeet runs on cuda or cpu, not {accelerator}")
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)   # cuBLAS's needs CUBLAS_WORKSPACE_CONFIG, set by node_nemo.py first
    return torch.device(accelerator)


def load(model_id: str, device):
    """NeMo's model in eval mode on `device`, decoding as the module docstring says."""
    from nemo.collections.asr.models import ASRModel
    from omegaconf import open_dict
    model = ASRModel.restore_from(checkpoint(model_id), map_location=device)
    model.eval()
    cfg = model.cfg.decoding
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        cfg.greedy.use_cuda_graph_decoder = False
        cfg.compute_timestamps = False       # token frames and durations are enough; words are raw_tokens.to_words'
        cfg.tdt_include_token_duration = True
        cfg.confidence_cfg = {"preserve_frame_confidence": True, "preserve_token_confidence": True, "exclude_blank": True,
                              "aggregation": "min", "tdt_include_duration": False,
                              "method_cfg": {"name": "entropy", "entropy_type": "gibbs", "alpha": 1.0, "entropy_norm": "lin"}}
    model.change_decoding_strategy(cfg, verbose=False)
    return model


def transcribe(model, samples, device) -> dict:
    """int16 samples at the model's rate -> the hypothesis as parallel lists (ids, frames, durations, confidences), the
    vocabulary, the frame's seconds parts and the model's durations: what nemo_tokens needs."""
    import torch
    audio = torch.from_numpy(samples.astype("float32") / 32768.0)[None].to(device)   # as the MLX runner scales them
    length = torch.tensor([audio.shape[1]], device=device)
    with torch.inference_mode():
        features, feature_lengths = model.preprocessor(input_signal=audio, length=length)
        encoded, encoded_lengths = model.encoder(audio_signal=features, length=feature_lengths)
        (hypothesis,) = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=encoded, encoded_lengths=encoded_lengths,
                                                                       return_hypotheses=True)
    as_list = lambda x: x.tolist() if torch.is_tensor(x) else list(x)
    pre = model.cfg.preprocessor
    return {"ids": as_list(hypothesis.y_sequence), "frames": as_list(hypothesis.timestamp),
            "durations": as_list(hypothesis.token_duration),
            "confidences": [float(c) for c in (hypothesis.token_confidence or [])],
            "vocabulary": list(model.cfg.joint.vocabulary), "durations_allowed": list(model.cfg.decoding.durations),
            "subsampling_factor": int(model.cfg.encoder.subsampling_factor), "sample_rate": int(pre.sample_rate),
            "window_stride": float(pre.window_stride), "frames_total": int(encoded_lengths[0])}
