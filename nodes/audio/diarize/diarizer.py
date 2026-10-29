"""pyannote's speaker-diarization-community-1 pipeline, called through pyannote.audio directly (the bake-off called it
through WhisperX's DiarizationPipeline, which only wraps it: same pipeline, same in-memory waveform, same output).

The pipeline is gated on Hugging Face: the account whose HF_TOKEN the node's prepare downloads it with must have
accepted its conditions at https://hf.co/pyannote/speaker-diarization-community-1. The node reads it from the Hugging
Face cache at a pinned revision (segmentation, embedding and PLDA models, 31 MB in all), offline (HF_HUB_OFFLINE), with
no token.

pyannote.audio 4 sends usage telemetry (pipeline name, audio duration, speaker counts) to otel.pyannote.ai unless
PYANNOTE_METRICS_ENABLED is false; it is set false here, before pyannote is imported: nothing leaves the machine.

Where it runs (the request's accelerator): mps (Apple GPU), cuda or cpu. A device torch cannot use here is an error,
not a quiet run on the CPU. On the 119 s OOO clip, an M5 Max: 6 s on MPS (13 s with other jobs on the GPU), 47-53 s on
the CPU, with identical turns.
"""

from __future__ import annotations

import os
import warnings

from perception_node import NodeError
from pcm16 import RATE

REPO = "pyannote/speaker-diarization-community-1"
REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
LICENCE = "CC-BY-4.0 (gated on Hugging Face)"


def load(token: str | None, accelerator: str = "cpu"):
    """The pipeline on the accelerator's device. Its weights come from the Hugging Face cache, or are downloaded into
    it (with `token`)."""
    os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
    # torchcodec (pyannote's file decoder) warns on import that it cannot find FFmpeg's shared libraries: the node never
    # asks it to decode, it passes the waveform in memory (pcm16.py)
    warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly")   # the message starts with a newline
    import torch
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError
    from pyannote.audio import Pipeline

    device = {"mps": "mps", "cuda": "cuda"}.get(accelerator, "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise NodeError("resources", "torch cannot use the Apple GPU (MPS) here: run Speaker diarization on another device")
    if device == "cuda" and not torch.cuda.is_available():
        raise NodeError("resources", "torch cannot use CUDA here (no NVIDIA GPU or driver): run Speaker diarization on another device")
    refused = NodeError("licence", f"{REPO} is gated, and Hugging Face refused this machine's HF_TOKEN for it: check the "
                                   f"token, and that its account accepted the conditions at https://hf.co/{REPO}")
    try:
        pipeline = Pipeline.from_pretrained(REPO, revision=REVISION, token=token)
    except GatedRepoError:   # before RepositoryNotFoundError, which it is a kind of. Also what an invalid token gets
        raise refused from None
    except RepositoryNotFoundError:   # the repository is not visible to this token
        raise NodeError("input", f"Hugging Face did not accept this machine's HF_TOKEN for {REPO}") from None
    except LocalEntryNotFoundError as e:
        from huggingface_hub import constants
        if constants.HF_HUB_OFFLINE:   # the node: its prepare downloads, and did not
            raise NodeError("resources", f"{REPO} is not in this machine's Hugging Face cache, which the node reads alone: "
                                         "its prepare downloads it (perception setup, or a run with --allow-downloads)") from None
        raise NodeError("transient", f"could not download {REPO} from Hugging Face: {e}") from None
    except HfHubHTTPError as e:   # the network, or the hub, failed: not in the cache either
        raise NodeError("transient", f"could not download {REPO} from Hugging Face: {e}") from None
    except OSError:   # a library's own answer, offline, for a file not in the cache (transformers: no LocalEntryNotFoundError)
        from huggingface_hub import constants
        if constants.HF_HUB_OFFLINE:
            raise NodeError("resources", f"{REPO} is not in this machine's Hugging Face cache, which the node reads alone: "
                                         "its prepare downloads it (perception setup, or a run with --allow-downloads)") from None
        raise
    if pipeline is None:   # from_pretrained's other answer when it gets no config (pyannote 3 gave it for a gated repo)
        raise refused
    return pipeline.to(torch.device(device)), device


def run(pipeline, waveform) -> dict:
    """Everything the pipeline returns (its DiarizeOutput), as plain values:
    - speakers: its labels, in its labels() order (SPEAKER_00, ...)
    - turns: speaker turns, [{start, end, speaker}] in pyannote's order: what WhisperX's DiarizationPipeline returned
      (its `speaker_diarization`, where turns of two speakers may overlap)
    - exclusive: its `exclusive_speaker_diarization`, turns of at most one speaker at a time (where turns overlap, one
      speaker's)
    - embeddings: its `speaker_embeddings`, a row per speaker in `speakers` order, as nested lists (None if it gave none)
    """
    import torch
    output = pipeline({"waveform": torch.from_numpy(waveform[None, :]), "sample_rate": RATE})

    def turns(annotation):
        return [{"start": seg.start, "end": seg.end, "speaker": label} for seg, _, label in annotation.itertracks(yield_label=True)]

    embeddings = output.speaker_embeddings
    return {"speakers": list(output.speaker_diarization.labels()), "turns": turns(output.speaker_diarization),
            "exclusive": turns(output.exclusive_speaker_diarization),
            "embeddings": None if embeddings is None else embeddings.tolist()}
