"""How loading the pipeline fails (diarizer.py), with torch, huggingface_hub and pyannote stood in for: a refused or
failed download is a NodeError the coordinator can act on, never an AttributeError, and telemetry is off before
pyannote is imported."""

import os
import sys
import types

import pytest

import diarizer
from perception_node import NodeError


class HubError(Exception):
    pass


class GatedRepoError(HubError):
    pass


class RepositoryNotFoundError(HubError):
    pass


class LocalEntryNotFoundError(Exception):
    pass


@pytest.fixture
def stubs(monkeypatch):
    """Stand-ins for the modules load() imports; `answer` is what Pipeline.from_pretrained does (a value, or an
    exception to raise)."""
    state = types.SimpleNamespace(answer=None, metrics_when_imported=None, mps=True, cuda=False,
                                  constants=types.SimpleNamespace(HF_HUB_OFFLINE=False))

    class Pipeline:
        @staticmethod
        def from_pretrained(repo, revision=None, token=None):
            if isinstance(state.answer, BaseException):
                raise state.answer
            return state.answer

    def import_pyannote():
        state.metrics_when_imported = os.environ.get("PYANNOTE_METRICS_ENABLED")
        return types.SimpleNamespace(Pipeline=Pipeline)

    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: state.mps))
    torch.cuda = types.SimpleNamespace(is_available=lambda: state.cuda)
    torch.device = lambda name: name
    errors = types.ModuleType("huggingface_hub.errors")
    errors.GatedRepoError, errors.HfHubHTTPError = GatedRepoError, HubError
    errors.RepositoryNotFoundError, errors.LocalEntryNotFoundError = RepositoryNotFoundError, LocalEntryNotFoundError
    pyannote = types.ModuleType("pyannote")
    audio = types.ModuleType("pyannote.audio")
    audio.__getattr__ = lambda name: import_pyannote().Pipeline if name == "Pipeline" else None
    monkeypatch.setitem(sys.modules, "torch", torch)
    hub = types.ModuleType("huggingface_hub")
    hub.constants = state.constants   # HF_HUB_OFFLINE: whether the node reads the cache alone (its prepare downloads)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    monkeypatch.setitem(sys.modules, "pyannote", pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)
    return state


class Loaded:
    def to(self, device):
        self.device = device
        return self


def test_the_pipeline_is_loaded_on_the_requested_device_with_telemetry_off(stubs):
    stubs.answer = Loaded()
    pipeline, device = diarizer.load("hf_x", "mps")
    assert device == "mps" and pipeline.device == "mps"
    assert stubs.metrics_when_imported == "false"


def test_pyannote_returning_no_pipeline_is_a_licence_error_not_an_attribute_error(stubs):
    """from_pretrained is typed Optional: it returns None when it cannot get the pipeline's config."""
    stubs.answer = None
    with pytest.raises(NodeError, match="gated") as e:
        diarizer.load("hf_x", "cpu")
    assert e.value.code == "licence"


@pytest.mark.parametrize("raised, code", [(GatedRepoError("403"), "licence"), (RepositoryNotFoundError("401"), "input"),
                                          (HubError("502"), "transient"), (LocalEntryNotFoundError("offline"), "transient")])
def test_a_refused_or_failed_download_is_a_node_error(stubs, raised, code):
    stubs.answer = raised
    with pytest.raises(NodeError) as e:
        diarizer.load("hf_x", "cpu")
    assert e.value.code == code


def test_a_pipeline_missing_from_the_cache_offline_is_said_so_not_retried_as_a_failed_download(stubs):
    stubs.constants.HF_HUB_OFFLINE = True   # the node: its prepare downloads (download_secrets)
    stubs.answer = LocalEntryNotFoundError("offline")
    with pytest.raises(NodeError, match="not in this machine's Hugging Face cache") as e:
        diarizer.load(None, "cpu")
    assert e.value.code == "resources"


@pytest.mark.parametrize("accelerator", ["mps", "cuda"])
def test_a_device_torch_cannot_use_is_refused_not_replaced_by_the_cpu(stubs, accelerator):
    stubs.mps = stubs.cuda = False
    stubs.answer = Loaded()
    with pytest.raises(NodeError) as e:
        diarizer.load("hf_x", accelerator)
    assert e.value.code == "resources"


class Segment:
    def __init__(self, start, end):
        self.start, self.end = start, end


class Annotation:
    """pyannote.core's Annotation, as far as run() reads it."""
    def __init__(self, turns):
        self.turns = turns

    def itertracks(self, yield_label=False):
        return [(Segment(s, e), i, label) for i, (s, e, label) in enumerate(self.turns)]

    def labels(self):
        return sorted({label for _, _, label in self.turns})


def test_run_keeps_everything_the_pipeline_returns(stubs):
    """Its overlapping turns (the voices' source), its exclusive turns and its speaker embeddings, in its orders."""
    import numpy as np
    sys.modules["torch"].from_numpy = lambda a: a
    turns = [(0.5, 2.0, "SPEAKER_01"), (1.0, 1.5, "SPEAKER_00"), (2.0, 3.0, "SPEAKER_00")]
    output = types.SimpleNamespace(speaker_diarization=Annotation(turns),
                                   exclusive_speaker_diarization=Annotation([turns[0], turns[2]]),
                                   speaker_embeddings=np.array([[0.25] * 4, [-0.5] * 4]))
    given = {}

    def pipeline(file):
        given.update(file)
        return output

    found = diarizer.run(pipeline, np.zeros(16000, np.float32))
    assert given["sample_rate"] == 16000 and given["waveform"].shape == (1, 16000)
    assert found == {"speakers": ["SPEAKER_00", "SPEAKER_01"],
                     "turns": [{"start": s, "end": e, "speaker": label} for s, e, label in turns],
                     "exclusive": [{"start": s, "end": e, "speaker": label} for s, e, label in (turns[0], turns[2])],
                     "embeddings": [[0.25] * 4, [-0.5] * 4]}
