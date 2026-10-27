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
    state = types.SimpleNamespace(answer=None, metrics_when_imported=None, mps=True, cuda=False)

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
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub"))
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


@pytest.mark.parametrize("accelerator", ["mps", "cuda"])
def test_a_device_torch_cannot_use_is_refused_not_replaced_by_the_cpu(stubs, accelerator):
    stubs.mps = stubs.cuda = False
    stubs.answer = Loaded()
    with pytest.raises(NodeError) as e:
        diarizer.load("hf_x", accelerator)
    assert e.value.code == "resources"
