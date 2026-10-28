"""How fetching the NeMo checkpoint fails (nemo_model.checkpoint), with huggingface_hub stood in for: a missing repo,
revision or file, a gated repo or a refused token is a terminal NodeError, never retried; only the network or the hub
failing is transient."""

import sys
import types

import pytest

import nemo_model
from perception_node import NodeError


class HfHubHTTPError(Exception):
    def __init__(self, status: int):
        super().__init__(f"{status} Client Error")
        self.response = types.SimpleNamespace(status_code=status)


class RepositoryNotFoundError(HfHubHTTPError):
    pass


class GatedRepoError(RepositoryNotFoundError):
    pass


class RevisionNotFoundError(HfHubHTTPError):
    pass


class RemoteEntryNotFoundError(HfHubHTTPError):
    pass


class LocalEntryNotFoundError(FileNotFoundError):
    pass


def offline() -> LocalEntryNotFoundError:
    """What hf_hub_download raises when the hub could not be reached (or answered 5xx) and the file is not cached."""
    try:
        raise LocalEntryNotFoundError("cannot find the requested files in the local cache") from ConnectionError("offline")
    except LocalEntryNotFoundError as e:
        return e


def refused_head(status: int) -> LocalEntryNotFoundError:
    """hf_hub_download wraps a HEAD answer it has no class for (a plain 403, a 5xx) the same way, the answer its cause."""
    try:
        raise LocalEntryNotFoundError("cannot find the requested files in the local cache") from HfHubHTTPError(status)
    except LocalEntryNotFoundError as e:
        return e


@pytest.fixture
def hub(monkeypatch):
    """huggingface_hub with hf_hub_download raising state.raised."""
    state = types.SimpleNamespace(raised=None)

    def hf_hub_download(repo, file, revision=None):
        raise state.raised

    errors = types.ModuleType("huggingface_hub.errors")
    for cls in (HfHubHTTPError, RepositoryNotFoundError, GatedRepoError, RevisionNotFoundError, RemoteEntryNotFoundError,
                LocalEntryNotFoundError):
        setattr(errors, cls.__name__, cls)
    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return state


@pytest.mark.parametrize("raised, code", [
    (GatedRepoError(403), "licence"),
    (RepositoryNotFoundError(401), "input"),
    (HfHubHTTPError(401), "input"),              # hf_hub_download raises a refused token as it gets it
    (HfHubHTTPError(403), "input"),
    (refused_head(403), "input"),
    (RevisionNotFoundError(404), "internal"),    # the pinned commit is gone: the node's pin needs changing
    (RemoteEntryNotFoundError(404), "internal"),  # the file was renamed or removed at that commit
    (HfHubHTTPError(502), "transient"),
    (HfHubHTTPError(429), "transient"),
    (refused_head(503), "transient"),
    (offline(), "transient"),
])
def test_only_the_network_or_the_hub_failing_is_retried(hub, raised, code):
    hub.raised = raised
    with pytest.raises(NodeError) as e:
        nemo_model.checkpoint("mlx-community/parakeet-tdt-0.6b-v3")
    assert e.value.code == code
    assert nemo_model.REPO in str(e.value)
