"""How the track step's loads fail (models._from_hub), with huggingface_hub stood in for: a weight missing from the
cache is said so when the step reads the cache alone (offline: its prepare downloads), and is a failed download, worth
retrying, when it does not."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from perception_node import NodeError

HERE = Path(__file__).parent


class LocalEntryNotFoundError(Exception):
    pass


@pytest.fixture
def models(monkeypatch):
    errors = types.ModuleType("huggingface_hub.errors")
    errors.HfHubHTTPError = type("HfHubHTTPError", (Exception,), {})
    errors.GatedRepoError = type("GatedRepoError", (errors.HfHubHTTPError,), {})
    errors.RepositoryNotFoundError = type("RepositoryNotFoundError", (errors.HfHubHTTPError,), {})
    errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    hub = types.ModuleType("huggingface_hub")
    hub.constants = types.SimpleNamespace(HF_HUB_OFFLINE=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    spec = importlib.util.spec_from_file_location("objects_models_hub", HERE / "models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hub.constants


def missing():
    raise LocalEntryNotFoundError("not in the cache")


@pytest.mark.parametrize("offline, code, said", [(True, "resources", "not in this machine's Hugging Face cache"),
                                                 (False, "transient", "could not download")])
def test_a_weight_missing_from_the_cache_is_said_so_offline_and_retried_online(models, offline, code, said):
    module, constants = models
    constants.HF_HUB_OFFLINE = offline
    with pytest.raises(NodeError, match=said) as e:
        module._from_hub(missing, "facebook/sam3")
    assert e.value.code == code


def test_transformers_plain_oserror_offline_is_said_so_too_and_otherwise_left_alone(models):
    module, constants = models

    def transformers_miss():   # what from_pretrained raises offline for weights not in the cache (transformers 5.20.2)
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")
    constants.HF_HUB_OFFLINE = True
    with pytest.raises(NodeError, match="not in this machine's Hugging Face cache") as e:
        module._from_hub(transformers_miss, "facebook/dinov3-vitl16-pretrain-lvd1689m")
    assert e.value.code == "resources"
    constants.HF_HUB_OFFLINE = False
    with pytest.raises(OSError, match="couldn't connect"):
        module._from_hub(transformers_miss, "facebook/dinov3-vitl16-pretrain-lvd1689m")
