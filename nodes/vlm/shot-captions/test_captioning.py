"""The captions call (captioning.caption) with Claude stood in for: a reply cut off at max_tokens is said so, not
parsed (its JSON is incomplete), and is still billed.
Run: uv run --no-project --with pytest --with-editable ./sdk pytest -q nodes/vlm/shot-captions"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def anthropic(monkeypatch):
    m = types.ModuleType("anthropic")
    for name in ("RateLimitError", "InternalServerError", "APIConnectionError", "AuthenticationError"):
        setattr(m, name, type(name, (Exception,), {}))
    monkeypatch.setitem(sys.modules, "anthropic", m)


def client(stop_reason: str, text: str):
    reply = SimpleNamespace(stop_reason=stop_reason, stop_details=None, content=[SimpleNamespace(type="text", text=text)],
                            usage=SimpleNamespace(input_tokens=1000, output_tokens=4000))
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **request: reply))


def test_a_reply_cut_off_at_max_tokens_is_said_so_and_billed(anthropic):
    import captioning
    from perception_node import NodeError
    billed = []
    with pytest.raises(NodeError, match="max_tokens") as e:
        captioning.caption(client("max_tokens", '{"description": "Two people'), "claude-haiku-5-5", "low", "prompt", [],
                           lambda n_in, n_out: billed.append((n_in, n_out)))
    assert e.value.code == "internal" and billed == [(1000, 4000)]
    assert captioning.caption(client("end_turn", '{"description": "x"}'), "claude-haiku-5-5", "low", "prompt", [],
                              lambda *_: None) == {"description": "x"}
