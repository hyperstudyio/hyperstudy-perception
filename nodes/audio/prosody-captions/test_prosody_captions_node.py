"""audio.prosody-captions run as the coordinator runs it, with OpenAI stood in for: every caption made once, then the
tags, then a reply that was not the schema's asked again once, only within what the step reserved (review of #53).
The stand-in answers by what it is asked (the target's line, from the prompt), never by when.
Run: uv run --no-project --with pytest --with numpy --with-editable ./sdk pytest -q nodes/audio/prosody-captions"""

import json
import runpy
import sys
import threading
import types
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import style_captions as sc
from test_style_captions import N, PROSODY, TRANSCRIPT

HERE = Path(__file__).resolve().parent
GOOD = {"description": "Calm and even.", "tags": ["calm"], "emphasized_words": []}


class OpenAI:
    """The client: a caption (the first one for `bad`'s line cut short, as gpt-audio's was on a pod), or tags. calls:
    every request, in order; usd: what each call is billed (prompt tokens, at the models' prices)."""
    calls: list = []
    bad: set = set()
    prompt_tokens = 100
    lock = threading.Lock()

    def __init__(self, api_key):
        assert api_key == "key"
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        usage = SimpleNamespace(prompt_tokens=self.prompt_tokens, completion_tokens=20,
                                prompt_tokens_details=SimpleNamespace(audio_tokens=0), model_dump=lambda mode: {})
        if "tools" in request:
            text = request["messages"][1]["content"][0]["text"]
            target = text.split('Target transcript (automatic, may contain errors): "')[1].split('"')[0]
            with self.lock:
                self.calls.append(request)
                asked = sum(1 for c in self.calls if "tools" in c and c["messages"][1]["content"][0]["text"] == text)
            arguments = '{"description": "Calm" "tags": []}' if target in self.bad and asked == 1 else json.dumps(GOOD)
            message = SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(arguments=arguments))])
            return SimpleNamespace(model=request["model"], usage=usage,
                                   choices=[SimpleNamespace(message=message, finish_reason="tool_calls")])
        with self.lock:
            self.calls.append(request)
        return SimpleNamespace(model=request["model"], usage=usage,
                               choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"tags": ["calm"]})),
                                                        finish_reason="stop")])


@pytest.fixture
def openai(monkeypatch):
    module = types.ModuleType("openai")
    module.OpenAI = OpenAI
    for name in ("RateLimitError", "InternalServerError", "APIConnectionError", "AuthenticationError"):
        setattr(module, name, type(name, (Exception,), {}))
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    OpenAI.calls, OpenAI.bad, OpenAI.prompt_tokens = [], set(), 100
    return OpenAI


def run(tmp_path, monkeypatch) -> dict:
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sc.RATE)
        w.writeframes(b"\0\0" * N)
    (tmp_path / "transcript.json").write_text(json.dumps(TRANSCRIPT))
    (tmp_path / "prosody.json").write_text(json.dumps(PROSODY))
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"protocol": 1, "out_dir": str(tmp_path / "out"), "accelerator": "cpu", "config": {},
                                   "secrets": ["OPENAI_API_KEY"], "producer": {"stage": "prosody", "stage_version": "1.0.0"},
                                   "inputs": {"audio": {"path": str(audio)}, "transcript": {"path": str(tmp_path / "transcript.json")},
                                              "prosody": {"path": str(tmp_path / "prosody.json")}}}))
    monkeypatch.setattr(sys, "argv", ["node.py", "--request", str(request)])
    try:
        runpy.run_path(str(HERE / "node.py"), run_name="__main__")
    except SystemExit:
        pass
    return json.loads((tmp_path / "response.json").read_text())


def test_a_reply_that_is_not_the_schemas_is_asked_again_after_every_other_call(tmp_path, monkeypatch, openai):
    openai.bad = {"orange lamp."}   # utt:3's line
    got = run(tmp_path, monkeypatch)
    assert got["ok"], got.get("error")
    kinds = ["caption" if "tools" in c else "tag" for c in openai.calls]
    # three captions, two tags (the good ones), then utt:3 asked again and tagged
    assert kinds == ["caption"] * 3 + ["tag"] * 2 + ["caption", "tag"]
    first = got["usage"]["asked_again"]["utt:3"]
    assert "delimiter" in first["problem"] and first["arguments"] == '{"description": "Calm" "tags": []}'
    assert got["usage"]["calls"] == 7


def test_a_reply_is_not_asked_again_past_the_steps_reservation(tmp_path, monkeypatch, openai):
    openai.bad = {"orange lamp."}
    openai.prompt_tokens = 4000   # each call near the utterance's reservation: a second caption would pass it
    got = run(tmp_path, monkeypatch)
    assert not got["ok"] and "not asked again" in got["error"]["message"] and "utt:3" in got["error"]["message"]
    assert ["caption" if "tools" in c else "tag" for c in openai.calls] == ["caption"] * 3 + ["tag"] * 2
