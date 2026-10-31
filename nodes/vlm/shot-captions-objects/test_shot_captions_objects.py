"""vlm.shot-captions-objects (run-anywhere 1b, B) against the two steps it would replace, with Claude stood in for: for
the same answers, its vocabulary is the names step's (vision.objects-names) byte for byte, and its shots and text
layers are the captions step's (vlm.shot-captions). The stand-in answers by what it is asked (the shot, from the
prompt or the keyframes' bytes), never by when.
Run: uv run --no-project --with pytest --with numpy --with pillow --with-editable ./sdk pytest -q nodes/vlm/shot-captions-objects"""

import base64
import io
import json
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
CAPTIONS = HERE.parent / "shot-captions"
OBJECTS = HERE.parents[1] / "vision" / "objects"

# what Claude names in each shot: repeats across shots, a garment and "person" (the body pass's own), a blank
NAMED = {0: ["Chair", "mug", "jacket", "helmet", " "], 1: ["chair", "poster", "person", "helmet", "chair"]}
CLASSES = {"chair": ("chair", "individual", False), "mug": ("cup", "individual", False),
           "helmet": ("helmet", "individual", True), "poster": ("poster", "individual", False)}


def caption_of(shot: int) -> dict:
    return {"description": f"Two people talk in shot {shot}.", "setting": "a studio", "mood": "calm",
            "people": [{"label": "0", "appearance": "a red top", "action": "talking", "speaking": True}],
            "onscreen_text": ["OOO"] if shot == 0 else []}


class Anthropic:
    """The client: a caption (with the shot's names when its schema asks for objects), the names alone (the names
    step: the shot found by its keyframes' bytes), or the names' classes. calls: every request, in order."""
    calls: list = []
    shot_of_image: dict = {}

    def __init__(self, api_key):
        assert api_key == "key"
        self.messages = self

    def create(self, **request):
        self.calls.append(request)
        content = request["messages"][0]["content"]
        text = content[-1]["text"]
        schema = request["output_config"]["format"]["schema"]["properties"]
        if "description" in schema:
            shot = int(text.split("Shot ")[1].split(" ")[0])
            answer = {**caption_of(shot), **({"objects": NAMED[shot]} if "objects" in schema else {})}
        elif "objects" in schema:
            images = [base64.b64decode(b["source"]["data"]) for b in content if b["type"] == "image"]
            answer = {"objects": NAMED[self.shot_of_image[images[0]]]}
        else:
            names = text.split("Names:\n")[1].split("\n")
            answer = {"names": [dict(zip(("name", "family", "kind", "worn"), (n, *CLASSES[n]))) for n in names]}
        return SimpleNamespace(stop_reason="end_turn", stop_details=None,
                               content=[SimpleNamespace(type="text", text=json.dumps(answer))],
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=50))


@pytest.fixture
def stand_in(monkeypatch):
    m = types.ModuleType("anthropic")
    m.Anthropic = Anthropic
    for name in ("RateLimitError", "InternalServerError", "APIConnectionError", "AuthenticationError"):
        setattr(m, name, type(name, (Exception,), {}))
    monkeypatch.setitem(sys.modules, "anthropic", m)
    monkeypatch.setattr(Anthropic, "calls", [])
    monkeypatch.setattr(Anthropic, "shot_of_image", {})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    for folder in (CAPTIONS, OBJECTS):
        monkeypatch.syspath_prepend(str(folder))
    return Anthropic


def inputs(folder: Path) -> dict[str, Path]:
    """Two shots, each with two real keyframes, one face track a shot (person 0), and a word of speech in each."""
    from PIL import Image
    (folder / "keyframes" / "keyframes").mkdir(parents=True)
    shots = []
    for k in range(2):
        frames = []
        for i, x in enumerate("ab"):
            buf = io.BytesIO()
            Image.new("RGB", (64, 48), (40 * k + 20 * i, 90, 160)).save(buf, format="JPEG")
            (folder / "keyframes" / "keyframes" / f"{k}_{x}.jpg").write_bytes(buf.getvalue())
            Anthropic.shot_of_image[buf.getvalue()] = k
            frames.append({"file": f"assets/keyframes/{k}_{x}.jpg", "t": k + 0.25 + 0.5 * i})
        shots.append({"id": f"shot:{k}", "index": k, "start": float(k), "end": float(k + 1), "keyframes": frames})
    docs = {"media": {"sha256": "0" * 64, "duration": 2.0, "width": 64, "height": 48, "fps": 10},
            "shots": {"layer": "shots", "schema_version": "0.1.0", "shots": shots},
            "faces": {"layer": "faces", "people": [{"id": "person:0", "first_seen": 0.1}],
                      "tracks": [{"id": f"face:{k}", "shot": f"shot:{k}", "person": "person:0",
                                  "samples": [{"t": k + 0.25, "box": [10, 8, 12, 14]}]} for k in range(2)]},
            "transcript": {"words": [{"w": "hello", "start": 0.3, "end": 0.6, "voice": "voice:0"},
                                     {"w": "again", "start": 1.4, "end": 1.7, "voice": "voice:0"}]}}
    for name, doc in docs.items():
        (folder / f"{name}.json").write_text(json.dumps(doc))
    return {"keyframes": folder / "keyframes", **{name: folder / f"{name}.json" for name in docs}}


def run(script: Path, folder: Path, given: dict, config: dict, monkeypatch) -> dict:
    """A node script's response, run here as the coordinator runs it (its request, its own folder)."""
    folder.mkdir(parents=True)
    request = folder / "request.json"
    request.write_text(json.dumps({"protocol": 1, "out_dir": str(folder / "out"), "accelerator": "cpu", "config": config,
                                   "secrets": ["ANTHROPIC_API_KEY"], "producer": {"stage": "captions", "stage_version": "1.0.0"},
                                   "inputs": {name: {"path": str(path)} for name, path in given.items()}}))
    monkeypatch.setattr(sys, "argv", [script.name, "--request", str(request)])
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit:   # how a node ends after writing a failed response
        pass
    response = json.loads((folder / "response.json").read_text())
    assert response["ok"], response["error"]
    return response


CONFIG = {"model": "claude-haiku-5-5", "effort": "medium"}


def test_its_vocabulary_is_the_names_steps_and_its_layers_are_the_captions_steps(tmp_path, monkeypatch, stand_in):
    given = inputs(tmp_path / "in")
    run(OBJECTS / "names.py", tmp_path / "names", {k: given[k] for k in ("shots", "keyframes", "media")}, {}, monkeypatch)
    run(CAPTIONS / "node.py", tmp_path / "captions", given, CONFIG, monkeypatch)
    plain = [c for c in stand_in.calls if "description" in c["output_config"]["format"]["schema"]["properties"]]
    stand_in.calls.clear()
    both = run(HERE / "node.py", tmp_path / "both", given, CONFIG, monkeypatch)
    out = lambda step, name: (tmp_path / step / "out" / name).read_bytes()
    assert out("both", "vocabulary.json") == out("names", "vocabulary.json")
    vocabulary = json.loads(out("both", "vocabulary.json"))
    assert vocabulary["shots"][1]["names"] == ["chair", "poster", "helmet"] and vocabulary["shots"][1]["body_named"] == ["person"]
    assert out("both", "shots.json") == out("captions", "shots.json") and out("both", "text.json") == out("captions", "text.json")
    # one call a shot, as the captions make, with the names part added; then the names' classes
    calls = stand_in.calls
    assert len(calls) == 3 and "names" in calls[2]["output_config"]["format"]["schema"]["properties"]
    for mine, theirs in zip(calls[:2], plain):
        assert mine["messages"][0]["content"][:-1] == theirs["messages"][0]["content"][:-1]   # the same keyframes
        assert mine["messages"][0]["content"][-1]["text"].startswith(theirs["messages"][0]["content"][-1]["text"])
    assert "Names used in earlier shots: (none yet)" in calls[0]["messages"][0]["content"][-1]["text"]
    # every name given before (cleaned; a garment too, as the names step tells it), never a later shot's
    assert "Names used in earlier shots: chair, helmet, jacket, mug" in calls[1]["messages"][0]["content"][-1]["text"]
    assert both["usage"]["calls"] == 2 and both["usage"]["class_calls"] == 1 and both["usage"]["usd"] > 0
