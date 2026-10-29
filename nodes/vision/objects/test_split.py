"""The objects steps (names.py, track.py, link.py) against the node they replace (vision.objects 1.1.0, kept as
testdata/node_1_1_0.py), run whole on a synthetic clip with SAM 3, DINOv3 and Haiku stood in for: for the same Haiku
answers the three steps make the same layer, side file and crops, byte for byte. The stand-ins are deterministic and
answer by what they are given, never by when they are asked, so a crop check whose calls finish in another order still
gets the same answers; a step that sent other images would get other answers and fail the comparison.
Run: uv run --no-project --with pytest --with numpy --with av==16.1.0 --with pillow --with pyarrow==26.0.0 --with opencv-python-headless==5.0.0.93 --with-editable ./sdk pytest -q nodes/vision/objects"""

import base64
import hashlib
import importlib.util
import json
import random
import runpy
import sys
import threading
import time
import types
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2", reason="geometry.py draws with OpenCV (envs/gaze); CI installs opencv-python-headless")
pytest.importorskip("pyarrow", reason="the masks side file is Parquet (envs/gaze); CI installs pyarrow")

HERE = Path(__file__).resolve().parent
W, H, FPS, SECONDS = 128, 96, 10, 2
# 1.1.0's settings at their defaults, in its manifest's order: the track step has the first two, the link step the rest
CONFIG = {"fps": 6, "clothing_fps": 3, "link_threshold": 0.62, "cross_name_margin": 0.05, "same_wearer_bonus": 0.05,
          "min_link_px": 48, "wearer_share": 0.5, "different_person_distance": 0.8, "duplicate_iou": 0.6}
TRACK_KEYS = ("fps", "clothing_fps")
PRODUCER = {"stage": "objects", "stage_version": "1.1.0", "models": []}   # what the coordinator gives the layer's step
# what Haiku names in each shot, by its first keyframe's bytes: repeats, a garment and "person" (the body pass's), a
# thing seen on one frame only (too short), a graphic the crop check rejects
NAMED = {b"keyframe 0 a": ["Chair", "mug", "jacket", "helmet", "lamp", " "],
         b"keyframe 1 a": ["chair", "poster", "person", "helmet", "chair"]}
CLASSES = {"chair": ("chair", "individual", False), "mug": ("cup", "individual", False),
           "helmet": ("helmet", "individual", True), "lamp": ("lamp", "lookalike", False),
           "poster": ("poster", "individual", False)}
# where each name is, as (x, y, w, h) in the frame; the chair is over body 0's person mask's edge (trimmed), the helmet
# on body 1's head (worn), the mug too small to embed
PLACES = {"chair": (10, 50, 35, 40), "mug": (100, 60, 12, 12), "helmet": (90, 2, 12, 10), "lamp": (60, 60, 30, 30),
          "poster": (60, 10, 25, 30)}
BODIES = ((40, "person:0"), (96, "person:1"))   # (centre x, person) in every shot


# --- the clip and its layers ----------------------------------------------------------------------------------------

def make_video(path):
    """2 s of 128x96 H.264 at 10 fps: a grey that brightens frame by frame, a red block (the chair) and a blue one."""
    import av
    with av.open(str(path), "w") as out:
        v = out.add_stream("libx264", rate=FPS)
        v.width, v.height, v.pix_fmt = W, H, "yuv420p"
        for i in range(FPS * SECONDS):
            rgb = np.full((H, W, 3), 40 + 5 * i, np.uint8)
            rgb[50:90, 10:45] = (200, 30, 30)
            rgb[10:40, 60:85] = (30, 30, 200)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, FPS)
            out.mux(v.encode(frame))
        out.mux(v.encode())


def figure(cx, top, scale=0.1):
    """133 COCO-WholeBody keypoints of a person standing at x = cx (test_garments.figure, smaller)."""
    from test_garments import figure as standing
    return standing(cx, top, scale)


def layers(folder: Path) -> dict[str, Path]:
    """The inputs: video, media, shots (two, with keyframes), keyframes, faces and bodies (two people in each shot)."""
    from test_garments import box_of
    folder.mkdir(parents=True)
    make_video(folder / "clip.mp4")
    shots = [{"id": f"shot:{k}", "index": k, "start": float(k), "end": float(k + 1),
              "keyframes": [{"file": f"assets/keyframes/{k}_{x}.jpg"} for x in "ab"]} for k in range(2)]
    (folder / "keyframes" / "keyframes").mkdir(parents=True)
    for k in range(2):
        for x in "ab":
            (folder / "keyframes" / "keyframes" / f"{k}_{x}.jpg").write_bytes(f"keyframe {k} {x}".encode())
    times = [round(0.1 * i, 3) for i in range(FPS * SECONDS)]
    faces, bodies = [], []
    for k in range(2):
        for b, (cx, person) in enumerate(BODIES):
            n = 2 * k + b
            at = [t for t in times if k <= t < k + 1]
            faces.append({"id": f"face:{n}", "shot": f"shot:{k}", "person": person,
                          "samples": [{"t": t, "box": [cx - 5, 9, 10, 11]} for t in at]})
            bodies.append({"id": f"body:{n}", "shot": f"shot:{k}", "face": f"face:{n}",
                           "samples": [{"t": t, "box": box_of(cx, 8, 0.1), "kpts": figure(cx, 8), "score": 0.9} for t in at]})
    docs = {"media": {"sha256": "0" * 64, "duration": 2.0, "width": W, "height": H, "fps": FPS, "time_base": "1/10",
                      "variable_frame_rate": False},
            "shots": {"layer": "shots", "shots": shots},
            "faces": {"layer": "faces", "tracks": faces, "person_pairs": [{"a": "person:0", "b": "person:1", "distance": 1.2}]},
            "bodies": {"layer": "bodies", "score_threshold": 3.0, "tracks": bodies}}
    for name, doc in docs.items():
        (folder / f"{name}.json").write_text(json.dumps(doc))
    return {"video": folder / "clip.mp4", "keyframes": folder / "keyframes",
            **{name: folder / f"{name}.json" for name in docs}}


# --- the stand-ins ---------------------------------------------------------------------------------------------------

def f32(x) -> float:
    """A score as SAM 3 gives one (a float32 tensor's .tolist())."""
    return float(np.float32(x))


def rect(h, w, r0, r1, c0, c1):
    m = np.zeros((h, w), bool)
    m[int(r0 * h):int(r1 * h), int(c0 * w):int(c1 * w)] = True
    return m


class CropDetector:
    """SAM 3's detector in each body crop: a t-shirt over the torso, a jacket under the keep score, pants, glasses at
    the crop's edge (nobody's), shoes on the brighter frames only (some too few to be a track), and the person, each
    where the crop's body has that region."""

    def __init__(self, segmenter, prompts, device, token):
        self.prompts = list(prompts)

    def detect(self, crops, min_score):
        out = []
        for crop in crops:
            h, w = crop.shape[:2]
            bright = crop.mean() > 75
            found = []
            for noun in self.prompts:
                if noun == "t-shirt":
                    found.append({"noun": noun, "score": f32(0.85), "mask": rect(h, w, .25, .45, .35, .65)})
                elif noun == "jacket":
                    found.append({"noun": noun, "score": f32(0.4), "mask": rect(h, w, .22, .5, .3, .7)})
                elif noun == "pants":
                    found.append({"noun": noun, "score": f32(0.75), "mask": rect(h, w, .55, .85, .38, .62)})
                elif noun == "glasses":   # in nobody's eye region: not this body's
                    found.append({"noun": noun, "score": f32(0.7), "mask": rect(h, w, .05, .15, .9, 1)})
                elif noun == "shoe" and bright:
                    found.append({"noun": noun, "score": f32(0.6), "mask": rect(h, w, .9, .96, .35, .45)})
                elif noun == "person":
                    found.append({"noun": noun, "score": f32(0.95), "mask": rect(h, w, .05, .95, .4, .6)})
            out.append([d for d in found if d["score"] >= min_score])
        return out


def segment_shot(segmenter, frames, times, nouns, device, on_mask):
    """SAM 3 video: each name where PLACES puts it, on every frame (the lamp on the first only), in frame order."""
    frames.clear()
    for i, t in enumerate(times):
        for j, noun in enumerate(nouns):
            if noun == "lamp" and i:
                continue
            x, y, w, h = PLACES[noun]
            mask = np.zeros((H, W), bool)
            mask[y:y + h, x:x + w] = True
            on_mask(t, noun, 100 * j, f32(0.9 - 0.05 * j), f32(0.8 + 0.01 * i), mask)


PROJECTION = np.random.default_rng(7).normal(size=(10, 16))


def embed_crops(model, crops, device):
    """DINOv3: a unit float32 vector a crop, from its colours and layout, and their normalised mean."""
    feats = []
    for c in crops:
        c = c.astype(np.float64) / 255
        f = [c[..., k].mean() for k in range(3)] + [c[:112].mean(), c[112:].mean(), c[:, :112].mean(), c[:, 112:].mean(),
                                                   (c == 128 / 255).mean(), c.std(), 1.0]
        feats.append(np.tanh(np.array(f) @ PROJECTION))
    feats = np.asarray(feats, np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    mean = feats.mean(0)
    return (mean / np.linalg.norm(mean)).astype(np.float32), feats


def models_module():
    """models.py's names, its constants as they are, its model calls stood in for."""
    spec = importlib.util.spec_from_file_location("objects_models_constants", HERE / "models.py")
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)
    m = types.ModuleType("models")
    for name in ("SEGMENTER", "SEGMENTER_REVISION", "SEGMENTER_LICENCE", "EMBEDDER", "EMBEDDER_REVISION", "EMBEDDER_LICENCE"):
        setattr(m, name, getattr(real, name))
    m.device_for = lambda accelerator: SimpleNamespace(type="cpu")
    m.segmenter_dtype = lambda device: np.dtype("float32")
    m.synchronize = lambda device: None
    m.check_access = lambda token: None
    m.load_segmenter = lambda device, token: "sam3"
    m.load_embedder = lambda device, token: "dinov3"
    m.CropDetector, m.segment_shot, m.embed_crops = CropDetector, segment_shot, embed_crops
    return m


def verdict(label: str, image: bytes) -> str:
    """The crop check's answer for one view: posters are graphics, shoes absent; a pants crop is absent or not by its
    bytes, so an answer put on another track's view changes the layer (which pants are rejected depends on the JPEG
    encoder's bytes, so on the platform: the comparison below is within one machine)."""
    if label == "poster":
        return "graphic"
    if label == "shoe" or (label == "pants" and hashlib.sha256(image).digest()[0] % 3 == 0):
        return "absent"
    return "object"


class Anthropic:
    """The Anthropic client: answers by the request's schema and content, after a random short wait (so the crop
    check's calls finish out of order). calls: every request, in the order they came."""
    calls: list = []
    lock = threading.Lock()

    def __init__(self, api_key):
        assert api_key == "key"
        self.messages = self

    def create(self, **request):
        with self.lock:
            self.calls.append(request)
        time.sleep(random.random() * 0.02)
        content = request["messages"][0]["content"]
        images = [base64.b64decode(b["source"]["data"]) for b in content if b["type"] == "image"]
        text = content[-1]["text"]
        schema = request["output_config"]["format"]["schema"]["properties"]
        if "objects" in schema:
            answer = {"objects": NAMED[images[0]]}
        elif "names" in schema:
            names = text.split("Names:\n")[1].split("\n")
            answer = {"names": [dict(zip(("name", "family", "kind", "worn"), (n, *CLASSES[n]))) for n in names]}
        else:
            label = text.split('labelled "')[1].split('"')[0]
            answer = {"views": [{"view": k, "verdict": verdict(label, img)} for k, img in enumerate(images, 1)]}
        return SimpleNamespace(stop_reason="end_turn", stop_details=None,
                               content=[SimpleNamespace(type="text", text=json.dumps(answer))],
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=50))


def anthropic_module():
    m = types.ModuleType("anthropic")
    m.Anthropic = Anthropic
    for name in ("RateLimitError", "InternalServerError", "APIConnectionError", "AuthenticationError"):
        setattr(m, name, type(name, (Exception,), {}))
    return m


@pytest.fixture
def stand_ins(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    monkeypatch.setitem(sys.modules, "models", models_module())
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_module())
    monkeypatch.setattr(Anthropic, "calls", [])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.setenv("HF_TOKEN", "token")
    return Anthropic


def run(script: Path, folder: Path, inputs: dict, config: dict, secrets: list, monkeypatch, producer=None) -> dict:
    """A node script's response, run here as the coordinator runs it (its request, its own folder)."""
    folder.mkdir(parents=True)
    request = folder / "request.json"
    request.write_text(json.dumps({"protocol": 1, "out_dir": str(folder / "out"), "accelerator": "cpu", "config": config,
                                   "secrets": secrets, "producer": producer or {"stage": script.stem, "stage_version": "1.0.0"},
                                   "inputs": {name: {"path": str(path)} for name, path in inputs.items()}}))
    monkeypatch.setattr(sys, "argv", [script.name, "--request", str(request)])
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit:   # how a node ends after writing a failed response
        pass
    response = json.loads((folder / "response.json").read_text())
    assert response["ok"], response["error"]
    return response


def run_unsplit(tmp_path, monkeypatch, given) -> Path:
    run(HERE / "testdata" / "node_1_1_0.py", tmp_path / "unsplit", given, CONFIG, ["ANTHROPIC_API_KEY", "HF_TOKEN"],
        monkeypatch, PRODUCER)
    return tmp_path / "unsplit" / "out"


def run_split(tmp_path, monkeypatch, given) -> dict[str, Path]:
    run(HERE / "names.py", tmp_path / "names", {k: given[k] for k in ("shots", "keyframes", "media")}, {},
        ["ANTHROPIC_API_KEY"], monkeypatch)
    vocabulary = tmp_path / "names" / "out" / "vocabulary.json"
    run(HERE / "track.py", tmp_path / "track",
        {"vocabulary": vocabulary, **{k: given[k] for k in ("video", "media", "shots", "faces", "bodies")}},
        {k: CONFIG[k] for k in TRACK_KEYS}, ["HF_TOKEN"], monkeypatch)
    tracks = tmp_path / "track" / "out" / "tracks"
    run(HERE / "link.py", tmp_path / "link", {"tracks": tracks, **{k: given[k] for k in ("shots", "faces", "bodies", "media")}},
        {k: v for k, v in CONFIG.items() if k not in TRACK_KEYS}, ["ANTHROPIC_API_KEY"], monkeypatch, PRODUCER)
    return {"vocabulary": vocabulary, "tracks": tracks, "out": tmp_path / "link" / "out"}


def files(folder: Path) -> dict[str, bytes]:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


# --- the parity ------------------------------------------------------------------------------------------------------

def test_the_three_steps_make_the_unsplit_nodes_layer_side_file_and_crops_byte_for_byte(tmp_path, monkeypatch, stand_ins):
    given = layers(tmp_path / "in")
    unsplit = run_unsplit(tmp_path, monkeypatch, given)
    unsplit_calls = list(stand_ins.calls)
    stand_ins.calls.clear()
    split = run_split(tmp_path, monkeypatch, given)
    want, got = files(unsplit), files(split["out"])
    assert sorted(got) == sorted(want)
    assert {"assets/crops/objects/chair_1.jpg", "assets/crops/objects/t_shirt_1.jpg", "objects.json",
            "objects_masks.parquet"} <= set(got)
    for name in want:
        assert got[name] == want[name], name

    # the same calls with the same content, the crop check's in another order
    key = lambda r: json.dumps(r, sort_keys=True)   # noqa: E731
    assert sorted(map(key, stand_ins.calls)) == sorted(map(key, unsplit_calls))

    # and the clip exercises what the split carries between steps: every reason and outcome, trimmed samples, worn things
    # with wearers, embedded and unembedded tracks, rejected tracks and a link across shots
    import object_raw
    layer = json.loads(got["objects.json"])
    rows, _ = object_raw.read(split["out"] / "objects_masks.parquet")
    assert {r["outcome"] for r in rows} == {"tracks", "rejected", "too_short", "none"}
    assert {r["reason"] for r in rows} == {"kept", "trimmed", "worn", "item", "person", "below_keep", "not_this_body"}
    assert {(t["label"], t["verdict"]) for t in layer["rejected_tracks"]} >= {("poster", "graphic"), ("shoe", "absent")}
    assert {t["wearer"] for t in layer["tracks"] if t["label"] == "helmet"} == {"person:1"}   # worn: assign_wearers
    linked = {i["id"]: (i["tracks"], i["worn_by"]) for i in layer["identities"] if len(i["tracks"]) > 1}
    assert linked["objid:t-shirt 1"] == (["obj:3", "obj:10"], ["person:0"])   # one wearer's, across the cut
    assert next(i for i in layer["identities"] if i["label"] == "mug")["crop"] is None   # never big enough to embed
    assert layer["producer"]["models"][0]["id"] == "facebook/sam3"


def test_the_crop_checks_answers_go_to_their_own_tracks_whatever_order_they_come_in(tmp_path, monkeypatch, stand_ins):
    """The link step run with one call at a time and with every call at once, the calls finishing in random order: the
    same verdicts, the same layer."""
    import haiku
    given = layers(tmp_path / "in")
    split = run_split(tmp_path, monkeypatch, given)
    rest = {k: given[k] for k in ("shots", "faces", "bodies", "media")}
    config = {k: v for k, v in CONFIG.items() if k not in TRACK_KEYS}
    monkeypatch.setattr(haiku, "WORKERS", 1)
    run(HERE / "link.py", tmp_path / "one", {"tracks": split["tracks"], **rest}, config, ["ANTHROPIC_API_KEY"], monkeypatch, PRODUCER)
    assert files(tmp_path / "one" / "out") == files(split["out"])


# --- the steps' own checks -------------------------------------------------------------------------------------------

def test_the_vocabulary_names_every_shot_and_classes_exactly_its_prompts(tmp_path, monkeypatch, stand_ins):
    given = layers(tmp_path / "in")
    run(HERE / "names.py", tmp_path / "names", {k: given[k] for k in ("shots", "keyframes", "media")}, {},
        ["ANTHROPIC_API_KEY"], monkeypatch)
    doc = json.loads((tmp_path / "names" / "out" / "vocabulary.json").read_text())
    assert doc == {"model": "claude-haiku-5-5",
                   "shots": [{"shot": "shot:0", "names": ["chair", "mug", "helmet", "lamp"], "body_named": ["jacket"]},
                             {"shot": "shot:1", "names": ["chair", "poster", "helmet"], "body_named": ["person"]}],
                   "classes": {n: dict(zip(("family", "kind", "worn"), CLASSES[n])) for n in sorted(CLASSES)}}


def test_a_vocabulary_for_other_shots_or_with_names_unclassed_is_refused():
    from vocabulary import session_prompts
    shots = [{"id": "shot:0", "index": 0}, {"id": "shot:1", "index": 1}]
    good = {"shots": [{"shot": "shot:0", "names": ["cup"], "body_named": []}, {"shot": "shot:1", "names": []}],
            "classes": {"cup": {"family": "cup", "kind": "individual", "worn": False}}}
    assert session_prompts(good, shots) == ({0: ["cup"], 1: []}, {0: [], 1: []}, good["classes"])
    with pytest.raises(ValueError, match=r"missing \['shot:1'\]"):
        session_prompts({**good, "shots": good["shots"][:1]}, shots)
    with pytest.raises(ValueError, match="some named twice"):
        session_prompts({**good, "shots": good["shots"] + good["shots"][1:]}, shots)
    with pytest.raises(ValueError, match=r"unclassed \['cup'\]"):
        session_prompts({**good, "classes": {}}, shots)
    with pytest.raises(ValueError, match=r"unused \['mug'\]"):
        session_prompts({**good, "classes": {**good["classes"], "mug": good["classes"]["cup"]}}, shots)


def test_a_track_step_given_a_vocabulary_of_other_shots_fails_on_its_input(tmp_path, monkeypatch, stand_ins):
    given = layers(tmp_path / "in")
    (tmp_path / "vocabulary.json").write_text(json.dumps({"shots": [{"shot": "shot:0", "names": []}], "classes": {}}))
    with pytest.raises(AssertionError, match="'code': 'input'.*missing \\['shot:1'\\]"):
        run(HERE / "track.py", tmp_path / "track",
            {"vocabulary": tmp_path / "vocabulary.json", **{k: given[k] for k in ("video", "media", "shots", "faces", "bodies")}},
            {k: CONFIG[k] for k in TRACK_KEYS}, ["HF_TOKEN"], monkeypatch)


def test_the_tracks_keep_their_embeddings_and_crops_exactly(tmp_path):
    import object_tracks
    rng = np.random.default_rng(1)
    mean, views = rng.normal(size=1024).astype(np.float32), rng.normal(size=(2, 1024)).astype(np.float32)
    tracks = [{"id": "obj:0", "label": "cup", "shot": 0, "samples": [], "rows": [3, 4], "embedding": mean,
               "view_embeddings": views, "views": [b"jpeg 0", b"jpeg 1"], "context": b"strip 0"},
              {"id": "obj:1", "label": "cup", "shot": 1, "samples": [], "rows": [], "embedding": None,
               "view_embeddings": None, "views": [], "context": b"strip 1"}]
    object_tracks.write(tmp_path, tracks, {"fps": 6, "classes": {}})
    back, info = object_tracks.read(tmp_path)
    assert info == {"fps": 6, "classes": {}}
    assert back[0]["embedding"].dtype == np.float32 and np.array_equal(back[0]["embedding"], mean)
    assert np.array_equal(back[0]["view_embeddings"], views) and back[0]["view_embeddings"].dtype == np.float32
    assert [p.read_bytes() for p in back[0]["views"]] == [b"jpeg 0", b"jpeg 1"] and back[0]["context"] == b"strip 0"
    assert back[1]["embedding"] is None and back[1]["view_embeddings"] is None and back[1]["views"] == []
    assert {k: back[0][k] for k in ("id", "label", "shot", "rows")} == {"id": "obj:0", "label": "cup", "shot": 0, "rows": [3, 4]}


# --- the crop check's calls at once ------------------------------------------------------------------------------------

def test_fan_out_keeps_the_items_order_whatever_order_they_finish_in():
    from haiku import fan_out

    def slow_first(i):
        time.sleep(0.002 * (20 - i))
        return i * i
    assert fan_out(slow_first, range(20), workers=8) == [i * i for i in range(20)]


def test_fan_out_starts_no_call_after_one_fails():
    """As prosody-captions' fan_out: those running finish, none starts (one worker, so the count is exact)."""
    from haiku import fan_out
    started = []

    def call(i):
        started.append(i)
        if i == 3:
            raise RuntimeError("call 3 failed")
        return i
    with pytest.raises(RuntimeError, match="call 3 failed"):
        fan_out(call, range(50), workers=1)
    assert started == [0, 1, 2, 3]
