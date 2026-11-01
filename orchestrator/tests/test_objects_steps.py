"""The objects node split into three steps (vision.objects-names, vision.objects-track, vision.objects 2.0.0;
RUN_ANYWHERE_PLAN.md item 1): together they declare what 1.1.0 declared, the graph wires them so the app and gaze and
gestures see one objects step, and the artifacts between them are checked before they are published. That they make
1.1.0's layer is nodes/vision/objects/test_split.py's."""

import json
from pathlib import Path

import pytest
import yaml

from perception import artifacts
from perception.cli import only
from perception.manifests import Registry

REPO = Path(__file__).resolve().parents[2]
STEPS = ("vision.objects-names@1.0.0", "vision.objects-track@1.0.0", "vision.objects@2.0.0")
# what vision.objects 1.1.0's manifest declared (nodes/vision/objects/manifest.yaml at the split's parent commit)
UNSPLIT = {"external_upload": {"frames", "faces"}, "secrets": {"ANTHROPIC_API_KEY", "HF_TOKEN"}, "usd_per_minute": 0.05,
           "config": {"fps": 6, "clothing_fps": 3, "link_threshold": 0.62, "cross_name_margin": 0.05,
                      "same_wearer_bonus": 0.05, "min_link_px": 48, "wearer_share": 0.5, "different_person_distance": 0.8,
                      "duplicate_iou": 0.6},
           "models": {"facebook/sam3", "facebook/dinov3-vitl16-pretrain-lvd1689m", "claude-haiku-5-5"}}


@pytest.fixture(scope="module")
def registry():
    return Registry.load(REPO / "nodes")


def test_the_three_steps_declare_together_what_the_unsplit_node_did(registry):
    names, track, link = (registry.get(ref) for ref in STEPS)
    steps = (names, track, link)
    assert set().union(*(m.privacy.external_upload for m in steps)) == UNSPLIT["external_upload"]
    assert set().union(*(m.secrets for m in steps), *(m.download_secrets for m in steps)) == UNSPLIT["secrets"]
    # the track step sends nothing and holds no key: its HF_TOKEN is its prepare's (phase 5), so it can run on a host
    assert track.privacy.external_upload == [] and track.secrets == [] and track.download_secrets == ["HF_TOKEN"]
    assert track.cost is None
    assert names.secrets == link.secrets == ["ANTHROPIC_API_KEY"]
    assert names.privacy.external_upload == link.privacy.external_upload == ["frames", "faces"]
    # the two that spend reserve at least what 1.1.0 did, per minute of the video
    for m in (names, link):
        assert (m.cost.per, m.inputs[m.cost.input]) == ("minute", "media.v1")
    assert names.cost.unit_usd({}) + link.cost.unit_usd({}) >= UNSPLIT["usd_per_minute"]
    # 1.1.0's settings, each in one step, at 1.1.0's defaults and in its order
    assert {**track.defaults(), **link.defaults()} == UNSPLIT["config"]
    assert list(track.defaults()) + list(link.defaults()) == list(UNSPLIT["config"])
    assert not set(track.config) & set(link.config) and names.config == {}
    # every model 1.1.0 used, downloaded by the track step only
    assert {x.id for m in steps for x in m.models} == UNSPLIT["models"]
    assert [x.id for x in track.models if x.gated and x.source] == ["facebook/sam3", "facebook/dinov3-vitl16-pretrain-lvd1689m"]
    assert not [x for m in (names, link) for x in m.models if x.source or x.gated]
    assert track.prepare and not names.prepare and not link.prepare
    assert track.resources.gpu == "exclusive" and names.resources.gpu == link.resources.gpu == "none"


def test_the_graph_keeps_one_objects_step_that_brings_the_others(registry):
    doc = yaml.safe_load((REPO / "graphs" / "full-objects-names.yaml").read_text())
    uses = {n["id"]: n["uses"] for n in doc["nodes"]}
    assert [uses[i] for i in ("objects_names", "objects_track", "objects")] == list(STEPS)
    # gaze and gestures read the objects layer from the step still called objects
    for reader in ("gaze", "gestures"):
        assert next(n for n in doc["nodes"] if n["id"] == reader)["inputs"]["objects"] == "$nodes.objects.objects"
    chosen = {n["id"] for n in only(doc, ["objects"], registry)["nodes"]}
    assert chosen == {"decode", "faces", "bodies", "objects_names", "objects_track", "objects"}
    assert {n["id"] for n in only(doc, ["objects_names"], registry)["nodes"]} == {"decode", "objects_names"}
    assert not {n["id"] for n in only(doc, ["gaze"], registry)["nodes"]} & {"objects", "objects_track", "objects_names"}


def test_the_full_graph_names_the_objects_in_the_captions_call(registry):
    """graphs/full.yaml (run-anywhere 1b, B): the captions step names the objects, whose vocabulary the track step
    reads, and there is no names step; full-objects-names.yaml (A) is the same graph with one. Both compile, and
    Objects alone brings the captions."""
    from perception.ir import compile_graph
    full, names = (yaml.safe_load((REPO / "graphs" / f).read_text()) for f in ("full.yaml", "full-objects-names.yaml"))
    nodes = {n["id"]: n for n in full["nodes"]}
    compile_graph(full, registry)
    compile_graph(names, registry)
    assert set(nodes) == {n["id"] for n in names["nodes"]} - {"objects_names"}
    assert nodes["captions"]["uses"] == "vlm.shot-captions-objects@1.0.0"
    assert nodes["objects_track"]["inputs"]["vocabulary"] == "$nodes.captions.vocabulary"
    chosen = {n["id"] for n in only(full, ["objects"], registry)["nodes"]}
    assert "captions" in chosen and "objects_names" not in chosen


def test_the_names_and_track_steps_are_steps_of_objects(registry):
    """What the app shows as one feature, Objects: the names and track steps say they are its (part_of), and objects is
    a feature of its own. The plan's part_of, in the graph's ids, is test_cli's."""
    names, track, link = (registry.get(ref) for ref in STEPS)
    assert (names.part_of, track.part_of, link.part_of) == ("vision.objects", "vision.objects", None)
    assert [m.id for m in registry if m.part_of] == ["vision.objects-names", "vision.objects-track"]


def vocabulary(tmp_path, doc) -> list[str]:
    path = tmp_path / "vocabulary.json"
    path.write_text(json.dumps(doc))
    return artifacts.get("vocabulary.v1").validate(path)


def test_a_vocabulary_names_each_shot_once_and_classes_exactly_its_names(tmp_path):
    cup = {"family": "cup", "kind": "individual", "worn": False}
    good = {"model": "claude-haiku-5-5", "classes": {"cup": cup},
            "shots": [{"shot": "shot:0", "names": ["cup"], "body_named": ["jacket"]}, {"shot": "shot:1", "names": []}]}
    assert vocabulary(tmp_path, good) == []
    assert vocabulary(tmp_path, {**good, "classes": {}}) == ["$.classes: no class for ['cup']"]
    assert vocabulary(tmp_path, {**good, "classes": {"cup": cup, "mug": cup}}) == ["$.classes: ['mug'] name no shot's prompt"]
    assert vocabulary(tmp_path, {**good, "shots": good["shots"] + good["shots"][:1]}) == ["$.shots: a shot is named twice"]
    assert vocabulary(tmp_path, {**good, "shots": [{"shot": "shot:0", "names": [f"n{i}" for i in range(9)]}]})   # over 8
    assert vocabulary(tmp_path, {**good, "classes": {"cup": {**cup, "kind": "several"}}})
    assert vocabulary(tmp_path, {**good, "shots": [{"shot": "0", "names": ["cup"]}]})   # not a shots layer id
    assert artifacts.get("vocabulary.v1").kinds == ()   # words: no picture, voice or identity


def tracks_tree(folder: Path) -> Path:
    """A track step's output, as nodes/vision/objects/object_tracks.py writes one (one track, one embedded crop)."""
    (folder / "views").mkdir(parents=True)
    (folder / "context").mkdir()
    (folder / "views" / "0_0.jpg").write_bytes(b"\xff\xd8 crop")
    (folder / "context" / "0.jpg").write_bytes(b"\xff\xd8 strip")
    for array in ("embeddings.npy", "view_embeddings.npy"):   # NumPy's format: its magic, then a header (no NumPy here)
        (folder / array).write_bytes(b"\x93NUMPY\x01\x00" + b"v\x00{'descr': '<f4', 'fortran_order': False, 'shape': (1, 4), }")
    (folder / "masks.parquet").write_bytes(b"PAR1" + b"\0" * 8 + (4).to_bytes(4, "little") + b"PAR1")
    (folder / "tracks.json").write_text(json.dumps({"fps": 6, "tracks": [
        {"id": "obj:0", "embedding": 0, "views": [{"crop": "views/0_0.jpg", "embedding": 0}], "context": "context/0.jpg"}]}))
    return folder


def test_object_tracks_hold_their_named_files_and_nothing_else(tmp_path):
    check = artifacts.get("object_tracks.v1").validate
    good = tracks_tree(tmp_path / "good")
    assert check(good) == []
    (good / "views" / "notes.txt").write_text("a file nobody named")
    assert check(good) == ["unexpected files: ['views/notes.txt']"]
    missing = tracks_tree(tmp_path / "missing")
    (missing / "context" / "0.jpg").unlink()
    assert check(missing) == ["tracks.json: track 'obj:0' names context/0.jpg, which is not here"]
    outside = tracks_tree(tmp_path / "outside")
    doc = json.loads((outside / "tracks.json").read_text())
    for name in ("../secret.jpg", "views/../../secret.jpg", "context/0.png"):
        doc["tracks"][0]["context"] = name
        (outside / "tracks.json").write_text(json.dumps(doc))
        assert check(outside)[0] == f"tracks.json: track 'obj:0' names {name!r}, not views/<n>_<k>.jpg or context/<n>.jpg"
    bad = tracks_tree(tmp_path / "bad")
    (bad / "embeddings.npy").write_bytes(b"not numpy")
    (bad / "masks.parquet").write_bytes(b"not parquet")
    assert check(bad)[0] == "embeddings.npy: not a NumPy array file" and check(bad)[1].startswith("masks.parquet: not Parquet")
    assert check(tmp_path / "good" / "tracks.json") == ["not a directory"]
    # crops of the people on screen, and DINOv3 vectors of them
    assert artifacts.get("object_tracks.v1").kinds == ("frames", "faces", "embeddings")


def test_the_track_step_takes_a_card_of_its_own_and_the_api_steps_none(registry):
    """scheduler.card_for (#42) with the steps' own resources and runners: on a host with two cards the track step takes
    the larger free card at least its CUDA runner's minimum, waits while another GPU node is on every card that fits, and
    the names and link steps hold no card at all."""
    from perception.backends.base import Gpu
    from perception.scheduler import card_for
    names, track, link = (registry.get(ref) for ref in STEPS)
    cuda = next(r for r in track.runners if r.accelerator == "cuda")
    cards = [Gpu(0, "small", 12.0), Gpu(1, "large", 48.0)]
    assert (track.resources.gpu, cuda.minimum_vram_gb) == ("exclusive", 16)
    assert card_for(track.resources.gpu, cuda.minimum_vram_gb, cards, []) == (True, cards[1])
    assert card_for(track.resources.gpu, cuda.minimum_vram_gb, cards, [("shared", 1, 4.0)]) == (False, None)   # 12 GB is too small
    assert card_for(track.resources.gpu, cuda.minimum_vram_gb, [Gpu(None, "Apple GPU", 0.0)], []) == (True, Gpu(None, "Apple GPU", 0.0))
    for m in (names, link):
        assert card_for(m.resources.gpu, 0, cards, [("exclusive", 1, 16.0)]) == (True, None)
