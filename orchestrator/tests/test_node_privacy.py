"""What the pipeline's own API nodes declare they send, against what they read."""

from pathlib import Path

from perception import artifacts
from perception.manifests import Registry

NODES = Path(__file__).resolve().parents[2] / "nodes"


def test_an_api_node_that_reads_images_declares_every_kind_they_hold():
    """The API nodes here send the images they read (keyframes, crops) to their API. assets.v1 can hold frames and
    faces (people on screen), so a node that reads one and uploads must declare both: ir.py checks policy.allow_upload
    against the declaration only, and a graph allowing frames alone would otherwise let faces leave."""
    image_kinds = set(artifacts.get("assets.v1").kinds)
    checked = 0
    for m in Registry.load(NODES):
        if not m.privacy.external_upload:
            continue
        if any(t == "assets.v1" for t in m.inputs.values()):
            assert image_kinds <= set(m.privacy.external_upload), f"{m.ref} sends images but declares {m.privacy.external_upload}"
            checked += 1
    assert checked >= 2   # shot-captions and objects
