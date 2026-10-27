"""Weights fetched from a URL (perception_node.weights): one shared cache keyed by sha256, checked before use, and
plan's view of it (a URL weight with its sha256 is a download until it is in that cache)."""

import hashlib
import zipfile

import pytest

from perception.manifests import ModelRef
from perception.plan import fetched, model_cached, weights_cached
from perception_node import weights


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PERCEPTION_WEIGHTS", str(tmp_path / "weights"))
    return tmp_path / "weights"


def zipped(tmp_path, content=b"onnx bytes", member="model/end2end.onnx"):
    path = tmp_path / "model.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(member, content)
        z.writestr("model/deploy.json", b"{}")
    return path, hashlib.sha256(content).hexdigest()


def test_a_zips_file_is_fetched_once_checked_and_kept_under_its_sha256(tmp_path, cache):
    source, sha = zipped(tmp_path)
    got = weights.fetch(source.as_uri(), sha, member="end2end.onnx")
    assert got == cache / sha / "end2end.onnx" and got.read_bytes() == b"onnx bytes"
    source.unlink()                                                            # a second fetch never goes to the URL
    assert weights.fetch(source.as_uri(), sha, member="end2end.onnx") == got
    assert not [p for p in cache.iterdir() if p.name.startswith(".fetch-")]   # nothing left half-written


def test_a_file_with_another_checksum_is_refused_and_not_kept(tmp_path, cache):
    source, _ = zipped(tmp_path)
    with pytest.raises(ValueError, match="has sha256"):
        weights.fetch(source.as_uri(), "0" * 64, member="end2end.onnx")
    assert weights.cached("0" * 64) is None


def test_a_zip_without_exactly_one_such_file_is_refused(tmp_path, cache):
    source, sha = zipped(tmp_path, member="model/other.bin")
    with pytest.raises(ValueError, match="0 files end in"):
        weights.fetch(source.as_uri(), sha, member="end2end.onnx")


def test_plan_and_the_node_agree_where_a_weight_is_cached(tmp_path, cache):
    source, sha = zipped(tmp_path)
    model = ModelRef(id="m", source="https://example.org/m.zip", sha256=sha, licence_mode="commercial-ok")
    assert fetched(model) and not weights_cached(sha) and not model_cached(model)   # a download, before
    weights.fetch(source.as_uri(), sha, member="end2end.onnx")
    assert weights_cached(sha) and model_cached(model)                              # not one, after


def test_a_url_weight_needs_its_sha256_and_is_then_fetched_by_a_run():
    with pytest.raises(ValueError, match="needs its sha256"):                  # it would download unseen by plan
        ModelRef(id="u", source="https://example.org/m.zip", licence_mode="commercial-ok")
    assert fetched(ModelRef(id="u", source="https://example.org/m.zip", sha256="a" * 64, licence_mode="commercial-ok"))
    for other in ("http://example.org/m.zip", "HTTPS://example.org/m.zip", "ftp://example.org/m.zip"):
        with pytest.raises(ValueError, match="lower case"):                    # plan would not count it a download
            ModelRef(id="h", source=other, sha256="a" * 64, licence_mode="commercial-ok")
    assert not fetched(ModelRef(id="p", source="py-feat==2.1.5 (weights from hf://py-feat)", licence_mode="academic-only"))
    assert fetched(ModelRef(id="h", source="hf://org/repo", licence_mode="commercial-ok"))
    assert not fetched(ModelRef(id="p", source="py-feat==2.1.5", licence_mode="academic-only"))
    with pytest.raises(ValueError):
        ModelRef(id="bad", source="https://example.org/m.zip", sha256="not-a-sha", licence_mode="commercial-ok")


def test_every_node_fetches_the_weights_its_manifest_declares():
    """plan decides from the manifest whether a run would download; the node fetches what its WEIGHTS say. A file
    changed in one and not the other would download without --allow-downloads (or always look missing)."""
    import importlib.util
    from pathlib import Path

    from perception.manifests import Registry
    nodes = Path(__file__).resolve().parents[2] / "nodes"
    checked = 0
    for m in Registry.load(nodes):
        declared = {(x.source, x.sha256) for x in m.models if x.sha256}
        if not (m.directory / "models.py").exists():
            assert not declared, f"{m.ref} declares URL weights but has no models.py WEIGHTS to fetch them"
            continue
        spec = importlib.util.spec_from_file_location(f"weights_of_{m.id}", m.directory / "models.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert {(w["url"], w["sha256"]) for w in getattr(module, "WEIGHTS", {}).values()} == declared, m.ref
        checked += 1
    assert checked   # body-pose at least
