import sqlite3

from perception import artifacts
from perception.store import MIGRATIONS, Store, _statements


def test_a_store_made_by_an_older_version_is_migrated_in_place(tmp_path):
    """A store created before migration 2 existed (the user's real store was) gains the column, keeping its rows."""
    root = tmp_path / "store"
    root.mkdir()
    db = sqlite3.connect(root / "perception.db")
    for statement in _statements(MIGRATIONS[0]):
        db.execute(statement)
    db.execute("PRAGMA user_version = 1")
    db.execute("INSERT INTO runs (id, pipeline, graph, inputs, status, created_at, max_cost_usd) VALUES ('r', 'p', '{}', '{}', 'running', 0, 0)")
    db.execute("INSERT INTO tasks (run_id, node_id, state) VALUES ('r', 'a', 'pending')")
    db.commit()
    db.close()

    store = Store(root)
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    row = store.db.execute("SELECT state, first_attempt FROM tasks WHERE run_id='r'").fetchone()
    assert (row["state"], row["first_attempt"]) == ("pending", 1)


def test_publishing_is_content_addressed_and_keeps_the_file_name(tmp_path):
    store = Store(tmp_path / "store")
    a, b = tmp_path / "a" / "out.json", tmp_path / "b" / "other.json"
    for p in (a, b):
        p.parent.mkdir()
        p.write_text('{"value": 1}')
    first, second = store.publish(a, "json.v1"), store.publish(b, "json.v1")
    assert first == second                              # same content: one object, the first name kept
    assert first.path.name == "out.json" and first.hash == artifacts.sha256_file(a)
    assert not first.path.stat().st_mode & 0o222        # read-only once published


def test_a_tree_is_published_whole_under_its_tree_hash(tmp_path):
    store = Store(tmp_path / "store")
    tree = tmp_path / "assets"
    (tree / "crops").mkdir(parents=True)
    (tree / "crops" / "x.jpg").write_bytes(b"jpeg")
    art = store.publish(tree, "assets.v1")
    assert art.shape == "tree" and art.hash == artifacts.sha256_tree(tree)
    assert (art.path / "crops" / "x.jpg").read_bytes() == b"jpeg"


def test_a_symlink_is_never_published(tmp_path):
    """Publishing follows nothing: a symlink in an output would copy a host file into the store and the bundle."""
    import pytest
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    store = Store(tmp_path / "store")
    tree = tmp_path / "assets"
    tree.mkdir()
    (tree / "x.jpg").symlink_to(secret)
    with pytest.raises(ValueError, match="symlink"):
        store.publish(tree, "assets.v1")
    link = tmp_path / "out.json"
    link.symlink_to(secret)
    with pytest.raises(ValueError, match="symlink"):
        store.publish(link, "json.v1")
    assert not [p for p in (store.root / "objects").rglob("*") if p.is_file()]
    assert not [p for p in (store.root / "trees").rglob("*") if p.is_file()]
