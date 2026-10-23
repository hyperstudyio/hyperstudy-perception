"""The artifact store and run state, under one root directory:

    <root>/perception.db                 run state, artifacts, cache (SQLite, WAL)
    <root>/objects/ab/<sha256>/<name>    published files, read-only, keeping their file name (tools read extensions)
    <root>/trees/ab/<sha256>/...         published directories, read-only
    <root>/work/<run>/<node>/<attempt>/  attempt directories (request.json, out/, response.json, log.txt)
    <root>/tmp/                          publication staging (same filesystem, so publication is a rename)

Publication is effectively-once: an output is copied into tmp/, hashed, and renamed into objects/ or trees/ under its
hash. A second publication of the same content finds the object already there. Nothing reads an object before its
row is committed, and a crash leaves at most an orphan file in tmp/ (cleared on open).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import artifacts

MIGRATIONS = [
    """
    CREATE TABLE artifacts (
        hash TEXT PRIMARY KEY, shape TEXT NOT NULL, type TEXT NOT NULL, bytes INTEGER NOT NULL, created_at REAL NOT NULL
    );
    -- a source file (the video) hashed once per (path, size, mtime): films are large
    CREATE TABLE sources (
        path TEXT NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, sha256 TEXT NOT NULL,
        PRIMARY KEY (path, size, mtime_ns)
    );
    CREATE TABLE runs (
        id TEXT PRIMARY KEY, pipeline TEXT NOT NULL, graph TEXT NOT NULL, graph_path TEXT, inputs TEXT NOT NULL,
        status TEXT NOT NULL, created_at REAL NOT NULL, finished_at REAL, max_cost_usd REAL NOT NULL, spent_usd REAL NOT NULL DEFAULT 0
    );
    CREATE TABLE tasks (
        run_id TEXT NOT NULL REFERENCES runs(id), node_id TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('pending','ready','leased','running','succeeded','cached','failed','canceled')),
        attempt INTEGER NOT NULL DEFAULT 0,       -- increases for ever: each attempt gets its own directory
        cache_key TEXT, runner TEXT,
        lease_owner TEXT, lease_expires REAL, handle TEXT,
        not_before REAL, started_at REAL, finished_at REAL, error_code TEXT, error TEXT,
        cost_usd REAL NOT NULL DEFAULT 0, usage TEXT, outputs TEXT,
        skip_cache INTEGER NOT NULL DEFAULT 0,   -- rerun --no-cache: run even if the cache has the result
        PRIMARY KEY (run_id, node_id)
    );
    CREATE TABLE cache (key TEXT PRIMARY KEY, node TEXT NOT NULL, outputs TEXT NOT NULL, usage TEXT, created_at REAL NOT NULL);
    CREATE TABLE events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, node_id TEXT, at REAL NOT NULL, kind TEXT NOT NULL, message TEXT
    );
    CREATE INDEX events_by_run ON events(run_id, id);
    """,
    # 2: retries count from the first attempt since the last rerun; attempt numbers (directories) keep increasing
    """
    ALTER TABLE tasks ADD COLUMN first_attempt INTEGER NOT NULL DEFAULT 1;
    """,
]


@dataclass(frozen=True)
class Artifact:
    hash: str
    shape: str          # file | tree | source
    type: str
    path: Path          # where it can be read

    def to_json(self) -> dict:
        return {"hash": self.hash, "shape": self.shape, "type": self.type, "path": str(self.path)}

    @classmethod
    def from_json(cls, d: dict) -> "Artifact":
        return cls(d["hash"], d["shape"], d["type"], Path(d["path"]))


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        for sub in ("objects", "trees", "work", "tmp"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        for leftover in (self.root / "tmp").iterdir():   # staging from a crashed publication
            shutil.rmtree(leftover) if leftover.is_dir() else leftover.unlink()
        self.db = sqlite3.connect(self.root / "perception.db", isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self):
        current = self.db.execute("PRAGMA user_version").fetchone()[0]
        for version, sql in enumerate(MIGRATIONS, start=1):
            if version > current:
                with self.tx():   # executescript would commit on its own; one statement at a time stays in the transaction
                    for statement in _statements(sql):
                        self.db.execute(statement)
                    self.db.execute(f"PRAGMA user_version = {version}")

    def tx(self):
        return _Tx(self.db)

    def close(self):
        self.db.close()

    # --- sources and objects --------------------------------------------------------------------------------

    def source(self, path: Path, type_id: str) -> Artifact:
        """A file that stays where it is (the video), identified by its sha256."""
        st = path.stat()
        row = self.db.execute("SELECT sha256 FROM sources WHERE path=? AND size=? AND mtime_ns=?",
                              (str(path), st.st_size, st.st_mtime_ns)).fetchone()
        sha = row["sha256"] if row else artifacts.sha256_file(path)
        if not row:
            self.db.execute("INSERT OR REPLACE INTO sources VALUES (?,?,?,?)", (str(path), st.st_size, st.st_mtime_ns, sha))
        return Artifact(sha, "source", type_id, path)

    def object_dir(self, shape: str, h: str) -> Path:
        return self.root / ("trees" if shape == "tree" else "objects") / h[:2] / h

    def publish(self, path: Path, type_id: str) -> Artifact:
        """Copy a node's output into the store under its content hash. Idempotent: the same content published again
        (under any name) resolves to the object already there."""
        bad = artifacts.special_files(path)
        if bad:   # never follow a link out of a node's output: it would publish a host file
            raise ValueError(f"refusing to publish {path.name}: contains a symlink or special file: {', '.join(bad[:5])}")
        shape = "tree" if path.is_dir() else "file"
        staged = self.root / "tmp" / uuid.uuid4().hex
        if shape == "tree":
            shutil.copytree(path, staged, symlinks=True)
            h = artifacts.sha256_tree(staged)
        else:
            staged.mkdir()
            shutil.copyfile(path, staged / path.name)
            h = artifacts.sha256_file(staged / path.name)
        final = self.object_dir(shape, h)
        if final.exists():
            shutil.rmtree(staged)
        else:
            final.parent.mkdir(parents=True, exist_ok=True)
            _fsync(staged)
            _read_only(staged)
            os.replace(staged, final)   # atomic: the object appears whole or not at all
        readable = final if shape == "tree" else next(final.iterdir())
        size = sum(p.stat().st_size for p in final.rglob("*") if p.is_file())
        self.db.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?,?,?,?)", (h, shape, type_id, size, time.time()))
        return Artifact(h, shape, type_id, readable)

    def has(self, a: Artifact) -> bool:
        return a.path.exists()

    # --- events ---------------------------------------------------------------------------------------------

    def event(self, run_id: str, node_id: str | None, kind: str, message: str = ""):
        self.db.execute("INSERT INTO events (run_id, node_id, at, kind, message) VALUES (?,?,?,?,?)",
                        (run_id, node_id, time.time(), kind, message))

    def events(self, run_id: str, after: int = 0) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id", (run_id, after)).fetchall()


class _Tx:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        self.db.execute("BEGIN IMMEDIATE")
        return self.db

    def __exit__(self, exc_type, exc, tb):
        self.db.execute("ROLLBACK" if exc_type else "COMMIT")
        return False


def _statements(sql: str) -> list[str]:
    return [s.strip() for s in sql.split(";") if s.strip() and not all(l.strip().startswith("--") for l in s.strip().splitlines())]


def _fsync(path: Path):
    files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
    for p in files:
        with open(p, "rb") as f:
            os.fsync(f.fileno())


def _read_only(path: Path):
    files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
    for p in files:
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def dumps(value) -> str:
    return json.dumps(value, separators=(",", ":"))
