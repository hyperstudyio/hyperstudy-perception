"""The side files (asd_raw.py): the scores file has the columns the bundle schema documents, holds LR-ASD's float32
scores as they were, is the same bytes for the same scores, and gives back the scored tracks the layer is made from;
the votes give back the layer's voice links. Run: uv run --project envs/gaze --with pytest pytest nodes/vision/active-speaker"""

import json
from pathlib import Path

import numpy as np
import pytest

from asd_crops import DURATIONS, FPS
from asd_raw import (ROW_GROUP_ROWS, SCORES_DESCRIPTION, SCORES_FILE, VOTES_DESCRIPTION, VOTES_FILE, WINDOWS,
                     read_scores, round_scores, scored_tracks, votes_doc, write_scores)
from speaker_links import layer_content, link_voices, speaking_intervals, tally, voice_votes
from test_speaker_links import best_assignment

SCHEMA = Path(__file__).resolve().parents[3] / "schema" / "bundle-0.1.schema.json"
MODEL = "LR-ASD (finetuning_TalkSet: AVA-ActiveSpeaker, then TalkSet)"


def columns(name):
    return [(c["name"], c["arrow"]) for c in json.loads(SCHEMA.read_text())["$defs"]["parquet"][name]["columns"]]


def track(face, first, n, seed, n_frames=None):
    """(face, frames, score, windows) as the node has them: float32 windows and numpy's float32 mean over them."""
    windows = np.random.default_rng(seed).normal(0, 2, (WINDOWS, n)).astype(np.float32)
    return face, np.arange(first, first + (n_frames or n)), np.mean(windows, axis=0), windows


def test_the_file_has_the_documented_columns_and_holds_the_scores_as_float32(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/gaze); CI installs it")
    # face:3's plan has 12 frames, of which the clip's audio covers 10: the scores cover those
    tracks = [track("face:0", 5, 20, 1), track("face:3", 40, 10, 2, n_frames=12)]
    assert write_scores(tmp_path / "s.parquet", tracks, MODEL) == 30
    table = pq.read_table(tmp_path / "s.parquet")
    assert [(f.name, str(f.type)) for f in table.schema] == columns("speakers_scores")
    assert all(not f.nullable for f in table.schema)
    assert table["face"].to_pylist() == ["face:0"] * 20 + ["face:3"] * 10
    assert table["frame"].to_pylist() == list(range(5, 25)) + list(range(40, 50))
    assert table["t"].to_pylist() == [round(k / FPS, 3) for k in table["frame"].to_pylist()]
    assert table["score"].to_pylist() == [float(v) for _, _, s, _ in tracks for v in s]          # float32, exactly
    assert table["windows"].to_pylist()[22] == [float(v) for v in tracks[1][3][:, 2]]
    # the stored mean is the node's: numpy's float32 mean over the window lengths, as [lengths, frames]. Averaged per
    # row, in another order, it can differ in the last bit, which is why the mean is stored beside the windows
    windows, score = np.array(table["windows"].to_pylist(), dtype=np.float32), np.array(table["score"].to_pylist(), np.float32)
    assert np.array_equal(np.mean(np.ascontiguousarray(windows.T), axis=0), score)
    assert np.allclose(np.mean(windows, axis=1), score, rtol=0, atol=1e-6)
    meta = table.schema.metadata
    assert (meta[b"model"], meta[b"fps"], meta[b"windows"]) == (MODEL.encode(), b"25", b"1,1,1,2,2,2,3,3,4,5,6")
    assert len(DURATIONS) == WINDOWS == 11
    assert SCORES_FILE == "layers/speakers_scores.parquet" and VOTES_FILE == "layers/speakers_votes.json"
    assert SCORES_DESCRIPTION and VOTES_DESCRIPTION


def test_no_scored_track_is_an_empty_file_with_the_columns(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/gaze); CI installs it")
    assert write_scores(tmp_path / "s.parquet", [], MODEL) == 0
    table = pq.read_table(tmp_path / "s.parquet")
    assert table.num_rows == 0 and [(f.name, str(f.type)) for f in table.schema] == columns("speakers_scores")


def test_the_same_scores_are_the_same_bytes(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/gaze); CI installs it")
    tracks = [track(f"face:{i}", 100 * i, 9000, i) for i in range(8)]          # 72,000 rows: two row groups
    write_scores(tmp_path / "a.parquet", tracks, MODEL)
    write_scores(tmp_path / "b.parquet", tracks, MODEL)
    assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()
    meta = pq.ParquetFile(tmp_path / "a.parquet").metadata
    assert meta.num_rows == 72000 > ROW_GROUP_ROWS and meta.num_row_groups == 2
    assert meta.row_group(0).column(0).compression == "ZSTD"


@pytest.mark.parametrize("bad", [
    lambda t: (t[0], t[1], t[2].astype(np.float64), t[3]),                            # not the model's float32
    lambda t: (t[0], t[1], t[2], t[3][:10]),                                          # 10 window lengths, not 11
    lambda t: (t[0], t[1][:5], t[2], t[3]),                                           # more scores than frames
    lambda t: (t[0], t[1], np.where(np.arange(len(t[2])) == 3, np.float32("nan"), t[2]).astype(np.float32), t[3]),
])
def test_scores_that_do_not_fit_or_are_not_finite_are_refused(tmp_path, bad):
    pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/gaze); CI installs it")
    with pytest.raises(ValueError):
        write_scores(tmp_path / "x.parquet", [track("face:0", 0, 20, 1), bad(track("face:1", 0, 20, 2))], MODEL)
    assert not (tmp_path / "x.parquet").exists()


def example():
    """A faces layer of four tracks in two shots (face:2 too short to score: no rows), a diarized transcript whose two
    voices speak while faces of P and Q do, and LR-ASD's scores: (faces, transcript, tracks)."""
    faces = {"people": [{"id": "person:P"}, {"id": "person:Q"}],
             "tracks": [{"id": "face:0", "person": "person:P", "shot": "shot:0"},
                        {"id": "face:1", "person": "person:Q", "shot": "shot:0"},
                        {"id": "face:2", "person": None, "shot": "shot:0"},
                        {"id": "face:3", "person": "person:Q", "shot": "shot:1"}]}
    rng = np.random.default_rng(3)
    p = np.where(np.arange(100) < 50, 1.5, -1.0)[None] + rng.normal(0, 0.8, (WINDOWS, 100))
    q = np.where(np.arange(100) < 50, -1.0, 1.0)[None] + rng.normal(0, 0.8, (WINDOWS, 100))
    tracks = [("face:0", np.arange(0, 100), *mean_of(p)), ("face:1", np.arange(0, 102), *mean_of(q)),
              ("face:3", np.arange(110, 160), *mean_of(rng.normal(0.2, 1, (WINDOWS, 50))))]
    words = [{"w": "x", "start": round(0.1 + 0.15 * k, 3), "end": round(0.2 + 0.15 * k, 3),
              "voice": "voice:A" if k < 13 else "voice:B"} for k in range(26)]
    words += [{"w": "long", "start": 5.0, "end": 7.5, "voice": "voice:B"}, {"w": "y", "start": 4.5, "end": 4.6, "voice": None}]
    utterances = [{"id": "utt:0", "start": 0.1, "end": 2.0, "shot": "shot:0", "voice": "voice:A"},
                  {"id": "utt:1", "start": 2.05, "end": 4.0, "shot": "shot:0", "voice": "voice:B"},
                  {"id": "utt:2", "start": 4.5, "end": 6.0, "shot": "shot:1", "voice": "voice:B"},
                  {"id": "utt:3", "start": 7.0, "end": 7.5, "shot": "shot:2", "voice": "voice:A"}]
    return faces, {"words": words, "utterances": utterances}, tracks


def mean_of(windows):
    windows = windows.astype(np.float32)
    return np.mean(windows, axis=0), windows


def scored_before(faces, tracks):
    """The scored tracks as vision.active-speaker 1.0.0 built them, from the model's arrays (no side file)."""
    position = {t["id"]: i for i, t in enumerate(faces["tracks"])}
    return [{"index": position[f], "face": f, "person": faces["tracks"][position[f]]["person"],
             "shot": faces["tracks"][position[f]]["shot"], "start": round(frames[0] / FPS, 3), "fps": FPS,
             "scores": [round(float(s), 2) for s in score]} for f, frames, score, _ in tracks]


def test_the_layer_is_what_the_stored_scores_and_votes_give(tmp_path):
    pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/gaze); CI installs it")
    faces, transcript, tracks = example()
    write_scores(tmp_path / "s.parquet", tracks, MODEL)
    scored = scored_tracks(*read_scores(tmp_path / "s.parquet"), faces)
    assert scored == scored_before(faces, tracks)                     # the scored tracks 1.0.0 made the layer from
    voices = voice_votes(speaking_intervals(transcript["words"]), scored)
    stored = json.loads(json.dumps(votes_doc(["person:P", "person:Q"], voices), separators=(",", ":")))
    content = layer_content(transcript, faces, scored, 0.3, best_assignment, stored["voices"])
    # the same layer as 1.0.0 made without side files, so its content does not change
    assert content == layer_content(transcript, faces, scored_before(faces, tracks), 0.3, best_assignment)
    assert [r["predicted"]["via"] for r in content["utterances"]] == ["asd", "asd", "asd", "voice_link"]
    assert [(v["voice"], v["person"], v["trusted"]) for v in content["voice_links"]] == [
        ("voice:A", "person:P", True), ("voice:B", "person:Q", True)]
    # the votes: every word with a voice and at most 2 s, its index into the words; each voice's counts are its words'
    assert stored["max_word_seconds"] == 2.0 and stored["people"] == ["person:P", "person:Q"]
    assert [w["word"] for v in stored["voices"] for w in v["words"]] == list(range(26))
    for v in stored["voices"]:
        assert tally(v["words"]) == (v["votes"], v["visible"])
    # the links from the stored votes alone: shares are votes over visible words, matched one to one
    links = link_voices(stored["voices"], stored["people"], best_assignment)
    assert [{"voice": k, "person": l["person"], "word_share": l["word_share"]} for k, l in sorted(links.items())] == [
        {k: v for k, v in link.items() if k != "trusted"} for link in content["voice_links"]]
    # the candidates from the stored scores alone: each face's frames' mean over the utterance, as rows of the file
    utt = transcript["utterances"][1]
    for cand in content["utterances"][1]["candidates"]:
        f = next(t for t in scored if t["face"] == cand["face"])
        i0, i1 = int((utt["start"] - f["start"]) * FPS), int((utt["end"] - f["start"]) * FPS) + 1
        assert cand["score"] == round(float(np.mean(f["scores"][max(0, i0):i1])), 3)


def test_scored_tracks_read_the_file_in_track_order_and_refuse_what_the_faces_layer_lacks():
    faces, _, _ = example()
    # rows out of track order (face:3 first) are read back in it; scores to 2 decimals from float32
    scored = scored_tracks(["face:3", "face:3", "face:0"], [7, 8, 2], [0.125, float(np.float32(0.335)), -1.0], faces)
    assert [(t["index"], t["face"], t["start"], t["scores"]) for t in scored] == [(0, "face:0", 0.08, [-1.0]),
                                                                                  (3, "face:3", 0.28, [0.12, 0.34])]
    # rounded from the float32 value, as the node always has: float32 0.335 is 0.33500001, and -0.005 is -0.0049999999
    assert round_scores(np.array([0.335, -0.005], dtype=np.float32)) == [0.34, -0.0]
    with pytest.raises(ValueError, match="face:9"):
        scored_tracks(["face:9"], [0], [0.0], faces)
    with pytest.raises(ValueError, match="not consecutive"):
        scored_tracks(["face:0", "face:0"], [0, 2], [0.0, 0.0], faces)
