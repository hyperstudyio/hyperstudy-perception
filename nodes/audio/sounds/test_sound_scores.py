"""The raw side files (sound_scores.py): PANNs' segments give back its frame-wise scores exactly, the files have the
columns the bundle schema documents and are the same bytes for the same scores, and re-thresholding what they hold
gives the layer's events."""

import json
from pathlib import Path

import numpy as np
import pytest

from sound_events import (PANNS_FPS, PE_HOP, QUERIES, audioset_labels, plan_chunks, sound_events, spans, stitch,
                          step_max, tag_rows)
from sound_scores import (CHECK_SEGMENTS, PANNS_SEGMENT, QUERIES_DESCRIPTION, QUERIES_FILE, ROW_GROUP_ROWS, TAGS_DESCRIPTION, TAGS_FILE,
                          panns_framewise, panns_segments, read_queries, read_tags, write_queries, write_tags)

SCHEMA = Path(__file__).resolve().parents[3] / "schema" / "bundle-0.1.schema.json"
LABELS = audioset_labels()
HOP = 320                  # PANNs' frame, in samples at 32 kHz
GRID = HOP * PANNS_SEGMENT


def cnn14_framewise(segments_of, n_samples):
    """Cnn14_DecisionLevelMax's frame-wise output for n samples, as panns/pytorch_utils.py builds it: a centred STFT
    has n // hop + 1 frames, the CNN gives one segment per 32 of them, each repeated 32 times and the last to the end."""
    frames = n_samples // HOP + 1
    segments = segments_of(frames // PANNS_SEGMENT)
    out = np.repeat(segments, PANNS_SEGMENT, axis=0)
    return np.concatenate([out, np.repeat(out[-1:], frames - len(out), axis=0)])


def scores(rng, rows, width):
    return rng.random((rows, width), dtype=np.float32)


# --- PANNs: segments and frames ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("n_samples", [32000, 32000 * 119 + 1280, 32000 * 3 + GRID - HOP, 32000 * 7 + 5])
def test_panns_segments_give_back_its_frame_wise_scores_exactly(n_samples):
    rng = np.random.default_rng(n_samples)
    framewise = cnn14_framewise(lambda n: scores(rng, n, 527), n_samples)
    segments = panns_segments(framewise)
    assert len(segments) == len(framewise) // PANNS_SEGMENT
    assert np.array_equal(panns_framewise(segments, len(framewise)), framewise)


def test_chunked_panns_scores_are_segments_too():
    """Chunks start on Cnn14's 32-frame grid (sound_events.PANNS_GRID), so the stitched output is still whole segments
    repeated, the last chunk's last one to the end."""
    rng = np.random.default_rng(0)
    n = 37 * GRID + 4321
    pieces = [(c, cnn14_framewise(lambda k: scores(rng, k, 8), c.end - c.start)) for c in plan_chunks(n, 8 * GRID, 2 * GRID)]
    assert len(pieces) > 2
    framewise = stitch(pieces, HOP, axis=0)
    assert np.array_equal(panns_framewise(panns_segments(framewise), len(framewise)), framewise)


def test_frame_wise_scores_that_are_not_repeated_segments_are_refused():
    rng = np.random.default_rng(1)
    framewise = cnn14_framewise(lambda n: scores(rng, n, 4), 32000 * 2)
    framewise[40, 2] += 0.01
    with pytest.raises(ValueError, match="not its 32-frame segments repeated"):
        panns_segments(framewise)
    with pytest.raises(ValueError, match="do not make"):
        panns_framewise(framewise[:3], 3 * PANNS_SEGMENT + PANNS_SEGMENT)
    with pytest.raises(ValueError, match="no whole 32-frame segment"):
        panns_segments(framewise[:PANNS_SEGMENT - 1])


@pytest.mark.parametrize("frame", [5, (CHECK_SEGMENTS + 3) * PANNS_SEGMENT + 7, -1])
def test_every_frame_is_checked_and_the_segments_do_not_hold_the_frames(frame):
    """Compared CHECK_SEGMENTS at a time: a frame off in a later block, or in the frames past the last segment, is
    refused. The segments are a copy, so the frames can be freed."""
    rng = np.random.default_rng(2)
    framewise = cnn14_framewise(lambda n: scores(rng, n, 3), (CHECK_SEGMENTS + 40) * GRID + 9 * HOP)
    segments = panns_segments(framewise)
    assert segments.base is None and len(segments) > CHECK_SEGMENTS
    framewise[frame, 1] = np.nextafter(framewise[frame, 1], np.float32(2))
    with pytest.raises(ValueError, match="not its 32-frame segments repeated"):
        panns_segments(framewise)


# --- the files --------------------------------------------------------------------------------------------------------

def soundtrack(seed, seconds=7.3):
    """Scores shaped as the node has them for a soundtrack of `seconds`: PANNs frame-wise, mn40 per step, PE per frame."""
    rng = np.random.default_rng(seed)
    n32, n48 = round(seconds * 32000), round(seconds * 48000)
    # a few loud labels, so steps clear the thresholds
    framewise = cnn14_framewise(lambda n: scores(rng, n, 527) ** 6, n32)
    mn40 = scores(rng, int(np.ceil(seconds)), 527) ** 6
    probs = scores(rng, len(QUERIES), n48 // PE_HOP + 1) ** 3
    return framewise, mn40, probs


def write(tmp_path, seed, name="a"):
    framewise, mn40, probs = soundtrack(seed)
    tags = write_tags(tmp_path / f"{name}_tags.parquet", panns_segments(framewise), len(framewise), mn40, LABELS)
    queries = write_queries(tmp_path / f"{name}_queries.parquet", probs, QUERIES)
    return tags, queries


def test_the_files_have_the_columns_the_bundle_schema_documents(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side files (envs/audio-events); CI installs it")
    tags, queries = write(tmp_path, 2)
    spec = json.loads(SCHEMA.read_text())["$defs"]["parquet"]
    for path, name in ((tags, "sounds_tags"), (queries, "sounds_queries")):
        table = pq.read_table(path)
        assert [(f.name, str(f.type)) for f in table.schema] == [(c["name"], c["arrow"]) for c in spec[name]["columns"]]
        assert all(not f.nullable for f in table.schema)
        assert table.schema.metadata[b"side"] == name.encode()
    framewise, mn40, probs = soundtrack(2)
    t = pq.read_table(tags)
    n_panns = len(framewise) // PANNS_SEGMENT
    assert t["model"].to_pylist() == ["panns"] * n_panns + ["mn40"] * len(mn40)
    starts, ends = t["start"].to_pylist(), t["end"].to_pylist()
    assert starts[:3] == [0.0, 0.32, 0.64] and ends[n_panns - 1] == len(framewise) / PANNS_FPS
    assert starts[n_panns:] == [float(i) for i in range(len(mn40))] and ends[-1] == len(mn40)
    assert json.loads(t.schema.metadata[b"labels"]) == LABELS
    q = pq.read_table(queries)
    assert q["start"].to_pylist()[:3] == [0.0, 0.04, 0.08] and len(q) == probs.shape[1]
    assert json.loads(q.schema.metadata[b"queries"]) == QUERIES
    assert (TAGS_FILE, QUERIES_FILE) == ("layers/sounds_tags.parquet", "layers/sounds_queries.parquet")
    assert TAGS_DESCRIPTION and QUERIES_DESCRIPTION


def test_what_the_files_hold_is_the_models_scores_bit_for_bit_and_gives_back_the_layer(tmp_path):
    pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side files (envs/audio-events); CI installs it")
    framewise, mn40, probs = soundtrack(3)
    tags, queries = write(tmp_path, 3)
    got, got_q = read_tags(tags), read_queries(queries)
    assert got["labels"] == LABELS and got_q["queries"] == QUERIES
    back = panns_framewise(got["panns"], got["panns_frames"])
    assert back.dtype == np.float32 and np.array_equal(back, framewise)
    assert np.array_equal(got["mn40"], mn40) and np.array_equal(got_q["probs"], probs)

    def layer(framewise, mn40, probs):
        steps = {"panns": tag_rows(step_max(framewise, len(mn40)), LABELS), "mn40": tag_rows(mn40, LABELS)}
        return sound_events(steps, {q: spans(p, 48000) for q, p in zip(QUERIES, probs)})
    events = layer(framewise, mn40, probs)
    assert {e["sources"][0] for e in events} >= {"panns", "mn40", "pe-a-frame"}
    assert layer(back, got["mn40"], got_q["probs"]) == events


def test_the_same_scores_are_the_same_bytes(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side files (envs/audio-events); CI installs it")
    for first, second in zip(write(tmp_path, 4, "a"), write(tmp_path, 4, "b")):
        assert first.read_bytes() == second.read_bytes()
    rng = np.random.default_rng(5)
    big = write_queries(tmp_path / "big.parquet", scores(rng, len(QUERIES), ROW_GROUP_ROWS + 9), QUERIES)
    meta = pq.ParquetFile(big).metadata
    assert meta.num_rows == ROW_GROUP_ROWS + 9 and meta.num_row_groups == 2   # groups of ROW_GROUP_ROWS
    probs = meta.row_group(0).column(2)
    assert probs.compression == "ZSTD" and "BYTE_STREAM_SPLIT" in probs.encodings   # the option names a real column


def test_scores_that_are_not_float32_probabilities_are_refused(tmp_path):
    pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side files (envs/audio-events); CI installs it")
    rng = np.random.default_rng(6)
    good = scores(rng, len(QUERIES), 10)
    for bad, problem in ((good.astype(np.float64), "not float32"), (good[:-1], r"not float32 \(rows, 21\)"),
                         (np.where(good > 0.5, np.nan, good).astype(np.float32), "outside"), (good + 1, "outside")):
        with pytest.raises(ValueError, match=problem):
            write_queries(tmp_path / "x.parquet", bad, QUERIES)
    with pytest.raises(ValueError, match="do not make"):
        write_tags(tmp_path / "t.parquet", scores(rng, 3, 527), 200, scores(rng, 2, 527), LABELS)
