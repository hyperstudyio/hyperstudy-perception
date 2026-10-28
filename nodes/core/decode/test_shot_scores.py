"""The shot scores side file (shot_scores.py): the rows the detector's metrics make, the cuts the stored scores give,
and that the Parquet file has the columns the bundle schema documents and is the same bytes for the same rows. The
last tests run PySceneDetect itself (envs/core has it; the CI node tests do not install it, and skip them):
uv run --project envs/core --with pytest pytest nodes/core/decode"""

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

import shot_scores
from shot_scores import COLUMNS, DESCRIPTION, FILE, ROW_GROUP_ROWS, Recorder, Writer, cuts, frames_between

SCHEMA = Path(__file__).resolve().parents[3] / "schema" / "bundle-0.1.schema.json"
PARAMETERS = {"adaptive_threshold": 3.0, "min_content_val": 15.0, "min_scene_len": 15, "window_width": 2,
              "weights": {"delta_hue": 1.0, "delta_sat": 1.0, "delta_lum": 1.0, "delta_edges": 0.0},
              "luma_only": False, "kernel_size": 5, "frame_size": [256, 144], "frame_rate": "30000/1001"}
RATIO = "adaptive_ratio (w=2)"


def pq():
    return pytest.importorskip("pyarrow.parquet", reason="pyarrow writes the side file (envs/core); CI installs it")


@dataclass(frozen=True)
class Timecode:
    """What the Recorder reads of scenedetect's FrameTimecode: hashable, with seconds."""
    seconds: float


def rows(n, rate=30000 / 1001):
    """n frames' (t, content_val, deltas, adaptive_ratio), a full window from frame 2 to n - 3, as the detector makes them."""
    rng = np.random.default_rng(n)
    out = []
    for k in range(n):
        deltas = [0.0] * 4 if k == 0 else [float(x) for x in rng.uniform(0, 40, 4)]
        out.append((round(k / rate, 6), sum(deltas[:3]) / 3, deltas, float(rng.uniform(0, 6)) if 2 <= k < n - 2 else 0.0))
    return out


def write(path, frames):
    with Writer(path, PARAMETERS) as w:
        for k, (t, score, deltas, ratio) in enumerate(frames):
            w.add(k, t, score, deltas, ratio)
    return path


def test_the_file_has_the_columns_the_bundle_schema_documents(tmp_path):
    frames = rows(9)
    table = pq().read_table(write(tmp_path / "shots_scores.parquet", frames))
    spec = json.loads(SCHEMA.read_text())["$defs"]["parquet"]["shots_scores"]
    assert [(f.name, str(f.type)) for f in table.schema] == [(c["name"], c["arrow"]) for c in spec["columns"]]
    assert tuple(table.column_names) == COLUMNS and all(not f.nullable for f in table.schema)
    assert table["frame"].to_pylist() == list(range(9)) and table["t"].to_pylist() == [f[0] for f in frames]
    assert table["delta_lum"].to_pylist() == [f[2][2] for f in frames]
    assert table["adaptive_ratio"].to_pylist() == [f[3] for f in frames]
    assert json.loads(table.schema.metadata[b"detector"]) == PARAMETERS
    assert shot_scores.read(tmp_path / "shots_scores.parquet")[1] == PARAMETERS
    assert FILE == "layers/shots_scores.parquet" and DESCRIPTION


def test_the_same_rows_are_the_same_bytes(tmp_path):
    frames = rows(ROW_GROUP_ROWS + 7)
    first, second = write(tmp_path / "a.parquet", frames), write(tmp_path / "b.parquet", frames)
    assert first.read_bytes() == second.read_bytes()
    meta = pq().ParquetFile(first).metadata
    assert meta.num_rows == len(frames) and meta.num_row_groups == 2   # groups of ROW_GROUP_ROWS as rows come
    assert meta.row_group(0).column(1).compression == "ZSTD"


def test_a_frame_out_of_order_a_score_that_is_not_finite_or_missing_parameters_are_refused(tmp_path):
    pq()
    with pytest.raises(ValueError, match="frame 2 where frame 1 comes next"):
        with Writer(tmp_path / "x.parquet", PARAMETERS) as w:
            w.add(0, 0.0, 0.0, [0.0] * 4, 0.0)
            w.add(2, 0.1, 1.0, [1.0] * 4, 0.0)
    with pytest.raises(ValueError, match="not finite"):
        with Writer(tmp_path / "y.parquet", PARAMETERS) as w:
            w.add(0, 0.0, float("nan"), [0.0] * 4, 0.0)
    with pytest.raises(ValueError, match="lack"):
        Writer(tmp_path / "z.parquet", {k: v for k, v in PARAMETERS.items() if k != "frame_rate"})


def test_the_recorder_writes_each_frame_once_its_ratio_has_come_and_zero_for_what_the_detector_did_not_compute(tmp_path):
    """As AdaptiveDetector reports: a frame's content_val and deltas when it scores it (none for the first frame), its
    adaptive_ratio window_width frames later, when the frame is centred (none for the first and last window_width)."""
    pq()
    path, n, w = tmp_path / "r.parquet", 8, 2
    rec = Recorder(lambda: Writer(path, PARAMETERS), w, RATIO)
    rec.register_metrics(["content_val", "delta_hue", "delta_sat", "delta_lum", "delta_edges", RATIO])
    times = [Timecode(k / 10) for k in range(n)]
    for k, tc in enumerate(times):
        score = 0.0 if k == 0 else float(10 * k)
        if k:
            rec.set_metrics(tc, {"content_val": score, "delta_hue": k + 0.1, "delta_sat": k + 0.2, "delta_lum": k + 0.3,
                                 "delta_edges": k + 0.4})
        if k >= 2 * w:
            rec.set_metrics(times[k - w], {RATIO: 100.0 + k - w})
        rec.frame(tc, score)
        assert (rec.writer.rows if rec.writer else 0) == max(0, k + 1 - w)   # a frame is written w frames later
    assert rec.finish() == n
    table = pq().read_table(path).to_pydict()
    assert table["frame"] == list(range(n)) and table["t"] == [k / 10 for k in range(n)]
    assert table["content_val"] == [10.0 * k for k in range(n)] and table["delta_edges"] == [0.0] + [k + 0.4 for k in range(1, n)]
    assert table["adaptive_ratio"] == [0.0, 0.0, 102.0, 103.0, 104.0, 105.0, 0.0, 0.0]
    assert not rec._metrics   # nothing left behind: a film's metrics are not held to the end


def test_the_recorder_refuses_a_score_that_is_not_the_one_the_detector_reported(tmp_path):
    pq()
    rec = Recorder(lambda: Writer(tmp_path / "r.parquet", PARAMETERS), 0, RATIO)
    rec.frame(Timecode(0.0), 0.0)
    rec.set_metrics(Timecode(0.1), {"content_val": 5.0})
    with pytest.raises(ValueError, match="scored 6.0, but reported content_val 5.0"):
        rec.frame(Timecode(0.1), 6.0)


def test_a_recorder_that_saw_no_frame_writes_no_file(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        Recorder(lambda: Writer(tmp_path / "r.parquet", PARAMETERS), 2, RATIO).finish()
    assert not (tmp_path / "r.parquet").exists()


def test_frames_between_counts_as_scenedetect_does_from_microsecond_times():
    rate = Fraction(30000, 1001)
    assert frames_between(0.0, 0.5005, rate) == 15 and frames_between(0.0, 0.467133, rate) == 14
    assert frames_between(1.0, 1.0, rate) == 0 and frames_between(0.0, 1.0, Fraction(10)) == 10


def scores(n, peaks, base=1.0):
    """t at 10 fps; content_val `base` but at the peaks, {frame: (content_val, adaptive_ratio)}."""
    t = [k / 10 for k in range(n)]
    content = [base] * n
    ratio = [0.0, 0.0] + [1.0] * (n - 4) + [0.0, 0.0]
    for k, (c, r) in peaks.items():
        content[k], ratio[k] = c, r
    return t, content, ratio


P10 = {**PARAMETERS, "frame_rate": "10/1"}


def test_a_cut_is_a_centred_frame_over_both_thresholds_min_scene_len_after_the_last():
    t, content, ratio = scores(80, {16: (20.0, 4.0), 20: (40.0, 5.0), 30: (14.9, 9.0), 40: (15.0, 3.0), 60: (30.0, 2.99)})
    # 20 is 20 frames after the first frame but only 4 after 16; 30's content_val is under 15; 60's ratio under 3
    assert cuts(t, content, ratio, P10) == [16, 40]
    assert cuts(t, content, ratio, P10, threshold=2.9) == [16, 40, 60]
    assert cuts(t, content, ratio, P10, threshold=4.5) == [20]    # without 16, 20 is a cut
    assert cuts(t, content, ratio, {**P10, "min_scene_len": 5}) == [16, 40]
    assert cuts(t, content, ratio, {**P10, "min_scene_len": 4}) == [16, 20, 40]


def test_frames_without_a_full_window_are_never_cuts_even_at_threshold_zero():
    t, content, ratio = scores(40, {}, base=20.0)
    every = cuts(t, content, ratio, {**P10, "min_scene_len": 1}, threshold=0.0)
    assert every == list(range(2, 38))   # not 0, 1 (the first frame is where counting starts), 38 or 39


def test_a_cut_needs_min_scene_len_frames_after_the_first_frame():
    t, content, ratio = scores(40, {14: (30.0, 9.0), 15: (30.0, 9.0)})
    assert cuts(t, content, ratio, P10) == [15]


# PySceneDetect itself: envs/core has it (with OpenCV); the CI node tests do not install it

FPS, W, H = 10, 240, 136   # under 256 wide: the SceneManager scores it as it is, not downscaled
COLOURS = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (220, 220, 220)]


@pytest.fixture(scope="module")
def cuts_clip(tmp_path_factory):
    """8 s at 10 fps: four 2 s shots of different colours, with a moving bar so the frames within a shot differ."""
    path = tmp_path_factory.mktemp("shots") / "cuts.mp4"
    with av.open(str(path), "w") as out:
        v = out.add_stream("libx264", rate=FPS)
        v.width, v.height, v.pix_fmt = W, H, "yuv420p"
        for i in range(8 * FPS):
            img = np.zeros((H, W, 3), np.uint8)
            img[:] = COLOURS[i // (2 * FPS)]
            img[:, (4 * i) % W:(4 * i) % W + 4] = 0
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            out.mux(v.encode(frame))
        out.mux(v.encode())
    return path


def test_the_detector_cuts_as_scenedetect_detect_does_and_the_stored_scores_give_its_cuts(cuts_clip, tmp_path):
    scenedetect = pytest.importorskip("scenedetect", reason="envs/core has PySceneDetect; the CI node tests do not")
    pq()
    scenes, parameters = shot_scores.detect(cuts_clip, 3.0, tmp_path / "s.parquet")
    plain = scenedetect.detect(str(cuts_clip), scenedetect.AdaptiveDetector(adaptive_threshold=3.0))
    assert [(s.seconds, e.seconds) for s, e in scenes] == [(s.seconds, e.seconds) for s, e in plain]
    assert [s.seconds for s, _ in scenes[1:]] == [2.0, 4.0, 6.0]
    columns, stored = shot_scores.read(tmp_path / "s.parquet")
    assert stored == parameters and parameters["frame_rate"] == "10/1" and parameters["frame_size"] == [W, H]
    assert columns["frame"] == list(range(8 * FPS))
    found = cuts(columns["t"], columns["content_val"], columns["adaptive_ratio"], stored)
    assert [columns["t"][k] for k in found] == [s.seconds for s, _ in scenes[1:]] and found == [20, 40, 60]
    w = stored["window_width"]
    assert columns["adaptive_ratio"][:w] == [0.0] * w == columns["adaptive_ratio"][-w:]
    assert columns["content_val"][0] == 0.0 and all(c > 0 for c in columns["content_val"][1:])   # the bar moves
    weights = stored["weights"]
    for k in range(1, 8 * FPS):   # content_val is the weighted mean of the deltas, to the bit
        # added in order, one by one, as scenedetect's sum() of numpy floats does (Python 3.12's sum() of floats
        # compensates, and can differ in the last bit)
        total = 0.0
        for d in shot_scores.DELTAS:
            total = total + columns[d][k] * weights[d]
        assert columns["content_val"][k] == total / sum(abs(x) for x in weights.values())


def test_the_detector_writes_the_same_bytes_twice(cuts_clip, tmp_path):
    pytest.importorskip("scenedetect", reason="envs/core has PySceneDetect; the CI node tests do not")
    pq()
    shot_scores.detect(cuts_clip, 3.0, tmp_path / "a.parquet")
    shot_scores.detect(cuts_clip, 3.0, tmp_path / "b.parquet")
    assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()


@pytest.fixture(scope="module")
def vfr_clip(tmp_path_factory):
    """8 s at a variable frame rate (each frame 1/60, 1/30 or 1/20 s), its colour changing every 14 or 15 frames: more
    often than min_scene_len lets the detector cut, so the frames between cuts are counted from uneven times."""
    path = tmp_path_factory.mktemp("vfr") / "vfr.mp4"
    rng = np.random.default_rng(0)
    with av.open(str(path), "w") as out:
        v = out.add_stream("libx264", rate=30)
        v.width, v.height, v.pix_fmt = W, H, "yuv420p"
        v.codec_context.time_base = v.time_base = Fraction(1, 90000)
        pts = 0
        for i in range(240):
            img = np.zeros((H, W, 3), np.uint8)
            img[:] = COLOURS[(i // 15) % 4 if (i // 15) % 2 else (i // 14) % 4]
            img[:, (4 * i) % W:(4 * i) % W + 4] = 0
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 90000)
            pts += int(rng.choice([1500, 3000, 4500]))
            out.mux(v.encode(frame))
        out.mux(v.encode())
    return path


def test_on_variable_frame_rate_video_the_stored_scores_give_scenedetects_cuts_at_any_threshold(vfr_clip, tmp_path):
    scenedetect = pytest.importorskip("scenedetect", reason="envs/core has PySceneDetect; the CI node tests do not")
    pq()
    shot_scores.detect(vfr_clip, 3.0, tmp_path / "s.parquet")
    columns, stored = shot_scores.read(tmp_path / "s.parquet")
    steps = {round(b - a, 4) for a, b in zip(columns["t"], columns["t"][1:])}
    assert steps == {0.0167, 0.0333, 0.05}   # the frame times are uneven
    counts = set()
    for threshold in (1.5, 2.5, 3.0, 5.0, 8.0):
        plain = scenedetect.detect(str(vfr_clip), scenedetect.AdaptiveDetector(adaptive_threshold=threshold))
        found = cuts(columns["t"], columns["content_val"], columns["adaptive_ratio"], stored, threshold)
        assert [columns["t"][k] for k in found] == [s.seconds for s, _ in plain[1:]], threshold
        counts.add(len(found))
    assert len(counts) > 1   # the thresholds tell cuts apart
    every = cuts(columns["t"], columns["content_val"], columns["adaptive_ratio"], {**stored, "min_scene_len": 1})
    assert len(every) > len(cuts(columns["t"], columns["content_val"], columns["adaptive_ratio"], stored))   # it binds
