"""Gaze targets and events (gaze_targets.py), without the model. Run: uv run --project envs/gaze --with pytest pytest nodes/vision/gaze"""

import numpy as np
import pytest

from gaze_targets import (HEATMAP_SIZE, MASS_MIN, Box, Polygon, candidates_at, events_from, face_index, head_box,
                          heads_by_frame, interpolate_box, layer_content, layer_target, object_at, objects_by_shot, peak_of,
                          sample_of, target_for, tracks_of)

W, H = 1920, 1080


def cell_heat(col, row, value=1.0, floor=0.0):
    """A heatmap that is `floor` everywhere and `value` at one cell."""
    heat = np.full((HEATMAP_SIZE, HEATMAP_SIZE), floor, np.float32)
    heat[row, col] = value
    return heat


def cell_box(col, row, cols=1, rows=1):
    """A box covering exactly these heatmap cells of a W x H frame."""
    return Box(col * W / 64, row * H / 64, (col + cols) * W / 64, (row + rows) * H / 64)


def test_a_head_box_is_the_face_box_scaled_about_its_centre_and_clipped_to_the_frame():
    assert head_box([100, 200, 50, 50], W, H, 1.2) == pytest.approx([95, 195, 155, 255])
    assert head_box([0, 1060, 40, 40], W, H, 1.2) == pytest.approx([0, 1056, 44, 1080])


def test_coverage_is_the_share_of_each_heatmap_cell_a_box_covers():
    whole = Box(0, 0, W, H).coverage(W, H)
    assert whole.dtype == np.float32 and whole.shape == (64, 64) and (whole == 1).all()
    one = cell_box(5, 7).coverage(W, H)
    assert one[7, 5] == 1 and one.sum() == 1
    half = Box(5 * W / 64, 7 * H / 64, 5.5 * W / 64, 8 * H / 64).coverage(W, H)   # half a cell wide: 2 of its 4 grid columns
    assert half[7, 5] == 0.5 and half.sum() == 0.5


def test_coverage_fills_a_box_narrower_than_a_grid_column_as_opencv_does():
    """cv2.rectangle(grid, (round(x1), ...), (round(x2) - 1, ...), 1, -1), the bake-off's raster, takes its corners in
    either order and fills both ends: a box whose ends round to the same grid column fills that column and the one
    before it. Checked against OpenCV 5.0 on 20000 random boxes (all equal)."""
    x = 40 * W / 256   # grid column 40
    cov = Box(x, 0, x, H).coverage(W, H)
    assert cov[0, 9] == 0.25 and cov[0, 10] == 0.25 and cov.sum() == 0.5 * 64
    assert Box(W + 100, 0, W + 200, H).coverage(W, H).sum() == 0   # outside the frame: nothing


def test_a_track_box_between_samples_is_interpolated_and_none_outside_its_span():
    samples = [{"t": 1.0, "box": [0, 0, 10, 10]}, {"t": 2.0, "box": [10, 20, 30, 40]}]
    assert interpolate_box(samples, 1.5) == [5, 10, 20, 25]
    assert interpolate_box(samples, 1.0) == [0, 0, 10, 10] and interpolate_box(samples, 2.0) == [10, 20, 30, 40]
    assert interpolate_box(samples, 0.9) is None and interpolate_box(samples, 2.1) is None


def test_the_peak_is_the_centre_of_the_heatmaps_hottest_cell_in_frame_pixels():
    assert peak_of(cell_heat(10, 20), W, H) == (10.5 / 64 * W, 20.5 / 64 * H)


def test_a_low_in_frame_score_is_out_whatever_the_heatmap():
    target, score, _ = target_for(cell_heat(5, 7), 0.2, [(3, cell_box(5, 7))], W, H)
    assert (target, score) == ("out", pytest.approx(0.8))


def test_the_candidate_holding_the_peak_is_the_target_with_its_share_of_the_mass():
    heat = cell_heat(5, 7, value=3.0, floor=0.0)
    heat[0, 0] = 1.0
    target, score, _ = target_for(heat, 0.9, [(1, cell_box(30, 30)), (2, cell_box(5, 7))], W, H)
    assert (target, score) == (2, pytest.approx(0.75))
    # two candidates hold the peak: the one with more mass wins
    heat[7, 6] = 2.0
    target, score, _ = target_for(heat, None, [(1, cell_box(5, 7)), (2, cell_box(5, 7, cols=2))], W, H)
    assert (target, score) == (2, pytest.approx(5 / 6))


def test_with_the_peak_in_no_candidate_a_candidate_needs_mass_min_of_the_heatmap():
    heat = cell_heat(40, 40, value=1.0)
    heat[10, 10] = 0.9   # beside the peak, inside a candidate
    near = [(4, cell_box(10, 10))]
    target, score, _ = target_for(heat, 0.9, near, W, H)
    assert score == pytest.approx(0.9 / 1.9) and score >= MASS_MIN and target == 4
    heat[10, 10] = 0.2
    target, score, _ = target_for(heat, 0.9, near, W, H)
    assert target is None and score == pytest.approx(0.2 / 1.2)
    assert target_for(heat, 0.9, [], W, H)[:2] == (None, 0.0)


def faces_layer():
    return {"sample_fps": 8, "tracks": [
        {"id": "face:0", "shot": "shot:0", "samples": [{"t": 0.0, "box": [100, 100, 100, 100]}, {"t": 0.133, "box": [110, 100, 100, 100]}]},
        {"id": "face:1", "shot": "shot:0", "samples": [{"t": 0.133, "box": [1000, 100, 100, 100]}]},
        {"id": "face:2", "shot": "shot:1", "samples": [{"t": 0.133, "box": [500, 500, 100, 100]}]}]}


def test_candidates_are_the_other_faces_of_the_shot_present_at_that_time_as_head_boxes():
    tracks = tracks_of(faces_layer())
    found = candidates_at(0, "shot:0", 0.133, tracks, {}, W, H, 1.2)
    assert [c for c, _ in found] == [1] and found[0][1].geom == pytest.approx((990, 90, 1110, 210))
    assert candidates_at(0, "shot:0", 0.0, tracks, {}, W, H, 1.2) == []   # face:1 starts later
    assert candidates_at(2, "shot:1", 0.133, tracks, {}, W, H, 1.2) == []  # another shot


def objects_layer():
    square = [[600, 600], [700, 600], [700, 700], [600, 700]]
    return {"tracks": [
        {"id": "obj:3", "shot": "shot:0", "samples": [{"t": 0.0, "box": [600, 600, 100, 100], "poly": square},
                                                       {"t": 0.333, "box": [0, 0, 10, 10], "poly": [[0, 0], [10, 0]]}]},
        {"id": "obj:1", "shot": "shot:1", "samples": [{"t": 0.1, "box": [10, 20, 30, 40], "poly": []}]},
        {"id": "obj:4", "shot": "shot:0", "samples": []}],
        "rejected_tracks": [{"id": "obj:9", "shot": "shot:0", "samples": [{"t": 0.0, "box": [0, 0, 5, 5]}]}]}


def test_object_candidates_are_the_shots_object_tracks_sampled_within_a_sixth_of_a_second():
    objects = objects_by_shot(objects_layer())
    assert {shot: [o["id"] for o in tracks] for shot, tracks in objects.items()} == {"shot:0": ["obj:3"], "shot:1": ["obj:1"]}
    tracks = tracks_of(faces_layer())
    found = candidates_at(0, "shot:0", 0.133, tracks, objects, W, H, 1.2)
    assert [c for c, _ in found] == [1, "obj:3"]          # faces first, then objects, each in their layer's order
    assert isinstance(found[1][1], Polygon) and found[1][1].geom.tolist() == objects_layer()["tracks"][0]["samples"][0]["poly"]
    # the nearer sample at 0.2 is the one at 0.333, with a two-point outline: its box stands for it
    [(_, region)] = [c for c in candidates_at(1, "shot:0", 0.2, tracks, objects, W, H, 1.2) if c[0] == "obj:3"]
    assert isinstance(region, Box) and region.geom == (0, 0, 10, 10)
    assert [c for c, _ in candidates_at(2, "shot:1", 0.133, tracks, objects, W, H, 1.2)] == ["obj:1"]   # box only
    assert [c for c, _ in candidates_at(2, "shot:1", 0.3, tracks, objects, W, H, 1.2)] == []           # 0.2 s away
    samples = [{"t": 0.0}, {"t": 0.2}, {"t": 0.4}]
    assert object_at(samples, 0.1) == 0 and object_at(samples, 0.3) == 1   # equally near: the first
    assert object_at(samples, 0.4 + 1 / 6) == 2 and object_at(samples, 0.6) is None


def test_an_object_outline_holding_the_peak_is_the_target():
    heat = cell_heat(21, 38, value=3.0)       # peak in frame pixels (645, 649.7): inside the square
    heat[0, 0] = 1.0
    square = Polygon([[600, 600], [700, 600], [700, 700], [600, 700]])
    cov = square.coverage(W, H)
    assert cov[38, 21] == 1 and square.contains(645, 649.7) and not square.contains(599.9, 650)
    assert square.coverage(W, H) is cov           # kept: every head of the frame asks for it
    target, score, _ = target_for(heat, 0.9, [(1, cell_box(40, 40)), ("obj:3", square)], W, H)
    assert (target, score) == ("obj:3", pytest.approx(0.75))


def test_heads_are_grouped_by_the_frame_their_sample_time_names():
    frames = heads_by_frame(tracks_of(faces_layer()), 29.97002997002997)
    assert sorted(frames) == [0, 4] and [f for f, _, _ in frames[4]] == [0, 1, 2]
    with pytest.raises(ValueError, match="not a frame time"):
        heads_by_frame({0: {"shot": "shot:0", "samples": [{"t": 0.05, "box": [0, 0, 1, 1]}]}}, 30.0)
    with pytest.raises(ValueError, match="not a face track id"):
        face_index("body:3")


def run(face, targets, start=0.0, step=0.133, score=0.8):
    return {"face_track": face, "samples": [{"t": round(start + i * step, 3), "target": t, "target_score": score}
                                            for i, t in enumerate(targets)]}


def test_looks_at_needs_three_samples_of_one_target_without_a_gap():
    events = events_from([run(0, [1, 1, None, 1, 1, 1, "out", "out", "out"])])
    assert [(e["kind"], e["to_face_track"], e["start"], e["end"]) for e in events] == [("looks_at", 1, 0.399, 0.665)]
    gap = {"face_track": 0, "samples": [{"t": t, "target": 1, "target_score": 0.5} for t in (0.0, 0.133, 0.6, 0.733)]}
    assert events_from([gap]) == []   # 0.133 -> 0.6 is more than MAX_GAP_S apart: two runs of two


def test_mutual_gaze_and_joint_attention_from_overlapping_looks():
    events = events_from([run(0, [1, 1, 1, 1]), run(1, [0, 0, 0], start=0.133, score=0.6), run(2, [1, 1, 1, 1], score=0.4)])
    kinds = [(e["kind"], e["from_face_track"], e["to_face_track"], e["also_face_track"], e["start"], e["end"], e["score"])
             for e in events if e["kind"] != "looks_at"]
    assert kinds == [("joint_attention", 0, 1, 2, 0.0, 0.399, 0.6), ("mutual_gaze", 0, 1, None, 0.133, 0.399, 0.7)]


def test_events_sort_by_track_number_not_by_the_ids_text():
    """The bake-off's tracks were integers: face 10 sorts after face 2 (as text, "face:10" < "face:2")."""
    events = events_from([run(10, [3, 3, 3]), run(2, [3, 3, 3])])
    assert [(e["kind"], e["from_face_track"]) for e in events] == [("joint_attention", 2), ("looks_at", 2), ("looks_at", 10)]
    assert events[0]["also_face_track"] == 10


def test_object_targets_make_looks_and_joint_attention_on_the_object():
    events = events_from([run(0, ["obj:3"] * 3), run(1, ["obj:3"] * 4)])
    assert [(e["kind"], e["from_face_track"], e["to_face_track"], e["object"], e["also_face_track"]) for e in events] == [
        ("joint_attention", 0, None, "obj:3", 1), ("looks_at", 0, None, "obj:3", None), ("looks_at", 1, None, "obj:3", None)]


def test_the_layer_writes_ids_as_convert_bakeoff_does():
    sample = sample_of(0.133, None, 0.12345, 1, 0.98765, (100.04, 200.06))
    assert sample == {"t": 0.133, "peak": [100.0, 200.1], "inout": 0.123, "target": 1, "target_score": 0.988}
    tracks = [{"face_track": 0, "samples": [sample]}]
    events = [{"kind": "joint_attention", "from_face_track": 0, "to_face_track": None, "object": "obj:3",
               "also_face_track": 2, "start": 0.0, "end": 0.4, "score": 0.5}]
    out = layer_content(8, tracks, events)
    assert out["tracks"] == [{"face": "face:0", "samples": [{"t": 0.133, "peak": [100.0, 200.1], "in_frame": 0.123,
                                                             "target": "face:1", "target_score": 0.988}]}]
    assert out["events"] == [{"id": "gaze:0", "kind": "joint_attention", "from": "face:0", "to": None, "object": "obj:3",
                              "also": "face:2", "start": 0.0, "end": 0.4, "score": 0.5}]
    assert [layer_target(t) for t in (None, "out", "obj:1", 4)] == [None, "out", "obj:1", "face:4"]
    with pytest.raises(ValueError):
        layer_target("camera?")
