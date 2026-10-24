"""Thumbnail choice (thumbnails.py). Run: uv run --project envs/faces --with pytest pytest nodes/vision/face-track"""

import numpy as np

from thumbnails import choose, shortlist

rng = np.random.default_rng(0)


def face(colour, size=64):
    """A textured face-like crop: a base colour with detail, so sharpness is comparable between crops."""
    base = np.array(colour, dtype=np.float64)
    texture = rng.normal(0, 18, (size, size, 1))
    return np.clip(base + texture, 0, 255).astype(np.uint8)


def with_graphic(image):
    """White on-screen graphics over the left third of the face (clip 2's panelist circles and name cards)."""
    out = image.copy()
    out[:, :22] = 255
    out[::8, :22] = 0   # hard edges: graphics are much "sharper" than a face
    return out


def row(t, side, yaw=0.0, score=0.99):
    return {"t": t, "box": [0, 0, side, side], "yaw": yaw, "score": score}


RED_LIT = (120, 8, 10)        # a face under red stage light: dark, almost one channel
WELL_LIT = (200, 160, 140)    # a face under white light (warm skin, but no strong cast)
DIM = (110, 90, 80)           # a face against clip 2's black backdrop, fairly dark but neutral


def pick_time(tracks, image_at):
    candidates = shortlist(tracks)
    index, _ = choose(candidates, [image_at(c["t"]) for c in candidates])
    return candidates[index]["t"]


def test_a_smaller_well_lit_face_beats_a_larger_red_lit_one():
    tracks = [{"rows": [row(1.0, 200)]}, {"rows": [row(5.0, 150)]}]
    assert pick_time(tracks, {1.0: face(RED_LIT), 5.0: face(WELL_LIT)}.get) == 5.0


def test_graphics_over_a_face_do_not_win_on_sharpness():
    tracks = [{"rows": [row(1.0, 214)]}, {"rows": [row(5.0, 217)]}]
    images = {1.0: with_graphic(face(DIM)), 5.0: face(DIM)}
    assert pick_time(tracks, images.get) == 5.0


def test_geometry_still_decides_between_faces_of_similar_quality():
    tracks = [{"rows": [row(1.0, 200)]}, {"rows": [row(5.0, 120)]}, {"rows": [row(9.0, 200, yaw=1.2)]}]
    assert pick_time(tracks, lambda t: face(WELL_LIT)) == 1.0   # largest and frontal


def test_a_person_with_one_track_still_gets_a_choice():
    track = {"rows": [row(t, 240 if t < 3 else 200) for t in np.arange(0, 6, 0.25).round(2)]}
    times = [c["t"] for c in shortlist([track])]
    assert len(times) == 6 and all(abs(a - b) >= 1.0 for i, a in enumerate(times) for b in times[i + 1:])
    assert pick_time([track], lambda t: face(RED_LIT) if t < 3 else face(WELL_LIT)) >= 3


def test_shortlist_goes_round_the_tracks():
    many_red = {"rows": [row(t, 200) for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0)]}
    small_clean = {"rows": [row(20.0, 100)]}
    assert 20.0 in [c["t"] for c in shortlist([many_red, small_clean])]
