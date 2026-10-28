"""No people inside object masks (person_masks.py), without the models.
Run: uv run --no-project --with pytest --with numpy --with av==16.1.0 --with pillow --with pyarrow==26.0.0 --with-editable ./sdk pytest -q nodes/vision/objects"""

import numpy as np
import pytest

from person_masks import COVER_SHARE, ENCLOSE_SHARE, PersonMasks, check

H, W = 40, 60


def rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def outer_box(mask):
    """A stand-in for geometry.filled_outline: the mask's bounding box, filled (an outline around every hole)."""
    out = np.zeros_like(mask)
    if mask.any():
        ys, xs = np.nonzero(mask)
        out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return out


def test_a_mask_with_no_person_pixels_is_kept_as_it_is():
    chair = rect(0, 0, 10, 10)
    reason, kept, share, enclosed = check(chair, [rect(30, 0, 40, 40)], outer_box)
    assert (reason, share, enclosed) == ("kept", 0.0, 0.0) and kept is chair
    assert check(chair, [], outer_box)[0] == "kept"


def test_a_chair_with_its_sitter_loses_the_sitter():
    """The chair's mask holds the sitter's legs (a third of it): those pixels go, the chair stays."""
    chair = rect(0, 10, 30, 40)                 # 900 px
    sitter = rect(20, 0, 40, 40)                # overlaps the chair in x 20-29: 300 px
    reason, kept, share, enclosed = check(chair, [sitter], outer_box)
    assert reason == "trimmed" and share == pytest.approx(1 / 3)
    assert kept.sum() == 600 and not (kept & sitter).any()
    assert enclosed < ENCLOSE_SHARE


def test_a_mask_that_is_mostly_people_is_dropped():
    blob = rect(0, 0, 20, 20)
    person = rect(0, 0, 20, 12)                 # 60% of the mask
    assert check(blob, [person], outer_box)[:1] == ("covers_person",) and 0.6 >= COVER_SHARE


def test_an_outline_around_a_person_is_dropped():
    """A neon frame around the set: its mask is a ring with no person pixels, its outline (the filled outer contour,
    which the layer's polygon draws) holds a whole person."""
    ring = rect(0, 0, 60, 40) & ~rect(4, 4, 56, 36)
    person = rect(20, 10, 30, 30)
    reason, kept, share, enclosed = check(ring, [person], outer_box)
    assert reason == "encloses_person" and share == 0.0 and enclosed == 1.0


def test_people_are_kept_packed_and_found_at_the_nearest_time():
    people = PersonMasks(W, H)
    people.add(1.0, np.ones((5, 5), bool), (10, 20))
    people.add(1.0, np.ones((2, 2), bool), (0, 0))
    people.add(1.333, np.ones((1, 1), bool), (59, 39))
    assert people.nearest(1.1, 1 / 6) == 1.0 and people.nearest(1.2, 1 / 6) == 1.333
    assert people.nearest(1.6, 1 / 6) is None and PersonMasks(W, H).nearest(1.0, 1.0) is None
    first, second = people.people(1.0)
    assert first.shape == (H, W) and first.sum() == 25 and first[20:25, 10:15].all()
    assert second.sum() == 4 and people.people(1.333)[0][39, 59]
