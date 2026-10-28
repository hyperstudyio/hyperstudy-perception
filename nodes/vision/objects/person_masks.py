"""No people inside object masks. NumPy only: test_person_masks.py runs without the models.

SAM 3 outlines what a text prompt names, and for a seated person's chair that is the chair with its sitter (OOO and
clip 2's director's chairs); a neon frame around the set has an outline (the outer contour, geometry.mask_polygon) that
holds every panelist. The bodies give the people: each body crop's "person" mask (garments.py) is that body's pixels.
Before an object sample becomes part of a track, its mask is checked against the people in the frame (PersonMasks, at
the body pass's times; a 6 fps object sample takes the nearest of them, at most half a body-pass step away):

- covers_person: at least COVER_SHARE of the mask is person pixels (the mask is mostly people): the sample is dropped.
- trimmed: some of it is: those pixels are taken out, and the sample is the rest (a chair without its sitter).
- encloses_person: the outline of what is left (its outer contour, filled: what the layer's polygon draws) holds at least
  ENCLOSE_SHARE of one person's pixels (a frame around people, a chair back seen through its sitter): dropped.
- kept: no person pixels.

Things worn or carried (the name's class says so, and every body garment) are on a person by nature and are not
checked. A track left with fewer samples than the node's minimum is dropped whole. Every sample's numbers and outcome
are kept in the side file (object_raw.py), so the thresholds can be revisited.
"""

from __future__ import annotations

import numpy as np

COVER_SHARE = 0.5     # a mask this much person is the person, not an object
ENCLOSE_SHARE = 0.5   # an outline holding this much of one person's pixels draws around that person
REASONS = ("kept", "trimmed", "covers_person", "encloses_person")


class PersonMasks:
    """The people's masks at each of a shot's body-pass times, full-frame, as packed bits (an eighth of a frame's
    bytes each): {t: [one per person]}. people(t) unpacks them, keeping the last time's."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.at: dict[float, list[np.ndarray]] = {}
        self._last: tuple[float, list[np.ndarray]] | None = None

    def add(self, t: float, mask, offset):
        """One person's mask (a crop's, at offset (x0, y0) in the frame)."""
        full = np.zeros((self.height, self.width), bool)
        x0, y0 = offset
        h, w = mask.shape
        full[y0:y0 + h, x0:x0 + w] = mask
        self.at.setdefault(t, []).append(np.packbits(full))
        self._last = None

    def nearest(self, t: float, within: float) -> float | None:
        """The body-pass time nearest t, if within `within` seconds (the earlier on a tie)."""
        if not self.at:
            return None
        best = min(self.at, key=lambda u: (abs(u - t), u))
        return best if abs(best - t) <= within + 1e-6 else None

    def people(self, t: float) -> list[np.ndarray]:
        if self._last is None or self._last[0] != t:
            n = self.height * self.width
            self._last = (t, [np.unpackbits(p, count=n).reshape(self.height, self.width).view(bool) for p in self.at[t]])
        return self._last[1]


def bounds(mask) -> tuple[slice, slice]:
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def check(mask, people: list[np.ndarray], fill) -> tuple[str, np.ndarray, float, float]:
    """(reason, the mask to keep, person share, enclosed share) for one object sample's full-frame mask against the
    people in its frame (full-frame masks). fill(mask) -> the filled outer contour of a mask (geometry.filled_outline),
    the same shape. Person share: the part of the mask that is person pixels; enclosed share: the most of one person's
    pixels inside the filled outline of what is kept."""
    if not people or not mask.any():
        return "kept", mask, 0.0, 0.0
    window = bounds(mask)   # the outline of a part of the mask lies inside the mask's bounds
    m = mask[window]
    anyone = np.logical_or.reduce([p[window] for p in people])
    on_people = int(np.logical_and(m, anyone).sum())
    share = on_people / int(m.sum())
    if share >= COVER_SHARE:
        return "covers_person", mask, share, 0.0
    kept = mask
    if on_people:
        kept = np.zeros_like(mask)
        kept[window] = np.logical_and(m, ~anyone)
    outline = fill(kept[window])
    enclosed = max((np.logical_and(outline, p[window]).sum() / total for p in people if (total := p.sum())), default=0.0)
    if enclosed >= ENCLOSE_SHARE:
        return "encloses_person", kept, share, float(enclosed)
    return ("trimmed" if on_people else "kept"), kept, share, float(enclosed)
