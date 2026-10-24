"""Choosing each person's thumbnail from their face detections.

Size and frontality alone picked a large frontal face under red stage light on OOO clip 2 (Person A). So the
geometry score only shortlists, and each shortlisted face box (no padding: the backdrop must not count) is scored on
its pixels. Measured on clip 2:
  - exposure and colour cast reject the red-lit and the near-black faces;
  - sharpness only gates blur (min(1, sharpness / SHARP_ENOUGH)): as a relative score, the edges of on-screen
    graphics over a face made it look 40x sharper than a clean, larger view of the same person;
  - blown-out pixels (white graphics, logos) are penalised directly.
The shortlist goes round the person's tracks, best geometry first, taking detections at least SPACING_S apart, so a
person with one track still gets a choice and one badly lit shot cannot fill the whole list.
"""

import numpy as np

SHORTLIST = 12
SPACING_S = 1.0          # shortlisted detections of one track are at least this far apart
QUALITY_PX = 64          # face boxes are scored at this size (square)
SHARP_ENOUGH = 3e-4      # Laplacian variance of a 64 px face box above which blur no longer counts against it
CLIPPED_LUMA = 0.92      # a pixel this bright is blown out (white graphics, logos, overexposure)
LUMA = np.array([0.2126, 0.7152, 0.0722])


def geometry(row) -> float:
    """Large and frontal: box area times the cosine of the head yaw (capped near profile)."""
    return row["box"][2] * row["box"][3] * float(np.cos(min(abs(row["yaw"]), 1.4)))


def shortlist(tracks: list[dict], k: int = SHORTLIST) -> list[dict]:
    per_track = []
    for t in tracks:
        picked = []
        for r in sorted(t["rows"], key=geometry, reverse=True):
            if all(abs(r["t"] - p["t"]) >= SPACING_S for p in picked):
                picked.append(r)
        per_track.append(picked)
    per_track.sort(key=lambda rows: geometry(rows[0]), reverse=True)
    out, depth = [], 0
    while len(out) < k and any(depth < len(rows) for rows in per_track):
        out += [rows[depth] for rows in per_track if depth < len(rows)][: k - len(out)]
        depth += 1
    return out


def quality(rgb: np.ndarray) -> dict:
    """Pixel measures of one face box (H x W x 3, uint8): exposure, neutral and unclipped in 0..1; sharpness is the
    Laplacian variance."""
    pixels = rgb.astype(np.float64) / 255
    luma = pixels @ LUMA
    exposure = 1 - min(1.0, abs(float(luma.mean()) - 0.5) / 0.4)     # 1 at mid-grey; 0 below 0.1 or above 0.9
    means = pixels.reshape(-1, 3).mean(axis=0)
    cast = float((means.max() - means.min()) / max(means.max(), 1e-6))  # 0 neutral; 1 a single channel
    lap = luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:] - 4 * luma[1:-1, 1:-1]
    return {"exposure": round(exposure, 3), "neutral": round(1 - cast, 3), "sharpness": round(float(lap.var()), 6),
            "unclipped": round(1 - min(1.0, 4 * float((luma > CLIPPED_LUMA).mean())), 3)}


def score(row, measures: dict, top_geometry: float) -> float:
    return (geometry(row) / top_geometry * measures["exposure"] * measures["neutral"] * measures["unclipped"]
            * min(1.0, measures["sharpness"] / SHARP_ENOUGH) * row["score"])


def choose(candidates: list[dict], images: list[np.ndarray]) -> tuple[int, list[dict]]:
    """The shortlisted detection to use as the thumbnail, given each one's face box; also returns the measures.
    Geometry counts relative to the best of this shortlist, so the choice is made among one person's faces only."""
    measures = [quality(image) for image in images]
    top_geometry = max(geometry(r) for r in candidates)
    return int(np.argmax([score(r, m, top_geometry) for r, m in zip(candidates, measures)])), measures
