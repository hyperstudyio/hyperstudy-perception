"""Masks to outlines and boxes, and the crops that are embedded, shown and checked: the bake-off's objects.py
(mask_polygon, object_views, context_strip...), with its constants, in OpenCV as it ran them."""

from __future__ import annotations

import cv2
import numpy as np

EMBED_PX = 224                 # DINOv3's pretraining resolution; 448 was not better in the bake-off
GREY = 128                     # letterbox bars in embedded crops
POLY_MAX_POINTS = 40           # polygon budget per sample
CROP_PAD = 0.1                 # fraction of the box added on each side of a displayed crop
CROP_MAX_PX = 384              # displayed crops are kept at most this size
CONTEXT_SCALE = 2.6            # each crop-check view: this many box sides around the box, with the box outlined
CONTEXT_MIN_PX = 200           # ... and at least this many pixels, so tiny objects show their surroundings
CONTEXT_PX = 256               # each moment is resized to this side; three are tiled left to right
CONTEXT_MOMENTS = (0.2, 0.5, 0.8)   # positions in the track's samples


def mask_polygon(mask) -> list[list[int]]:
    """Largest external contour, simplified until it has at most POLY_MAX_POINTS points."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(contour, True)
    poly = cv2.approxPolyDP(contour, epsilon, True)
    while len(poly) > POLY_MAX_POINTS:
        epsilon *= 1.5
        poly = cv2.approxPolyDP(contour, epsilon, True)
    return [[int(x), int(y)] for x, y in poly[:, 0]]


def mask_box(mask) -> list[int]:
    ys, xs = np.nonzero(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    return [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1]


def frame_sample(t, mask, score, offset=(0, 0)) -> dict:
    """A layer sample from a mask; a crop's mask at offset (x0, y0) gives frame pixels."""
    x0, y0 = offset
    x, y, w, h = mask_box(mask)
    return {"t": t, "box": [x + x0, y + y0, w, h], "poly": [[x + x0, y + y0] for x, y in mask_polygon(mask)],
            "score": round(float(score), 3)}


def filled_outline(mask):
    """What a mask's layer polygon (mask_polygon) covers, as a mask of the same shape; empty for an empty mask."""
    out = np.zeros(mask.shape, np.uint8)
    if mask.any():
        cv2.fillPoly(out, [np.array(mask_polygon(mask), np.int32)], 1)
    return out.astype(bool)


def padded_crop(frame, box):
    x, y, w, h = box
    px, py = int(w * CROP_PAD), int(h * CROP_PAD)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(frame.shape[1], x + w + px), min(frame.shape[0], y + h + py)
    return frame[y0:y1, x0:x1]


def shrink(image):
    h, w = image.shape[:2]
    scale = CROP_MAX_PX / max(h, w)
    return image.copy() if scale >= 1 else cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def letterbox(image, size):
    """Scale the longer side to size and pad the shorter one with grey: the whole object, undistorted."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    out = np.full((size, size, 3), GREY, np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def object_views(frame, box):
    """(embedded crop, displayed crop). The embedded one is the tight box, letterboxed to a square: centre-cropping
    cut thin poles and posts down to their middle, and in the bake-off neither masking out the background nor adding
    context linked better than the plain box. The displayed one has a margin, which is what a person wants to see."""
    x, y, w, h = box
    return letterbox(frame[y:y + h, x:x + w], EMBED_PX), shrink(padded_crop(frame, box))


def context_view(frame, box):
    """The crop check's view: CONTEXT_SCALE box sides around the box (at least CONTEXT_MIN_PX), box outlined."""
    x, y, w, h = box
    side = max(max(w, h) * CONTEXT_SCALE, CONTEXT_MIN_PX)
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(frame.shape[1], cx + side / 2)), int(min(frame.shape[0], cy + side / 2))
    view = frame[y0:y1, x0:x1].copy()
    thickness = max(2, round(max(view.shape[:2]) / 120))
    cv2.rectangle(view, (x - x0, y - y0), (x - x0 + w, y - y0 + h), (255, 230, 0), thickness)   # RGB yellow
    scale = CONTEXT_PX / max(view.shape[:2])
    view = cv2.resize(view, (round(view.shape[1] * scale), round(view.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    out = np.zeros((CONTEXT_PX, CONTEXT_PX, 3), np.uint8)
    out[:view.shape[0], :view.shape[1]] = view
    return out


def context_samples(track) -> list[dict]:
    """The samples at CONTEXT_MOMENTS of a track: the crop check's three moments."""
    samples = track["samples"]
    return [samples[min(len(samples) - 1, int(m * len(samples)))] for m in CONTEXT_MOMENTS]


def context_strip(track, frame_at):
    """The crop check's image for a track: context_view at its context_samples, left to right, separated by white
    bars."""
    bar = np.full((CONTEXT_PX, 6, 3), 255, np.uint8)
    views = [context_view(frame_at[s["t"]], s["box"]) for s in context_samples(track)]
    return np.hstack([x for v in views for x in (v, bar)][:-1])
