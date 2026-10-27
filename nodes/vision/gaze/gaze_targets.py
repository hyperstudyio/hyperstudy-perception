"""Gaze targets and events from Gaze-LLE heatmaps (numpy only): the bake-off's gaze_stage.py, reading the bundle's faces
layer instead of the bake-off's faces.json, and writing the gaze layer as pipeline/bundle/convert_bakeoff.py's
gaze_layer does.

  1. Every face sample is one head. Its head box is the face box scaled by `head_box_scale` (1.2: py-feat's square
     RetinaFace boxes span brow to chin; x1.2 encloses the head, hair included, as Gaze-LLE's training boxes do) about
     its centre, clipped to the frame. One model call per sampled frame carries every head in that frame.
  2. Target per sample (target_for): an in-frame score below INOUT_THRESHOLD is "out" (target_score = 1 - in_frame).
     Otherwise the candidates are the other face tracks of the same shot (their head boxes, linearly interpolated to
     the sample time). The candidate containing the heatmap peak wins; when several contain it, the one holding the
     most heatmap mass. When the peak is in none, the candidate holding the most mass wins if that mass is at least
     MASS_MIN. Otherwise the target is null (something in frame that is not a tracked face). target_score is the
     winner's share of the heatmap mass (for null, the best candidate's share; 0.0 without candidates).
  3. Events (events_from): looks_at = a run of at least MIN_LOOK_SAMPLES samples of one face track with the same face
     (or object) target, consecutive samples at most MAX_GAP_S apart; mutual_gaze = A looks_at B and B looks_at A
     sharing at least one sample time; joint_attention = A and C look_at the same target X sharing at least one sample
     time. Face tracks never cross a cut, so every event lies within one shot.

Objects (a later wave): the bake-off also offered the object tracks of the same shot as candidates (the polygon of the
object sample nearest in time, within 1/6 s; its box when it has none). The events here already carry object targets
("obj:N", as `object`). Adding them is a candidates_at argument with those objects' regions, a Polygon region beside
Box (rasterised like Box.coverage, at COVERAGE_GRID), and a bundle.objects@0.1 input to the node.

Inside this module face tracks are their integer index (face:N -> N), as in the bake-off, so events sort exactly as
its did (by track number, not by the id's text).
"""

from __future__ import annotations

import numpy as np

# Both models' in/out heads are sigmoids trained with BCE on in/out labels, so 0.5 is the decision boundary.
INOUT_THRESHOLD = 0.5
# With the peak outside every candidate, the candidate holding at least this share of the heatmap mass is still the
# target (a peak landing just beside a head). On OOO, samples left null have a best-candidate share of median 0.03,
# 99th percentile 0.25; the rule fired for 16 of 1463 face targets.
MASS_MIN = 0.3
MIN_LOOK_SAMPLES = 3     # a single-sample argmax flip between neighbouring heads is not a look
MAX_GAP_S = 0.3          # consecutive samples of a run at most this far apart (one missed face sample at 7.5/s)
HEATMAP_SIZE = 64        # Gaze-LLE's heatmap side
COVERAGE_GRID = 256      # candidate regions are rasterised on this grid, then area-averaged to the heatmap's grid


def face_index(face_id: str) -> int:
    """face:N -> N."""
    kind, _, number = face_id.partition(":")
    if kind != "face" or not number.isdigit():
        raise ValueError(f"{face_id!r} is not a face track id (face:N)")
    return int(number)


def face_id(index: int | None) -> str | None:
    return None if index is None else f"face:{index}"


def head_box(face_box, width, height, scale):
    """Face box [x, y, w, h] -> head box [x1, y1, x2, y2] in pixels, scaled about the centre, clipped to the frame."""
    x, y, w, h = face_box
    cx, cy = x + w / 2, y + h / 2
    return [max(0.0, cx - w * scale / 2), max(0.0, cy - h * scale / 2),
            min(float(width), cx + w * scale / 2), min(float(height), cy + h * scale / 2)]


def interpolate_box(samples, t):
    """Box of a face track at time t, linearly interpolated between its samples (None outside its span)."""
    ts = [s["t"] for s in samples]
    if t < ts[0] or t > ts[-1]:
        return None
    i = next(k for k in range(len(ts)) if ts[k] >= t)
    if ts[i] == t or i == 0:
        return samples[i]["box"]
    f = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return [a + f * (b - a) for a, b in zip(samples[i - 1]["box"], samples[i]["box"])]


class Box:
    """A candidate region [x1, y1, x2, y2] in pixels."""

    def __init__(self, x1, y1, x2, y2):
        self.geom = (x1, y1, x2, y2)

    def coverage(self, width, height):
        """Fraction of each heatmap cell the box covers: the box filled on a COVERAGE_GRID raster (as OpenCV's
        cv2.rectangle(grid, (round(x1), round(y1)), (round(x2) - 1, round(y2) - 1), 1, -1) fills it: corners in
        either order, both ends inclusive, clipped to the grid), then each 4x4 block averaged (cv2.INTER_AREA at an
        integer factor)."""
        grid = np.zeros((COVERAGE_GRID, COVERAGE_GRID), np.float32)
        sx, sy = COVERAGE_GRID / width, COVERAGE_GRID / height
        x1, y1, x2, y2 = self.geom
        xa, xb = sorted((int(round(x1 * sx)), int(round(x2 * sx)) - 1))
        ya, yb = sorted((int(round(y1 * sy)), int(round(y2 * sy)) - 1))
        xa, ya, xb, yb = max(xa, 0), max(ya, 0), min(xb, COVERAGE_GRID - 1), min(yb, COVERAGE_GRID - 1)
        if xa <= xb and ya <= yb:
            grid[ya:yb + 1, xa:xb + 1] = 1.0
        k = COVERAGE_GRID // HEATMAP_SIZE
        return grid.reshape(HEATMAP_SIZE, k, HEATMAP_SIZE, k).mean(axis=(1, 3), dtype=np.float32)

    def contains(self, x, y):
        x1, y1, x2, y2 = self.geom
        return x1 <= x <= x2 and y1 <= y <= y2


def candidates_at(face, shot, t, face_tracks, width, height, scale):
    """[(face index, Box)]: the other face tracks of the shot at time t, as head boxes at the scale the heads fed to
    the model have."""
    out = []
    for other, track in face_tracks.items():
        if other == face or track["shot"] != shot:
            continue
        box = interpolate_box(track["samples"], t)
        if box is not None:
            out.append((other, Box(*head_box(box, width, height, scale))))
    return out


def peak_of(heat, width, height):
    """Heatmap argmax as the centre of its cell, in frame pixels."""
    row, col = np.unravel_index(int(heat.argmax()), heat.shape)
    return (col + 0.5) / HEATMAP_SIZE * width, (row + 0.5) / HEATMAP_SIZE * height


def target_for(heat, inout, candidates, width, height):
    """(target, target_score, peak) for one head; see the module docstring, step 2."""
    peak = peak_of(heat, width, height)
    if inout is not None and inout < INOUT_THRESHOLD:
        return "out", 1.0 - float(inout), peak
    total = float(heat.sum())
    scored = [(float((heat * region.coverage(width, height)).sum()) / total, cid, region) for cid, region in candidates]
    hit = [(mass, cid) for mass, cid, region in scored if region.contains(*peak)]
    if hit:
        mass, cid = max(hit, key=lambda x: x[0])
        return cid, mass, peak
    if scored:
        mass, cid, _ = max(scored, key=lambda x: x[0])
        return (cid, mass, peak) if mass >= MASS_MIN else (None, mass, peak)
    return None, 0.0, peak


def sample_of(t, heat, inout, target, target_score, peak):
    """One gaze sample as the bake-off stored it (gaze_lle.json)."""
    return {"t": t, "peak": [round(peak[0], 1), round(peak[1], 1)], "inout": None if inout is None else round(inout, 3),
            "target": target, "target_score": round(target_score, 3)}


# ---------------------------------------------------------------- frames

def heads_by_frame(face_tracks, fps):
    """frame index -> [(face index, sample index, face box)]. Every sample time must be a frame time (the faces node
    samples frames, at t = round(index / fps, 3))."""
    frames = {}
    for face, track in face_tracks.items():
        for j, s in enumerate(track["samples"]):
            index = round(s["t"] * fps)
            if abs(round(index / fps, 3) - s["t"]) > 1e-6:
                raise ValueError(f"face:{face} sample t={s['t']} is not a frame time at {fps} fps")
            frames.setdefault(index, []).append((face, j, s["box"]))
    return frames


def frame_heads(entries, width, height, scale):
    """heads_by_frame entries of one frame -> the model's input [(face index, head box)]."""
    return [(face, head_box(box, width, height, scale)) for face, _, box in entries]


def tracks_of(faces_layer):
    """The faces layer's tracks as {face index: {"shot", "samples"}}, in the layer's order."""
    return {face_index(t["id"]): {"shot": t["shot"], "samples": t["samples"]} for t in faces_layer["tracks"]}


# ---------------------------------------------------------------- events

def looks_at_runs(tracks):
    """looks_at events: runs of one face or object target per face track (MIN_LOOK_SAMPLES, MAX_GAP_S)."""
    events = []
    for tr in tracks:
        run = []
        for s in tr["samples"] + [None]:  # sentinel closes the last run
            continues = (s is not None and run and s["target"] == run[-1]["target"]
                         and s["t"] - run[-1]["t"] <= MAX_GAP_S + 1e-6)
            if continues:
                run.append(s)
                continue
            if run and len(run) >= MIN_LOOK_SAMPLES and run[0]["target"] not in (None, "out"):
                target = run[0]["target"]
                is_obj = isinstance(target, str)
                events.append({"kind": "looks_at", "from_face_track": tr["face_track"],
                               "to_face_track": None if is_obj else target, "object": target if is_obj else None,
                               "also_face_track": None, "start": run[0]["t"], "end": run[-1]["t"],
                               "score": round(float(np.mean([x["target_score"] for x in run])), 3),
                               "_times": {x["t"] for x in run}})
            run = [s] if s is not None else []
    return events


def events_from(tracks):
    """looks_at, mutual_gaze and joint_attention events, sorted by start, kind and looking track."""
    looks = looks_at_runs(tracks)
    pairs = []
    for i, a in enumerate(looks):
        for b in looks[i + 1:]:
            shared = sorted(a["_times"] & b["_times"])
            if not shared or a["from_face_track"] == b["from_face_track"]:
                continue
            base = {"start": shared[0], "end": shared[-1], "score": round((a["score"] + b["score"]) / 2, 3)}
            first, second = sorted((a, b), key=lambda e: e["from_face_track"])
            if a["to_face_track"] == b["from_face_track"] and b["to_face_track"] == a["from_face_track"]:
                pairs.append({"kind": "mutual_gaze", "from_face_track": first["from_face_track"],
                              "to_face_track": second["from_face_track"], "object": None, "also_face_track": None, **base})
            elif (a["to_face_track"], a["object"]) == (b["to_face_track"], b["object"]):
                pairs.append({"kind": "joint_attention", "from_face_track": first["from_face_track"],
                              "to_face_track": a["to_face_track"], "object": a["object"],
                              "also_face_track": second["from_face_track"], **base})
    for e in looks:
        del e["_times"]
    return sorted(looks + pairs, key=lambda e: (e["start"], e["kind"], e["from_face_track"]))


# ---------------------------------------------------------------- layer

def layer_target(target):
    """A sample's target as the layer writes it: face:N, obj:N, "out" or null."""
    if target is None or target == "out" or (isinstance(target, str) and target.startswith("obj:")):
        return target
    if isinstance(target, int):
        return face_id(target)
    raise ValueError(f"unknown gaze target {target!r}")


def layer_content(fps, tracks, events):
    """The gaze layer's fps, tracks and events (convert_bakeoff.gaze_layer's mapping of the bake-off's gaze_lle.json)."""
    out_tracks = [{"face": face_id(t["face_track"]),
                   "samples": [{"t": s["t"], "peak": s["peak"], "in_frame": s["inout"], "target": layer_target(s["target"]),
                                "target_score": s["target_score"]} for s in t["samples"]]} for t in tracks]
    out_events = [{"id": f"gaze:{i}", "kind": e["kind"], "from": face_id(e["from_face_track"]), "to": face_id(e["to_face_track"]),
                   "object": e["object"], "also": face_id(e["also_face_track"]), "start": e["start"], "end": e["end"],
                   "score": e["score"]} for i, e in enumerate(events)]
    return {"fps": fps, "tracks": out_tracks, "events": out_events}
