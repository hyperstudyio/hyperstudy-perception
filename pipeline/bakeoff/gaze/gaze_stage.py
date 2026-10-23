"""Model-agnostic part of the gaze-target stage: heads in, targets and events out. Used by gaze_lle.py and gaze_page.py.

The contract between a gaze model and this module is one method,

    model.predict(frame_bgr, heads) -> [(heatmap, inout), ...]

where heads is [(face_track, head box [x1, y1, x2, y2] in pixels), ...] for one frame, heatmap is a 64x64 float array
over the whole frame and inout an in-frame probability in [0, 1] (None for a model without an in/out head). The model
also keeps model.forward_s, the seconds spent in its forward passes (device-synchronised), and model.device. Everything
after the heatmaps (targets, events, the JSON) is here and identical for every model.

  1. Every face sample in faces.json (py-feat tracks, sample_fps) is one head. Its head box is the face box scaled
     by HEAD_BOX_SCALE about its centre and clipped to the frame. One predict call per sampled frame carries every
     head in that frame.
  2. Target per sample (target_for): inout below INOUT_THRESHOLD is "out" (target_score = 1 - inout). Otherwise
     the candidates are the other face tracks of the same shot (their head boxes, linearly interpolated to the
     sample time, as annotator/build.py box_at does) and, where objects.json exists, the object tracks of the same
     shot (the polygon of the object sample nearest in time, within OBJECT_MAX_DT_S; its box when it has none).
     The candidate containing the heatmap peak wins; when several contain it, the one holding the most heatmap mass.
     When the peak is in none, the candidate holding the most mass wins if that mass is at least MASS_MIN.
     Otherwise the target is null (looking at something in frame that is not a tracked face or object).
     target_score is the winner's share of the heatmap mass (for null, the best candidate's share, 0.0 without
     candidates).
  3. Events (events_from): looks_at = a run of at least MIN_LOOK_SAMPLES samples of one face track with the same
     face or object target, consecutive samples at most MAX_GAP_S apart; mutual_gaze = A looks_at B and B looks_at
     A sharing at least one sample time; joint_attention = A and C looks_at the same target X (X neither A nor C)
     sharing at least one sample time. Face tracks never cross a cut, so every event lies within one shot.
     Event start/end are the first/last sample times of the (overlapping part of the) runs; score = mean
     target_score of the run (mean of the two runs for the pair events).

Timings exclude model loading and one warm-up frame. model_s = forward passes (model.forward_s);
total_s = decode + preprocessing + forwards + targets + events; realtime_factor = total_s / video duration.
"""

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
BAKEOFF = HERE.parent  # extract.json's video.source is relative to this
VENDOR = HERE / "vendor"

# Head box = py-feat face box x this, about the face centre. Gaze-LLE and PaGE are both trained on annotated head boxes
# (whole head, hair included: GazeFollow / VAT / ChildPlay). py-feat's RetinaFace boxes are square and already span
# brow to chin; drawn on OOO frames, x1.0 cuts off the crown and hair, x1.3 takes in neck and background, x1.2
# encloses the head. check_device.py --scales measures how much the result moves between 1.0 and 1.4.
HEAD_BOX_SCALE = 1.2
# In/out-of-frame score below this is "out". Both models' in/out heads are sigmoids trained with BCE on in/out labels,
# so 0.5 is the decision boundary; neither paper reports a threshold (both report AP).
INOUT_THRESHOLD = 0.5
# With the peak outside every candidate, the candidate holding at least this share of the heatmap mass is still the
# target (a peak landing just beside a head or on a thin object). Below it the gaze lands on something untracked.
# On OOO (Gaze-LLE), samples left null have a best-candidate share of median 0.03, 99th percentile 0.25; the rule
# fired for 16 of 1463 face targets.
MASS_MIN = 0.3
OBJECT_MAX_DT_S = 1 / 6  # objects.json samples at 6 fps; an object sample this close in time stands for the face sample
MIN_LOOK_SAMPLES = 3     # looks_at needs this many consecutive samples (0.27 s span at 7.5 samples/s): a single-sample
                         # argmax flip between neighbouring heads is not a look
MAX_GAP_S = 0.3          # consecutive samples of a run at most this far apart (one missed face sample at 7.5/s)
HEATMAP_SIZE = 64        # heatmap side of both models
COVERAGE_GRID = 256      # candidate regions are rasterised on this grid, then area-averaged to the 64x64 heatmap grid
PROGRESS_EVERY = 100     # frames


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def head_box(face_box, width, height, scale):
    """Face box [x, y, w, h] -> head box [x1, y1, x2, y2] in pixels, scaled about the centre, clipped to the frame."""
    x, y, w, h = face_box
    cx, cy = x + w / 2, y + h / 2
    return [max(0.0, cx - w * scale / 2), max(0.0, cy - h * scale / 2),
            min(float(width), cx + w * scale / 2), min(float(height), cy + h * scale / 2)]


# ---------------------------------------------------------------- targets

def interpolate_box(series, t):
    """Box of a face track at time t, linearly interpolated between its samples (None outside its span)."""
    ts = [s["t"] for s in series]
    if t < ts[0] or t > ts[-1]:
        return None
    i = next(k for k in range(len(ts)) if ts[k] >= t)
    if ts[i] == t or i == 0:
        return series[i]["box"]
    f = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return [a + f * (b - a) for a, b in zip(series[i - 1]["box"], series[i]["box"])]


def object_at(track, t):
    """The object sample nearest t within OBJECT_MAX_DT_S, else None."""
    best = min(track["samples"], key=lambda s: abs(s["t"] - t))
    return best if abs(best["t"] - t) <= OBJECT_MAX_DT_S else None


def candidates_at(face_id, shot, t, face_tracks, objects_by_shot, width, height, scale):
    """[(target id, region)] for one head at time t; region = ("box", [x1, y1, x2, y2]) or ("poly", Nx2 array).

    Other people are their head boxes at the same scale as the heads fed to the model."""
    out = []
    for other_id, other in face_tracks.items():
        if other_id == face_id or other["shot"] != shot:
            continue
        box = interpolate_box(other["series"], t)
        if box is not None:
            out.append((other_id, ("box", head_box(box, width, height, scale))))
    for obj in objects_by_shot.get(shot, []):
        sample = object_at(obj, t)
        if sample is None:
            continue
        if sample.get("poly") and len(sample["poly"]) >= 3:
            out.append((obj["id"], ("poly", np.array(sample["poly"], dtype=np.float32))))
        else:
            x, y, w, h = sample["box"]
            out.append((obj["id"], ("box", [x, y, x + w, y + h])))
    return out


def coverage(region, width, height):
    """Fraction of each 64x64 heatmap cell covered by the region."""
    grid = np.zeros((COVERAGE_GRID, COVERAGE_GRID), np.float32)
    sx, sy = COVERAGE_GRID / width, COVERAGE_GRID / height
    kind, geom = region
    if kind == "box":
        x1, y1, x2, y2 = geom
        cv2.rectangle(grid, (int(round(x1 * sx)), int(round(y1 * sy))), (int(round(x2 * sx)) - 1, int(round(y2 * sy)) - 1), 1.0, -1)
    else:
        cv2.fillPoly(grid, [np.round(geom * [sx, sy]).astype(np.int32)], 1.0)
    return cv2.resize(grid, (HEATMAP_SIZE, HEATMAP_SIZE), interpolation=cv2.INTER_AREA)


def contains(region, x, y):
    kind, geom = region
    if kind == "box":
        return geom[0] <= x <= geom[2] and geom[1] <= y <= geom[3]
    return cv2.pointPolygonTest(geom.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0


def peak_of(heat, width, height):
    """Heatmap argmax as the centre of its cell, in full-resolution pixels."""
    row, col = np.unravel_index(int(heat.argmax()), heat.shape)
    return (col + 0.5) / HEATMAP_SIZE * width, (row + 0.5) / HEATMAP_SIZE * height


def target_for(heat, inout, candidates, width, height):
    """(target, target_score, peak) for one head; see the module docstring, step 2."""
    peak = peak_of(heat, width, height)
    if inout is not None and inout < INOUT_THRESHOLD:
        return "out", 1.0 - float(inout), peak
    total = float(heat.sum())
    scored = [(float((heat * coverage(region, width, height)).sum()) / total, cid, region) for cid, region in candidates]
    hit = [(mass, cid) for mass, cid, region in scored if contains(region, *peak)]
    if hit:
        mass, cid = max(hit, key=lambda x: x[0])
        return cid, mass, peak
    if scored:
        mass, cid, _ = max(scored, key=lambda x: x[0])
        return (cid, mass, peak) if mass >= MASS_MIN else (None, mass, peak)
    return None, 0.0, peak


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
    looks = looks_at_runs(tracks)
    pairs = []
    for i, a in enumerate(looks):
        for b in looks[i + 1:]:
            shared = sorted(a["_times"] & b["_times"])
            if not shared or a["from_face_track"] == b["from_face_track"]:
                continue
            base = {"start": shared[0], "end": shared[-1], "score": round((a["score"] + b["score"]) / 2, 3)}
            first, second = sorted((a, b), key=lambda e: e["from_face_track"])
            if (a["to_face_track"] == b["from_face_track"] and b["to_face_track"] == a["from_face_track"]):
                pairs.append({"kind": "mutual_gaze", "from_face_track": first["from_face_track"],
                              "to_face_track": second["from_face_track"], "object": None, "also_face_track": None, **base})
            elif (a["to_face_track"], a["object"]) == (b["to_face_track"], b["object"]):
                pairs.append({"kind": "joint_attention", "from_face_track": first["from_face_track"],
                              "to_face_track": a["to_face_track"], "object": a["object"],
                              "also_face_track": second["from_face_track"], **base})
    for e in looks:
        del e["_times"]
    return sorted(looks + pairs, key=lambda e: (e["start"], e["kind"], e["from_face_track"]))


# ---------------------------------------------------------------- run

def frames_at(video_path, indices):
    """(frame index, BGR frame) for the sorted frame indices, decoded sequentially."""
    capture = cv2.VideoCapture(str(video_path))
    wanted = iter(sorted(indices))
    target = next(wanted, None)
    index = 0
    while target is not None:
        if index < target:
            if not capture.grab():
                break
        else:
            ok, frame = capture.read()
            if not ok:
                break
            yield index, frame
            target = next(wanted, None)
        index += 1
    capture.release()
    if target is not None:
        raise SystemExit(f"video ended at frame {index} before frame {target}")


def heads_by_frame(faces, fps):
    """frame index -> [(face track id, sample index, face box)]; checks that every sample time is a frame time."""
    frames = {}
    for tid, tr in enumerate(faces["tracks"]):
        for j, s in enumerate(tr["series"]):
            index = round(s["t"] * fps)
            if abs(round(index / fps, 3) - s["t"]) > 1e-6:
                raise SystemExit(f"face track {tid} sample t={s['t']} is not a frame time at {fps} fps")
            frames.setdefault(index, []).append((tid, j, s["box"]))
    return frames


def load_run(run):
    """(extract video dict, faces, objects or None, video path) of a run directory."""
    extract = json.loads((run / "extract.json").read_text())
    faces = json.loads((run / "faces.json").read_text())
    objects = json.loads((run / "objects.json").read_text()) if (run / "objects.json").exists() else None
    video = Path(extract["video"]["source"])
    return extract["video"], faces, objects, video if video.is_absolute() else BAKEOFF / video


def frame_heads(entries, width, height, scale=HEAD_BOX_SCALE):
    """heads_by_frame entries of one frame -> the predict() input [(face track, head box)]."""
    return [(tid, head_box(b, width, height, scale)) for tid, _, b in entries]


def run_stage(run, model, header, out_path, heatmaps_path=None):
    """Run model over every face sample of a run and write out_path in the gaze-target schema.

    header: {"model", "checkpoint", "licence_mode", "licence"} of the model; fps, device, timings, head_box_scale,
    tracks and events are added here. With heatmaps_path, every heatmap (float16) is saved there for draw_gaze.py."""
    meta, faces, objects, video = load_run(run)
    fps, width, height, duration = meta["fps"], meta["width"], meta["height"], meta["duration"]
    face_tracks = dict(enumerate(faces["tracks"]))  # face track id = index in faces.json
    objects_by_shot = {}
    for obj in (objects or {}).get("tracks", []):
        objects_by_shot.setdefault(obj["shot"], []).append(obj)
    heads = heads_by_frame(faces, fps)

    first_index = min(heads)
    _, first = next(frames_at(video, [first_index]))
    model.predict(first, frame_heads(heads[first_index], width, height))  # warm-up
    model.forward_s = 0.0

    results = {tid: [None] * len(tr["series"]) for tid, tr in face_tracks.items()}
    saved = {}
    total_start = time.perf_counter()
    for n, (index, frame) in enumerate(frames_at(video, heads)):
        if frame.shape[:2] != (height, width):
            raise SystemExit(f"frame {index} is {frame.shape[1]}x{frame.shape[0]}, extract.json says {width}x{height}")
        predictions = model.predict(frame, frame_heads(heads[index], width, height))
        for (tid, j, _), (heat, score) in zip(heads[index], predictions):
            sample = face_tracks[tid]["series"][j]
            cands = candidates_at(tid, face_tracks[tid]["shot"], sample["t"], face_tracks, objects_by_shot, width, height, HEAD_BOX_SCALE)
            target, target_score, peak = target_for(heat, score, cands, width, height)
            results[tid][j] = {"t": sample["t"], "peak": [round(peak[0], 1), round(peak[1], 1)],
                               "inout": None if score is None else round(score, 3), "target": target,
                               "target_score": round(target_score, 3)}
            if heatmaps_path:
                saved[f"{tid}_{j}"] = heat.astype(np.float16)
        if (n + 1) % PROGRESS_EVERY == 0:
            print(f"  {n + 1}/{len(heads)} frames, {time.perf_counter() - total_start:.0f} s", flush=True)

    tracks = [{"face_track": tid, "samples": samples} for tid, samples in results.items()]
    events = events_from(tracks)
    total_s = time.perf_counter() - total_start

    out = {**header, "fps": faces["sample_fps"], "device": model.device,
           "timings": {"model_s": round(model.forward_s, 2), "total_s": round(total_s, 2), "realtime_factor": round(total_s / duration, 3)},
           "head_box_scale": HEAD_BOX_SCALE, "tracks": tracks, "events": events}
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    if heatmaps_path:
        np.savez_compressed(heatmaps_path, **saved)

    samples = [s for tr in tracks for s in tr["samples"]]
    kinds = {}
    for s in samples:
        key = "face" if isinstance(s["target"], int) else s["target"] if s["target"] in (None, "out") else "object"
        kinds[str(key)] = kinds.get(str(key), 0) + 1
    by_kind = {k: sum(1 for e in events if e["kind"] == k) for k in ("looks_at", "mutual_gaze", "joint_attention")}
    print(f"{out_path.name}: {len(heads)} frames, {len(samples)} heads; targets {kinds}; events {by_kind}")
    print(f"timings: {out['timings']}")
