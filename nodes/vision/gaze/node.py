"""vision.gaze: where each tracked face looks, as the bundle's gaze layer: per face sample a heatmap peak, an in-frame
score and a target (another face track, "out" of the frame, or null), and the looks_at, mutual_gaze and
joint_attention events those targets make.

Every face sample of the faces layer is one head (its box scaled by `head_box_scale`); the frames holding them are
decoded in order with OpenCV, and one Gaze-LLE forward per frame scores every head in it (models.py). Targets and
events: gaze_targets.py.

Beside the layer, the side file layers/gaze_heatmaps.parquet keeps the rest of what the model returned for every head:
its whole 64x64 heatmap (uint8 relative to its maximum, with the maximum) and its unrounded in-frame score
(heatmaps.py). The layer points at it with raw.heatmaps.

Objects: with the objects layer (an optional input; the objects node can be left out of a run, and then gaze runs on
faces alone), each object track's outline at the sample time is a candidate too, as in the bake-off (gaze_targets.py,
gaze_polygons.py), and a target may be "obj:N". Without it the layer is the faces-only one, byte for byte.

Licence: DINOv2 is Apache-2.0, but Gaze-LLE's checkpoints state no licence and its training data's terms are unverified
(needs-review), and the head boxes are py-feat's (the faces layer, academic-only): the layer is academic-only.
"""

import time

import cv2

from perception_node import Node, NodeError
from gaze_targets import (candidates_at, events_from, face_id, frame_heads, heads_by_frame, layer_content, objects_by_shot,
                          sample_of, target_for, tracks_of)
import heatmaps
from models import DINOV2_ID, DINOV2_LICENCE, GAZELLE_ID, GAZELLE_LICENCE, WEIGHTS, GazeModel


def frames_at(video, indices):
    """(frame index, BGR frame) for the sorted frame indices, decoded in order."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise NodeError("input", f"OpenCV cannot open {video.name}")
    wanted = iter(sorted(indices))
    target = next(wanted, None)
    index = 0
    try:
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
    finally:
        capture.release()
    if target is not None:
        raise NodeError("input", f"{video.name} ended at frame {index}, before frame {target}, which has face samples")


with Node() as node:
    c = node.config
    scale = c["head_box_scale"]
    video, media, faces = node.input("video"), node.input_json("media"), node.input_json("faces")
    face_tracks = tracks_of(faces)
    has_objects = "objects" in node.request["inputs"]   # an optional input: absent when the run leaves objects out
    objects = objects_by_shot(node.input_json("objects")) if has_objects else {}
    try:
        heads = heads_by_frame(face_tracks, media["fps"])
    except ValueError as e:
        raise NodeError("input", str(e)) from None

    t0 = time.time()
    model = GazeModel(node.accelerator)
    load_s = round(time.time() - t0, 1)
    node.model(GAZELLE_ID, licence_mode="needs-review", source=WEIGHTS["gazelle"]["url"], revision=WEIGHTS["gazelle"]["sha256"],
               licence=GAZELLE_LICENCE)
    node.model(DINOV2_ID, licence_mode="commercial-ok", source=WEIGHTS["dinov2"]["url"], revision=WEIGHTS["dinov2"]["sha256"],
               licence=DINOV2_LICENCE)

    results = {face: [None] * len(track["samples"]) for face, track in face_tracks.items()}
    model_s, t_all = 0.0, time.time()
    with heatmaps.Writer(node.path("heatmaps", "gaze_heatmaps.parquet")) as raw:
        for index, frame in frames_at(video, heads):
            height, width = frame.shape[:2]
            tick = time.time()
            predictions = model.predict(frame, frame_heads(heads[index], width, height, scale))
            model_s += time.time() - tick
            for (face, j, _), (heat, inout) in zip(heads[index], predictions):
                t = face_tracks[face]["samples"][j]["t"]
                candidates = candidates_at(face, face_tracks[face]["shot"], t, face_tracks, objects, width, height, scale)
                results[face][j] = sample_of(t, heat, inout, *target_for(heat, inout, candidates, width, height))
                try:
                    raw.add(face_id(face), j, t, heat, inout)
                except ValueError as e:
                    raise NodeError("internal", f"Gaze-LLE's output for face:{face} at t={t}: {e}") from None
    tracks = [{"face_track": face, "samples": samples} for face, samples in results.items()]
    events = events_from(tracks)
    total_s = round(time.time() - t_all, 1)

    node.write_json("gaze", "gaze.json", {
        "layer": "gaze", "schema_version": "0.1.0",
        "producer": node.producer(timings={"load_s": load_s, "model_s": round(model_s, 1), "total_s": total_s},
                                  device=str(model.device)),
        **layer_content(faces["sample_fps"], tracks, events),
        "raw": {"heatmaps": {"file": heatmaps.FILE, "format": "parquet", "description": heatmaps.DESCRIPTION}}})
    samples = [s for tr in tracks for s in tr["samples"]]
    node.usage(frames=len(heads), heads=len(samples), events=len(events),
               targets={"face": sum(isinstance(s["target"], int) for s in samples), "out": sum(s["target"] == "out" for s in samples),
                        "object": sum(isinstance(s["target"], str) and s["target"].startswith("obj:") for s in samples),
                        "none": sum(s["target"] is None for s in samples)},
               objects=sum(len(v) for v in objects.values()) if has_objects else None,
               realtime_factor=round(total_s / media["duration"], 3))
