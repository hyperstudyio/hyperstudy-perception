"""vision.objects: what things are on screen, tracked within shots and linked across them, as the bundle's objects layer.

Ported from the bake-off's objects/objects.py (its interactive segment_server.py is not part of this). First, before
anything is uploaded or paid for: the gated models must be ours to use, and the longest shot's SAM 3 session must fit
in this machine's memory (budget.py). Then:

1. Vocabulary (haiku.py): Claude Haiku 5.5 sees each shot's three keyframes (decode's) and names up to 8 concrete,
   trackable objects, told the names it used in earlier shots so one bike stays "bicycle".
2. Segment and track (models.py): each shot is decoded at `fps` (shot_frames.py, the bake-off's ffmpeg frames) and run
   through SAM 3 video with the shot's names as text prompts. A session never spans a cut. Tracks seen on fewer than
   two frames are dropped (almost always a flicker detection).
3. Embed: each track is the mean DINOv3 ViT-L/16 CLS embedding of up to 8 tight box crops, letterboxed to a square
   (geometry.py), from the frames SAM 3 saw.
4. Checks (haiku.py): one call classifies the clip's names into families, lookalike or individual, and worn or not
   (made right after the vocabulary, before the GPU work: it needs nothing else); each track's three outlined moments
   are checked for really being its label (object / graphic / absent). Tracks judged graphic or absent are kept apart
   (rejected_tracks) and never linked.
5. Identities (linking.py): worn tracks get a wearer, the person (faces layer) whose posed body (bodies layer, linked to
   a face track) holds them; tracks of one family are merged by average-linkage clustering of adjusted cosines.
   Each identity's crop (assets/crops/objects/) is its embedded view closest to the identity's mean embedding.

Licence: SAM 3 and DINOv3 are under Meta's own licences (commercial use permitted, with conditions; gated on Hugging
Face), the names and checks are Claude's output under the Anthropic API terms; the layer refers to the faces layer's
people, so the bundle gives it the faces layer's academic-only mode.
"""

import time

import numpy as np

from perception_node import Node, NodeError
from budget import GB, physical_memory, too_long
from geometry import context_samples, context_strip, frame_sample, object_views
from haiku import MAX_OBJECTS, MODEL as HAIKU, VERIFY_BATCH, Haiku, spent
from linking import (assign_wearers, body_samples, check_classes, check_views, clean_names, crop_files, different_people,
                     layer_content, link_identities, representative_view)
from models import (EMBEDDER, EMBEDDER_LICENCE, EMBEDDER_REVISION, SEGMENTER, SEGMENTER_LICENCE, SEGMENTER_REVISION, check_access,
                    device_for, embed_crops, load_embedder, load_segmenter, segment_shot, segmenter_dtype, synchronize)
from shot_frames import shot_frames

MIN_TRACK_SAMPLES = 2   # a track seen on a single frame is almost always a flicker detection
EMBED_SAMPLES = 8       # crops averaged per track embedding, evenly spaced over the track
MIN_CROP_PX = 24        # crops smaller than this on both sides carry no identity information
RESERVED_GB = 8         # memory beside a SAM 3 session's frames: the models (3.4 and 1.2 GB on disk) and the rest


def open_video(path):
    """The video for shot_frames, decoded on several threads (the same frames, 7x faster on OOO)."""
    import av
    container = av.open(str(path))
    container.streams.video[0].thread_type = "AUTO"
    return container


def embed_picks(track) -> list[dict]:
    """The samples a track's embedding is made from: up to EMBED_SAMPLES big enough to see, evenly spaced."""
    usable = [s for s in track["samples"] if max(s["box"][2:]) >= MIN_CROP_PX]
    return [usable[int(i)] for i in np.linspace(0, len(usable) - 1, min(EMBED_SAMPLES, len(usable)))] if usable else []


def frames_needed(tracks) -> set[float]:
    """The times embed_shot reads frames at: each track's embedded samples and its crop-check moments."""
    return {s["t"] for t in tracks for s in embed_picks(t) + context_samples(t)}


def embed_shot(tracks, frame_at, model, device):
    """Embedding, view_embeddings, views (displayed crops) and context (the crop check's strip) for each track of one
    shot, from the frames SAM 3 saw (frame_at: {time: frame}, at least those at frames_needed(tracks))."""
    for track in tracks:
        picks = embed_picks(track)
        views = [object_views(frame_at[s["t"]], s["box"]) for s in picks]
        track["embedding"], track["view_embeddings"] = embed_crops(model, [e for e, _ in views], device) if views else (None, None)
        track["views"] = [shown for _, shown in views]
        track["context"] = context_strip(track, frame_at)


with Node() as node:
    from PIL import Image

    c = node.config
    fps = c["fps"]
    video = node.input("video")
    shots_doc, faces, bodies = node.input_json("shots"), node.input_json("faces"), node.input_json("bodies")
    media = node.input_json("media")
    keyframes_dir = node.input("keyframes").resolve()
    shots = shots_doc["shots"]
    shot_index = {s["id"]: s["index"] for s in shots}
    device = device_for(node.accelerator)
    hf_token = node.secret("HF_TOKEN")
    totals = {"in": 0, "out": 0, "calls": 0}

    def spend(n_in, n_out):
        """The running total after every billed call: if a later call fails, the response still reports it."""
        totals.update({"in": totals["in"] + n_in, "out": totals["out"] + n_out, "calls": totals["calls"] + 1})
        node.usage(usd=spent(totals["in"], totals["out"]), tokens_in=totals["in"], tokens_out=totals["out"], calls=totals["calls"])

    haiku = Haiku(node.secret("ANTHROPIC_API_KEY"), spend)
    timings = {}

    # 0. before anything is uploaded or paid for: the gated models are ours to use, and every shot fits in memory
    check_access(hf_token)
    long = too_long(shots, fps, media["width"], media["height"], segmenter_dtype(device).itemsize, physical_memory() - RESERVED_GB * GB)
    if long:
        raise NodeError("resources", f"shot {long['index']} is {long['frames']} frames at {fps} fps: SAM 3 holds every frame of "
                                     f"a shot at once ({long['gb']} GB), more than this machine has. Lower this node's fps, or run "
                                     "it on a machine with more memory (budget.py)")

    # 1. vocabulary: one call per shot, in order, each told the names used before it (never after)
    t0 = time.time()
    vocabulary, earlier = {}, set()
    for shot in shots:
        images = []
        for kf in shot.get("keyframes") or []:
            path = (keyframes_dir / kf["file"].removeprefix("assets/")).resolve()   # follows links: one out is caught too
            if not path.is_relative_to(keyframes_dir):
                raise NodeError("input", f"keyframe {kf['file']} is outside the keyframes input; only those images are sent")
            images.append(path.read_bytes())
        if not images:
            raise NodeError("input", f"shot {shot['index']} has no keyframes: the object names come from them")
        names = clean_names(haiku.vocabulary(images, earlier), MAX_OBJECTS)
        vocabulary[shot["index"]] = names
        earlier.update(names)
    # the names' families, kinds and whether they are worn: from the vocabulary alone, so before the GPU work
    names = sorted({n for shot_names in vocabulary.values() for n in shot_names})
    try:
        classes = check_classes(names, haiku.name_classes(names)) if names else {}
    except ValueError as e:
        raise NodeError("internal", str(e)) from None
    timings["vocabulary_s"] = round(time.time() - t0, 1)   # with the name classification

    # 2. segment and track, shot by shot
    t0 = time.time()
    segmenter = load_segmenter(device, hf_token)
    timings["segmenter_load_s"] = round(time.time() - t0, 1)
    tracks, per_shot, segment_s = [], [], 0.0
    with open_video(video) as container:
        for shot in shots:
            nouns = vocabulary[shot["index"]]
            if not nouns:
                continue
            t0 = time.time()
            frames, times = shot_frames(container, shot["start"], shot["end"], fps)
            n_frames = len(frames)
            found = segment_shot(segmenter, frames, times, nouns, device, frame_sample) if frames else {}   # empties frames
            synchronize(device)
            seg = time.time() - t0
            segment_s += seg
            kept = 0
            for _, track in sorted(found.items()):
                if len(track["samples"]) < MIN_TRACK_SAMPLES:
                    continue
                tracks.append({"id": f"obj:{len(tracks)}", "label": track["label"], "shot": shot["index"],
                               "start": track["samples"][0]["t"], "end": track["samples"][-1]["t"],
                               "score": round(float(track["score"]), 3), "samples": track["samples"]})
                kept += 1
            per_shot.append({"shot": shot["index"], "frames": n_frames, "nouns": len(nouns), "segment_s": round(seg, 2), "tracks": kept})
    del segmenter
    timings["segment_s"] = round(segment_s, 1)

    # 3. embed, from the same frames
    t0 = time.time()
    embedder = load_embedder(device, hf_token)
    by_shot = {}
    for t in tracks:
        by_shot.setdefault(t["shot"], []).append(t)
    with open_video(video) as container:
        for shot in shots:
            if shot["index"] in by_shot:
                frames, times = shot_frames(container, shot["start"], shot["end"], fps, frames_needed(by_shot[shot["index"]]))
                embed_shot(by_shot[shot["index"]], {t: f for t, f in zip(times, frames) if f is not None}, embedder, device)
                synchronize(device)
    del embedder
    timings["embed_s"] = round(time.time() - t0, 1)   # with decoding the shots again and loading the model

    # 4. the crop check
    t0 = time.time()
    verdict = {}
    for label in sorted({t["label"] for t in tracks}):
        group = [t for t in tracks if t["label"] == label]
        for start in range(0, len(group), VERIFY_BATCH):
            batch = group[start:start + VERIFY_BATCH]
            try:
                answers = check_views(haiku.check_crops(label, [t["context"] for t in batch]), len(batch))
            except ValueError as e:
                raise NodeError("internal", f"{label}: {e}") from None
            verdict.update({t["id"]: answers[k] for k, t in enumerate(batch, 1)})
    timings["checks_s"] = round(time.time() - t0, 1)
    rejected = [t for t in tracks if verdict.get(t["id"]) in ("graphic", "absent")]
    tracks = [t for t in tracks if verdict.get(t["id"]) not in ("graphic", "absent")]

    # 5. wearers and identities
    try:
        bodies_at = body_samples(bodies, faces, shot_index)
    except KeyError as e:
        raise NodeError("input", f"a body track's shot {e} is not in the shots layer") from None
    assign_wearers(tracks, bodies_at, bodies["fps"], classes, c["wearer_share"])
    identities = link_identities(tracks, classes, different_people(faces, c["different_person_distance"]), c)
    assets = node.path("crops", "assets")
    (assets / "crops" / "objects").mkdir(parents=True, exist_ok=True)
    by_id = {t["id"]: t for t in tracks}
    crops = {}
    for identity, file in crop_files(identities).items():
        view = representative_view(next(i for i in identities if i["id"] == identity), by_id)
        if view is not None:
            Image.fromarray(view).save(assets / file, quality=90)
            crops[identity] = f"assets/{file}"

    node.model(SEGMENTER, licence_mode="commercial-ok", source=f"hf://{SEGMENTER}@{SEGMENTER_REVISION}", revision=SEGMENTER_REVISION,
               licence=SEGMENTER_LICENCE)
    node.model(EMBEDDER, licence_mode="commercial-ok", source=f"hf://{EMBEDDER}@{EMBEDDER_REVISION}", revision=EMBEDDER_REVISION,
               licence=EMBEDDER_LICENCE)
    node.model(HAIKU, licence_mode="api-terms", licence="Anthropic API terms")
    dtype = str(segmenter_dtype(device)).removeprefix("torch.")
    node.write_json("objects", "objects.json", {
        "layer": "objects", "schema_version": "0.1.0",
        "producer": node.producer(timings=timings, device=f"{device.type} (SAM 3 in {dtype})"),
        **layer_content(fps, tracks, rejected, identities, verdict, crops)})
    node.usage(vocabulary={str(k): v for k, v in vocabulary.items()}, classes=classes, per_shot=per_shot,
               tracks=len(tracks), rejected=len(rejected), identities=len(identities),
               linked=sum(len(i["tracks"]) > 1 for i in identities), with_wearer=sum(1 for t in tracks if t["wearer"]),
               verdicts={v: sum(1 for x in verdict.values() if x == v) for v in ("object", "graphic", "absent")})
