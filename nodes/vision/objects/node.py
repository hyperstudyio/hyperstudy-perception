"""vision.objects: what things are on screen, tracked within shots and linked across them, as the bundle's objects layer.

Ported from the bake-off's objects/objects.py (its interactive segment_server.py is not part of this), with clothing
found on every posed person and no people inside object masks. First, before anything is uploaded or paid for: the
gated models must be ours to use, and the longest shot's SAM 3 session must fit in this machine's memory (budget.py).
Then:

1. Vocabulary (haiku.py): Claude Haiku 5.5 sees each shot's three keyframes (decode's) and names up to 8 concrete,
   trackable objects in generic detector nouns, told the names it used in earlier shots so one bike stays "bicycle".
   Clothing, shoes, hats, glasses and necklaces are not its to name (step 2 finds them on every person); a name it gives
   anyway that the body pass asks for (a garment noun, or "person") is left out (garments.session_names).
2. Segment and track (models.py), shot by shot; each shot is decoded at `fps` (shot_frames.py, the bake-off's ffmpeg
   frames):
   a. Clothing from bodies (garments.py): at `clothing_fps`, every body of the bodies layer present at that frame is
      cropped around its box and SAM 3's detector is asked there for fixed garment nouns and "person". Each body keeps
      the detections in its own keypoint regions: its items, whose wearer is its person (the face it is linked to). A
      body track's items of one kind are one object track.
   b. The shot runs through SAM 3 video with its names as text prompts. A session never spans a cut. Each mask of a thing
      not worn is checked against the people's masks of 2a (person_masks.py): one that is mostly people is dropped, one
      that holds some has them taken out, one whose outline draws around a person is dropped.
   Tracks with fewer than two samples are dropped (almost always a flicker detection).
3. Embed: each track is the mean DINOv3 ViT-L/16 CLS embedding of up to 8 tight box crops, letterboxed to a square
   (geometry.py), from the frames SAM 3 saw.
4. Checks (haiku.py): one call classifies the clip's names into families, lookalike or individual, and worn or not
   (made right after the vocabulary, before the GPU work: it needs nothing else; the body nouns' classes are fixed,
   garments.classes); each track's three outlined moments are checked for really being its label (object / graphic /
   absent). Tracks judged graphic or absent are kept apart (rejected_tracks) and never linked.
5. Identities (linking.py): the vocabulary's worn tracks get a wearer, the person (faces layer) whose posed body (bodies
   layer, linked to a face track) holds them; tracks of one family are merged by average-linkage clustering of adjusted
   cosines. Each identity's crop (assets/crops/objects/) is its embedded view closest to the identity's mean embedding.

Every mask SAM 3 gave, with its scores and what became of it, goes to the side file layers/objects_masks.parquet
(object_raw.py), which the layer points at (raw.masks).

Licence: SAM 3 and DINOv3 are under Meta's own licences (commercial use permitted, with conditions; gated on Hugging
Face), the names and checks are Claude's output under the Anthropic API terms; the layer refers to the faces layer's
people, so the bundle gives it the faces layer's academic-only mode.
"""

import time

import numpy as np

from perception_node import Node, NodeError
import garments
import object_raw
from budget import GB, physical_memory, too_long
from geometry import context_samples, context_strip, filled_outline, frame_sample, mask_box, mask_polygon, object_views
from haiku import MAX_OBJECTS, MODEL as HAIKU, VERIFY_BATCH, Haiku, spent
from linking import (assign_wearers, by_shot, check_classes, check_views, clean_names, crop_files, different_people,
                     layer_content, link_identities, representative_view)
from models import (EMBEDDER, EMBEDDER_LICENCE, EMBEDDER_REVISION, SEGMENTER, SEGMENTER_LICENCE, SEGMENTER_REVISION,
                    CropDetector, check_access, device_for, embed_crops, load_embedder, load_segmenter, segment_shot,
                    segmenter_dtype, synchronize)
from garments import body_person
from person_masks import PersonMasks, check as check_people
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


def clothe_frame(frame, t, shot_id, present, detector, threshold, people):
    """Step 2a at one frame: (items, rows). present: [(body track, its sample at t)]. Every body is cropped
    (garments.crop_box) and SAM 3's detector asked for garments.PROMPTS in each crop; a detection is the body's when
    its centroid is in the body's region for its slot (its keypoints in the body's mask, for "person"), and pick()
    chooses the body's items among its own. Each body's person mask goes to `people`. items: [{body, key, t, noun,
    score, sample, rows}] (rows: indices into rows of the detections that made it); rows: object_raw rows, one per
    detection."""
    height, width = frame.shape[:2]
    boxes = [garments.crop_box(s["box"], width, height) for _, s in present]
    found = detector.detect([np.ascontiguousarray(frame[y0:y1, x0:x1]) for x0, y0, x1, y1 in boxes], garments.STORE_SCORE)
    regions = {b["id"]: garments.regions(s["kpts"], s["box"], threshold) for b, s in present}
    kpts = {b["id"]: s["kpts"] for b, s in present}
    items, rows = [], []
    for (body, _), (x0, y0, _, _), detections in zip(present, boxes, found):
        owners, base = [], len(rows)
        for d in detections:
            mask = d["mask"]
            if d["noun"] == garments.PERSON:
                owners.append(garments.person_owner(mask, (x0, y0), kpts, threshold)[0])
            else:
                ys, xs = np.nonzero(mask)
                owners.append(garments.owner(garments.NOUNS[d["noun"]], (xs.mean() + x0, ys.mean() + y0), regions))
            x, y, w, h = mask_box(mask)
            rows.append(object_raw.row(t, shot_id, "body", d["noun"], d["score"], (x + x0, y + y0, w, h),
                                       [[px + x0, py + y0] for px, py in mask_polygon(mask)], "", body=body["id"]))
        picked = garments.pick(body["id"], detections, owners)
        taken = {i: key for key, idx in picked.items() for i in idx}
        for i, d in enumerate(detections):
            rows[base + i]["reason"] = ("person" if taken.get(i) == "person" else "item") if i in taken else (
                "below_keep" if d["score"] < garments.KEEP_SCORE else
                "not_this_body" if owners[i] != body["id"] else "not_picked")
        for key, idx in picked.items():
            mask = np.logical_or.reduce([detections[i]["mask"] for i in idx])
            if key == "person":
                people.add(t, mask, (x0, y0))
                continue
            top = max(idx, key=lambda i: (detections[i]["score"], -i))
            items.append({"body": body["id"], "key": key, "t": t, "noun": detections[top]["noun"],
                          "score": detections[top]["score"], "sample": frame_sample(t, mask, detections[top]["score"], (x0, y0)),
                          "rows": [base + i for i in idx]})
    return items, rows


with Node() as node:
    from PIL import Image

    c = node.config
    fps = c["fps"]
    body_step = max(1, round(fps / c["clothing_fps"]))   # every body_step-th frame of a shot gets the body pass
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

    # 0. before anything is uploaded or paid for: the gated models are ours to use, every shot fits in memory, and every
    # body's shot is in the shots layer
    check_access(hf_token)
    long = too_long(shots, fps, media["width"], media["height"], segmenter_dtype(device).itemsize, physical_memory() - RESERVED_GB * GB)
    if long:
        raise NodeError("resources", f"shot {long['index']} is {long['frames']} frames at {fps} fps: SAM 3 holds every frame of "
                                     f"a shot at once ({long['gb']} GB), more than this machine has. Lower this node's fps, or run "
                                     "it on a machine with more memory (budget.py)")
    try:
        bodies_of_shot, faces_of_shot = by_shot(bodies, shot_index), by_shot(faces, shot_index)
    except KeyError as e:
        raise NodeError("input", f"a body or face track's shot {e} is not in the shots layer") from None
    person_of_face = {t["id"]: t.get("person") for t in faces["tracks"]}
    threshold = bodies["score_threshold"]   # a keypoint is seen at this score

    # 1. vocabulary: one call per shot, in order, each told the names used before it (never after)
    t0 = time.time()
    vocabulary, earlier, body_named = {}, set(), {}
    for shot in shots:
        images = []
        for kf in shot.get("keyframes") or []:
            path = (keyframes_dir / kf["file"].removeprefix("assets/")).resolve()   # follows links: one out is caught too
            if not path.is_relative_to(keyframes_dir):
                raise NodeError("input", f"keyframe {kf['file']} is outside the keyframes input; only those images are sent")
            images.append(path.read_bytes())
        if not images:
            raise NodeError("input", f"shot {shot['index']} has no keyframes: the object names come from them")
        named = clean_names(haiku.vocabulary(images, earlier), MAX_OBJECTS)
        earlier.update(named)
        # the body pass's own nouns (garments, "person") are not session prompts
        vocabulary[shot["index"]], body_named[shot["index"]] = garments.session_names(named)
    # the names' families, kinds and whether they are worn: from the vocabulary alone, so before the GPU work
    names = sorted({n for shot_names in vocabulary.values() for n in shot_names})
    try:
        classes = check_classes(names, haiku.name_classes(names)) if names else {}
    except ValueError as e:
        raise NodeError("internal", str(e)) from None
    classes.update(garments.classes())
    timings["vocabulary_s"] = round(time.time() - t0, 1)   # with the name classification

    # 2. segment and track, shot by shot: the body crops first (the clothing, and the people's masks), then the session
    t0 = time.time()
    segmenter = load_segmenter(device, hf_token)
    detector = CropDetector(segmenter, garments.PROMPTS, device, hf_token)
    timings["segmenter_load_s"] = round(time.time() - t0, 1)
    tracks, rows, per_shot, segment_s, clothing_s = [], [], [], 0.0, 0.0
    with open_video(video) as container:
        for shot in shots:
            nouns, shot_bodies, shot_id = vocabulary[shot["index"]], bodies_of_shot.get(shot["index"], []), f"shot:{shot['index']}"
            if not nouns and not shot_bodies:
                continue
            t0 = time.time()
            frames, times = shot_frames(container, shot["start"], shot["end"], fps)
            n_frames = len(frames)
            people, items = PersonMasks(media["width"], media["height"]), []
            for k in range(0, n_frames, body_step):
                present = [(b.track, s) for b in shot_bodies if (s := b.at(times[k])) is not None]
                if present:
                    found, found_rows = clothe_frame(frames[k], times[k], shot_id, present, detector, threshold, people)
                    for it in found:
                        it["rows"] = [len(rows) + i for i in it["rows"]]
                    items += found
                    rows += found_rows
            synchronize(device)
            clothe = time.time() - t0
            clothing_s += clothe
            t0 = time.time()
            session = {}

            def on_mask(t, noun, obj, det_score, tracker_score, mask):
                """One session mask: checked against the people unless worn, kept (or trimmed) as a sample or dropped."""
                track = session.setdefault(obj, {"label": noun, "score": det_score, "samples": [], "rows": []})
                x, y, w, h = mask_box(mask)
                if classes[noun]["worn"]:
                    reason, kept, share, enclosed = "worn", mask, -1.0, -1.0
                else:
                    near = people.nearest(t, body_step / (2 * fps))
                    reason, kept, share, enclosed = check_people(mask, people.people(near) if near is not None else [], filled_outline)
                sample = frame_sample(t, kept, tracker_score) if reason in ("kept", "trimmed", "worn") else None
                poly = sample["poly"] if sample is not None and kept is mask else mask_polygon(mask)
                r = object_raw.row(t, shot_id, "vocabulary", noun, det_score, (x, y, w, h), poly, reason, sam=obj,
                                   tracker_score=tracker_score, person_share=share, enclosed_share=enclosed)
                if sample is not None:
                    r["sample"] = len(track["samples"])
                    track["samples"].append(sample)
                track["rows"].append(len(rows))
                rows.append(r)

            if nouns and frames:
                segment_shot(segmenter, frames, times, nouns, device, on_mask)   # empties frames
            synchronize(device)
            seg = time.time() - t0
            segment_s += seg
            kept = 0
            for _, track in sorted(session.items()):
                if len(track["samples"]) < MIN_TRACK_SAMPLES:
                    for i in track["rows"]:
                        rows[i].update(outcome="too_short", sample=-1)
                    continue
                tracks.append({"id": f"obj:{len(tracks)}", "label": track["label"], "shot": shot["index"], "found": "vocabulary",
                               "start": track["samples"][0]["t"], "end": track["samples"][-1]["t"],
                               "score": round(float(track["score"]), 3), "samples": track["samples"], "rows": track["rows"]})
                kept += 1
            worn_kept = 0
            for g in garments.garment_tracks(items, MIN_TRACK_SAMPLES):
                body = next(b for b in shot_bodies if b.track["id"] == g["body"])
                samples = [it["sample"] for it in g["items"]]
                for j, it in enumerate(g["items"]):
                    for i in it["rows"]:
                        rows[i]["sample"] = j
                tracks.append({"id": f"obj:{len(tracks)}", "label": g["label"], "shot": shot["index"], "found": "body",
                               "body": g["body"], "family": g["family"], "start": samples[0]["t"], "end": samples[-1]["t"],
                               "wearer": body_person(body, [s["t"] for s in samples], faces_of_shot.get(shot["index"], []),
                                                     person_of_face, threshold),
                               "score": round(g["score"], 3), "samples": samples,
                               "rows": [i for it in g["items"] for i in it["rows"]]})
                worn_kept += 1
            in_track = {i for t in tracks if t["shot"] == shot["index"] and t["found"] == "body" for i in t["rows"]}
            for it in items:
                if it["rows"][0] not in in_track:
                    for i in it["rows"]:
                        rows[i].update(outcome="too_short", sample=-1)
            per_shot.append({"shot": shot["index"], "frames": n_frames, "nouns": len(nouns), "bodies": len(shot_bodies),
                             "clothing_s": round(clothe, 2), "segment_s": round(seg, 2), "tracks": kept, "clothing_tracks": worn_kept,
                             "body_named": body_named[shot["index"]]})
    del segmenter, detector
    timings["clothing_s"] = round(clothing_s, 1)
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
    for t in tracks:
        for i in t["rows"]:   # a sample dropped for holding people names no layer sample
            rows[i].update(outcome="tracks", track=t["id"] if rows[i]["sample"] >= 0 else "")
    for t in rejected:
        for i in t["rows"]:
            rows[i].update(outcome="rejected", rejected=t["id"], sample=-1)

    # 5. wearers and identities
    assign_wearers(tracks, bodies_of_shot, faces_of_shot, person_of_face, classes, c["wearer_share"], threshold)
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
    try:
        object_raw.write(node.path("masks", "objects_masks.parquet"), rows,
                         {"segmenter": f"{SEGMENTER}@{SEGMENTER_REVISION}", "frame_width": media["width"],
                          "frame_height": media["height"], "fps": fps, "clothing_fps": fps / body_step,
                          "store_score": garments.STORE_SCORE, "keep_score": garments.KEEP_SCORE,
                          "prompts": " ".join(garments.PROMPTS)})
    except ValueError as e:
        raise NodeError("internal", f"the masks side file: {e}") from None
    node.write_json("objects", "objects.json", {
        "layer": "objects", "schema_version": "0.1.0",
        "producer": node.producer(timings=timings, device=f"{device.type} (SAM 3 in {dtype})"),
        "raw": {"masks": {"file": object_raw.FILE, "format": "parquet", "description": object_raw.DESCRIPTION}},
        **layer_content(fps, tracks, rejected, identities, verdict, crops)})
    reasons = {}
    for r in rows:
        reasons[f"{r['found']}/{r['reason']}"] = reasons.get(f"{r['found']}/{r['reason']}", 0) + 1
    node.usage(vocabulary={str(k): v for k, v in vocabulary.items()}, classes=classes, per_shot=per_shot,
               tracks=len(tracks), rejected=len(rejected), identities=len(identities),
               clothing_tracks=sum(1 for t in tracks if t["found"] == "body"),
               linked=sum(len(i["tracks"]) > 1 for i in identities), with_wearer=sum(1 for t in tracks if t["wearer"]),
               verdicts={v: sum(1 for x in verdict.values() if x == v) for v in ("object", "graphic", "absent")},
               masks=reasons)
