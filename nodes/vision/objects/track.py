"""vision.objects-track: the named things and every posed person's clothing, outlined and tracked within shots by SAM 3
and embedded by DINOv3, as an object_tracks artifact (object_tracks.py) for the link step. The objects steps' GPU work:
it calls no API and sends nothing off the machine, so it waits on nothing but its inputs and can run on any machine
with a GPU (once the Hugging Face token is needed only to download the weights: RUN_ANYWHERE_PLAN.md phase 5).

Ported from the bake-off's objects/objects.py (its interactive segment_server.py is not part of this), with clothing
found on every posed person and no people inside object masks; steps 0, 2 and 3 of the objects node 1.1.0, unchanged.
First: the gated models must be ours to use, and the longest shot's SAM 3 session must fit in this machine's memory
(budget.py). Then:

2. Segment and track (models.py), shot by shot; each shot is decoded at `fps` (shot_frames.py, the bake-off's ffmpeg
   frames):
   a. Clothing from bodies (garments.py): at `clothing_fps`, every body of the bodies layer present at that frame is
      cropped around its box and SAM 3's detector is asked there for fixed garment nouns and "person". Each body keeps
      the detections in its own keypoint regions: its items, whose wearer is the person of the face holding the body's
      head keypoints over the body track (linking.person_of_body; the bodies layer's face link only for a body whose
      head is never seen). A body track's items of one kind are one object track.
   b. The shot runs through SAM 3 video with its vocabulary names (the names step's) as text prompts. A session never
      spans a cut. Each mask of a thing not worn (the name's class says) is checked against the people's masks of 2a
      (person_masks.py): one that is mostly people is dropped, one that holds some has them taken out, one whose
      outline draws around a person is dropped.
   Tracks with fewer than two samples are dropped (almost always a flicker detection).
3. Embed: each track is the mean DINOv3 ViT-L/16 CLS embedding of up to 8 tight box crops, letterboxed to a square
   (geometry.py), from the frames SAM 3 saw. The crops shown for each embedded sample and the crop check's strip (three
   outlined moments) are made from the same frames and kept as JPEG, so the link step needs neither the video nor a
   model.

Every mask SAM 3 gave goes to the artifact's masks.parquet (object_raw.py), as the side file will hold it if every track
passes the crop check; the link step writes the side file.

Licence: SAM 3 and DINOv3 are under Meta's own licences (commercial use permitted, with conditions; gated on Hugging
Face).
"""

import io
import time

import numpy as np

from perception_node import Node, NodeError
import garments
import object_raw
import object_tracks
import vocabulary
from budget import GB, physical_memory, too_long
from geometry import context_samples, context_strip, filled_outline, frame_sample, mask_box, mask_polygon, object_views
from haiku import jpeg
from linking import by_shot, person_of_body
from models import (EMBEDDER, EMBEDDER_LICENCE, EMBEDDER_REVISION, SEGMENTER, SEGMENTER_LICENCE, SEGMENTER_REVISION,
                    CropDetector, check_access, device_for, embed_crops, load_embedder, load_segmenter, segment_shot,
                    segmenter_dtype, synchronize)
from person_masks import PersonMasks, check as check_people
from shot_frames import shot_frames

MIN_TRACK_SAMPLES = 2   # a track seen on a single frame is almost always a flicker detection
EMBED_SAMPLES = 8       # crops averaged per track embedding, evenly spaced over the track
MIN_CROP_PX = 24        # crops smaller than this on both sides carry no identity information
RESERVED_GB = 8         # memory beside a SAM 3 session's frames: the models (3.4 and 1.2 GB on disk) and the rest
MODELS = [{"id": SEGMENTER, "source": f"hf://{SEGMENTER}@{SEGMENTER_REVISION}", "revision": SEGMENTER_REVISION,
           "licence": SEGMENTER_LICENCE, "licence_mode": "commercial-ok"},
          {"id": EMBEDDER, "source": f"hf://{EMBEDDER}@{EMBEDDER_REVISION}", "revision": EMBEDDER_REVISION,
           "licence": EMBEDDER_LICENCE, "licence_mode": "commercial-ok"}]


def open_video(path):
    """The video for shot_frames, decoded on several threads (the same frames, 7x faster on OOO)."""
    import av
    container = av.open(str(path))
    container.streams.video[0].thread_type = "AUTO"
    return container


def crop_jpeg(rgb) -> bytes:
    """A displayed crop as the objects layer's identity crops are saved: PIL JPEG, quality 90 (1.1.0 saved the chosen
    one with Image.save(path, quality=90), which encodes the same bytes)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def embed_picks(track) -> list[dict]:
    """The samples a track's embedding is made from: up to EMBED_SAMPLES big enough to see, evenly spaced."""
    usable = [s for s in track["samples"] if max(s["box"][2:]) >= MIN_CROP_PX]
    return [usable[int(i)] for i in np.linspace(0, len(usable) - 1, min(EMBED_SAMPLES, len(usable)))] if usable else []


def frames_needed(tracks) -> set[float]:
    """The times embed_shot reads frames at: each track's embedded samples and its crop-check moments."""
    return {s["t"] for t in tracks for s in embed_picks(t) + context_samples(t)}


def embed_shot(tracks, frame_at, model, device):
    """Embedding, view_embeddings, views (displayed crops, as JPEG) and context (the crop check's strip, as JPEG) for
    each track of one shot, from the frames SAM 3 saw (frame_at: {time: frame}, at least those at
    frames_needed(tracks))."""
    for track in tracks:
        picks = embed_picks(track)
        views = [object_views(frame_at[s["t"]], s["box"]) for s in picks]
        track["embedding"], track["view_embeddings"] = embed_crops(model, [e for e, _ in views], device) if views else (None, None)
        track["views"] = [crop_jpeg(shown) for _, shown in views]
        track["context"] = jpeg(context_strip(track, frame_at))


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
    c = node.config
    fps = c["fps"]
    body_step = round(fps / c["clothing_fps"])   # every body_step-th frame of a shot gets the body pass
    if body_step < 1 or abs(fps / c["clothing_fps"] - body_step) > 1e-6:
        raise NodeError("input", f"clothing_fps {c['clothing_fps']} is not a whole fraction of fps {fps} (fps / clothing_fps "
                                 "must be a whole number): the body pass runs on every n-th of the frames SAM 3 sees")
    video = node.input("video")
    shots_doc, faces, bodies = node.input_json("shots"), node.input_json("faces"), node.input_json("bodies")
    media = node.input_json("media")
    shots = shots_doc["shots"]
    shot_index = {s["id"]: s["index"] for s in shots}
    try:
        vocabulary_of, body_named, classes = vocabulary.session_prompts(node.input_json("vocabulary"), shots)
    except ValueError as e:
        raise NodeError("input", str(e)) from None
    classes.update(garments.classes())   # the body nouns' classes are fixed
    device = device_for(node.accelerator)
    hf_token = node.secret("HF_TOKEN")
    timings = {}

    # 0. before the GPU work: the gated models are ours to use, every shot fits in memory, and every body's shot is in
    # the shots layer
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

    # 2. segment and track, shot by shot: the body crops first (the clothing, and the people's masks), then the session
    t0 = time.time()
    segmenter = load_segmenter(device, hf_token)
    detector = CropDetector(segmenter, garments.PROMPTS, device, hf_token)
    timings["segmenter_load_s"] = round(time.time() - t0, 1)
    tracks, rows, per_shot, segment_s, clothing_s = [], [], [], 0.0, 0.0
    persons = {}   # each body track's person, decided once (linking.person_of_body)
    with open_video(video) as container:
        for shot in shots:
            nouns, shot_bodies, shot_id = vocabulary_of[shot["index"]], bodies_of_shot.get(shot["index"], []), f"shot:{shot['index']}"
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
                               "wearer": person_of_body(body, faces_of_shot.get(shot["index"], []), person_of_face,
                                                        threshold, persons),
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

    # 3. embed, from the same frames, with the crops the link step shows and checks
    t0 = time.time()
    embedder = load_embedder(device, hf_token)
    tracks_of_shot = {}
    for t in tracks:
        tracks_of_shot.setdefault(t["shot"], []).append(t)
    with open_video(video) as container:
        for shot in shots:
            if shot["index"] in tracks_of_shot:
                frames, times = shot_frames(container, shot["start"], shot["end"], fps, frames_needed(tracks_of_shot[shot["index"]]))
                embed_shot(tracks_of_shot[shot["index"]], {t: f for t, f in zip(times, frames) if f is not None}, embedder, device)
                synchronize(device)
    del embedder
    timings["embed_s"] = round(time.time() - t0, 1)   # with decoding the shots again and loading the model

    # the side file's rows as they stand if every track passes the crop check (link.py marks the ones that do not)
    for t in tracks:
        for i in t["rows"]:   # a sample dropped for holding people names no layer sample
            rows[i].update(outcome="tracks", track=t["id"] if rows[i]["sample"] >= 0 else "")
    for m in MODELS:
        node.model(**m)
    dtype = str(segmenter_dtype(device)).removeprefix("torch.")
    folder = node.path("tracks", "tracks")
    folder.mkdir(parents=True, exist_ok=True)
    try:
        object_raw.write(folder / object_tracks.MASKS, rows,
                         {"segmenter": f"{SEGMENTER}@{SEGMENTER_REVISION}", "frame_width": media["width"],
                          "frame_height": media["height"], "fps": fps, "clothing_fps": fps / body_step,
                          "store_score": garments.STORE_SCORE, "keep_score": garments.KEEP_SCORE,
                          "prompts": " ".join(garments.PROMPTS)})
    except ValueError as e:
        raise NodeError("internal", f"the masks side file: {e}") from None
    object_tracks.write(folder, tracks, {"fps": fps, "device": f"{device.type} (SAM 3 in {dtype})", "config": c,
                                         "models": MODELS, "classes": classes})
    node.usage(timings=timings, per_shot=per_shot, tracks=len(tracks), clothing_tracks=sum(1 for t in tracks if t["found"] == "body"),
               embedded=sum(1 for t in tracks if t["embedding"] is not None))
