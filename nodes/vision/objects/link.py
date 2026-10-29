"""vision.objects: what things are on screen, tracked within shots and linked across them, as the bundle's objects
layer. The last of the objects steps (names, track, link): from the track step's tracks (object_tracks.py), on the CPU, with
no model but Claude Haiku 5.5 over the API. Steps 4 and 5 of the objects node 1.1.0, which it replaces: the same layer,
side file and crops from the same tracks and Haiku answers (test_split.py).

4. Crop check (haiku.py): each track's three outlined moments are checked for really being its label (object / graphic
   / absent), VERIFY_BATCH tracks of one label a call. The calls are independent, so they go haiku.WORKERS at a time
   (1.1.0 made them one after another: 97-253 s, measured, that gaze and gestures waited on); each answer is matched to
   its tracks by the call it came from, never by when it came. Tracks judged graphic or absent are kept apart
   (rejected_tracks) and never linked.
5. Identities (linking.py): the vocabulary's worn tracks get a wearer, the person (faces layer) whose posed body (bodies
   layer, linked to a face track) holds them; tracks of one family are merged by average-linkage clustering of adjusted
   cosines. Each identity's crop (assets/crops/objects/) is its embedded view closest to the identity's mean embedding.

Every mask SAM 3 gave, with its scores and what became of it, goes to the side file layers/objects_masks.parquet
(object_raw.py), which the layer points at (raw.masks): the track step's rows, with the rejected tracks' marked.

The layer's producer names the three steps' work, as 1.1.0's did: the models (SAM 3 and DINOv3 as the track step
recorded them, then Haiku), the track step's settings with this step's, and SAM 3's device.

What leaves the machine: the crop check's images, crops of video frames around each track, which show the people on
screen (a worn item's crop, glasses or earrings, centres on its wearer's face): frames and faces. No data from the faces
or bodies layers is sent.

Licence: SAM 3 and DINOv3 are under Meta's own licences (commercial use permitted, with conditions; gated on Hugging
Face), the names and checks are Claude's output under the Anthropic API terms; the layer refers to the faces layer's
people, so the bundle gives it the faces layer's academic-only mode.
"""

import threading
import time

from perception_node import Node, NodeError
import object_raw
import object_tracks
from haiku import MODEL as HAIKU, VERIFY_BATCH, Haiku, fan_out, spent
from linking import (assign_wearers, by_shot, check_views, crop_files, different_people, layer_content, link_identities,
                     representative_view)

with Node() as node:
    c = node.config
    shots, faces, bodies = node.input_json("shots")["shots"], node.input_json("faces"), node.input_json("bodies")
    shot_index = {s["id"]: s["index"] for s in shots}
    folder = node.input("tracks")
    tracks, made = object_tracks.read(folder)
    rows, masks_meta = object_raw.read(folder / object_tracks.MASKS)
    classes = made["classes"]
    try:
        bodies_of_shot, faces_of_shot = by_shot(bodies, shot_index), by_shot(faces, shot_index)
    except KeyError as e:
        raise NodeError("input", f"a body or face track's shot {e} is not in the shots layer") from None
    if missing := sorted({t["shot"] for t in tracks} - set(shot_index.values())):
        raise NodeError("input", f"the tracks' shots {missing} are not in the shots layer")
    person_of_face = {t["id"]: t.get("person") for t in faces["tracks"]}
    threshold = bodies["score_threshold"]   # a keypoint is seen at this score
    totals, lock, timings = {"in": 0, "out": 0, "calls": 0}, threading.Lock(), {}

    def spend(n_in, n_out):
        """The running total after every billed call (from fan_out's threads, one at a time): if a later call fails, the
        response still reports it."""
        with lock:
            totals.update({"in": totals["in"] + n_in, "out": totals["out"] + n_out, "calls": totals["calls"] + 1})
            node.usage(usd=spent(totals["in"], totals["out"]), tokens_in=totals["in"], tokens_out=totals["out"],
                       calls=totals["calls"])

    haiku = Haiku(node.secret("ANTHROPIC_API_KEY"), spend)

    # 4. the crop check: the calls 1.1.0 made one after another (by label, VERIFY_BATCH tracks a call), several at once
    t0 = time.time()
    calls = []
    for label in sorted({t["label"] for t in tracks}):
        group = [t for t in tracks if t["label"] == label]
        calls += [(label, group[start:start + VERIFY_BATCH]) for start in range(0, len(group), VERIFY_BATCH)]

    def check(call):
        label, batch = call
        try:
            return check_views(haiku.check_crops(label, [t["context"] for t in batch]), len(batch))
        except ValueError as e:
            raise NodeError("internal", f"{label}: {e}") from None

    verdict = {}
    for (_, batch), answers in zip(calls, fan_out(check, calls)):   # in the calls' order, whichever finished first
        verdict.update({t["id"]: answers[k] for k, t in enumerate(batch, 1)})
    timings["checks_s"] = round(time.time() - t0, 1)
    rejected = [t for t in tracks if verdict.get(t["id"]) in ("graphic", "absent")]
    tracks = [t for t in tracks if verdict.get(t["id"]) not in ("graphic", "absent")]
    for t in rejected:   # the track step wrote every track's rows as kept
        for i in t["rows"]:
            rows[i].update(outcome="rejected", track="", rejected=t["id"], sample=-1)

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
            object_tracks.copy_crop(view, assets / file)
            crops[identity] = f"assets/{file}"

    for m in made["models"]:   # SAM 3 and DINOv3, as the track step recorded them
        node.model(**m)
    node.model(HAIKU, licence_mode="api-terms", licence="Anthropic API terms")
    try:
        object_raw.write(node.path("masks", "objects_masks.parquet"), rows,
                         {k: v for k, v in masks_meta.items() if k != "side"})
    except ValueError as e:
        raise NodeError("internal", f"the masks side file: {e}") from None
    # the settings the layer was made with: the track step's (fps, clothing_fps) and this step's
    producer = {**node.producer(timings=timings, device=made["device"]), "config": {**made["config"], **c}}
    node.write_json("objects", "objects.json", {
        "layer": "objects", "schema_version": "0.1.0", "producer": producer,
        "raw": {"masks": {"file": object_raw.FILE, "format": "parquet", "description": object_raw.DESCRIPTION}},
        **layer_content(made["fps"], tracks, rejected, identities, verdict, crops)})
    reasons = {}
    for r in rows:
        reasons[f"{r['found']}/{r['reason']}"] = reasons.get(f"{r['found']}/{r['reason']}", 0) + 1
    node.usage(tracks=len(tracks), rejected=len(rejected), identities=len(identities), checks=len(calls),
               clothing_tracks=sum(1 for t in tracks if t["found"] == "body"),
               linked=sum(len(i["tracks"]) > 1 for i in identities), with_wearer=sum(1 for t in tracks if t["wearer"]),
               verdicts={v: sum(1 for x in verdict.values() if x == v) for v in ("object", "graphic", "absent")},
               masks=reasons)
