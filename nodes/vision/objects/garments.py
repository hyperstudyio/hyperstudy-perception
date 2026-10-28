"""Clothing from bodies: what each posed person wears, found by SAM 3 inside that person's box with fixed garment
nouns, so the wearer is that person by construction. NumPy only: test_garments.py runs without the models.

Why: the vocabulary (Haiku, from keyframes) named clothing only when it found it notable, and differently from run to
run (14 clothing prompts in one OOO run, 57 in the next, on the same inputs); plain t-shirts were never named, so two
of OOO's seven panelists had no clothing in any run. And wearers came from whose body box held an object's box centre,
which gave seated neighbours each other's clothes. Here every body sample of the bodies layer is cropped (a square
around its box, CROP_SCALE box sides, so the model sees the person at its own aspect ratio) and SAM 3's detector is
asked for the same NOUNS in every crop.

The nouns are the ones SAM 3 grounded on OOO clip 1's panelists (the PR that added this measured 22 candidates on 28
crops of the seven panelists): "t-shirt" and "shirt" for plain and printed tops (0.85-0.98 on the two panelists the
vocabulary never clothed), "cardigan" and "jacket" for outer layers (cardigans score as cardigans, 0.72-0.90, never as
jackets; a zip jacket scores 0.75 as a jacket), "pants", "skirt" and "shorts" below, and "shoe", "hat", "glasses" and
"necklace". Left out: "top" (it fires on every upper garment, outer layers included, so it adds no region), "coat"
(never above "jacket"), "dress" (fired at 0.55-0.87 on cardigans), "earring", "watch", "bag" (below 0.6 or never),
"clothing" (the whole person's outfit as one region), and, after a full OOO pass, "sweater" and "jeans": another noun
found the same garment (box IoU >= 0.6, scoring >= 0.5) for 61 of sweater's 62 items and 113 of jeans' 117, and each
noun costs about a twentieth of a second a crop. "person" is asked for too: its mask is what an object's mask must not
swallow (person_masks.py).

A detection belongs to the person whose keypoint region for its slot holds the detection's mask centroid (regions(),
owner()): the crop shows neighbours too, and their shirts score as well as this person's. Per body sample a slot keeps
its best owned detection, scoring at least KEEP_SCORE (SAM 3 video's own detection threshold); the upper body keeps up
to two layers, an outer garment (OUTER nouns) and an inner one whose mask is not the outer one's (mask IoU under
SAME_REGION_IOU), so a cardigan over a white top gives both; shoes keep every owned shoe as one item. A body track's
items of one slot and layer over its samples are one object track (garment_tracks), labelled with its most common noun.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

# noun -> slot; the order is the order of the side file's rows and of label ties
NOUNS = {"t-shirt": "upper", "shirt": "upper", "cardigan": "upper", "jacket": "upper",
         "pants": "lower", "skirt": "lower", "shorts": "lower",
         "shoe": "feet", "hat": "head", "glasses": "eyes", "necklace": "neck"}
OUTER = ("cardigan", "jacket")   # the upper slot's outer layer; its other nouns are the inner layer
PERSON = "person"
PROMPTS = (*NOUNS, PERSON)   # what SAM 3 is asked for in every crop, in this order
# each slot's family: the vocabulary's name classification groups upper-body clothing the same way (haiku.py), so a
# clothing name the vocabulary gives anyway is one family with these, and a duplicate of one of them (linking.py)
FAMILIES = {"upper": "upper-body clothing", "lower": "lower-body clothing", "feet": "footwear", "head": "headwear",
            "eyes": "eyewear", "neck": "necklace"}
STORE_SCORE = 0.3        # detections kept in the side file (SAM 3's own image post-processing threshold)
KEEP_SCORE = 0.5         # detections that can be an item (SAM 3 video's detection threshold, score_threshold_detection)
SAME_REGION_IOU = 0.5    # an inner-layer detection with this mask IoU with the outer one is the same garment
CROP_SCALE = 1.1         # the crop's side, in body box sides (the longer one)
PERSON_MIN_KEYPOINTS = 3  # a "person" mask is this body's when it holds at least this many of its keypoints

# COCO-WholeBody keypoints (the bodies layer's keypoint_names): body 0-22, face 23-90 (face-8 is the chin)
NOSE, EYES, EARS = 0, (1, 2), (3, 4)
SHOULDERS, WRISTS, HIPS, KNEES, ANKLES, FEET = (5, 6), (9, 10), (11, 12), (13, 14), (15, 16), tuple(range(17, 23))
HANDS = tuple(range(91, 133))
FACE = tuple(range(23, 91))
CHIN = 23 + 8
BODY_POINTS = tuple(range(0, 23))


def session_names(names: list[str]) -> tuple[list[str], list[str]]:
    """(a shot's session prompts, the names left out): a vocabulary name the body pass asks for (PROMPTS) is not a
    session prompt. A garment would be found twice; "person" would make people objects (Haiku named it in 5 NES shots
    despite the rule against people), and the people are the faces and bodies layers'."""
    return [n for n in names if n not in PROMPTS], [n for n in names if n in PROMPTS]


def crop_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    """The crop around a body box [x, y, w, h]: a square CROP_SCALE times its longer side about its centre, inside the
    frame (cut where it leaves it), as (x0, y0, x1, y1) in whole pixels."""
    x, y, w, h = box
    half = round(max(w, h) * CROP_SCALE / 2, 6)   # 1.1 x 100 is 110.00000000000001: no stray pixel from that
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(max(0, np.floor(cx - half))), int(max(0, np.floor(cy - half)))
    x1, y1 = int(min(width, np.ceil(cx + half))), int(min(height, np.ceil(cy + half)))
    return x0, y0, x1, y1


def _points(kpts, idx, threshold):
    k = np.asarray(kpts, float)[list(idx)]
    return k[k[:, 2] >= threshold, :2]


def regions(kpts, box, threshold: float) -> dict[str, tuple[float, float, float, float]]:
    """{slot: (x0, y0, x1, y1)}: where a body's items of each slot are, from its keypoints scoring at least
    `threshold` (the bodies layer's score_threshold) and its box [x, y, w, h]. A slot whose keypoints are not seen
    (legs out of frame, a face turned away) has no region, and so no item. Scales: shoulder width and torso length
    (shoulders to hips; from the box when the hips are not seen). "hands" (wrists and hand keypoints) is no garment's
    slot: it is where a ring or a held thing is, for the vocabulary's worn things (linking.assign_wearers)."""
    bx, by, bw, bh = box
    shoulders, hips = _points(kpts, SHOULDERS, threshold), _points(kpts, HIPS, threshold)
    legs = _points(kpts, KNEES + ANKLES, threshold)
    feet = _points(kpts, ANKLES + FEET, threshold)
    face = _points(kpts, FACE, threshold)
    if len(face) < 3:   # the face's own points, else the eyes, ears and nose
        face = _points(kpts, (NOSE, *EYES, *EARS), threshold)
    out = {}
    width = float(np.ptp(shoulders[:, 0])) if len(shoulders) == 2 else 0.0
    width = max(width, 0.15 * bw)
    if len(shoulders):
        top = float(shoulders[:, 1].mean())
        bottom = float(hips[:, 1].mean()) if len(hips) else by + bh
        torso = max(bottom - top, 0.5 * width)
        xs = np.concatenate([shoulders[:, 0], hips[:, 0]]) if len(hips) else shoulders[:, 0]
        out["upper"] = (xs.min() - 0.35 * width, top - 0.2 * torso, xs.max() + 0.35 * width, bottom + 0.2 * torso)
        chin = _points(kpts, (CHIN,), threshold)
        neck_top = float(chin[0, 1]) if len(chin) else (float(face[:, 1].max()) if len(face) else top - 0.3 * torso)
        out["neck"] = (shoulders[:, 0].min() - 0.1 * width, neck_top - 0.1 * torso, shoulders[:, 0].max() + 0.1 * width,
                       top + 0.6 * torso)
    else:
        torso = 0.3 * bh
    if len(hips):
        hip_w = max(float(np.ptp(hips[:, 0])) if len(hips) == 2 else 0.0, 0.5 * width)
        xs = np.concatenate([hips[:, 0], legs[:, 0]]) if len(legs) else hips[:, 0]
        low = float(legs[:, 1].max()) if len(legs) else by + bh
        out["lower"] = (xs.min() - 0.5 * hip_w, float(hips[:, 1].mean()) - 0.2 * torso, xs.max() + 0.5 * hip_w,
                        low + 0.1 * torso)
    if len(feet):
        r = 0.12 * max(bw, bh)
        out["feet"] = (feet[:, 0].min() - r, feet[:, 1].min() - r, feet[:, 0].max() + r, feet[:, 1].max() + r)
    hands = _points(kpts, WRISTS + HANDS, threshold)
    if len(hands):
        r = 0.06 * max(bw, bh)
        out["hands"] = (hands[:, 0].min() - r, hands[:, 1].min() - r, hands[:, 0].max() + r, hands[:, 1].max() + r)
    if len(face) >= 2:
        x0, y0, x1, y1 = face[:, 0].min(), face[:, 1].min(), face[:, 0].max(), face[:, 1].max()
        fw, fh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        out["head"] = (x0 - 0.5 * fw, y0 - 1.2 * fh, x1 + 0.5 * fw, y0 + 0.5 * fh)
        out["eyes"] = (x0 - 0.2 * fw, y0 - 0.8 * fh, x1 + 0.2 * fw, y0 + 0.6 * fh)
    return {k: tuple(float(v) for v in r) for k, r in out.items()}


def inside(region, x: float, y: float) -> bool:
    x0, y0, x1, y1 = region
    return x0 <= x <= x1 and y0 <= y <= y1


def owner(slot: str, centroid, bodies: dict[str, dict]) -> str | None:
    """The body (id) a detection of `slot` with this mask centroid (x, y in frame pixels) belongs to: the one whose
    region for the slot holds the centroid, the nearest region centre where several do; None if none does. bodies:
    {body id: regions()} of every body in the frame."""
    x, y = centroid
    holding = [(np.hypot(x - (r[0] + r[2]) / 2, y - (r[1] + r[3]) / 2), body)
               for body, rs in bodies.items() if (r := rs.get(slot)) is not None and inside(r, x, y)]
    return min(holding)[1] if holding else None


def person_owner(mask, offset, bodies_kpts: dict[str, np.ndarray], threshold: float) -> tuple[str | None, int]:
    """(body id, keypoints held) for a "person" mask (a crop's, at offset (x0, y0) in the frame): the body with the most
    of its seen body keypoints (BODY_POINTS) inside the mask, if at least PERSON_MIN_KEYPOINTS; else (None, most)."""
    x0, y0 = offset
    h, w = mask.shape
    best, most = None, 0
    for body, kpts in bodies_kpts.items():
        p = _points(kpts, BODY_POINTS, threshold)
        cx, cy = np.floor(p[:, 0]).astype(int) - x0, np.floor(p[:, 1]).astype(int) - y0
        ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        n = int(mask[cy[ok], cx[ok]].sum())
        if n > most:
            best, most = body, n
    return (best, most) if most >= PERSON_MIN_KEYPOINTS else (None, most)


def mask_iou(a, b) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def pick(body: str, detections: list[dict], owners: list[str | None]) -> dict[str, list[int]]:
    """Which of one body crop's detections become this body's items: {item key: [detection indices]}. detections:
    [{noun, score, mask}] (masks of one crop), owners: each one's owner() (or person_owner() for "person"). Item keys:
    "upper/outer", "upper/inner", "lower", "feet", "head", "eyes", "neck" and "person" (the person mask). Every slot takes
    its best owned detection scoring at least KEEP_SCORE; feet take all of them (a person's shoes are one item); the
    inner upper layer takes the best whose mask is not the outer one's."""
    mine = [i for i, d in enumerate(detections) if owners[i] == body and d["score"] >= KEEP_SCORE]
    best = lambda idx: max(idx, key=lambda i: (detections[i]["score"], -i)) if idx else None   # noqa: E731
    out: dict[str, list[int]] = {}
    person = best([i for i in mine if detections[i]["noun"] == PERSON])
    if person is not None:
        out["person"] = [person]
    outer = best([i for i in mine if detections[i]["noun"] in OUTER])
    if outer is not None:
        out["upper/outer"] = [outer]
    inner = [i for i in mine if NOUNS.get(detections[i]["noun"]) == "upper" and detections[i]["noun"] not in OUTER]
    for i in sorted(inner, key=lambda i: (-detections[i]["score"], i)):
        if outer is None or mask_iou(detections[i]["mask"], detections[outer]["mask"]) < SAME_REGION_IOU:
            out["upper/inner"] = [i]
            break
    for slot in ("lower", "head", "eyes", "neck"):
        found = best([i for i in mine if NOUNS.get(detections[i]["noun"]) == slot])
        if found is not None:
            out[slot] = [found]
    shoes = sorted(i for i in mine if detections[i]["noun"] == "shoe")
    if shoes:
        out["feet"] = shoes
    return out


def slot_of(key: str) -> str:
    return key.split("/")[0]


def garment_tracks(items: list[dict], min_samples: int) -> list[dict]:
    """Object tracks from the items pick() chose: one per body track and item key, over that body's samples, in order
    of (body track, NOUNS' slot order, outer before inner); ones with fewer than min_samples samples are left out.
    items: [{body, key, t, noun, score, sample}] where sample is the layer sample (t, box, poly, score). A track's label
    is its most common noun (the first in NOUNS on a tie), its score the mean detection score."""
    order = {k: i for i, k in enumerate(("upper/outer", "upper/inner", "lower", "feet", "head", "eyes", "neck"))}
    groups: dict[tuple, list[dict]] = {}
    for it in items:
        if it["key"] != "person":
            groups.setdefault((it["body"], it["key"]), []).append(it)
    noun_rank = {n: i for i, n in enumerate(NOUNS)}
    out = []
    for (body, key), group in sorted(groups.items(), key=lambda kv: (_body_index(kv[0][0]), order[kv[0][1]])):
        group = sorted(group, key=lambda it: it["t"])
        if len(group) < min_samples:
            continue
        counts = Counter(it["noun"] for it in group)
        label = min(counts, key=lambda n: (-counts[n], noun_rank[n]))
        out.append({"body": body, "key": key, "label": label, "family": FAMILIES[slot_of(key)],
                    "score": float(np.mean([it["score"] for it in group])), "items": group})
    return out


def _body_index(body: str) -> int:
    return int(body.split(":")[1])


class Series:
    """A track's samples by time (a body's or a face's): at(t) is the sample nearest t, if within half the track's own
    sampling step (the median gap between its samples), else None. Its own step, not its layer's fps: body-pose
    samples at about 7.49 a second on 29.97 fps video, not the 8 its layer names, so half of 1/8 s missed about 6% of
    object samples (the bake-off's wearer window). Times are to the millisecond, so a sample half a step away can read
    as up to a millisecond more (ROUNDING). A track of one sample stands for its own time only."""

    ROUNDING = 0.001

    def __init__(self, track: dict):
        self.track = track
        self.times = np.array([s["t"] for s in track["samples"]], float)
        self.half = float(np.median(np.diff(self.times))) / 2 if len(self.times) > 1 else 0.0

    def at(self, t: float) -> dict | None:
        if not len(self.times):
            return None
        i = int(np.argmin(np.abs(self.times - t)))
        return self.track["samples"][i] if abs(self.times[i] - t) <= self.half + self.ROUNDING + 1e-9 else None


def head_point(kpts, threshold: float):
    """Where a body's face is: its nose, else the middle of its seen eyes; None if neither is seen."""
    nose = _points(kpts, (NOSE,), threshold)
    if len(nose):
        return nose[0]
    eyes = _points(kpts, EYES, threshold)
    return eyes.mean(axis=0) if len(eyes) else None


def face_holding(point, t: float, faces: list[Series]) -> str | None:
    """The face track (id) whose box at t holds the point, the one whose box centre is nearest on a tie; None if none."""
    best = None
    for face in faces:
        s = face.at(t)
        if s is None:
            continue
        x, y, w, h = s["box"]
        if x <= point[0] <= x + w and y <= point[1] <= y + h:
            d = np.hypot(point[0] - x - w / 2, point[1] - y - h / 2)
            if best is None or d < best[0]:
                best = (d, face.track["id"])
    return best[1] if best else None


def body_person(body: Series, times, faces: list[Series], person_of_face: dict, threshold: float) -> str | None:
    """Whose body this is at these times: the person of the face track holding the body's head keypoints (head_point)
    at most of the times where one does (the earliest such person on a tie). Where the head is seen but no face holds
    it, nobody's: the body is not shown to be anyone. Only a body whose head is never seen (from behind) is the person
    of the face the bodies layer linked it to. The layer's link alone gave seated neighbours each other's clothes: on
    OOO a panelist's body at the frame's edge was linked to the face of the panelist beside her, which was inside the
    body's box, while the body's nose was at the edge, where no face was found."""
    votes: dict[str, int] = {}
    seen = False
    for t in times:
        s = body.at(t)
        point = head_point(s["kpts"], threshold) if s is not None else None
        if point is None:
            continue
        seen = True
        face = face_holding(point, t, faces)
        person = person_of_face.get(face) if face else None
        if person:
            votes[person] = votes.get(person, 0) + 1
    if votes:
        return max(votes, key=lambda p: votes[p])
    linked = body.track.get("face")
    return person_of_face.get(linked) if linked and not seen else None


def classes() -> dict[str, dict]:
    """The name classes of the body nouns, as the vocabulary's are given ({name: {family, kind, worn}})."""
    return {n: {"family": FAMILIES[s], "kind": "individual", "worn": True} for n, s in NOUNS.items()}
