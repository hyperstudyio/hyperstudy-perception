"""Who wears each object track, which tracks are one object across shots, and the objects layer: the bake-off's
objects.py (assign_wearers, link_identities and their rules) and convert_bakeoff.objects_layer, with the bake-off's run
files replaced by the graph's layers. NumPy only: test_object_links.py runs without the models.

Tracks are dicts: id ("obj:N"), label, shot (the shot's index), found ("vocabulary": a session track of a name the
vocabulary gave; "body": a garment found in a body's crop, garments.py, with body, its body track, and wearer, that
body's person), start, end, score, samples [{t, box [x, y, w, h], poly, score}], and once embedded, embedding (unit
vector or None), view_embeddings and views (a displayed crop per embedded view). Linking adds the vocabulary tracks'
wearer.
"""

from __future__ import annotations

import numpy as np

from garments import Series, body_person, inside, regions


def clean_names(names: list[str], limit: int) -> list[str]:
    """A shot's object names as the bake-off kept them: stripped, lower case, the first of repeats, the first `limit`
    (the schema cannot cap a list; the prompt asks for the most prominent first)."""
    return list(dict.fromkeys(n.strip().lower() for n in names if n.strip()))[:limit]


def check_classes(names: list[str], answer: list[dict]) -> dict[str, dict]:
    """The name classification's answer as {name: {family, kind, worn}}; ValueError unless it covers exactly these
    names. Families are compared in lower case."""
    classes = {c["name"].strip().lower(): {k: c[k] for k in ("family", "kind", "worn")} for c in answer}
    missing, extra = set(names) - set(classes), set(classes) - set(names)
    if missing or extra:
        raise ValueError(f"name classification does not match the vocabulary: missing {sorted(missing)}, "
                         f"unexpected {sorted(extra)}")
    for c in classes.values():
        c["family"] = c["family"].strip().lower()
    return classes


def check_views(answer: list[dict], n: int) -> dict[int, str]:
    """A crop check's verdict per view (1..n); ValueError unless it answers every view once."""
    verdicts = {a["view"]: a["verdict"] for a in answer}
    if sorted(verdicts) != list(range(1, n + 1)) or len(answer) != n:
        raise ValueError(f"the crop check answered views {sorted(a['view'] for a in answer)}, expected 1..{n}")
    return verdicts


# --- geometry -------------------------------------------------------------------------------------------------------

def overlaps(a, b) -> bool:
    return a["shot"] == b["shot"] and a["start"] <= b["end"] and b["start"] <= a["end"]


def box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return inter / (aw * ah + bw * bh - inter)


def one_body(a, b) -> bool:
    """Two garments the body pass found on one body at once: different garments (garments.pick: an outer layer and the
    inner one under it share a box, not a mask)."""
    return a.get("found") == b.get("found") == "body" and a["body"] == b["body"] and overlaps(a, b)


def shared_iou(a, b) -> float | None:
    """Mean box IoU of two tracks over their shared sample times; None if they share none (or are one body's)."""
    if not overlaps(a, b) or one_body(a, b):
        return None
    boxes_b = {s["t"]: s["box"] for s in b["samples"]}
    shared = [box_iou(s["box"], boxes_b[s["t"]]) for s in a["samples"] if s["t"] in boxes_b]
    return float(np.mean(shared)) if shared else None


def duplicates(a, b, duplicate_iou: float) -> bool:
    """Two tracks of one object in the same frames (SAM 3 removes duplicates within one text prompt, not across
    prompts): mean box IoU over their shared sample times >= duplicate_iou. Two garments of one body never are."""
    iou = shared_iou(a, b)
    return iou is not None and iou >= duplicate_iou


def track_size(t) -> float:
    return float(np.median([max(s["box"][2:]) for s in t["samples"]]))


# --- wearers --------------------------------------------------------------------------------------------------------

def by_shot(layer: dict, shot_index: dict[str, int]) -> dict[int, list[Series]]:
    """A bodies or faces layer's tracks as {shot index: [Series]}."""
    out: dict[int, list[Series]] = {}
    for track in layer["tracks"]:
        out.setdefault(shot_index[track["shot"]], []).append(Series(track))
    return out


def different_people(faces_layer: dict, distance: float) -> set[frozenset]:
    """Person pairs that are certainly different people: ArcFace centroids at least `distance` apart (the faces
    layer's person_pairs, the bake-off's identity_similarity.json)."""
    return {frozenset((p["a"], p["b"])) for p in faces_layer.get("person_pairs", []) if p["distance"] >= distance}


def holder(x: float, y: float, t: float, bodies: list[Series], threshold: float) -> Series | None:
    """The body whose keypoint regions (garments.regions, any slot, hands too) hold the point at t: the one with the
    nearest region centre where several do."""
    best = None
    for body in bodies:
        s = body.at(t)
        if s is None:
            continue
        for r in regions(s["kpts"], s["box"], threshold).values():
            if inside(r, x, y):
                d = np.hypot(x - (r[0] + r[2]) / 2, y - (r[1] + r[3]) / 2)
                if best is None or d < best[0]:
                    best = (d, body)
    return best[1] if best else None


def assign_wearers(tracks, bodies: dict[int, list[Series]], faces: dict[int, list[Series]], person_of_face: dict,
                   classes: dict, wearer_share: float, threshold: float) -> None:
    """Each worn vocabulary track's wearer: the person whose posed body holds the object's box centre in >= wearer_share
    of its samples. A body holds a point when one of its keypoint regions does (holder: around its torso, legs, feet,
    head, eyes, neck or hands, not its whole box, which took in seated neighbours); the body's person at that moment is
    the face holding its head keypoints (garments.body_person). Bodies are each one's sample nearest the object
    sample, within half its own sampling step. Sets t["wearer"] (a person id or None) on every vocabulary track; a body
    garment keeps its body's person."""
    for t in tracks:
        if t.get("found") == "body":
            continue
        t["wearer"] = None
        if not classes[t["label"]]["worn"]:
            continue
        votes: dict[str, int] = {}
        for s in t["samples"]:
            x, y = s["box"][0] + s["box"][2] / 2, s["box"][1] + s["box"][3] / 2
            body = holder(x, y, s["t"], bodies.get(t["shot"], []), threshold)
            person = body_person(body, [s["t"]], faces.get(t["shot"], []), person_of_face, threshold) if body else None
            if person:
                votes[person] = votes.get(person, 0) + 1
        if votes:
            person, n = max(votes.items(), key=lambda kv: kv[1])
            if n >= wearer_share * len(t["samples"]):
                t["wearer"] = person


# --- identities -----------------------------------------------------------------------------------------------------

def pair_score(a, b, different: set[frozenset], c: dict, sizes: tuple[float, float] | None = None) -> float | None:
    """Adjusted cosine for linking a and b, or None when they may never be one object. c: the linking settings
    (cross_name_margin, same_wearer_bonus, min_link_px). sizes: their track_size, when already known.
    Two body garments of two different persons (face clusters) never link: their wearers are known by construction, and
    on OOO jeans, plain tops and shoes of different panelists are alike (lower-body garments of different panelists
    reached 0.62 in 15% of pairs, shoes in 78%), while the face distance cannot tell the panelists apart (0.54 between
    two of them, 0.91 between two clusters of one). A person split over clusters has their clothes split with them:
    a split is fixed in review, a merge misleads."""
    if overlaps(a, b) or a["embedding"] is None or b["embedding"] is None:
        return None
    wa, wb = a.get("wearer"), b.get("wearer")
    if wa and wb and wa != wb and frozenset((wa, wb)) in different:
        return None   # worn by two different people
    if wa and wb and wa != wb and a.get("found") == b.get("found") == "body":
        return None   # worn by two persons
    same_wearer = bool(wa) and wa == wb
    if min(sizes or (track_size(a), track_size(b))) < c["min_link_px"] and not same_wearer:
        return None   # too small to identify by appearance alone
    score = float(a["embedding"] @ b["embedding"])
    if a["label"] != b["label"]:
        score -= c["cross_name_margin"]
    if same_wearer:
        score += c["same_wearer_bonus"]
    return score


def duplicate_groups(members, duplicate_iou: float) -> list[list[int]]:
    """Groups of member indices that are one object tracked under two names at once (duplicates), in order of their
    first member. Only tracks of one shot can be duplicates, so only those pairs are compared. Pairs join best first
    (highest mean IoU), and never so that one group holds two garments of one body at once (one_body): a vocabulary
    track over a cardigan can match both it and the top under it, and would otherwise make them one."""
    group = list(range(len(members)))   # union-find
    held = {i: [i] for i in range(len(members))}   # each root's members

    def root(i):
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i
    by_shot: dict[int, list[int]] = {}
    for i, t in enumerate(members):
        by_shot.setdefault(t["shot"], []).append(i)
    pairs = []
    for same_shot in by_shot.values():
        for x, i in enumerate(same_shot):
            for j in same_shot[x + 1:]:
                iou = shared_iou(members[i], members[j])
                if iou is not None and iou >= duplicate_iou:
                    pairs.append((-iou, i, j))
    for _, i, j in sorted(pairs):
        ri, rj = root(i), root(j)
        if ri == rj or any(one_body(members[a], members[b]) for a in held[ri] for b in held[rj]):
            continue
        group[rj] = ri
        held[ri] += held.pop(rj)
    clusters: dict[int, list[int]] = {}
    for i in range(len(members)):
        clusters.setdefault(root(i), []).append(i)
    return list(clusters.values())


def average_linkage(members, groups: list[list[int]], different: set[frozenset], c: dict) -> list[list[int]]:
    """Merge groups while some pair of them has every cross pair linkable and an average score >= the threshold: the
    best such pair first (the first in group order on a tie), the later group appended to the earlier one, as the
    bake-off's loop did. Group-to-group sums, counts and vetoes are kept and updated on each merge, so a merge costs
    one pass over the groups rather than over every pair of tracks."""
    g = len(groups)
    of = {i: k for k, grp in enumerate(groups) for i in grp}
    total, count = np.zeros((g, g)), np.zeros((g, g))
    blocked = np.eye(g, dtype=bool)
    sizes = [track_size(t) for t in members]
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = of[i], of[j]
            if a == b:
                continue
            s = pair_score(members[i], members[j], different, c, (sizes[i], sizes[j]))
            if s is None:
                blocked[a, b] = blocked[b, a] = True
            else:
                total[a, b] += s
                total[b, a] += s
                count[a, b] += 1
                count[b, a] += 1
    alive = np.ones(g, dtype=bool)
    lower = np.tril(np.ones((g, g), dtype=bool))   # each pair once, as (earlier, later)
    while True:
        avg = np.where(blocked | lower | ~alive[:, None] | ~alive[None, :], -np.inf, total / np.maximum(count, 1))
        k = int(np.argmax(avg))   # the first maximum in row-major order: the bake-off's scan order
        i, j = divmod(k, g)
        if not avg[i, j] >= c["link_threshold"]:
            break
        groups[i] += groups[j]
        alive[j] = False
        for m in (total, count):
            m[i, :] += m[j, :]
            m[:, i] = m[i, :]
        blocked[i, :] |= blocked[j, :]
        blocked[:, i] = blocked[i, :]
        blocked[i, i] = True
    return [grp for k, grp in enumerate(groups) if alive[k]]


def link_identities(tracks, classes: dict, different: set[frozenset], c: dict) -> list[dict]:
    """Average-linkage clustering within each name family: a merge needs every cross pair linkable (pair_score not
    None) and an average adjusted score >= c["link_threshold"]. Clustering starts from groups of duplicate tracks,
    which are one object from the outset. Lookalike families never link by appearance: a shot's tracks of one are one
    identity ("light 3": the shot's lights), not one each, which made 86 of OOO's 108 identities single light tubes and
    194 of NES's trees. Each identity is named after its members' most common label ("necklace 2"), numbered per label
    in order of first appearance."""
    identities, numbering = [], {}
    for family in sorted({classes[t["label"]]["family"] for t in tracks}):
        members = [t for t in tracks if classes[t["label"]]["family"] == family]
        lookalike = all(classes[t["label"]]["kind"] == "lookalike" for t in members)
        groups = duplicate_groups(members, c["duplicate_iou"])
        if lookalike:
            per_shot: dict[int, list[int]] = {}
            for g in groups:
                per_shot.setdefault(members[g[0]]["shot"], []).extend(g)
            groups = list(per_shot.values())
        else:
            groups = average_linkage(members, groups, different, c)
        found = [[members[i] for i in g] for g in groups]
        found.sort(key=lambda g: min(t["start"] for t in g))
        for cluster in found:
            labels = [t["label"] for t in cluster]
            label = max(sorted(set(labels)), key=labels.count)
            numbering[label] = numbering.get(label, 0) + 1
            wearers = {t["wearer"] for t in cluster if t.get("wearer")}
            identities.append({"id": f"{label} {numbering[label]}", "label": label, "family": family,
                               "worn_by": sorted(wearers), "tracks": [t["id"] for t in sorted(cluster, key=lambda t: t["start"])]})
    return identities


def representative_view(identity: dict, by_id: dict):
    """The identity's embedded view closest to its members' mean embedding (which skips frames where a mask drifted),
    or None when no member was embedded."""
    members = [by_id[t] for t in identity["tracks"] if by_id[t]["embedding"] is not None]
    if not members:
        return None
    centre = np.mean([t["embedding"] for t in members], axis=0)
    views = [(float(e @ centre), v) for t in members for e, v in zip(t["view_embeddings"], t["views"])]
    return max(views, key=lambda x: x[0])[1]


def slug(text: str) -> str:
    """convert_bakeoff.slug: an identity's crop file name."""
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def crop_files(identities: list[dict]) -> dict[str, str]:
    """{identity id: crops/objects/<slug>.jpg}: convert_bakeoff's file name for each identity's crop, in a folder of its
    own (convert put it in crops/, beside face-track's face_<i>.jpg, which an object called "face" would collide with).
    Two ids with one slug ("t-shirt 1", "t shirt 1") would overwrite one crop there; here the later one gets _2."""
    files, used = {}, set()
    for identity in identities:
        base = slug(identity["id"])
        name, k = base, 1
        while name in used:
            k += 1
            name = f"{base}_{k}"
        used.add(name)
        files[identity["id"]] = f"crops/objects/{name}.jpg"
    return files


# --- the layer ------------------------------------------------------------------------------------------------------

def sample(s) -> dict:
    return {"t": s["t"], "box": s["box"], "poly": s.get("poly", []), "score": s.get("score")}


def found(t) -> dict:
    """How a track was found: {"found": "vocabulary"} or {"found": "body", "body": its body track}."""
    return {"found": "body", "body": t["body"]} if t.get("found") == "body" else {"found": "vocabulary"}


def layer_content(fps, tracks, rejected, identities, verdict, crops: dict[str, str | None]) -> dict:
    """The objects layer's fps, tracks, identities and rejected_tracks, as convert_bakeoff.objects_layer writes them
    from the bake-off's objects.json. crops: {identity id: its crop's path in the bundle (assets/...), or None}."""
    identity_of = {t: i["id"] for i in identities for t in i["tracks"]}
    return {
        "fps": fps,
        "tracks": [{"id": t["id"], "label": t["label"], "identity": f"objid:{identity_of[t['id']]}", "shot": f"shot:{t['shot']}",
                    "start": t["start"], "end": t["end"], "wearer": t["wearer"], "verdict": verdict.get(t["id"]), **found(t),
                    "samples": [sample(s) for s in t["samples"]]} for t in tracks],
        "identities": [{"id": f"objid:{i['id']}", "label": i["label"], "family": i["family"], "worn_by": i["worn_by"],
                        "tracks": i["tracks"], "crop": crops.get(i["id"])} for i in identities],
        "rejected_tracks": [{"id": t["id"], "label": t["label"], "verdict": verdict[t["id"]], "shot": f"shot:{t['shot']}",
                             "start": t["start"], "end": t["end"], **found(t), "samples": [sample(s) for s in t["samples"]]}
                            for t in rejected],
    }
