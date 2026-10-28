"""Who wears each object track, which tracks are one object across shots, and the objects layer: the bake-off's
objects.py (assign_wearers, link_identities and their rules) and convert_bakeoff.objects_layer, with the bake-off's run
files replaced by the graph's layers. NumPy only: test_object_links.py runs without the models.

Tracks are dicts: id ("obj:N"), label, shot (the shot's index), start, end, score, samples [{t, box [x, y, w, h], poly,
score}], and once embedded, embedding (unit vector or None), view_embeddings and views (a displayed crop per embedded
view). Linking adds wearer.
"""

from __future__ import annotations

import numpy as np


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


def contains(box, x, y) -> bool:
    bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh


def box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return inter / (aw * ah + bw * bh - inter)


def duplicates(a, b, duplicate_iou: float) -> bool:
    """Two tracks of one object in the same frames (SAM 3 removes duplicates within one text prompt, not across
    prompts): mean box IoU over their shared sample times >= duplicate_iou."""
    if not overlaps(a, b):
        return False
    boxes_b = {s["t"]: s["box"] for s in b["samples"]}
    shared = [box_iou(s["box"], boxes_b[s["t"]]) for s in a["samples"] if s["t"] in boxes_b]
    return bool(shared) and float(np.mean(shared)) >= duplicate_iou


def track_size(t) -> float:
    return float(np.median([max(s["box"][2:]) for s in t["samples"]]))


# --- wearers --------------------------------------------------------------------------------------------------------

def body_samples(bodies_layer: dict, faces_layer: dict, shot_index: dict[str, int]) -> dict[int, list]:
    """{shot index: [(t, box, person)]}: every body sample whose body is linked to a face track of a person (the
    bake-off's pose_rtmw.json face_track with faces.json's person, never its catch-all "Unknown": here, no person)."""
    person_of = {t["id"]: t.get("person") for t in faces_layer["tracks"]}
    out: dict[int, list] = {}
    for body in bodies_layer["tracks"]:
        person = person_of.get(body.get("face")) if body.get("face") else None
        if person is None:
            continue
        out.setdefault(shot_index[body["shot"]], []).extend((s["t"], s["box"], person) for s in body["samples"])
    return out


def different_people(faces_layer: dict, distance: float) -> set[frozenset]:
    """Person pairs that are certainly different people: ArcFace centroids at least `distance` apart (the faces
    layer's person_pairs, the bake-off's identity_similarity.json)."""
    return {frozenset((p["a"], p["b"])) for p in faces_layer.get("person_pairs", []) if p["distance"] >= distance}


def assign_wearers(tracks, bodies: dict[int, list], body_fps: float, classes: dict, wearer_share: float) -> None:
    """Each worn track's wearer: the person whose posed body holds the object's box centre in >= wearer_share of its
    samples (bodies within half a body sample of the object sample; the smallest containing body box wins where
    bodies overlap). Sets t["wearer"] (a person id or None) on every track."""
    step = 1 / body_fps
    for t in tracks:
        t["wearer"] = None
        if not classes[t["label"]]["worn"]:
            continue
        votes: dict[str, int] = {}
        for s in t["samples"]:
            x, y = s["box"][0] + s["box"][2] / 2, s["box"][1] + s["box"][3] / 2
            near = [(b[2] * b[3], person) for bt, b, person in bodies.get(t["shot"], [])
                    if abs(bt - s["t"]) <= step / 2 + 1e-6 and contains(b, x, y)]
            if near:
                person = min(near)[1]
                votes[person] = votes.get(person, 0) + 1
        if votes:
            person, n = max(votes.items(), key=lambda kv: kv[1])
            if n >= wearer_share * len(t["samples"]):
                t["wearer"] = person


# --- identities -----------------------------------------------------------------------------------------------------

def pair_score(a, b, different: set[frozenset], c: dict) -> float | None:
    """Adjusted cosine for linking a and b, or None when they may never be one object. c: the linking settings
    (cross_name_margin, same_wearer_bonus, min_link_px)."""
    if overlaps(a, b) or a["embedding"] is None or b["embedding"] is None:
        return None
    wa, wb = a.get("wearer"), b.get("wearer")
    if wa and wb and wa != wb and frozenset((wa, wb)) in different:
        return None   # worn by two different people
    same_wearer = bool(wa) and wa == wb
    if min(track_size(a), track_size(b)) < c["min_link_px"] and not same_wearer:
        return None   # too small to identify by appearance alone
    score = float(a["embedding"] @ b["embedding"])
    if a["label"] != b["label"]:
        score -= c["cross_name_margin"]
    if same_wearer:
        score += c["same_wearer_bonus"]
    return score


def link_identities(tracks, classes: dict, different: set[frozenset], c: dict) -> list[dict]:
    """Average-linkage clustering within each name family: a merge needs every cross pair linkable (pair_score not
    None) and an average adjusted score >= c["link_threshold"]. Clustering starts from groups of duplicate tracks,
    which are one object from the outset. Lookalike families never link across tracks. Each identity is named after
    its members' most common label ("necklace 2"), numbered per label in order of first appearance."""
    identities, numbering = [], {}
    for family in sorted({classes[t["label"]]["family"] for t in tracks}):
        members = [t for t in tracks if classes[t["label"]]["family"] == family]
        lookalike = all(classes[t["label"]]["kind"] == "lookalike" for t in members)
        n = len(members)
        score = [[pair_score(members[i], members[j], different, c) if i != j else None for j in range(n)] for i in range(n)]
        group = list(range(n))   # union-find over duplicate pairs

        def root(i):
            while group[i] != i:
                group[i] = group[group[i]]
                i = group[i]
            return i
        for i in range(n):
            for j in range(i + 1, n):
                if duplicates(members[i], members[j], c["duplicate_iou"]):
                    group[root(j)] = root(i)
        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(root(i), []).append(i)
        groups = list(clusters.values())
        while not lookalike:
            best = None
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    pairs = [score[a][b] for a in groups[i] for b in groups[j]]
                    if any(p is None for p in pairs):
                        continue
                    avg = float(np.mean(pairs))
                    if avg >= c["link_threshold"] and (best is None or avg > best[0]):
                        best = (avg, i, j)
            if best is None:
                break
            _, i, j = best
            groups[i] += groups.pop(j)
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
    """{identity id: crops/<slug>.jpg}, the path convert_bakeoff gave each identity's crop. Two ids with one slug
    ("t-shirt 1", "t shirt 1") would overwrite one crop there; here the later one gets a _2 suffix."""
    files, used = {}, set()
    for identity in identities:
        base = slug(identity["id"])
        name, k = base, 1
        while name in used:
            k += 1
            name = f"{base}_{k}"
        used.add(name)
        files[identity["id"]] = f"crops/{name}.jpg"
    return files


# --- the layer ------------------------------------------------------------------------------------------------------

def sample(s) -> dict:
    return {"t": s["t"], "box": s["box"], "poly": s.get("poly", []), "score": s.get("score")}


def layer_content(fps, tracks, rejected, identities, verdict, crops: dict[str, str | None]) -> dict:
    """The objects layer's fps, tracks, identities and rejected_tracks, as convert_bakeoff.objects_layer writes them
    from the bake-off's objects.json. crops: {identity id: its crop's path in the bundle (assets/...), or None}."""
    identity_of = {t: i["id"] for i in identities for t in i["tracks"]}
    return {
        "fps": fps,
        "tracks": [{"id": t["id"], "label": t["label"], "identity": f"objid:{identity_of[t['id']]}", "shot": f"shot:{t['shot']}",
                    "start": t["start"], "end": t["end"], "wearer": t["wearer"], "verdict": verdict.get(t["id"]),
                    "samples": [sample(s) for s in t["samples"]]} for t in tracks],
        "identities": [{"id": f"objid:{i['id']}", "label": i["label"], "family": i["family"], "worn_by": i["worn_by"],
                        "tracks": i["tracks"], "crop": crops.get(i["id"])} for i in identities],
        "rejected_tracks": [{"id": t["id"], "label": t["label"], "verdict": verdict[t["id"]], "shot": f"shot:{t['shot']}",
                             "start": t["start"], "end": t["end"], "samples": [sample(s) for s in t["samples"]]} for t in rejected],
    }
