"""Score face identity clustering and speaker identification against the owner's cleaned identities.

Ground truth (runs/<name>/identity_truth.json, resolved by the annotator from the owner's corrections):
  - every face track's real person (after merges and reassignments), or "not a person";
  - speakers: the owner's annotator choices (set_by_user) where made, else the blind hand labels from the
    labelling page (speaker_labels.json, predictions hidden), with a face label mapped to that track's real
    person. Unsure / several / not-speech utterances are left out. Where both sources label an utterance and
    disagree, the annotator choice wins and the disagreement is counted.

1. Face identity. Over tracks of real people, B-cubed precision and recall of the machine clusters (a track left
   "Unknown" counts as its own cluster), the number of clusters a reviewer has to merge, impure clusters, and how
   many not-a-person tracks were put into a person. Variants, all from the cached detections (load_tracks):
   the clusters as run (faces.json), cluster_tracks at other thresholds, and a fixed second pass that stops two
   tracks on screen together from attaching to one person (the known cause of clusters holding two people).
   Also: cluster-centroid ArcFace distance for same vs different real people (objects.py uses 0.8 as "certainly
   different"), and how many not-a-person tracks never link to a posed body (the false-face filter idea).
2. Speaker identification, scored by real person rather than track: LR-ASD and TalkNet (top face in the shot,
   mean score > 0, else "off screen"), the mouth-movement baseline, the pipeline's pyannote voice -> face links
   (speaker_links_lrasd.json, speaker_links.json), an oracle voice -> person mapping (each voice's majority real
   person on these labels: the best any voice linker could do), and LR-ASD falling back to its voice link.

Usage (from faces/): uv run python evaluate_identity.py ../runs/<name>
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from faces import ATTACH_COSINE, SAME_PERSON_COSINE, cluster_tracks, is_anchor, load_tracks, track_embedding

OFF = "off screen"


def real_name(track):
    """The owner's name for a truth track; None for not-a-person and for tracks left Unknown."""
    return None if track["not_person"] or track["name"] in ("Unknown", "Not a person") else track["name"]


def bcubed(pred, truth):
    """B-cubed precision/recall over items with a truth label; pred None = singleton."""
    items = [i for i in truth if truth[i] is not None]
    cluster = defaultdict(set)
    for i in items:
        cluster[pred[i] if pred[i] is not None else ("solo", i)].add(i)
    p = np.mean([sum(truth[j] == truth[i] for j in cluster[pred[i] if pred[i] is not None else ("solo", i)])
                 / len(cluster[pred[i] if pred[i] is not None else ("solo", i)]) for i in items])
    same = defaultdict(set)
    for i in items:
        same[truth[i]].add(i)
    r = np.mean([sum((pred[j] is not None and pred[j] == pred[i]) or j == i for j in same[truth[i]]) / len(same[truth[i]])
                 for i in items])
    return p, r


def cluster_tracks_fixed(tracks, threshold=SAME_PERSON_COSINE):
    """cluster_tracks with one change: a track attached in the second pass occupies its person's times, so a later
    track on screen at the same moment cannot attach to that person too. Tracks attach nearest-first."""
    from sklearn.cluster import AgglomerativeClustering

    emb = np.stack([track_embedding(t) for t in tracks])
    times = [{r["t"] for r in t["rows"]} for t in tracks]
    anchors = [i for i, t in enumerate(tracks) if is_anchor(t)]
    labels = [-1] * len(tracks)
    dist = np.clip(1 - emb[anchors] @ emb[anchors].T, 0, 2)
    for a in range(len(anchors)):
        for b in range(a + 1, len(anchors)):
            if times[anchors[a]] & times[anchors[b]]:
                dist[a, b] = dist[b, a] = 2.0
    anchor_labels = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                            distance_threshold=threshold).fit_predict(dist)
    for i, label in zip(anchors, anchor_labels):
        labels[i] = int(label)
    people = sorted(set(anchor_labels))
    centroid = {p: np.mean([emb[i] for i in anchors if labels[i] == p], axis=0) for p in people}
    centroid = {p: c / np.linalg.norm(c) for p, c in centroid.items()}
    occupied = {p: set().union(*(times[i] for i in anchors if labels[i] == p)) for p in people}
    rest = [(min(1 - float(emb[i] @ centroid[p]) for p in people), i) for i in range(len(tracks)) if labels[i] == -1]
    for _, i in sorted(rest):
        options = sorted((1 - float(emb[i] @ centroid[p]), p) for p in people if not (times[i] & occupied[p]))
        if options and options[0][0] <= ATTACH_COSINE:
            labels[i] = options[0][1]
            occupied[options[0][1]] |= times[i]
    return labels


def identity_report(run, truth_tracks):
    _, faces, tracks = load_tracks(run)
    truth = {t["track"]: real_name(t) for t in truth_tracks}        # None = not a person
    unresolved = {t["track"] for t in truth_tracks if not t["not_person"] and real_name(t) is None}
    for i in unresolved:                                             # left Unknown by the owner: not scored
        del truth[i]
    n_real, n_not = sum(v is not None for v in truth.values()), sum(v is None for v in truth.values())
    print(f"\n== 1. Face identity: {len(truth)} tracks scored: {n_real} of {len(set(v for v in truth.values() if v))} real "
          f"people, {n_not} not a person ({len(unresolved)} left Unknown by the owner, not scored)")

    def score(name, labels):
        pred = {i: (None if labels[i] in (-1, "Unknown") else labels[i]) for i in range(len(labels))}
        p, r = bcubed(pred, truth)
        clusters = {pred[i] for i in truth if truth[i] is not None and pred[i] is not None}
        impure = sum(1 for c in clusters if len({truth[i] for i in truth if pred[i] == c} - {None}) > 1)
        unknown_real = sum(1 for i in truth if truth[i] is not None and pred[i] is None)
        # a not-a-person track in a cluster whose other members are mostly real people pollutes that person;
        # one in a cluster of its own kind (posters, hands) is a junk "person" a reviewer has to mark once
        real_share = {c: np.mean([truth[i] is not None for i in truth if pred[i] == c]) for c in set(pred.values()) if c is not None}
        junk_in = sum(1 for i in truth if truth[i] is None and pred[i] is not None and real_share[pred[i]] > 0.5)
        junk_own = sum(1 for i in truth if truth[i] is None and pred[i] is not None and real_share[pred[i]] <= 0.5)
        print(f"  {name:34s} precision {p:.3f} recall {r:.3f} | person clusters {len(clusters):2d} (impure {impure}) | "
              f"real left Unknown {unknown_real:2d} | not-a-person: inside a real person {junk_in}, own clusters {junk_own}, Unknown {n_not - junk_in - junk_own}")

    score("as run (faces.json)", [t["person"] for t in faces["tracks"]])
    for th in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        score(f"cluster_tracks {th}", cluster_tracks(tracks, th))
    for th in (0.65, 0.7, 0.75):
        score(f"fixed second pass {th}", cluster_tracks_fixed(tracks, th))

    # centroid distances between machine clusters, by whether they are the same real person
    majority = {}
    for p in {t["person"] for t in faces["tracks"]}:
        names = Counter(truth.get(i) for i, t in enumerate(faces["tracks"]) if t["person"] == p and truth.get(i))
        majority[p] = names.most_common(1)[0][0] if names else None
    sim = json.loads((run / "identity_similarity.json").read_text())["pairs"]
    same = sorted(round(x["distance"], 2) for x in sim if majority.get(x["a"]) and majority.get(x["a"]) == majority.get(x["b"]))
    diff = sorted(round(x["distance"], 2) for x in sim if majority.get(x["a"]) and majority.get(x["b"])
                  and majority[x["a"]] != majority[x["b"]])
    print(f"  centroid distance, same real person: {same}")
    print(f"  centroid distance, different people: min {diff[0] if diff else None}, "
          f"{sum(d < 0.8 for d in diff)} of {len(diff)} below 0.8")

    pose = json.loads((run / "pose_rtmw.json").read_text())
    linked = {b["face_track"] for b in pose["tracks"] if b.get("face_track") is not None}
    print(f"  never linked to a posed body: not-a-person {sum(1 for i in truth if truth[i] is None and i not in linked)}/{n_not}, "
          f"real people {sum(1 for i in truth if truth[i] is not None and i not in linked)}/{n_real}")
    return majority


def speaker_truth(run, identity):
    by_track = {t["track"]: real_name(t) for t in identity["tracks"]}
    blind = {u["id"]: u["label"] for u in json.loads((run / "speaker_labels.json").read_text())["units"] if u["label"]}
    truth, source, visible, conflicts = {}, {}, {}, 0

    def resolve(label):
        """(real person or OFF, speaker visible on screen) or (None, None) for unsure / several / not speech."""
        if label["kind"] == "face":
            return by_track.get(label["track"]) or "not a person", True
        if label["kind"] == "person":
            return label.get("name") or label.get("person"), False
        if label["kind"] in ("offscreen", "unboxed"):
            return OFF, False
        return None, None

    for u in identity["utterances"]:
        user = {"kind": u["speaker"]["kind"], "track": u["speaker"].get("track"), "name": u["name"]} if u["set_by_user"] else None
        a, va = resolve(user) if user else (None, None)
        b, vb = resolve(blind[u["id"]]) if u["id"] in blind else (None, None)
        if a is not None and b is not None and a != b:
            conflicts += 1
        if a is not None:
            truth[u["id"]], source[u["id"]], visible[u["id"]] = a, "annotator", va
        elif b is not None:
            truth[u["id"]], source[u["id"]], visible[u["id"]] = b, "blind", vb
    return truth, source, visible, conflicts


def mean_score(track, start, end, fps):
    i0, i1 = int((start - track["start"]) * fps), int((end - track["start"]) * fps) + 1
    w = track["scores"][max(0, i0):max(0, i1)]
    return sum(w) / len(w) if w else None


def speaker_report(run, identity, majority):
    truth, source, visible, conflicts = speaker_truth(run, identity)
    by_track = {t["track"]: real_name(t) for t in identity["tracks"]}
    utts = {u["id"]: u for u in identity["utterances"]}
    shot = {u["id"]: u["shot"] for u in json.loads((run / "speaker_labels.json").read_text())["units"]}
    kinds = Counter("known person" if v != OFF else OFF for v in truth.values())
    print(f"\n== 2. Speaker identification: {len(truth)} utterances with truth "
          f"({Counter(source.values())}; {conflicts} where the two sources disagree, annotator kept); {dict(kinds)}")

    faces = json.loads((run / "faces.json").read_text())
    transcript = json.loads((run / "transcript_parakeet.json").read_text())
    words = [w for s in transcript["segments"] for w in s["words"]]

    def voice_of(u):
        c = Counter(w.get("speaker") for w in words if u["start"] <= w["start"] <= u["end"])
        return c.most_common(1)[0][0] if c else None

    def asd_pick(model):
        res = json.loads((run / f"asd_{model}.json").read_text())

        def pick(u):
            scored = [(mean_score(t, u["start"], u["end"], res["fps"]), t["track"]) for t in res["tracks"] if t["shot"] == shot[u["id"]]]
            scored = [x for x in scored if x[0] is not None]
            if not scored or max(scored)[0] <= 0:
                return OFF
            return by_track.get(max(scored)[1]) or "not a person"
        return pick

    def mouth_pick(u):
        scored = []
        for i, t in enumerate(faces["tracks"]):
            if t["shot"] != shot[u["id"]]:
                continue
            w = [s["mouth"] for s in t["series"] if u["start"] <= s["t"] <= u["end"]]
            if w:
                scored.append((sum(w) / len(w), i))
        return (by_track.get(max(scored)[1]) or "not a person") if scored else OFF

    def link_pick(file, min_share=0.0):
        """Voice -> linked person; with min_share, only links whose linked face got at least that share of the
        voice's visible words (speaker_links_lrasd.json word_share), else off screen."""
        links = json.loads((run / file).read_text())["links"]

        def pick(u):
            link = links.get(voice_of(u), {})
            return (majority.get(link.get("person")) or OFF) if link.get("word_share", 1.0) >= min_share else OFF
        return pick

    oracle_votes = defaultdict(Counter)
    for i, name in truth.items():
        oracle_votes[voice_of(utts[i])][name] += 1
    oracle = {v: c.most_common(1)[0][0] for v, c in oracle_votes.items()}
    lr, lr_link = asd_pick("lrasd"), link_pick("speaker_links_lrasd.json")
    lr_link30 = link_pick("speaker_links_lrasd.json", 0.3)
    print("  voice links (LR-ASD votes):", {v: (majority.get(l["person"]), l["word_share"]) for v, l in
                                            json.loads((run / "speaker_links_lrasd.json").read_text())["links"].items()})

    methods = {"LR-ASD": lr, "TalkNet": asd_pick("talknet"), "mouth movement (AU25/26)": mouth_pick,
               "voice link via LR-ASD votes": lr_link, "voice link via mouth AUs": link_pick("speaker_links.json"),
               "LR-ASD, else its voice link": lambda u: (lr(u) if lr(u) != OFF else lr_link(u)),
               "LR-ASD, else voice link if share >= 0.3": lambda u: (lr(u) if lr(u) != OFF else lr_link30(u)),
               "oracle voice -> person (upper bound)": lambda u: oracle.get(voice_of(u), OFF)}
    groups = {"visible speaker": [i for i, v in truth.items() if v != OFF and visible[i]],
              "known person off camera": [i for i, v in truth.items() if v != OFF and not visible[i]],
              "off-screen voice": [i for i, v in truth.items() if v == OFF]}
    print(f"  {'method':38s}" + "".join(f"{g} (n={len(ids)})".rjust(30) for g, ids in groups.items()) + "   all")
    for name, pick in methods.items():
        cells, right_all = [], 0
        for ids in groups.values():
            right = sum(pick(utts[i]) == truth[i] for i in ids)
            right_all += right
            cells.append(f"{right}/{len(ids)} ({right / max(1, len(ids)):.0%})".rjust(30))
        print(f"  {name:38s}" + "".join(cells) + f"   {right_all / len(truth):.0%}")
    print("  pyannote voices vs real people (utterances):")
    for v, c in sorted(oracle_votes.items(), key=lambda kv: str(kv[0])):
        total = sum(c.values())
        print(f"    {v}: {total} utterances, purity {c.most_common(1)[0][1] / total:.0%}, people {dict(c)}")


def main():
    run = Path(sys.argv[1])
    identity = json.loads((run / "identity_truth.json").read_text())
    majority = identity_report(run, identity["tracks"])
    speaker_report(run, identity, majority)


if __name__ == "__main__":
    main()
