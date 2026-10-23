"""Two gaze-target outputs of one run (same schema: gaze_lle.json, gaze_page.json, ...) side by side, sample by sample.

Prints the target-class confusion (face / object / none / out), agreement on the face or object chosen, the in/out
split, the peak distance (fraction of the frame diagonal), the same numbers split by head size (head box side in
pixels, from faces.json at the file's head_box_scale), the event counts and how many events of A have a matching
event of B (same kind and people/object, overlapping in time), and the samples where they disagree.

Usage: uv run python compare_models.py ../runs/<name> [--a gaze_lle.json] [--b gaze_page.json] [--list 20] [--small 100]
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def kind(target):
    if target is None:
        return "none"
    if isinstance(target, int):
        return "face"
    if target == "out":
        return "out"
    return "object"


def event_key(e):
    return (e["kind"], e["from_face_track"], e["to_face_track"], e["object"], e["also_face_track"])


def matched(events_a, events_b):
    """Events of A with an event of B of the same kind and participants overlapping in time."""
    by_key = {}
    for e in events_b:
        by_key.setdefault(event_key(e), []).append(e)
    return sum(any(b["start"] <= a["end"] and a["start"] <= b["end"] for b in by_key.get(event_key(a), [])) for a in events_a)


def summary(name, pairs, diag):
    n = len(pairs)
    classes = ["face", "object", "none", "out"]
    confusion = Counter((kind(a["target"]), kind(b["target"])) for _, _, a, b in pairs)
    print(f"{name}: {n} samples. rows = A, columns = B")
    print(f"{'':>10}" + "".join(f"{c:>9}" for c in classes) + f"{'total':>8}")
    for r in classes:
        row = [confusion[(r, c)] for c in classes]
        print(f"{r:>10}" + "".join(f"{v:>9}" for v in row) + f"{sum(row):>8}")
    print(f"{'total':>10}" + "".join(f"{sum(confusion[(r, c)] for r in classes):>9}" for c in classes))
    same = sum(a["target"] == b["target"] for _, _, a, b in pairs)
    both_face = [(a, b) for _, _, a, b in pairs if kind(a["target"]) == kind(b["target"]) == "face"]
    same_face = sum(a["target"] == b["target"] for a, b in both_face)
    dists = [math.dist(a["peak"], b["peak"]) / diag for _, _, a, b in pairs]
    print(f"  same target (incl. both out / both none): {same}/{n} ({same / max(1, n):.0%}); both pick a face: "
          f"{len(both_face)}, same face {same_face} ({same_face / max(1, len(both_face)):.0%})")
    if n:
        print(f"  peak distance / diagonal: median {np.median(dists):.3f}, p90 {np.percentile(dists, 90):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--a", default="gaze_lle.json")
    parser.add_argument("--b", default="gaze_page.json")
    parser.add_argument("--list", type=int, default=20)
    parser.add_argument("--small", type=float, default=100, help="head boxes with a side below this many pixels are 'small'")
    args = parser.parse_args()

    run = args.run_dir
    meta = json.loads((run / "extract.json").read_text())["video"]
    faces = json.loads((run / "faces.json").read_text())
    a = json.loads((run / args.a).read_text())
    b = json.loads((run / args.b).read_text())
    if a["head_box_scale"] != b["head_box_scale"]:
        raise SystemExit(f"head_box_scale differs: {a['head_box_scale']} vs {b['head_box_scale']}")
    diag = math.hypot(meta["width"], meta["height"])
    tracks_b = {tr["face_track"]: tr["samples"] for tr in b["tracks"]}
    pairs = []  # (face track, head side px, sample A, sample B)
    for tr in a["tracks"]:
        series = faces["tracks"][tr["face_track"]]["series"]
        for j, (sa, sb) in enumerate(zip(tr["samples"], tracks_b[tr["face_track"]])):
            if sa["t"] != sb["t"]:
                raise SystemExit(f"face {tr['face_track']} sample {j}: t {sa['t']} vs {sb['t']}")
            side = max(series[j]["box"][2], series[j]["box"][3]) * a["head_box_scale"]
            pairs.append((tr["face_track"], side, sa, sb))

    print(f"A = {args.a} ({a['checkpoint']})\nB = {args.b} ({b['checkpoint']})")
    summary("all", pairs, diag)
    summary(f"head side < {args.small:g} px", [p for p in pairs if p[1] < args.small], diag)
    summary(f"head side >= {args.small:g} px", [p for p in pairs if p[1] >= args.small], diag)

    inout_a = np.array([p[2]["inout"] for p in pairs], dtype=float)
    inout_b = np.array([p[3]["inout"] for p in pairs], dtype=float)
    if len(pairs) > 1:
        print(f"inout: A out {int((inout_a < 0.5).sum())}, B out {int((inout_b < 0.5).sum())}; "
              f"Pearson r {np.corrcoef(inout_a, inout_b)[0, 1]:.2f}")

    for label, x, y in (("A", a, b), ("B", b, a)):
        counts = Counter(e["kind"] for e in x["events"])
        print(f"events {label}: {dict(sorted(counts.items()))}; with a matching event in the other: "
              + ", ".join(f"{k} {matched([e for e in x['events'] if e['kind'] == k], y['events'])}/{counts[k]}" for k in sorted(counts)))

    if args.list:
        print(f"\nfirst {args.list} samples where the targets differ and one names a face (track, t, side px, A, B):")
        shown = 0
        for tid, side, sa, sb in pairs:
            if sa["target"] != sb["target"] and "face" in (kind(sa["target"]), kind(sb["target"])):
                print(f"  face {tid} t={sa['t']} side {side:.0f}: A {sa['target']} (inout {sa['inout']}, score {sa['target_score']})"
                      f"  B {sb['target']} (inout {sb['inout']}, score {sb['target_score']})")
                shown += 1
                if shown >= args.list:
                    break


if __name__ == "__main__":
    main()
