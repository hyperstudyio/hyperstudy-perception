"""Gaze-model targets (gaze_lle.json, or gaze_page.json with --gaze) against the annotator's 2D py-feat gaze-ray
heuristic (annotator/build.py gaze_target).

The heuristic is imported, not re-implemented, and fed exactly what build.py feeds it: face_detail.json samples,
same-shot face tracks (face boxes) and same-shot object tracks. Its classes are a face track, an object, "camera"
(gaze within 12 degrees of the camera axis) or None; the gaze model's are a face track, an object, "out" or null. "camera"
and "out" are put side by side because looking into the camera is looking out of frame, though "out" also covers
looking at anything off-screen.

Prints the class confusion, agreement on the face chosen where both pick a face, and samples where they disagree
(for draw_gaze.py --times).

Usage: uv run python compare_heuristic.py ../runs/<name> [--gaze gaze_lle.json] [--list 20]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "annotator"))
from build import gaze_target  # noqa: E402  the annotator's heuristic, unchanged


def heuristic_targets(run):
    """face track id -> [heuristic target per sample], built as annotator/build.py main() does."""
    faces = json.loads((run / "faces.json").read_text())
    detail = json.loads((run / "face_detail.json").read_text())
    objects = json.loads((run / "objects.json").read_text()) if (run / "objects.json").exists() else {"tracks": []}
    tracks = [{"id": i, "shot": t["shot"], "t": [s["t"] for s in t["series"]], "box": [s["box"] for s in t["series"]]}
              for i, t in enumerate(faces["tracks"])]
    object_boxes = {}
    for o in objects["tracks"]:
        object_boxes.setdefault(o["shot"], []).append({"id": o["id"], "t": [x["t"] for x in o["samples"]], "box": [x["box"] for x in o["samples"]]})
    out = {}
    for track, samples in zip(tracks, detail["tracks"]):
        if len(samples) != len(track["t"]):
            raise SystemExit("face_detail.json does not match faces.json; re-run faces/face_detail.py")
        candidates = [t for t in tracks if t["shot"] == track["shot"]] + object_boxes.get(track["shot"], [])
        out[track["id"]] = [gaze_target(track, s, candidates) for s in samples]
    return out


def kind(target):
    if target is None:
        return "none"
    if isinstance(target, int):
        return "face"
    if target in ("camera", "out"):
        return "camera/out"
    return "object"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--gaze", default="gaze_lle.json")
    parser.add_argument("--list", type=int, default=20)
    args = parser.parse_args()

    gaze = json.loads((args.run_dir / args.gaze).read_text())
    heur = heuristic_targets(args.run_dir)
    pairs = []
    for tr in gaze["tracks"]:
        h = heur[tr["face_track"]]
        for s, ht in zip(tr["samples"], h):
            pairs.append((tr["face_track"], s["t"], ht, s["target"], s["inout"], s["target_score"]))

    n = len(pairs)
    confusion = Counter((kind(ht), kind(gt)) for _, _, ht, gt, _, _ in pairs)
    classes = ["face", "object", "none", "camera/out"]
    print(f"{n} samples. rows = heuristic, columns = {args.gaze}")
    print(f"{'':>12}" + "".join(f"{c:>12}" for c in classes) + f"{'total':>9}")
    for r in classes:
        row = [confusion[(r, c)] for c in classes]
        print(f"{r:>12}" + "".join(f"{v:>12}" for v in row) + f"{sum(row):>9}")
    print(f"{'total':>12}" + "".join(f"{sum(confusion[(r, c)] for r in classes):>12}" for c in classes))
    both_face = [(ht, gt) for _, _, ht, gt, _, _ in pairs if kind(ht) == "face" and kind(gt) == "face"]
    same = sum(ht == gt for ht, gt in both_face)
    exact = sum(ht == gt or (kind(ht) == kind(gt) == "camera/out") for _, _, ht, gt, _, _ in pairs)
    print(f"both pick a face: {len(both_face)}; same face {same} ({same / max(1, len(both_face)):.0%})")
    print(f"exact agreement (same face/object, both none, or camera~out): {exact}/{n} ({exact / n:.0%})")

    if args.list:
        print(f"\nfirst {args.list} disagreements where at least one side names a face (track, t, heuristic, gaze model, inout, score):")
        shown = 0
        for tid, t, ht, gt, inout, score in pairs:
            if ht != gt and (kind(ht) == "face" or kind(gt) == "face"):
                print(f"  face {tid} t={t}: heuristic {ht}  model {gt}  inout {inout} score {score}")
                shown += 1
                if shown >= args.list:
                    break


if __name__ == "__main__":
    main()
