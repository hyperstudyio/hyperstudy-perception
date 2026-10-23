"""Check that no face track is linked to two bodies at once in any pose_<model>.json.

Two body tracks linked to the same face track must not overlap in time ([start, end] intervals),
and no (face track, time) pair may be claimed by two samples. Exits non-zero if any file fails.

Usage: python check_links.py ../runs/<name> [../runs/<name> ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

DUPLICATE_IOU = 0.45  # reported only: same-frame body pairs overlapping this much (YOLOX's rtmlib NMS IoU)


def overlapping_links(pose):
    """[(face track, track a, track b)] for every pair of tracks on one face whose time spans overlap."""
    by_face = defaultdict(list)
    for track in pose["tracks"]:
        if track["face_track"] is not None:
            by_face[track["face_track"]].append(track)
    overlaps = []
    for face, tracks in by_face.items():
        tracks.sort(key=lambda t: t["start"])
        for i, a in enumerate(tracks):
            overlaps += [(face, a["id"], b["id"]) for b in tracks[i + 1:] if b["start"] <= a["end"]]
    return overlaps


def iou(a, b):
    w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return w * h / (a[2] * a[3] + b[2] * b[3] - w * h) if w * h else 0.0


def duplicate_boxes(pose, threshold):
    """Body pairs in one frame whose boxes overlap above the detector NMS threshold (reported, not a failure)."""
    by_time = defaultdict(list)
    for track in pose["tracks"]:
        for s in track["samples"]:
            by_time[s["t"]].append(s["box"])
    return sum(1 for boxes in by_time.values() for i, a in enumerate(boxes) for b in boxes[i + 1:] if iou(a, b) > threshold)


def main():
    failed = False
    for run in map(Path, sys.argv[1:]):
        for path in sorted(run.glob("pose_*.json")):
            pose = json.loads(path.read_text())
            overlaps = overlapping_links(pose)
            linked = [t for t in pose["tracks"] if t["face_track"] is not None]
            print(f"{run.name}/{path.name}: {len(pose['tracks'])} tracks, {len(linked)} linked, "
                  f"{len(overlaps)} overlapping links" + (f" e.g. {overlaps[:3]}" if overlaps else "")
                  + f"; {duplicate_boxes(pose, DUPLICATE_IOU)} same-frame body pairs with box IoU > {DUPLICATE_IOU}")
            failed |= bool(overlaps)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
