"""Frame strips for judging grw.json by eye: the top-scoring events and a random sample of low-scoring words.

Each strip is one JPEG: STRIP_FRAMES frames evenly spaced over [start - MARGIN_S, end + MARGIN_S] (GRW reports
that gestures start before the word in 97.5% and end after it in 85.7% of cases), cropped to the speaker's body
box from pose_rtmw.json (widened by CROP_PAD), with the time of each frame and the words under it. Frames inside
the word/event span are outlined. A manifest (strips.json) lists every strip with its words and probability.

--threshold re-merges events from grw.json's words at another threshold (grw.events_from, transcript order
recovered from transcript_parakeet.json), for runs with fewer events than --top at grw.py's EVENT_THRESHOLD.

Usage: uv run python grw_review.py ../runs/<name> --out <dir> [--top 25] [--low 15] [--low-max 0.3] [--seed 0] [--threshold T]
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

import grw

STRIP_FRAMES = 8       # enough to see a stroke's start, peak and retraction over a ~1-2 s window
MARGIN_S = 0.5         # seconds shown before and after the span
CROP_PAD = 0.15        # body box widened by this fraction on each side so outstretched hands stay in view
TILE_HEIGHT = 300      # px per frame in the strip
TILES_PER_ROW = 4


def frame_at(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, bgr = cap.read()
    assert ok, f"no frame at {t:.2f} s"
    return bgr


def body_box(pose, face_track, t):
    for track in pose["tracks"]:
        if track["face_track"] == face_track and track["start"] - grw.EDGE_PAD_S <= t <= track["end"] + grw.EDGE_PAD_S:
            return grw.interp_samples(track, np.array([t * grw.FPS]), "box")[0]
    return None


def strip(cap, pose, item, path, width, height):
    start, end = item["start"], item["end"]
    times = np.linspace(start - MARGIN_S, end + MARGIN_S, STRIP_FRAMES)
    box = body_box(pose, item["face_track"], (start + end) / 2)
    x, y, w, h = box
    x0, x1 = int(max(0, x - CROP_PAD * w)), int(min(width, x + w * (1 + CROP_PAD)))
    y0, y1 = int(max(0, y - CROP_PAD * h)), int(min(height, y + h * (1 + CROP_PAD)))
    tiles = []
    for t in times:
        tile = frame_at(cap, t)[y0:y1, x0:x1]
        tile = cv2.resize(tile, (int(tile.shape[1] * TILE_HEIGHT / tile.shape[0]), TILE_HEIGHT))
        inside = start <= t <= end
        if inside:
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (0, 220, 255), 4)
        spoken = [wd["word"] for wd in item["word_list"] if wd["start"] - 0.05 <= t <= wd["end"] + 0.05]
        label = f"{t:.2f}s {' '.join(spoken)}"
        cv2.putText(tile, label, (6, TILE_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(tile, label, (6, TILE_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(tile)
    tile_w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, tile_w - t.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in tiles]
    rows = [np.hstack(tiles[i:i + TILES_PER_ROW]) for i in range(0, len(tiles), TILES_PER_ROW)]
    header = np.full((40, rows[0].shape[1], 3), 255, np.uint8)
    title = f"{item['id']}  p={item['score']:.3f}  face {item['face_track']}  \"{' '.join(item['words'])}\""
    cv2.putText(header, title[:110], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(str(path), np.vstack([header] + rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--low", type=int, default=15)
    parser.add_argument("--low-max", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    result = json.loads((args.run / "grw.json").read_text())
    pose = json.loads((args.run / "pose_rtmw.json").read_text())
    extract = json.loads((args.run / "extract.json").read_text())
    video = extract["video"]
    cap = cv2.VideoCapture(str(grw.BAKEOFF / video["source"]))
    args.out.mkdir(parents=True, exist_ok=True)

    words = result["words"]
    events = result["events"]
    if args.threshold is not None:
        transcript = json.loads((args.run / "transcript_parakeet.json").read_text())
        order = {(w["word"], w["start"], w["end"]): i for i, w in enumerate(x for s in transcript["segments"] for x in s["words"])}
        events = grw.events_from([{**w, "order": order[(w["word"], w["start"], w["end"])]} for w in words], args.threshold)
    items = []
    for i, e in enumerate(sorted(events, key=lambda e: -e["score"])[:args.top]):
        span = [w for w in words if w["face_track"] == e["face_track"] and e["start"] <= w["start"] and w["end"] <= e["end"]]
        items.append({**e, "id": f"event{i:02d}", "word_list": span})
    low = [w for w in words if w["prob"] < args.low_max]
    for i, w in enumerate(random.Random(args.seed).sample(low, min(args.low, len(low)))):
        items.append({"id": f"low{i:02d}", "kind": "low_word", "face_track": w["face_track"], "start": w["start"],
                      "end": w["end"], "words": [w["word"]], "score": w["prob"], "word_list": [w]})
    for item in items:
        item["file"] = str(args.out / f"{item['id']}.jpg")
        strip(cap, pose, item, item["file"], video["width"], video["height"])
    (args.out / "strips.json").write_text(json.dumps(
        [{k: v for k, v in it.items() if k != "word_list"} for it in items], indent=1))
    print(f"wrote {len(items)} strips to {args.out} ({sum(it['kind'] != 'low_word' for it in items)} events, "
          f"{sum(it['kind'] == 'low_word' for it in items)} low words of {len(low)} below {args.low_max})")


if __name__ == "__main__":
    main()
