"""Draw hands from several hands_<model>.json files on the same frames, for side-by-side visual comparison.

One image per time: a row per model, each row = the full frame at half size plus zoom tiles of the same hand
regions for every model (regions = all models' hand boxes at that time, merged by overlap, largest first).
Colours: subject's left hand blue, right hand red, unknown side yellow; box white for box_source wrist, magenta
for detector. Keypoints below the model's draw threshold are not drawn.

Times are snapped to the nearest sampled frame.

Usage: uv run python draw_hands.py ../runs/<name> --models rtmw,rtmpose_hand,mediapipe --times 12.3,45.6 --out DIR
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from hand_events import KPT_SCORE_THRESHOLD
from hands import BAKEOFF, iou

SIDE_COLOUR = {"left": (255, 128, 0), "right": (0, 0, 255), None: (0, 220, 255)}
SOURCE_COLOUR = {"wrist": (255, 255, 255), "detector": (255, 0, 255)}
TILES = 4         # zoom tiles per row
TILE_PX = 270     # tile side, half the 540 px row height
ZOOM = 2.2        # tile = this x the region's side, so the arm and face around the hand are visible


def frame_at(capture, fps, t):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = capture.read()
    assert ok, t
    return frame


def hands_at(data, t):
    out = []
    for hand in data["hands"]:
        for s in hand["samples"]:
            if abs(s["t"] - t) < 1e-3:
                out.append((hand, s))
    return out


def draw(image, data, hands, threshold, scale=1.0, offset=(0, 0)):
    thick = max(1, int(round(2 * scale)))
    for hand, s in hands:
        to = lambda x, y: (int(round((x - offset[0]) * scale)), int(round((y - offset[1]) * scale)))
        x, y, w, h = s["box"]
        cv2.rectangle(image, to(x, y), to(x + w, y + h), SOURCE_COLOUR[hand["box_source"]], 1)
        colour = SIDE_COLOUR[hand["side"]]
        k = s["kpts"]
        for a, b in data["skeleton"]:
            if k[a][2] >= threshold and k[b][2] >= threshold:
                cv2.line(image, to(*k[a][:2]), to(*k[b][:2]), colour, thick)
        for i, (px, py, c) in enumerate(k):
            if c >= threshold:
                cv2.circle(image, to(px, py), thick + (2 if i in (4, 8) else 1), (0, 255, 0) if i == 8 else colour, -1)
        cv2.putText(image, f"{hand['id']} {hand['side'] or '?'}", to(x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * max(scale, 1), colour, 1)


def regions(all_hands):
    boxes = sorted((s["box"] for hands in all_hands for _, s in hands), key=lambda b: -b[2])
    merged = []
    for b in boxes:
        if all(iou(b, m) < 0.2 for m in merged):
            merged.append(b)
    return merged[:TILES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--models", default="rtmw,rtmpose_hand,mediapipe")
    parser.add_argument("--times", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    models = [m for m in args.models.split(",") if (args.run_dir / f"hands_{m}.json").exists()]
    data = {m: json.loads((args.run_dir / f"hands_{m}.json").read_text()) for m in models}
    extract = json.loads((args.run_dir / "extract.json").read_text())
    video = Path(extract["video"]["source"])
    capture = cv2.VideoCapture(str(video if video.is_absolute() else BAKEOFF / video))
    fps = extract["video"]["fps"]
    step = max(1, round(fps / data[models[0]]["fps"]))
    for t in (float(x) for x in args.times.split(",")):
        t = round(round(t * fps / step) * step / fps, 3)  # snap to the sampled frame (hands.py's grid)
        frame = frame_at(capture, fps, t)
        found = {m: hands_at(data[m], t) for m in models}
        tiles = regions(found.values())
        rows = []
        for m in models:
            full = cv2.resize(frame, (960, 540))
            draw(full, data[m], found[m], KPT_SCORE_THRESHOLD[m], scale=0.5)
            cv2.putText(full, f"{m} t={t:.3f} hands={len(found[m])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            tile_imgs = []
            for x, y, w, h in tiles:
                side = w * ZOOM
                x0, y0 = x + w / 2 - side / 2, y + h / 2 - side / 2
                pad = int(side) + 2
                padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
                crop = padded[int(y0) + pad:int(y0 + side) + pad, int(x0) + pad:int(x0 + side) + pad].copy()
                scale = TILE_PX / crop.shape[0]
                crop = cv2.resize(crop, (TILE_PX, TILE_PX))
                draw(crop, data[m], found[m], KPT_SCORE_THRESHOLD[m], scale=scale, offset=(int(x0), int(y0)))
                tile_imgs.append(crop)
            while len(tile_imgs) < TILES:
                tile_imgs.append(np.zeros((TILE_PX, TILE_PX, 3), np.uint8))
            grid = np.vstack([np.hstack(tile_imgs[:2]), np.hstack(tile_imgs[2:4])])
            rows.append(np.hstack([full, grid]))
        path = args.out / f"{args.run_dir.name}_t{t:07.3f}.jpg"
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(path, {m: len(found[m]) for m in models})


if __name__ == "__main__":
    main()
