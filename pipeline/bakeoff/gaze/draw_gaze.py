"""Draw gaze_lle.json (or any gaze output with --gaze) on frames for checking by eye; --vs puts a second one beside it.

Per head: its head box, a line from the box centre to the heatmap peak (solid; dotted when the in/out score says
"out"), and a label "face -> target (inout)". The heatmaps of all heads in the frame (max over heads) are blended in
when --heatmaps (from gaze_lle.py --heatmaps) is given. With --heuristic, the annotator's 2D py-feat target is drawn
as a thin white line to that face's box centre and added to the label as "H:<target>".

Usage: uv run python draw_gaze.py ../runs/<name> --out DIR --times 14.147 52.586 [--heatmaps h.npz] [--heuristic]
       [--vs gaze_page.json --vs-heatmaps h2.npz] [--crop x1 y1 x2 y2]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from gaze_stage import BAKEOFF, frames_at, head_box, interpolate_box

COLOURS = [(255, 99, 71), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (210, 245, 60), (250, 190, 212)]  # RGB; drawn as BGR below
OUT_WIDTH = 1600


def annotate(frame, t, gaze, faces, heat, heur, objects, width, height):
    """frame with one gaze output's heads, gaze lines and labels at sample time t (heatmaps blended in when given)."""
    scale = gaze["head_box_scale"]
    heads = [(tr["face_track"], j, s) for tr in gaze["tracks"] for j, s in enumerate(tr["samples"]) if abs(s["t"] - t) < 1e-6]
    if heat is not None and heads:
        combined = np.max([heat[f"{tid}_{j}"].astype(np.float32) for tid, j, _ in heads], axis=0)
        overlay = cv2.applyColorMap((cv2.resize(combined, (width, height)) * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
    for k, (tid, j, s) in enumerate(heads):
        colour = COLOURS[k % len(COLOURS)][::-1]
        box = faces["tracks"][tid]["series"][j]["box"]
        x1, y1, x2, y2 = (int(v) for v in head_box(box, width, height, scale))
        centre = ((x1 + x2) // 2, (y1 + y2) // 2)
        peak = tuple(int(v) for v in s["peak"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        if s["target"] == "out":
            n = 12
            for a in range(0, n, 2):
                p = (centre[0] + (peak[0] - centre[0]) * a // n, centre[1] + (peak[1] - centre[1]) * a // n)
                q = (centre[0] + (peak[0] - centre[0]) * (a + 1) // n, centre[1] + (peak[1] - centre[1]) * (a + 1) // n)
                cv2.line(frame, p, q, colour, 3)
        else:
            cv2.line(frame, centre, peak, colour, 3)
        cv2.circle(frame, peak, 10, colour, -1)
        if isinstance(s["target"], str) and s["target"].startswith("obj:"):
            sample = min(objects[s["target"]]["samples"], key=lambda o: abs(o["t"] - t))
            cv2.polylines(frame, [np.array(sample["poly"], np.int32)], True, colour, 2)
        label = f"{tid}->{s['target']} ({s['inout']})"
        if heur is not None:
            h = heur[tid][j]
            label += f" H:{h}"
            if isinstance(h, int):
                hb = interpolate_box(faces["tracks"][h]["series"], t)
                cv2.line(frame, centre, (int(hb[0] + hb[2] / 2), int(hb[1] + hb[3] / 2)), (255, 255, 255), 1)
        cv2.putText(frame, label, (x1, max(30, y1 - 10 - 34 * (k % 2))), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(frame, label, (x1, max(30, y1 - 10 - 34 * (k % 2))), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    cv2.putText(frame, f"t={t:.3f}", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--gaze", default="gaze_lle.json")
    parser.add_argument("--heatmaps", type=Path)
    parser.add_argument("--vs", help="a second gaze output (e.g. gaze_page.json), drawn on the same frame to the right")
    parser.add_argument("--vs-heatmaps", type=Path)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="draw only this region (pixels)")
    parser.add_argument("--heuristic", action="store_true")
    args = parser.parse_args()

    run = args.run_dir
    extract = json.loads((run / "extract.json").read_text())
    faces = json.loads((run / "faces.json").read_text())
    panels = [(args.gaze, json.loads((run / args.gaze).read_text()), np.load(args.heatmaps) if args.heatmaps else None)]
    if args.vs:
        panels.append((args.vs, json.loads((run / args.vs).read_text()), np.load(args.vs_heatmaps) if args.vs_heatmaps else None))
    heur = None
    if args.heuristic:
        from compare_heuristic import heuristic_targets
        heur = heuristic_targets(run)
    objects = json.loads((run / "objects.json").read_text()) if (run / "objects.json").exists() else {"tracks": []}
    objects = {o["id"]: o for o in objects["tracks"]}
    fps, width, height = extract["video"]["fps"], extract["video"]["width"], extract["video"]["height"]
    video = BAKEOFF / extract["video"]["source"]
    args.out.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = args.crop or (0, 0, width, height)
    panel_width = OUT_WIDTH  # side by side, the image is len(panels) x OUT_WIDTH wide

    wanted = {round(t * fps): t for t in args.times}
    for index, frame in frames_at(video, wanted):
        t = wanted[index]
        drawn = []
        for name, gaze, heat in panels:
            image = annotate(frame.copy(), t, gaze, faces, heat, heur, objects, width, height)[y1:y2, x1:x2]
            image = cv2.resize(image, (panel_width, round(panel_width * (y2 - y1) / (x2 - x1))))
            if len(panels) > 1:
                cv2.putText(image, name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6)
                cv2.putText(image, name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
            drawn.append(image)
        suffix = "" if not args.crop else "_crop"
        path = args.out / f"{run.name}_{t:.3f}{suffix}.jpg"
        cv2.imwrite(str(path), np.hstack(drawn))
        print(path)


if __name__ == "__main__":
    main()
