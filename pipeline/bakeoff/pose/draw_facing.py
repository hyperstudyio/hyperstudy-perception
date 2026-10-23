"""Draw torso facing arrows (and active facing events) on video frames, for checking pose3d.py / facing.py by eye.

Per linked body: shoulders and hips, an arrow from the torso centre along the image-plane part of the facing
vector (length ARROW_PX x its image-plane fraction, so a body facing the camera gets a short arrow), and
"F<face track> yaw/pitch". A facing event active at the frame is a line from A's torso centre to B's, with
"A->B" (double line when mutual). Unlinked bodies are drawn grey.

Usage: uv run python draw_facing.py ../runs/<name> --pose pose_<model>.json [--facing facing_<model>.json]
       --times 6.0 85.4 ... --out <dir>
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from facing import facing_vector, torso_centre

HERE = Path(__file__).parent
ARROW_PX = 220
COLOURS = [(0, 200, 255), (255, 120, 0), (0, 255, 0), (255, 0, 255), (0, 0, 255), (255, 255, 0), (128, 128, 255), (0, 128, 255)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--facing")
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pose = json.loads((args.run_dir / args.pose).read_text())
    events = json.loads((args.run_dir / args.facing).read_text())["events"] if args.facing else []
    extract = json.loads((args.run_dir / "extract.json").read_text())
    video = Path(extract["video"]["source"])
    video = video if video.is_absolute() else HERE.parent / video
    names, threshold = pose["keypoints"], pose["score_threshold"]
    joints = [names.index(n) for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    capture = cv2.VideoCapture(str(video))
    fps = extract["video"]["fps"]
    args.out.mkdir(parents=True, exist_ok=True)
    for want in args.times:
        samples = [(tr, s) for tr in pose["tracks"] for s in tr["samples"] if abs(s["t"] - want) < 0.07]
        if not samples:
            print(f"t={want}: no pose samples")
            continue
        t = min({s["t"] for _, s in samples}, key=lambda x: abs(x - want))
        samples = [(tr, s) for tr, s in samples if s["t"] == t]
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
        ok, frame = capture.read()
        assert ok, f"cannot read frame at {t}"
        centres = {}
        for tr, s in samples:
            face = tr["face_track"]
            colour = COLOURS[face % len(COLOURS)] if face is not None else (160, 160, 160)
            k = np.array(s["kpts"])
            for a, b in ((0, 1), (0, 2), (1, 3), (2, 3)):
                ja, jb = joints[a], joints[b]
                if k[ja, 2] >= threshold and k[jb, 2] >= threshold:
                    cv2.line(frame, tuple(int(v) for v in k[ja, :2]), tuple(int(v) for v in k[jb, :2]), colour, 3)
            for j, c in ((joints[0], (255, 0, 0)), (joints[1], (0, 255, 255))):  # left shoulder blue, right yellow
                if k[j, 2] >= threshold:
                    cv2.circle(frame, tuple(int(v) for v in k[j, :2]), 8, c, -1)
            centre = torso_centre(s, joints, threshold)
            if face is not None:
                centres[face] = centre
            label = f"F{face}" if face is not None else "-"
            if s["facing"] is not None:
                f = facing_vector(*s["facing"])
                tip = centre + ARROW_PX * f[:2]
                cv2.arrowedLine(frame, tuple(int(v) for v in centre), tuple(int(v) for v in tip), colour, 6, tipLength=0.25)
                pitch = "-" if s["facing"][1] is None else f"{s['facing'][1]:.0f}"
                label += f" {s['facing'][0]:.0f}/{pitch} {s['facing_from'][0]}"
            org = (int(centre[0]) - 60, int(centre[1]) + 40)
            cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
        active = [e for e in events if e["start"] - 1e-6 <= t <= e["end"] + 1e-6]
        for i, e in enumerate(active):
            a, b = centres.get(e["from_face_track"]), centres.get(e["to_face_track"])
            if a is None or b is None:
                continue
            offset = np.array([0, -30 - 12 * i])
            cv2.line(frame, tuple(int(v) for v in a + offset), tuple(int(v) for v in b + offset), (255, 255, 255), 4 if e["mutual"] else 2)
            cv2.circle(frame, tuple(int(v) for v in b + offset), 10, (255, 255, 255), -1)
        text = f"t={t:.3f} {args.pose}  " + "  ".join(f"{e['from_face_track']}->{e['to_face_track']}{'*' if e['mutual'] else ''}" for e in active)
        cv2.putText(frame, text, (20, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(frame, text, (20, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        path = args.out / f"{args.run_dir.name}_{Path(args.pose).stem}_{t:07.3f}.jpg"
        cv2.imwrite(str(path), frame)
        print(path)


if __name__ == "__main__":
    main()
