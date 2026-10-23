"""Per-sample py-feat detail for the annotator: landmarks, action units, emotions, valence/arousal, gaze, mesh.

faces.json keeps only what the perception track needs (boxes, track-mean expression). The annotator
shows the full py-feat output for the face under the playhead, so this writes it per track sample,
in faces.json's track order:

  face_detail.json  68 landmarks, 20 AUs, 7 emotions, valence/arousal, head pose, gaze (small, loaded with the page)
  face_mesh.json    478-point 2D mesh per sample (large, fetched only when the mesh overlay is switched on)

Usage: uv run python face_detail.py ../runs/<name>
"""

import argparse
import json
from pathlib import Path

from faces import load_tracks

EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    fex, faces, tracks = load_tracks(args.run_dir)
    aus = sorted(c for c in fex.columns if c.startswith("AU"))
    lmk_cols = [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]
    mesh_cols = [f"mesh_x_{i}" for i in range(478)] + [f"mesh_y_{i}" for i in range(478)]
    missing = [c for c in lmk_cols + mesh_cols + EMOTIONS + ["valence", "arousal", "gaze_yaw", "gaze_pitch"] if c not in fex.columns]
    if missing:
        raise SystemExit(f"py-feat output lacks columns: {missing[:5]}")

    detail, mesh = [], []
    for track in tracks:
        samples, meshes = [], []
        for row in track["rows"]:
            r = fex.loc[row["index"]]
            samples.append({
                "t": row["t"],
                "lmk": [round(float(r[c]), 1) for c in lmk_cols],
                "au": [round(float(r[c]), 2) for c in aus],
                "emo": [round(float(r[c]), 3) for c in EMOTIONS],
                "va": [round(float(r["valence"]), 3), round(float(r["arousal"]), 3)],
                "head": [round(float(r["Yaw"]), 3), round(float(r["Pitch"]), 3), round(float(r["Roll"]), 3)],
                "gaze": [round(float(r["gaze_yaw"]), 3), round(float(r["gaze_pitch"]), 3)],
            })
            meshes.append([round(float(r[c])) for c in mesh_cols])
        detail.append(samples)
        mesh.append(meshes)

    conventions = ("radians; yaw + = subject's right (image left), pitch + = up; gaze is in camera coordinates, "
                   "(0, 0) = looking into the camera")
    (args.run_dir / "face_detail.json").write_text(json.dumps(
        {"au_names": aus, "emotion_names": EMOTIONS, "angles": conventions, "tracks": detail}, separators=(",", ":")))
    (args.run_dir / "face_mesh.json").write_text(json.dumps({"tracks": mesh}, separators=(",", ":")))
    print(f"{sum(len(t) for t in detail)} samples over {len(detail)} tracks; {len(aus)} AUs")


if __name__ == "__main__":
    main()
