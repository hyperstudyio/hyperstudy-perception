"""Per-shot social signals from py-feat output: expression and gaze.

For every known person on screen in a shot:
  - expression: the emotion that rises most above that person's own baseline
    (their mean over the whole clip), reported only when the rise is clear;
  - gaze: camera-relative horizontal direction. py-feat reports radians;
    positive gaze_yaw = looking toward image-left (checked by eye on two
    frames of OOO_GenZ_mellennials_clip1, not on a labelled set);
  - target: the other face on screen in that direction, nearest horizontally,
    or "camera" when gaze is close to straight ahead; mutual gaze is flagged.

Usage: uv run python social.py runs/<name>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EMOTION_RISE = 0.15     # shot mean must exceed the person's baseline by this much
EMOTION_FLOOR = 0.4     # and be at least this probability in absolute terms
CAMERA_YAW = 0.25       # |gaze_yaw| below this (radians, ~14 deg) counts as toward the camera
LOOK_YAW = 0.35         # |gaze_yaw| above this counts as looking sideways at someone


def main():
    run_dir = Path(sys.argv[1])
    faces = json.loads((run_dir / "faces.json").read_text())
    extract = json.loads((run_dir / "extract.json").read_text())
    tracks = [t for t in faces["tracks"] if t["person"] != "Unknown"]

    baseline = defaultdict(lambda: defaultdict(list))
    for t in tracks:
        for k, v in t["emotions"].items():
            baseline[t["person"]][k].extend([v] * t["samples"])
    baseline = {p: {k: float(np.mean(v)) for k, v in emo.items()} for p, emo in baseline.items()}

    shots_out = []
    for shot in extract["shots"]:
        here = [t for t in tracks if t["shot"] == shot["index"]]
        people = {}
        for t in here:
            p = people.setdefault(t["person"], {"x": [], "gaze_yaw": [], "emotions": defaultdict(list)})
            for s in t["series"]:
                p["x"].append(s["box"][0] + s["box"][2] / 2)
                p["gaze_yaw"].append(s["gaze_yaw"])
            for k, v in t["emotions"].items():
                p["emotions"][k].extend([v] * t["samples"])

        summary = {}
        for name, p in people.items():
            x, yaw = float(np.mean(p["x"])), float(np.median(p["gaze_yaw"]))
            emotions = {k: float(np.mean(v)) for k, v in p["emotions"].items()}
            rise = {k: emotions[k] - baseline[name][k] for k in emotions}
            top = max(rise, key=rise.get)
            expression = top if rise[top] >= EMOTION_RISE and emotions[top] >= EMOTION_FLOOR else None

            if abs(yaw) < CAMERA_YAW:
                target = "camera"
            elif abs(yaw) >= LOOK_YAW:
                # positive yaw looks toward image-left, i.e. smaller x
                side = [(abs(o["x_mean"] - x), other) for other, o in
                        ((n, {"x_mean": float(np.mean(q["x"]))}) for n, q in people.items() if n != name)
                        if (o["x_mean"] < x) == (yaw > 0)]
                target = min(side)[1] if side else ("off-screen left" if yaw > 0 else "off-screen right")
            else:
                target = None
            summary[name] = {"x": round(x), "gaze_yaw": round(yaw, 2), "target": target,
                             "expression": expression, "expression_rise": round(rise[top], 2)}

        for name, s in summary.items():
            other = summary.get(s["target"]) if s["target"] in summary else None
            s["mutual_gaze"] = bool(other and other["target"] == name)
        shots_out.append({"index": shot["index"], "people": summary})

    out = {"method": __doc__.strip().splitlines()[0], "baseline": baseline, "shots": shots_out}
    (run_dir / "social.json").write_text(json.dumps(out, indent=2))
    n = sum(len(s["people"]) for s in shots_out)
    looks = sum(1 for s in shots_out for p in s["people"].values() if p["target"] not in (None, "camera"))
    print(f"{n} person-shots, {looks} with a gaze target, "
          f"{sum(1 for s in shots_out for p in s['people'].values() if p['mutual_gaze'])} mutual-gaze person-shots, "
          f"{sum(1 for s in shots_out for p in s['people'].values() if p['expression'])} with a clear expression change")


if __name__ == "__main__":
    main()
