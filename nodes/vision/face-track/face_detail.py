"""The faces layer's side file, layers/faces_detail.json: py-feat's output per track sample, for the face under the
annotator's playhead (68 landmarks drawn over it; action units, emotions, valence and arousal, head pose and gaze in
its inspector), and for the gestures node, which reads the landmarks.

The faces layer keeps only what most readers need (boxes, mouth, gaze, track means); this file holds the rest, keyed
by the layer's face track ids, one entry per sample in the same order. Ported from the bake-off's
faces/face_detail.py, with the bundle's track ids as pipeline/bundle/convert_bakeoff.py keys it. The bake-off's
478-point mesh (face_mesh.json) never went into a bundle and is not written.

Standard library only: the node passes each sample's py-feat row as a mapping from column to value.
"""

import math
from collections.abc import Iterable, Mapping, Sequence

FILE = "layers/faces_detail.json"                  # where the bundle holds it; the layer's detail.file
FIELDS = ["lmk", "au", "emo", "va", "head", "gaze"]  # the per-sample fields; the layer's detail.fields
EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
LANDMARKS = [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]   # all 68 x, then all 68 y
OTHER = ["valence", "arousal", "Yaw", "Pitch", "Roll", "gaze_yaw", "gaze_pitch"]
# the units of head pose and gaze, here and in the faces layer's `angles`
ANGLES = ("radians; yaw + = subject's right (image left), pitch + = up; gaze is in camera coordinates, "
          "(0, 0) = looking into the camera")


def au_names(columns: Iterable[str]) -> list[str]:
    """py-feat's action unit columns (AU01 ... AU43), sorted: the order of every sample's `au`."""
    return sorted(c for c in columns if c.startswith("AU"))


def missing_columns(columns: Iterable[str]) -> list[str]:
    """The columns the side file needs that py-feat's output lacks ("AU*" when it has no action units)."""
    have = set(columns)
    return [c for c in LANDMARKS + EMOTIONS + OTHER if c not in have] + ([] if au_names(have) else ["AU*"])


def sample(t: float, row: Mapping[str, float], aus: Sequence[str]) -> dict:
    """One sample, rounded as the bake-off rounded it. A value that is not a finite number is an error: JSON has no
    NaN, and the annotator could not read the file."""
    def value(column: str, digits: int) -> float:
        x = float(row[column])
        if not math.isfinite(x):
            raise ValueError(f"py-feat gave {column} = {x} for the face at {t} s")
        return round(x, digits)

    return {"t": t,
            "lmk": [value(c, 1) for c in LANDMARKS],
            "au": [value(c, 2) for c in aus],
            "emo": [value(c, 3) for c in EMOTIONS],
            "va": [value("valence", 3), value("arousal", 3)],
            "head": [value("Yaw", 3), value("Pitch", 3), value("Roll", 3)],
            "gaze": [value("gaze_yaw", 3), value("gaze_pitch", 3)]}


def side_file(tracks: Sequence[Sequence[tuple[float, Mapping[str, float]]]], aus: Sequence[str]) -> dict:
    """The side file for the faces layer's tracks, in the layer's order (track i is face:i): each track is its
    samples' (t, py-feat row)."""
    return {"au_names": list(aus), "emotion_names": EMOTIONS, "angles": ANGLES,
            "tracks": {f"face:{i}": [sample(t, row, aus) for t, row in rows] for i, rows in enumerate(tracks)}}


def reference() -> dict:
    """The faces layer's `detail`: where its side file is and what each sample holds."""
    return {"file": FILE, "fields": FIELDS}
