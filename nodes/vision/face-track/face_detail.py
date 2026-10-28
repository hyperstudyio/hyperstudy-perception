"""The faces layer's side files: py-feat's output per track sample, for the face under the annotator's playhead.

layers/faces_detail.json  68 landmarks (drawn over the face), action units, emotions, valence and arousal, head pose
                          and gaze (in the inspector); the gestures node reads the landmarks. Small: the annotator
                          loads it with the page. The layer's `detail` points at it.
layers/faces_mesh.json    py-feat's 478-point 2D face mesh, rounded to whole pixels. Large: the annotator fetches it
                          only when its mesh overlay is switched on. The layer's `mesh` points at it.
layers/faces_embeddings.json
                          py-feat's 512-d ArcFace identity embedding, as py-feat returns it (not L2-normalised): the
                          vectors tracking.py clusters into people and measures person_pairs with, kept for reuse (the
                          same person across videos, re-clustering). Identity data: a step that reads it on another
                          machine needs the run's consent for embeddings. The layer's `embeddings` points at it.

The faces layer keeps only what most readers need (boxes, mouth, gaze, track means); these files hold the rest, keyed
by the layer's face track ids, one entry per sample in the same order. The detail and mesh are ported from the
bake-off's faces/face_detail.py (which wrote face_detail.json and face_mesh.json), with the bundle's track ids as
pipeline/bundle/convert_bakeoff.py keys them; the bake-off never wrote the embeddings.

Standard library only: the node passes each sample's py-feat row as a mapping from column to value (the detail), the
row's mesh values in MESH order (the mesh), or the face's embedding, EMBEDDING's columns in order (the embeddings).
"""

import math
from collections.abc import Iterable, Mapping, Sequence

FILE = "layers/faces_detail.json"                  # where the bundle holds it; the layer's detail.file
FIELDS = ["lmk", "au", "emo", "va", "head", "gaze"]  # the per-sample fields; the layer's detail.fields
EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
LANDMARKS = [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]   # all 68 x, then all 68 y
OTHER = ["valence", "arousal", "Yaw", "Pitch", "Roll", "gaze_yaw", "gaze_pitch"]
MESH_FILE = "layers/faces_mesh.json"               # the layer's mesh.file
MESH_POINTS = 478
MESH = [f"mesh_x_{i}" for i in range(MESH_POINTS)] + [f"mesh_y_{i}" for i in range(MESH_POINTS)]   # all x, then all y
MESH_LAYOUT = f"all {MESH_POINTS} x, then all {MESH_POINTS} y, in source pixels, rounded to integers"
EMBEDDINGS_FILE = "layers/faces_embeddings.json"   # the layer's embeddings.file
EMBEDDING_MODEL = "ArcFace (InsightFace, via py-feat Detectorv2)"
EMBEDDING_DIMS = 512
EMBEDDING = [f"Identity_{i + 1}" for i in range(EMBEDDING_DIMS)]   # py-feat's identity columns, in order
# py-feat's ArcFace returns the backbone's output without L2 normalisation (feat/identity_detectors/arcface), and
# tracking.py averages these raw vectors before normalising a track's mean, so the file keeps them as they are
EMBEDDING_NORMALISED = False
# decimal places kept. Values are up to about 4.4 (vector norms 8 to 27) on OOO clip 1, where 4 places keep every
# cosine distance between two samples within 2e-5 of full precision and every person centroid distance within 6e-6;
# 3 places drift up to 1.6e-4
EMBEDDING_DIGITS = 4
# the units of head pose and gaze, here and in the faces layer's `angles`
ANGLES = ("radians; yaw + = subject's right (image left), pitch + = up; gaze is in camera coordinates, "
          "(0, 0) = looking into the camera")


def au_names(columns: Iterable[str]) -> list[str]:
    """py-feat's action unit columns (AU01 ... AU43), sorted: the order of every sample's `au`."""
    return sorted(c for c in columns if c.startswith("AU"))


def missing_columns(columns: Iterable[str]) -> list[str]:
    """The columns the side files need that py-feat's output lacks ("AU*" when it has no action units)."""
    have = set(columns)
    return ([c for c in LANDMARKS + EMOTIONS + OTHER + MESH + EMBEDDING if c not in have]
            + ([] if au_names(have) else ["AU*"]))


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


def mesh_sample(t: float, values: Sequence[float]) -> dict:
    """One mesh sample: `values` are the row's MESH columns in that order, each rounded to a whole pixel as the
    bake-off rounded it (Python's round, halves to even). A value that is not a finite number is an error, as in
    `sample`."""
    if len(values) != len(MESH):
        raise ValueError(f"{len(values)} mesh values for the face at {t} s, not {len(MESH)}")
    xy = []
    for column, v in zip(MESH, values):
        x = float(v)
        if not math.isfinite(x):
            raise ValueError(f"py-feat gave {column} = {x} for the face at {t} s")
        xy.append(round(x))
    return {"t": t, "xy": xy}


def mesh_file(tracks: Sequence[Sequence[tuple[float, Sequence[float]]]]) -> dict:
    """The mesh side file for the faces layer's tracks, in the layer's order (track i is face:i): each track is its
    samples' (t, mesh values in MESH order)."""
    return {"points": MESH_POINTS, "layout": MESH_LAYOUT,
            "tracks": {f"face:{i}": [mesh_sample(t, values) for t, values in rows] for i, rows in enumerate(tracks)}}


def mesh_reference() -> dict:
    """The faces layer's `mesh`: where its mesh side file is."""
    return {"file": MESH_FILE}


def embedding_sample(t: float, values: Sequence[float]) -> dict:
    """One embeddings sample: `values` are the face's embedding (EMBEDDING's columns, in order), each rounded to
    EMBEDDING_DIGITS decimal places. A value that is not a finite number is an error, as in `sample`."""
    if len(values) != EMBEDDING_DIMS:
        raise ValueError(f"{len(values)} embedding values for the face at {t} s, not {EMBEDDING_DIMS}")
    v = []
    for column, value in zip(EMBEDDING, values):
        x = float(value)
        if not math.isfinite(x):
            raise ValueError(f"py-feat gave {column} = {x} for the face at {t} s")
        v.append(round(x, EMBEDDING_DIGITS))
    return {"t": t, "v": v}


def embeddings_file(tracks: Sequence[Sequence[tuple[float, Sequence[float]]]]) -> dict:
    """The embeddings side file for the faces layer's tracks, in the layer's order (track i is face:i): each track is
    its samples' (t, embedding)."""
    return {"model": EMBEDDING_MODEL, "dims": EMBEDDING_DIMS, "normalised": EMBEDDING_NORMALISED,
            "tracks": {f"face:{i}": [embedding_sample(t, values) for t, values in rows] for i, rows in enumerate(tracks)}}


def embeddings_reference() -> dict:
    """The faces layer's `embeddings`: where its embeddings side file is."""
    return {"file": EMBEDDINGS_FILE}
