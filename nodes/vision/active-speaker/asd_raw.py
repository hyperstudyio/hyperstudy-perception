"""The speakers layer's raw side files, what it drops of LR-ASD's output and of the voice linking (schema:
#/$defs/parquet/speakers_scores and #/$defs/side/speakers_votes):

layers/speakers_scores.parquet  LR-ASD's score for every 25 fps frame of every scored face track: each window
                                length's score (asd_crops.DURATIONS) and their mean, the per-frame score the layer's
                                candidates and votes are means of (at 2 decimals). A row per frame, in the faces
                                layer's track order, then frame order: face, frame (the 25 fps frame index from the
                                video's first frame), t (frame / 25 to 3 places), score, windows.
layers/speakers_votes.json      the word votes behind the voice links (speaker_links.voice_votes): per voice, each
                                word's top visible face, its person and mean score, the votes per person and the
                                number of words with a face visible, from which the links' word shares come.

Both reproduce the layer exactly, because the node makes the layer from them: it reads its scored tracks back from the
scores file once written (read_scores, scored_tracks), takes the votes from them, and links the voices from the votes.
A reader doing the same gets the layer's candidates, predictions and voice links (test_asd_raw.py). The scores are
float32, as the model returned them (the mean as the node took it, numpy's float32 mean over the window lengths: a
reader averaging the windows in another order can differ in the last bit, so the mean is stored too). LR-ASD's scores
are not median-filtered: the node's only median filter smooths the crop boxes (asd_crops.crop_boxes).

Parquet for the scores (OOO clip 1: 9066 frames of 164 tracks in 119 s, 171 KB, about 5.2 MB an hour), JSON for the
votes (a record per word: 328 words, 27 KB, about 0.8 MB an hour). Deterministic: identical scores give identical
bytes. Every writer option that shapes the file is fixed below, and the file carries no time; the pyarrow version
(envs/gaze's lock, part of the node's cache key) names itself in the footer. The node holds every track's scores until
the end (about 56 bytes a frame, some 30 MB for a two-hour film of OOO's density), as it held the per-frame means the
layer needs, and the file is written in track order in row groups of ROW_GROUP_ROWS. numpy only; pyarrow is imported
where the file is written and read.
"""

from __future__ import annotations

import numpy as np

from asd_crops import DURATIONS, FPS
from speaker_links import MAX_WORD_SECONDS

WINDOWS = len(DURATIONS)
ROW_GROUP_ROWS = 65536
ZSTD_LEVEL = 9
SCORES_FILE = "layers/speakers_scores.parquet"
VOTES_FILE = "layers/speakers_votes.json"
SCORES_DESCRIPTION = ("LR-ASD's score for every 25 fps frame of every scored face track, per window length and their "
                      "mean (float32), a row per frame")
VOTES_DESCRIPTION = ("The word votes behind the voice links: per voice, each word's top visible face, its person and "
                     "mean score, and the votes and visible words the word shares come from")


def round_scores(score) -> list[float]:
    """A track's per-frame scores as the layer reads them: each float32 score to 2 decimals."""
    return [round(float(s), 2) for s in score]


def check_track(face: str, frames, score, windows) -> None:
    """ValueError unless a scored track's frames, mean scores (one a frame) and window scores (WINDOWS x frames) fit
    together and are finite float32 (what the model returns, so the file holds exactly what the layer read)."""
    frames, score, windows = np.asarray(frames), np.asarray(score), np.asarray(windows)
    if score.dtype != np.float32 or windows.dtype != np.float32:
        raise ValueError(f"{face}: scores of {score.dtype} and {windows.dtype}, not float32 (the model's, stored as they are)")
    if score.ndim != 1 or len(frames) < len(score) or windows.shape != (WINDOWS, len(score)):
        raise ValueError(f"{face}: {len(score)} scores for {len(frames)} frames and windows of shape {windows.shape}, "
                         f"not ({WINDOWS}, {len(score)})")
    if not (np.isfinite(score).all() and np.isfinite(windows).all()):
        raise ValueError(f"{face}: a score is not finite")


def scores_schema(model: str):
    import pyarrow as pa
    return pa.schema([pa.field("face", pa.string(), nullable=False), pa.field("frame", pa.int32(), nullable=False),
                      pa.field("t", pa.float64(), nullable=False), pa.field("score", pa.float32(), nullable=False),
                      pa.field("windows", pa.list_(pa.field("element", pa.float32(), nullable=False), WINDOWS),
                               nullable=False)],
                     metadata={"side": "speakers_scores", "model": model, "fps": str(FPS),
                               "windows": ",".join(str(d) for d in DURATIONS),
                               "score": "numpy's float32 mean of windows (the layer reads it to 2 decimals)"})


def write_scores(path, tracks: list[tuple], model: str) -> int:
    """tracks: (face, frames, score, windows) per scored track in the faces layer's track order: its 25 fps frame
    indices (the plan's: the scores cover the first len(score)), its float32 mean score per frame, and its float32
    scores per window length (WINDOWS x frames). Writes a row per scored frame; returns the number of rows. A track
    whose arrays do not fit or hold a value that is not finite is a ValueError, and no file is written."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    for face, frames, score, windows in tracks:
        check_track(face, frames, score, windows)
    schema = scores_schema(model)
    n = [len(score) for _, _, score, _ in tracks]
    frame = np.concatenate([np.asarray(f[:k], dtype=np.int32) for (_, f, _, _), k in zip(tracks, n)] or [np.zeros(0, np.int32)])
    score = np.concatenate([np.asarray(s, dtype=np.float32) for _, _, s, _ in tracks] or [np.zeros(0, np.float32)])
    windows = np.concatenate([np.asarray(w, dtype=np.float32).T for *_, w in tracks] or [np.zeros((0, WINDOWS), np.float32)])
    columns = {
        "face": pa.array([face for (face, *_), k in zip(tracks, n) for _ in range(k)], pa.string()),
        "frame": pa.array(frame, pa.int32()),
        "t": pa.array(np.round(frame / FPS, 3), pa.float64()),
        "score": pa.array(score, pa.float32()),
        "windows": pa.FixedSizeListArray.from_arrays(pa.array(windows.reshape(-1), pa.float32()),
                                                     type=schema.field("windows").type),
    }
    table = pa.table(columns, schema=schema)
    with pq.ParquetWriter(path, schema, compression="zstd", compression_level=ZSTD_LEVEL, use_dictionary=["face"],
                          write_statistics=["face", "frame", "t"], version="2.6", data_page_version="1.0",
                          store_schema=True, write_page_index=False, use_byte_stream_split=False) as writer:
        writer.write_table(table, row_group_size=ROW_GROUP_ROWS)
    return len(frame)


def read_scores(path) -> tuple[list[str], list[int], list[float]]:
    """The scores file's face, frame and score columns (scored_tracks reads them)."""
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["face", "frame", "score"])
    return table["face"].to_pylist(), table["frame"].to_pylist(), table["score"].to_pylist()


def scored_tracks(face: list[str], frame: list[int], score: list[float], faces: dict) -> list[dict]:
    """The scored tracks the layer was made from (speaker_links' `scored`), rebuilt from the scores file's face, frame
    and score columns and the faces layer: in track order, each starting at its first frame, its scores at 2
    decimals. A face the faces layer does not have, or a track's rows out of order, is a ValueError."""
    position = {t["id"]: i for i, t in enumerate(faces["tracks"])}
    rows: dict[str, list[tuple[int, float]]] = {}
    for f, k, s in zip(face, frame, score):
        if f not in position:
            raise ValueError(f"the scores name {f}, which the faces layer does not have")
        rows.setdefault(f, []).append((k, s))
    scored = []
    for f in sorted(rows, key=position.get):
        frames = [k for k, _ in rows[f]]
        if frames != list(range(frames[0], frames[0] + len(frames))):
            raise ValueError(f"{f}'s frames are not consecutive from its first")
        track = faces["tracks"][position[f]]
        scored.append({"index": position[f], "face": f, "person": track["person"], "shot": track["shot"],
                       "start": round(frames[0] / FPS, 3), "fps": FPS, "scores": round_scores(s for _, s in rows[f])})
    return scored


def votes_doc(people: list[str], voices: list[dict]) -> dict:
    """layers/speakers_votes.json: the people the voices were matched to (the assignment's columns, the faces layer's
    order) and voice_votes' records."""
    return {"max_word_seconds": MAX_WORD_SECONDS, "people": people, "voices": voices}
