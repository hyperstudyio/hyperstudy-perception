"""How long a node will take, from how long it took here: seconds per minute of video, per node and place, measured
from this store's finished runs. A node that has not run on this machine in that place gets no estimate, not a guess.

The video's length comes from a run's media.v1 artifact (decode writes it), or for a run being planned, from its
cached decode or the file itself, read with PyAV as decode reads it (no ffprobe on the machine: an app started from
the Finder has none on its PATH).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .store import Artifact, Store


def media_seconds(outputs: dict) -> float | None:
    """The video length in a media.v1 output, if these outputs hold one."""
    for a in outputs.values():
        art = Artifact.from_json(a)
        if art.type == "media.v1" and art.path.is_file():
            duration = json.loads(art.path.read_text()).get("duration")
            if duration:
                return float(duration)
    return None


def run_seconds(store: Store, run_id: str) -> float | None:
    for t in store.db.execute("SELECT outputs FROM tasks WHERE run_id=? AND outputs IS NOT NULL", (run_id,)):
        seconds = media_seconds(json.loads(t["outputs"]))
        if seconds:
            return seconds
    return None


def probe_seconds(video: Path) -> float | None:
    """A video's length, as decode's media.v1 will say it (the container's duration), when no decode output says it
    yet; None for a file that is not a video."""
    import av

    if not video.is_file():
        return None
    try:
        with av.open(str(video)) as container:
            return container.duration / av.time_base if container.duration else None
    except av.FFmpegError:
        return None


def estimate(store: Store, ref: str, place: str, video_seconds: float | None) -> tuple[float | None, str]:
    """Seconds this node should take here on a video this long, and what that rests on."""
    rows = store.db.execute(
        """SELECT t.run_id, t.usage FROM tasks t JOIN cache c ON c.key = t.cache_key
           WHERE t.state = 'succeeded' AND c.node = ? AND t.runner = ?""", (ref, place)).fetchall()
    rates = []
    for r in rows:
        # the seconds the node itself ran (the SDK reports them), not the coordinator's clock: a node that finished
        # while no coordinator watched would otherwise count that time too
        ran = json.loads(r["usage"] or "{}").get("seconds")
        seconds = run_seconds(store, r["run_id"])
        if ran and seconds:
            rates.append(ran / (seconds / 60))
    if not rates:
        return None, f"no runs of {ref} on {place} here yet"
    if not video_seconds:
        return None, f"the video's length is not known yet ({len(rates)} runs of {ref} on {place} here)"
    return round(statistics.median(rates) * video_seconds / 60, 1), f"median of {len(rates)} runs of {ref} on {place} here"
