"""How long a node will take, from how long it took here: seconds per minute of video, per node and place, measured
from this store's finished runs. A node that has not run on this machine in that place gets no estimate, not a guess.

The video's length comes from a run's media.v1 artifact (decode writes it), or for a run being planned, from its
cached decode or ffprobe.
"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
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
    """A video's length by ffprobe (when no decode output says it yet)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not video.is_file():
        return None
    out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                         capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def estimate(store: Store, ref: str, place: str, video_seconds: float | None) -> tuple[float | None, str]:
    """Seconds this node should take here on a video this long, and what that rests on."""
    rows = store.db.execute(
        """SELECT t.run_id, t.started_at, t.finished_at FROM tasks t JOIN cache c ON c.key = t.cache_key
           WHERE t.state = 'succeeded' AND c.node = ? AND t.runner = ? AND t.started_at IS NOT NULL""", (ref, place)).fetchall()
    rates = []
    for r in rows:
        seconds = run_seconds(store, r["run_id"])
        if seconds:
            rates.append((r["finished_at"] - r["started_at"]) / (seconds / 60))
    if not rates:
        return None, f"no runs of {ref} on {place} here yet"
    if not video_seconds:
        return None, f"the video's length is not known yet ({len(rates)} runs of {ref} on {place} here)"
    return round(statistics.median(rates) * video_seconds / 60, 1), f"median of {len(rates)} runs of {ref} on {place} here"
