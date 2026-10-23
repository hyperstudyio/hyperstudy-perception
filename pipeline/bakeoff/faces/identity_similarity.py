"""Pairwise similarity between the people in faces.json, for merge suggestions in the annotator.

Clustering is tuned to over-split (a wrong merge misleads agents, a split is fixable in review),
so the same panelist can end up as two people. For each pair of people this writes the cosine
distance between their ArcFace centroids and whether they are ever on screen at the same time
(two faces visible together cannot be one person). Tracks are rebuilt from the cached detections
exactly as faces.py built them, so track i here is track i in faces.json.

Usage: uv run python identity_similarity.py ../runs/<name>
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from faces import load_tracks, track_embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    _, faces, tracks = load_tracks(args.run_dir)

    members = {}
    for i, saved in enumerate(faces["tracks"]):
        if saved["person"] != "Unknown":
            members.setdefault(saved["person"], []).append(i)
    centroid = {}
    for person, idx in members.items():
        c = np.mean([track_embedding(tracks[i]) for i in idx], axis=0)
        centroid[person] = c / np.linalg.norm(c)
    times = {p: {round(r["t"], 3) for i in idx for r in tracks[i]["rows"]} for p, idx in members.items()}

    pairs = []
    for a, b in combinations(sorted(members), 2):
        pairs.append({"a": a, "b": b, "distance": round(float(1 - centroid[a] @ centroid[b]), 3),
                      "co_visible": bool(times[a] & times[b])})
    pairs.sort(key=lambda p: p["distance"])
    out = {"method": "cosine distance between ArcFace track-mean centroids; co_visible = ever detected in the same sample",
           "cluster_threshold": faces["threshold"], "pairs": pairs}
    (args.run_dir / "identity_similarity.json").write_text(json.dumps(out, indent=2))
    for p in pairs[:15]:
        print(f"  {p['a']} ~ {p['b']}: {p['distance']:.3f}{'  (on screen together)' if p['co_visible'] else ''}")


if __name__ == "__main__":
    main()
