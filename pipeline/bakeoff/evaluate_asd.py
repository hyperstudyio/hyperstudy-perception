"""Score active-speaker detection and voice→face links against hand labels.

Reads runs/<name>/speaker_labels.json (exported from label_speakers.html) and
compares it with asd_talknet.json, asd_lrasd.json, the mouth-AU baseline in
faces.json, and the pyannote voices in transcript_parakeet.json.

  1. Utterance accuracy: for utterances labelled with an on-screen face, does
     the method's top face (highest mean score over the utterance, > 0) match?
  2. Silent-face false positives: frames of faces that are on screen during an
     utterance labelled as someone else; how often is the face scored > 0?
     Speaking-face hit rate is the matching true-positive rate.
  3. Diarization: for each pyannote voice, which labelled people its words
     belong to (purity), and how many labelled people it merges.

Usage: uv run python evaluate_asd.py runs/<name>
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MODELS = ("talknet", "lrasd")


def mean_score(track, start, end, fps):
    i0, i1 = int((start - track["start"]) * fps), int((end - track["start"]) * fps) + 1
    window = track["scores"][max(0, i0):max(0, i1)]
    return (sum(window) / len(window)) if window else None


def mouth_score(track, start, end):
    window = [s["mouth"] for s in track["series"] if start <= s["t"] <= end]
    return (sum(window) / len(window)) if window else None


def main():
    run_dir = Path(sys.argv[1])
    labelled = json.loads((run_dir / "speaker_labels.json").read_text())
    faces = json.loads((run_dir / "faces.json").read_text())
    units = [u for u in labelled["units"] if u["label"]]
    kinds = Counter(u["label"]["kind"] for u in units)
    print(f"{len(units)}/{len(labelled['units'])} utterances labelled: {dict(kinds)}")
    face_units = [u for u in units if u["label"]["kind"] == "face"]

    asd = {m: json.loads((run_dir / f"asd_{m}.json").read_text()) for m in MODELS
           if (run_dir / f"asd_{m}.json").exists()}

    # 1. utterance accuracy: compare by track id (unidentified faces count too)
    print("\nUtterance accuracy (labelled on-screen speaker = method's top-scoring face):")
    for name in list(asd) + ["mouth_au"]:
        right = wrong = abstain = 0
        for u in face_units:
            if name == "mouth_au":
                scored = [(mouth_score(t, u["start"], u["end"]), i) for i, t in enumerate(faces["tracks"])
                          if t["shot"] == u["shot"]]
                threshold = None  # the baseline has no calibrated zero; it always picks the most open mouth
            else:
                scored = [(mean_score(t, u["start"], u["end"], asd[name]["fps"]), t["track"])
                          for t in asd[name]["tracks"] if t["shot"] == u["shot"]]
                threshold = 0.0
            scored = [(s, i) for s, i in scored if s is not None]
            if not scored or (threshold is not None and max(scored)[0] <= threshold):
                abstain += 1
            elif max(scored)[1] == u["label"]["track"]:
                right += 1
            else:
                wrong += 1
        n = len(face_units)
        print(f"  {name:9s} right {right}/{n} ({right / max(1, n):.0%}), wrong {wrong}, no speaker found {abstain}")

    # 2. frame-level rates on labelled utterances
    print("\nFrame rates on labelled utterances (score > 0):")
    for name, result in asd.items():
        speaking, silent = [], []
        for u in face_units:
            for t in result["tracks"]:
                if t["shot"] != u["shot"]:
                    continue
                i0, i1 = int((u["start"] - t["start"]) * result["fps"]), int((u["end"] - t["start"]) * result["fps"]) + 1
                frames = [s > 0 for s in t["scores"][max(0, i0):max(0, i1)]]
                (speaking if t["track"] == u["label"]["track"] else silent).extend(frames)
        off = [s > 0 for u in units if u["label"]["kind"] == "offscreen" for t in result["tracks"]
               if t["shot"] == u["shot"]
               for s in t["scores"][max(0, int((u["start"] - t["start"]) * result["fps"])):
                                    max(0, int((u["end"] - t["start"]) * result["fps"]) + 1)]]
        print(f"  {name:9s} speaking face hit rate {np.mean(speaking):.0%} (n={len(speaking)}); "
              f"silent co-visible face false alarms {np.mean(silent) if silent else float('nan'):.0%} (n={len(silent)}); "
              f"visible faces while an off-screen voice talks {np.mean(off) if off else float('nan'):.0%} (n={len(off)})")

    # 3. diarization against labelled people
    transcript = json.loads((run_dir / "transcript_parakeet.json").read_text())
    words = [w for s in transcript["segments"] for w in s["words"]]
    who = {}
    for u in units:
        label = u["label"]
        if label["kind"] == "face":
            person = label["person"] if label["person"] != "Unknown" else f"face #{label['track']}"
        elif label["kind"] in ("offscreen", "unboxed") and label.get("voice"):
            person = label["voice"]
        else:
            continue
        who[(u["start"], u["end"])] = person
    per_voice = defaultdict(Counter)
    for w in words:
        for (start, end), person in who.items():
            if start <= w["start"] <= end:
                per_voice[w.get("speaker") or "none"][person] += 1
                break
    print("\npyannote voices vs labelled people (words):")
    for voice, people in sorted(per_voice.items()):
        total = sum(people.values())
        top, n = people.most_common(1)[0]
        print(f"  {voice}: {total} words, purity {n / total:.0%} ({top}); people merged: {len(people)} {dict(people)}")
    labelled_people = {p for c in per_voice.values() for p in c}
    print(f"  labelled distinct speakers {len(labelled_people)} vs pyannote voices {len(per_voice)}")


if __name__ == "__main__":
    main()
