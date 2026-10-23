"""Link diarized voices (SPEAKER_00, ...) to face identities (Person A, ...).

A first, model-free active-speaker signal: py-feat's mouth action units
(AU25 lips part, AU26 jaw drop). For every voice and every person, compare the
person's mouth activity while that voice is talking against while it is not,
using only moments the person's face is on screen. Voices are then matched to
people one-to-one (Hungarian assignment) where the evidence is positive.
A voice with no positive match stays voice-only (off-screen or narrator).

Light-ASD (audio-visual active speaker detection) is the planned replacement;
this baseline shows how far py-feat output alone gets.

With --asd MODEL it instead uses asd/asd_scores.py output (asd_<MODEL>.json,
MODEL = talknet or lrasd): each word of a voice votes for the visible face the
detector scores as speaking, and voices whose words split across several faces
are flagged (a sign that diarization merged people). Writes
speaker_links_<MODEL>.json.

Usage: uv run python link_speakers.py runs/<name> [--asd talknet|lrasd]
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

MIN_SAMPLES = 4  # face samples needed on each side (talking / not talking) to score a pair


def speaking_intervals(transcript):
    spans = {}
    for segment in transcript["segments"]:
        for word in segment["words"]:
            speaker = word.get("speaker") or segment.get("speaker")
            if speaker and word["start"] is not None and word["end"] - word["start"] <= 2.0:
                spans.setdefault(speaker, []).append((word["start"], word["end"]))
    return spans


def is_talking(t, spans, pad=0.15):
    return any(s - pad <= t <= e + pad for s, e in spans)


MIN_SHARE = 0.2  # a person taking at least this share of a voice's on-screen words suggests the voice merges people


def asd_votes(run_dir, model, spans):
    """For each word of each voice, the visible face the ASD model scores highest (if > 0) gets the vote."""
    asd = json.loads((run_dir / f"asd_{model}.json").read_text())
    votes = {speaker: {} for speaker in spans}
    visible_words = {speaker: 0 for speaker in spans}
    for speaker, words in spans.items():
        for start, end in words:
            best, best_score, seen = None, 0.0, False
            for track in asd["tracks"]:
                if track["person"] == "Unknown":
                    continue
                i0 = int((start - track["start"]) * asd["fps"])
                i1 = int((end - track["start"]) * asd["fps"]) + 1
                window = track["scores"][max(0, i0):max(0, i1)]
                if not window:
                    continue
                seen = True
                score = sum(window) / len(window)
                if score > best_score:
                    best, best_score = track["person"], score
            visible_words[speaker] += seen
            if best:
                votes[speaker][best] = votes[speaker].get(best, 0) + 1
    return votes, visible_words


def main_asd(run_dir, model, spans, people):
    """Link voices to faces from ASD word votes, and flag voices whose words split across several faces."""
    votes, visible = asd_votes(run_dir, model, spans)
    speakers = sorted(spans)
    share = np.array([[votes[s].get(p, 0) / max(1, visible[s]) for p in people] for s in speakers])
    rows, cols = linear_sum_assignment(-share)
    links = {}
    for i, j in zip(rows, cols):
        if share[i, j] > 0:
            links[speakers[i]] = {"person": people[j], "word_share": round(float(share[i, j]), 2),
                                  "visible_words": visible[speakers[i]]}
    for i, speaker in enumerate(speakers):
        links.setdefault(speaker, {"person": None, "note": f"{model} found no speaking face for this voice"})
        links[speaker]["also_speaking"] = {people[j]: round(float(share[i, j]), 2) for j in np.argsort(-share[i])
                                           if share[i, j] >= MIN_SHARE and people[j] != links[speaker]["person"]}
    out = {"method": f"{model} per-word votes (top visible face with score > 0), Hungarian on word share",
           "links": links, "votes": votes, "visible_words": visible}
    (run_dir / f"speaker_links_{model}.json").write_text(json.dumps(out, indent=2))
    for speaker, link in links.items():
        print(f"{speaker} -> {link['person']}  share {link.get('word_share')} of {visible[speaker]} visible words; "
              f"also {link['also_speaking']}")


def main():
    run_dir = Path(sys.argv[1])
    transcript = json.loads((run_dir / "extract.json").read_text())["transcript"]
    faces = json.loads((run_dir / "faces.json").read_text())
    spans = speaking_intervals(transcript)
    if "--asd" in sys.argv:
        model = sys.argv[sys.argv.index("--asd") + 1]
        return main_asd(run_dir, model, spans, [p["label"] for p in faces["people"]])
    speakers = sorted(spans)
    people = [p["label"] for p in faces["people"]]

    samples = {p: [] for p in people}
    for track in faces["tracks"]:
        if track["person"] == "Unknown":  # unattributed tracks cannot vote for a person
            continue
        samples[track["person"]].extend((s["t"], s["mouth"]) for s in track["series"])

    score = np.zeros((len(speakers), len(people)))
    evidence = {}
    for i, speaker in enumerate(speakers):
        for j, person in enumerate(people):
            talking = [m for t, m in samples[person] if is_talking(t, spans[speaker])]
            silent = [m for t, m in samples[person] if not is_talking(t, spans[speaker])]
            if len(talking) >= MIN_SAMPLES and len(silent) >= MIN_SAMPLES:
                score[i, j] = np.mean(talking) - np.mean(silent)
                evidence[(speaker, person)] = (len(talking), len(silent), round(float(score[i, j]), 3))

    rows, cols = linear_sum_assignment(-score)
    links = {}
    for i, j in zip(rows, cols):
        if score[i, j] > 0:
            n_talk, n_silent, diff = evidence[(speakers[i], people[j])]
            links[speakers[i]] = {"person": people[j], "mouth_diff": diff, "talking_samples": n_talk,
                                  "silent_samples": n_silent}
    for speaker in speakers:
        links.setdefault(speaker, {"person": None, "note": "no on-screen face with positive mouth evidence"})

    out = {"method": "py-feat AU25/AU26 talking-vs-silent difference, Hungarian assignment",
           "links": links,
           "matrix": {speakers[i]: {people[j]: round(float(score[i, j]), 3) for j in range(len(people)) if score[i, j]}
                      for i in range(len(speakers))}}
    (run_dir / "speaker_links.json").write_text(json.dumps(out, indent=2))
    for speaker, link in links.items():
        print(f"{speaker} -> {link['person']}  {link.get('mouth_diff', link.get('note'))}")


if __name__ == "__main__":
    main()
