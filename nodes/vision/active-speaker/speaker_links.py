"""The speakers layer from LR-ASD's per-frame scores: for each utterance, the face tracks in its shot ranked by their
mean score over it (the top one, if positive, is the speaker), and otherwise the person its voice is linked to.

As the bake-off made it: annotator/build.py's asd_candidates, link_speakers.py's --asd votes and
convert_bakeoff.py's speakers_layer, with one change: voices are linked on the transcript this layer describes (the
diarize node's: Parakeet's words, pyannote's voices), where link_speakers.py read WhisperX's transcript, whose voice
labels were another diarization's.

Voice links: each word of a voice (shorter than MAX_WORD_SECONDS) votes for the visible person whose face LR-ASD scores
highest, if above 0, over the word; a voice's share for a person is that person's votes over the voice's words with
any face visible. Voices and people are then matched one to one on those shares (Hungarian), and a link is trusted, and
used for an utterance without a speaking face, at a share of `min_share` or more (0.3: faces/evaluate_identity.py on
OOO, 72% right against 65% ungated). The votes are kept as records per word (voice_votes: each word's top visible face,
its person and score), which the links are made from and the node stores as layers/speakers_votes.json. numpy only;
the assignment is scipy's, passed in.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

MAX_WORD_SECONDS = 2.0   # longer "words" are alignment errors, and vote for no one


def require_diarized(transcript: dict) -> None:
    """ValueError unless the transcript is a diarized one (its producer's stage names diarization, as the diarize
    node's "transcribe+diarize" does): asr's has no voices to link. A diarized transcript whose words all have no
    voice (pyannote found no speech turns) is valid: its utterances are then off-screen voices."""
    stage = (transcript.get("producer") or {}).get("stage") or ""
    if "diarize" not in stage:
        raise ValueError(f"the transcript comes from stage {stage!r}, not a diarization: speakers reads the diarize "
                         "node's transcript, whose voices it links to people")


def mean_score(track: dict, start: float, end: float) -> float | None:
    """A scored track's mean per-frame score over [start, end] (the frames from start's to end's, inclusive), or None
    where the track has no frames then."""
    i0, i1 = int((start - track["start"]) * track["fps"]), int((end - track["start"]) * track["fps"]) + 1
    window = track["scores"][max(0, i0):max(0, i1)]
    return sum(window) / len(window) if window else None


def candidates(scored: list[dict], utterance: dict) -> list[tuple[float, int]]:
    """(mean score, track position) of the scored tracks in the utterance's shot that have frames during it, highest
    first (ties: the later track first)."""
    found = [(mean_score(t, utterance["start"], utterance["end"]), t["index"]) for t in scored if t["shot"] == utterance["shot"]]
    return sorted(((s, i) for s, i in found if s is not None), reverse=True)


def speaking_intervals(words: list[dict]) -> dict[str, list[tuple[int, float, float]]]:
    """voice -> its words' (index in `words`, start, end), for words with a voice and at most MAX_WORD_SECONDS long."""
    spans: dict[str, list[tuple[int, float, float]]] = {}
    for k, w in enumerate(words):
        if w.get("voice") and w["start"] is not None and w["end"] - w["start"] <= MAX_WORD_SECONDS:
            spans.setdefault(w["voice"], []).append((k, w["start"], w["end"]))
    return spans


def top_face(scored: list[dict], start: float, end: float) -> tuple[dict | None, float | None]:
    """The scored track with a person whose mean score over [start, end] is highest (of equals, the first in track
    order), and that score; (None, None) when no person's face has frames then. Faces with no person do not count."""
    best, best_score = None, None
    for track in scored:
        if track["person"] is None:
            continue
        score = mean_score(track, start, end)
        if score is not None and (best_score is None or score > best_score):
            best, best_score = track, score
    return best, best_score


def tally(words: list[dict]) -> tuple[dict[str, int], int]:
    """A voice's votes per person and its number of words with a person's face visible, from its word records
    (voice_votes): a word votes for its top face's person when that face's mean score is above 0."""
    votes: dict[str, int] = {}
    for w in words:
        if w["score"] is not None and w["score"] > 0:
            votes[w["person"]] = votes.get(w["person"], 0) + 1
    return votes, sum(w["score"] is not None for w in words)


def voice_votes(spans: dict, scored: list[dict]) -> list[dict]:
    """The votes behind the voice links, per voice (sorted): {voice, visible, votes: {person: n}, words: [{word,
    face, person, score}]}. Each word (its index in the transcript's words) records its top visible face, that
    face's person and its mean score over the word (unrounded; all None when no person's face is visible); it votes
    for that person if the score is above 0. visible counts the words with a face visible (tally)."""
    voices = []
    for voice in sorted(spans):
        words = []
        for k, start, end in spans[voice]:
            track, score = top_face(scored, start, end)
            words.append({"word": k, "face": track and track["face"], "person": track and track["person"], "score": score})
        votes, visible = tally(words)
        voices.append({"voice": voice, "visible": visible, "votes": dict(sorted(votes.items())), "words": words})
    return voices


def votes_for(spans: dict, scored: list[dict]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Per voice: the votes per person (each word's top visible person, if that face's mean score is above 0), and the
    number of its words when any person's face was visible."""
    voices = voice_votes(spans, scored)
    return {v["voice"]: v["votes"] for v in voices}, {v["voice"]: v["visible"] for v in voices}


def link_voices(voices: list[dict], people: list[str], assign: Callable) -> dict[str, dict]:
    """voice -> {person, word_share} for the voices the one-to-one matching gives a person with a share above 0, from
    voice_votes' records (a voice's share for a person: its votes over its visible words). `assign(cost)` -> (rows,
    cols) solves the assignment (scipy.optimize.linear_sum_assignment)."""
    if not voices or not people:
        return {}
    names = [v["voice"] for v in voices]
    share = np.array([[v["votes"].get(p, 0) / max(1, v["visible"]) for p in people] for v in voices])
    rows, cols = assign(-share)
    return {names[i]: {"person": people[j], "word_share": round(float(share[i, j]), 2)}
            for i, j in zip(rows, cols) if share[i, j] > 0}


def utterance_row(utterance: dict, scored: list[dict], person_of: dict[int, str | None], links: dict, min_share: float) -> dict:
    """One utterance's speaker: the top candidate face if its mean score is above 0, else the person its voice is
    linked to (trusted links only), else off screen."""
    ranked = candidates(scored, utterance)
    face_of = {t["index"]: t["face"] for t in scored}
    if ranked and ranked[0][0] > 0:
        track = ranked[0][1]
        predicted = {"kind": "face", "face": face_of[track], "via": "asd"}
        if person_of.get(track):
            predicted["person"] = person_of[track]
        margin = ranked[0][0] - (ranked[1][0] if len(ranked) > 1 else 0.0)
    else:
        margin = -ranked[0][0] if ranked else None
        voice = utterance.get("voice")
        link = links.get(voice or "", {})
        if link and link["word_share"] >= min_share and link["person"]:
            predicted = {"kind": "person", "person": link["person"], "voice": voice, "via": "voice_link"}
        else:
            predicted = {"kind": "offscreen", "voice": voice, "via": "none"}
    return {"utt": utterance["id"], "predicted": predicted,
            "candidates": [{"face": face_of[i], "score": round(s, 3)} for s, i in ranked],
            "margin": None if margin is None else round(margin, 3), "second_opinion_agrees": None}


def layer_content(transcript: dict, faces: dict, scored: list[dict], min_share: float, assign: Callable,
                  voices: list[dict] | None = None) -> dict:
    """The speakers layer's utterances and voice_links. `scored`: [{index (track position), face, person, shot,
    start, fps, scores}] for the tracks LR-ASD scored, in track order. `voices`: voice_votes' records the links are
    made from (by default, voice_votes over the transcript's words and `scored`; the votes side file stores them)."""
    people = [p["id"] for p in faces["people"]]
    person_of = {i: t["person"] for i, t in enumerate(faces["tracks"])}
    if voices is None:
        voices = voice_votes(speaking_intervals(transcript["words"]), scored)
    links = link_voices(voices, people, assign)
    rows = [utterance_row(u, scored, person_of, links, min_share) for u in transcript["utterances"]]
    voice_links = [{"voice": v, "person": l["person"], "word_share": l["word_share"], "trusted": l["word_share"] >= min_share}
                   for v, l in sorted(links.items())]
    return {"utterances": rows, "voice_links": voice_links}
