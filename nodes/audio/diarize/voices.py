"""Diarization turns onto a transcript's words and utterances: the voices of the bundle's transcript layer.

As the bake-off made them: each word takes the speaker whose turns overlap it longest (transcribe_parakeet.py's
speaker_for), and each utterance the most common speaker of its words (convert_bakeoff.py's transcript_layer). Ties go
to the speaker met first: the earlier turn for a word, the earlier word for an utterance. A word that no turn overlaps
has no voice, and neither has an utterance most of whose words have none. Standard library only.
"""

from __future__ import annotations

from collections import Counter

from perception_node import NodeError


def speaker_for(word: dict, turns: list[dict]) -> str | None:
    """The speaker whose turns overlap the word longest (summed over that speaker's turns), or None."""
    overlaps: dict[str, float] = {}
    for turn in turns:
        overlap = min(word["end"], turn["end"]) - max(word["start"], turn["start"])
        if overlap > 0:
            overlaps[turn["speaker"]] = overlaps.get(turn["speaker"], 0) + overlap
    return max(overlaps, key=overlaps.get) if overlaps else None


def voice_id(speaker: str | None) -> str | None:
    return f"voice:{speaker}" if speaker else None


def diarize(transcript: dict, turns: list[dict]) -> dict:
    """The transcript layer with `voices` and every word's and utterance's voice filled from `turns`
    ([{start, end, speaker}], in the diarization's order). Everything else is kept as it was."""
    words = [{**w, "voice": voice_id(speaker_for(w, turns))} for w in transcript["words"]]
    utterances = []
    for u in transcript["utterances"]:
        first, last = u["words"]
        if not 0 <= first <= last < len(words):
            raise NodeError("input", f"{u['id']} spans words {first}-{last}, but the transcript has {len(words)}")
        voice = Counter(words[i]["voice"] for i in range(first, last + 1)).most_common(1)[0][0]
        utterances.append({**u, "voice": voice})
    voices = sorted({w["voice"] for w in words if w["voice"]})
    return {**transcript, "voices": [{"id": v} for v in voices], "words": words, "utterances": utterances}
