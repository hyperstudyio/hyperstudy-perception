"""The utterances prosody measures, with their words, voice and speaker, from the diarized transcript layer and the
speakers layer. Standard library only.

Utterances are the transcript layer's own (asr splits Parakeet's words at a pause over 0.25 s, a sentence end or a shot
cut, as the bake-off's label_page.utterances() did), read by id, so every prosody row joins the app's utterance. Their
words are the layer's [first, last] word range.

The speaker of an utterance, whose pitch range and median its pitch is measured against (pitch is normalised per
speaker), comes from the speakers layer's prediction:
- the person it names: LR-ASD's speaking face (via asd) or the person the utterance's voice is linked to (via
  voice_link): speaker "<person label>" (person:Person G -> "Person G", as the app names it), speaker_source
  "asd_face" or "asd_voice_link";
- else, when it predicts an off-screen voice, the diarized voice as an off-screen one: "off-screen SPEAKER_00"
  (speaker_source "offscreen_voice"), the name the bake-off gave an off-screen voice, which the app draws neutral;
- else (a speaking face with no person) the diarized voice: "SPEAKER_00" (speaker_source "diarized_voice"), the
  bake-off's fallback.
An utterance with no voice (no diarization turn under most of its words) is "off-screen unknown voice" or "unknown
voice". The bake-off's first choice, a hand label (speaker_labels.json), has no counterpart in the pipeline: the
pipeline has no hand labels, so that branch is not ported.
"""

from __future__ import annotations

WORD_MAX_S = 1.2   # see analysed_end
SOURCES = ("asd_face", "asd_voice_link", "offscreen_voice", "diarized_voice")


def label(prefixed: str | None) -> str | None:
    """'person:Person G' -> 'Person G', 'voice:SPEAKER_01' -> 'SPEAKER_01'."""
    return None if prefixed is None else prefixed.split(":", 1)[1]


def word_end(word: dict) -> float:
    """The end of the part of a word that is analysed. Parakeet stretches a sentence-final word over the silence or
    music after it (the bake-off's NES clip: "now." lasts 49 s; the 99th percentile of OOO's words is 0.88 s), so every
    analysis (pitch, rates, emotion, captions) stops a word WORD_MAX_S after its onset; the word keeps its transcript
    times and is flagged (timing_clamped)."""
    return min(word["end"], word["start"] + WORD_MAX_S)


def utterance_index(utt_id: str) -> int:
    """'utt:17' -> 17."""
    kind, _, n = utt_id.partition(":")
    if kind != "utt" or not n.isdigit():
        raise ValueError(f"{utt_id!r} is not an utterance id (utt:N)")
    return int(n)


def speaker_of(predicted: dict | None, voice: str | None) -> tuple[str, str]:
    """(speaker, speaker_source) from a speakers layer row's prediction and the utterance's voice id."""
    if predicted and predicted.get("person"):
        source = {"asd": "asd_face", "voice_link": "asd_voice_link"}.get(predicted.get("via"))
        if source is None:
            raise ValueError(f"the speakers layer names {predicted['person']} via {predicted.get('via')!r}, "
                             "neither asd nor voice_link")
        return label(predicted["person"]), source
    name = label(voice) or "unknown voice"
    if predicted and predicted.get("kind") == "offscreen":
        return f"off-screen {name}", "offscreen_voice"
    return name, "diarized_voice"


def units(transcript: dict, speakers: dict) -> list[dict]:
    """The transcript layer's utterances, each with its word indices, analysed end, voice and speaker:
    [{id, n, start, end, text, words: [i, ...], analysed_end, voice, speaker, speaker_source}]. ValueError when the
    two layers do not describe the same utterances or a word range is out of order."""
    words = transcript["words"]
    rows = {r["utt"]: r for r in speakers["utterances"]}
    if set(rows) != {u["id"] for u in transcript["utterances"]}:
        missing = sorted({u["id"] for u in transcript["utterances"]} - set(rows))
        extra = sorted(set(rows) - {u["id"] for u in transcript["utterances"]})
        raise ValueError(f"the speakers layer does not label the transcript's utterances (missing {missing[:5]}, "
                         f"extra {extra[:5]}): it must come from the same transcript")
    out, previous_last = [], -1
    for u in transcript["utterances"]:
        first, last = u["words"]
        if not previous_last < first <= last < len(words):
            raise ValueError(f"{u['id']} spans words {first}-{last}, after word {previous_last}, of {len(words)}: "
                             "utterances must partition the words in order")
        previous_last = last
        index = list(range(first, last + 1))
        if (words[first]["start"], words[last]["end"]) != (u["start"], u["end"]):
            raise ValueError(f"{u['id']} runs {u['start']}-{u['end']}, but its words run "
                             f"{words[first]['start']}-{words[last]['end']}")
        speaker, source = speaker_of(rows[u["id"]].get("predicted"), u.get("voice"))
        out.append({"id": u["id"], "n": utterance_index(u["id"]), "start": u["start"], "end": u["end"],
                    "text": u["text"], "words": index, "analysed_end": round(word_end(words[last]), 3),
                    "voice": label(u.get("voice")), "speaker": speaker, "speaker_source": source})
    return out
