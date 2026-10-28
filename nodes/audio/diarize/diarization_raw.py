"""The diarization's full output beside the transcript layer, which keeps only each word's and utterance's voice:
two JSON side files (schema: #/$defs/side/transcript_diarization and #/$defs/side/transcript_voice_embeddings).

pyannote.audio 4's pipeline returns three things (DiarizeOutput), all kept:
- layers/transcript_diarization.json: its speaker turns (speaker_diarization, where two speakers' turns overlap in
  crosstalk), the turns the layer's voices are assigned from (voices.py), and its exclusive turns
  (exclusive_speaker_diarization: one speaker at a time, pyannote's version for aligning a transcript), each in
  pyannote's order with its times unrounded, so the stored turns give exactly the layer's voices.
- layers/transcript_voice_embeddings.json: its speaker embedding per speaker (speaker_embeddings), the centroid of the
  clustered 256-d WeSpeaker ResNet34 embeddings (VBx responsibilities as weights), raw (not L2-normalised), to 4 decimal
  places (cosine similarities within 3e-5 of full precision on OOO clip 1). A speaker pyannote has no centroid for, whose
  row it pads with zeros, is null. Identity data: a voiceprint, kept apart from the turns so that a step reading the
  turns elsewhere does not need consent to embeddings.

Not kept, the pipeline's intermediates (its hook's): the segmentation, binary for this powerset model (each frame's
class taken by argmax: no scores reach the pipeline) in chunk-local speaker slots of 10 s windows a second apart, and
the per-(chunk, slot) embeddings, both used only to find the clusters; and the frame-level diarization they are
aggregated into, which the turns encode exactly (min_duration_off is 0: each turn covers the frames whose middle is in
[start, end); on OOO clip 1, 0 of 35,560 frame-speaker cells differ).

Speakers are the layer's voice ids (voice:SPEAKER_00, ...), pyannote's labels with the bundle's prefix. Standard library
only.
"""

from __future__ import annotations

import math

from perception_node import NodeError
from voices import voice_id

DIARIZATION_FILE = "layers/transcript_diarization.json"
DIARIZATION_DESCRIPTION = ("pyannote's speaker turns (overlapping in crosstalk), which the voices are assigned from, and "
                           "its exclusive turns (one speaker at a time), unrounded")
EMBEDDINGS_FILE = "layers/transcript_voice_embeddings.json"
EMBEDDINGS_DESCRIPTION = "pyannote's 256-d speaker embedding (WeSpeaker ResNet34 centroid) per voice, raw, to 4 places"
EMBEDDING_MODEL = "WeSpeaker ResNet34 (pyannote/speaker-diarization-community-1's embedding)"
DIMS = 256
PLACES = 4


def turns_of(turns: list[dict]) -> list[dict]:
    """[{start, end, speaker}] -> [{start, end, voice}], in the same order, the times as they are."""
    return [{"start": t["start"], "end": t["end"], "voice": voice_id(t["speaker"])} for t in turns]


def diarization(model: str, speakers: list[str], turns: list[dict], exclusive: list[dict]) -> dict:
    """The transcript_diarization side file: pyannote's speakers (its labels() order, the order of the embeddings) and
    its turns and exclusive turns ([{start, end, speaker}] as pyannote gave them)."""
    known = set(speakers)
    for t in turns + exclusive:
        if t["speaker"] not in known:
            raise NodeError("internal", f"pyannote gave a turn to {t['speaker']}, not one of its speakers {speakers}")
        if not (math.isfinite(t["start"]) and math.isfinite(t["end"]) and 0 <= t["start"] <= t["end"]):
            raise NodeError("internal", f"pyannote gave {t['speaker']} a turn from {t['start']} to {t['end']}")
    return {"model": model, "voices": [voice_id(s) for s in speakers], "turns": turns_of(turns),
            "exclusive_turns": turns_of(exclusive)}


def voice_embeddings(speakers: list[str], embeddings) -> dict:
    """The transcript_voice_embeddings side file from pyannote's speaker_embeddings (one row of DIMS values per speaker,
    in `speakers` order, as nested lists): each voice's embedding to PLACES places, or null for a row of zeros."""
    if embeddings is None:   # what pyannote gives only with oracle clustering, which community-1 does not use
        raise NodeError("internal", "pyannote returned no speaker embeddings")
    if len(embeddings) != len(speakers):
        raise NodeError("internal", f"pyannote returned {len(embeddings)} speaker embeddings for {len(speakers)} speakers")
    voices = {}
    for speaker, row in zip(speakers, embeddings):
        if len(row) != DIMS:
            raise NodeError("internal", f"{speaker}'s speaker embedding has {len(row)} values, not {DIMS}")
        if not all(math.isfinite(x) for x in row):
            raise NodeError("internal", f"{speaker}'s speaker embedding holds a value that is not finite")
        voices[voice_id(speaker)] = [round(float(x), PLACES) for x in row] if any(row) else None
    return {"model": EMBEDDING_MODEL, "dims": DIMS, "normalised": False, "voices": voices}


def pointers() -> dict:
    """The layer's raw pointers at the two files."""
    return {"diarization": {"file": DIARIZATION_FILE, "format": "json", "description": DIARIZATION_DESCRIPTION},
            "voice_embeddings": {"file": EMBEDDINGS_FILE, "format": "json", "description": EMBEDDINGS_DESCRIPTION}}
