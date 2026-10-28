"""The prosody layer and its two JSON side files, from what prosody_measure and odyssey computed. Standard library only.

The layer is what convert_bakeoff.py's prosody_layer made of the bake-off's prosody.json, with the pipeline's models:
the 20 Hz contour (semitones), and per utterance (by the transcript layer's id) its speaker, measures, words and
emotion; captions stay empty here (the prosody-captions node fills them). Its emotion key is the bake-off's,
dimensional_odyssey, and the default the app draws.

Side files (schema: #/$defs/side/prosody_speakers and #/$defs/side/prosody_emotion), beside the Parquet frames
(prosody_tracks.py):
- layers/prosody_speakers.json: each speaker's pitch analysis (floor and ceiling of the second pass and where they come
  from, the median the semitones are relative to, voiced seconds, octave jumps in both passes), the settings, the
  Parselmouth and Praat versions and the contour in Hz;
- layers/prosody_emotion.json: Odyssey's raw output per utterance (the float32 values unrounded, where the layer keeps
  6 places), the samples it read and the normalisation.
"""

from __future__ import annotations

EMOTION_KEY = "dimensional_odyssey"
SPEAKERS_FILE = "layers/prosody_speakers.json"
SPEAKERS_DESCRIPTION = ("each speaker's pitch range, median and octave-jump counts (the semitones' reference), the "
                        "settings, and the 20 Hz contour in Hz")
EMOTION_FILE = "layers/prosody_emotion.json"
EMOTION_DESCRIPTION = "Odyssey's raw arousal, dominance and valence per utterance (float32, unrounded) and the samples it read"
PLACES = 6


def pointers(tracks_file: str, tracks_description: str) -> dict:
    return {"tracks": {"file": tracks_file, "format": "parquet", "description": tracks_description},
            "speakers": {"file": SPEAKERS_FILE, "format": "json", "description": SPEAKERS_DESCRIPTION},
            "emotion": {"file": EMOTION_FILE, "format": "json", "description": EMOTION_DESCRIPTION}}


def emotion_row(labels: list[str], raw: list[float] | None) -> dict:
    """The layer's emotion for one utterance: {dimensional_odyssey: {arousal, dominance, valence}} to 6 places, or {}
    where the model had no output (a clip too short for it)."""
    if raw is None:
        return {}
    if len(raw) != len(labels):
        raise ValueError(f"{len(raw)} emotion outputs for {len(labels)} labels")
    return {EMOTION_KEY: {k: round(float(v), PLACES) for k, v in zip(labels, raw)}}


def layer(units: list[dict], measured: dict, labels: list[str], raw_emotion: list, emotion_model: dict, producer: dict,
          raw: dict) -> dict:
    if not (len(units) == len(measured["utterances"]) == len(raw_emotion)):
        raise ValueError(f"{len(units)} utterances, {len(measured['utterances'])} measured, {len(raw_emotion)} emotions")
    c = measured["contour"]
    utterances = [{"utt": u["id"], "speaker": u["speaker"], "speaker_source": u["speaker_source"],
                   "measures": m["measures"], "words": m["words"], "emotion": emotion_row(labels, e), "captions": {}}
                  for u, m, e in zip(units, measured["utterances"], raw_emotion)]
    return {"layer": "prosody", "schema_version": "0.1.0", "producer": producer,
            "contour": {"step": c["step"], "t0": c["t0"], "speakers": c["speakers"], "f0_st": c["f0_st"], "speaker": c["speaker"]},
            "default_dimensional": EMOTION_KEY,
            "emotion_models": {EMOTION_KEY: {**emotion_model, "kind": "dimensional"}},
            "caption_models": {}, "utterances": utterances, "raw": raw}


def speakers_side(measured: dict, settings: dict, parselmouth: str, praat: str) -> dict:
    return {"model": f"Parselmouth {parselmouth} (Praat {praat})",
            "pitch": "To Pitch (ac), standard settings, two passes (per-speaker floor/ceiling)",
            "settings": settings, "speakers": measured["speakers"],
            "contour": {"step": measured["contour"]["step"], "t0": measured["contour"]["t0"],
                        "speakers": measured["contour"]["speakers"], "f0_hz": measured["contour"]["f0_hz"]}}


def emotion_side(units: list[dict], spans: list[tuple[int, int]], labels: list[str], raw_emotion: list, model: str,
                 config: dict, epsilon: float, min_samples: int, rate: int) -> dict:
    return {"model": model, "outputs": labels, "rate": rate, "min_samples": min_samples,
            "input": "the utterance's own audio, samples [a, b) from its start to its analysed end, as int16 / 32768",
            "input_norm": {"mean": config["mean"], "std": config["std"], "epsilon": epsilon},
            "utterances": [{"utt": u["id"], "samples": [a, b], "values": e}
                           for u, (a, b), e in zip(units, spans, raw_emotion)]}
