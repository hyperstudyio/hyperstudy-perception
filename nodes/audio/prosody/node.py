"""audio.prosody: how each utterance is said, as the bundle's prosody layer: measured prosody (Parselmouth, i.e.
Praat) and speech emotion (the Odyssey 2024 baseline), ported from the bake-off's prosody/prosody.py, which chose them.
The bake-off's third part, style captions from gpt-audio, is the prosody-captions node: it sends audio to OpenAI, and a
node's uploads are all or nothing (the run's --allow-uploads consents per node), so these local measures run whether or
not the user lets anything leave the machine.

Utterances and speakers: prosody_units.py (the transcript layer's utterances, by id; the speaker from the speakers
layer). Measures: prosody_measure.py over praat.py. Emotion: odyssey.py. The layer and its JSON side files:
prosody_layer.py; the Parquet frames: prosody_tracks.py. The layer points at all three with raw.

Times: the transcript's words count from the audio's first sample, as decode's audio does, so the word times index it
directly (the bake-off read the same 16 kHz mono audio, the same samples on OOO clip 1).

Licence: Parselmouth and Praat are GPL-3 (used as a separate program; needs-review, as the bake-off recorded); the
Odyssey weights MIT, trained on MSP-Podcast, whose academic-only data licence may carry over (needs-review). The
speakers come from the speakers layer, which reads py-feat's face tracks (academic-only): export gives the layer the
strictest of its upstream nodes.
"""

import time
import wave

import numpy as np

import odyssey
import prosody_layer
import prosody_tracks
from perception_node import Node, NodeError
from praat import LICENCE as PRAAT_LICENCE, Praat, versions
from prosody_measure import SAMPLE_RATE, SETTINGS, measure
from prosody_units import units as make_units

PARSELMOUTH, PRAAT = "0.4.7", "6.1.38"   # envs/prosody's lock
PARSELMOUTH_ID = f"Parselmouth {PARSELMOUTH} (Praat {PRAAT})"


def read_samples(path) -> np.ndarray:
    """Decode's audio (16 kHz mono 16-bit WAV) as float32 int16 / 32768, the bake-off's soundfile reading of it."""
    with wave.open(str(path)) as w:
        got = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        if got != (1, 2, SAMPLE_RATE):
            raise NodeError("input", f"the audio is {got[0]} channel(s), {8 * got[1]}-bit, {got[2]} Hz; "
                                     f"decode writes 1 channel, 16-bit, {SAMPLE_RATE} Hz")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


with Node() as node:
    transcript, speakers = node.input_json("transcript"), node.input_json("speakers")
    stage = (transcript.get("producer") or {}).get("stage") or ""
    if "diarize" not in stage:
        raise NodeError("input", f"the transcript comes from stage {stage!r}, not a diarization: prosody reads the diarize "
                                 "node's transcript, whose utterances the speakers layer labels")
    samples = read_samples(node.input("audio"))
    try:
        units = make_units(transcript, speakers)
    except ValueError as e:
        raise NodeError("input", str(e)) from None
    if versions() != (PARSELMOUTH, PRAAT):
        raise NodeError("internal", f"Parselmouth {versions()} is not the locked {PARSELMOUTH} (Praat {PRAAT})")

    t0 = time.time()
    measured = measure(units, transcript["words"], len(samples), Praat(samples, SAMPLE_RATE))
    measure_s = round(time.time() - t0, 2)

    device = odyssey.device_for(node.accelerator)
    spans = [odyssey.clip_span(u) for u in units]
    t0 = time.time()
    model, config = odyssey.load(device)
    load_s = round(time.time() - t0, 2)
    t0 = time.time()
    raw_emotion = odyssey.run(model, config, [samples[a:b] for a, b in spans], device)
    emotion_s = round(time.time() - t0, 2)
    labels = odyssey.labels(config)

    node.model(PARSELMOUTH_ID, licence_mode="needs-review", licence=PRAAT_LICENCE)
    node.model(odyssey.REPO, licence_mode="needs-review", source=f"hf://{odyssey.REPO}@{odyssey.REVISION}",
               revision=odyssey.REVISION, licence=odyssey.LICENCE)
    node.model(odyssey.WAVLM, licence_mode="commercial-ok", source=f"hf://{odyssey.WAVLM}@{odyssey.WAVLM_REVISION}/config.json",
               revision=odyssey.WAVLM_REVISION, licence="MIT (Microsoft); only its config.json: the weights are Odyssey's")

    rows = prosody_tracks.write(node.path("tracks", "prosody_tracks.parquet"), measured["tracks"], units)
    node.write_json("speakers", "prosody_speakers.json", prosody_layer.speakers_side(measured, SETTINGS, PARSELMOUTH, PRAAT))
    node.write_json("emotion", "prosody_emotion.json", prosody_layer.emotion_side(
        units, spans, labels, raw_emotion, odyssey.MODEL_ID, config, odyssey.EPSILON, odyssey.MIN_SAMPLES, odyssey.RATE))
    timings = {"measure_s": measure_s, "emotion_load_s": load_s, "emotion_s": emotion_s}
    node.write_json("prosody", "prosody.json", prosody_layer.layer(
        units, measured, labels, raw_emotion, {"model": odyssey.MODEL_ID, "licence_mode": "needs-review"},
        node.producer(timings=timings, device=str(device)),
        prosody_layer.pointers(prosody_tracks.FILE, prosody_tracks.DESCRIPTION)))
    audio_s = len(samples) / SAMPLE_RATE
    node.usage(audio_seconds=round(audio_s, 1), utterances=len(units), speakers=len(measured["speakers"]),
               track_rows=rows, emotion_skipped=sum(e is None for e in raw_emotion),
               speaker_sources={s: sum(u["speaker_source"] == s for u in units)
                                for s in sorted({u["speaker_source"] for u in units})},
               realtime_factor=round((measure_s + emotion_s) / audio_s, 3) if audio_s else None)
