"""Prosody stage of the bake-off: how each utterance is said, at three levels.

Utterances are the annotator's: label_page.utterances() splits Parakeet's words at pauses over 0.25 s,
sentence ends and cuts, and is imported here (not copied), so every record joins the annotator's
utterances by the same id, start and end.

1. Measured prosody (Parselmouth, i.e. Praat 6.1.38; GPL-3). Pitch is Praat's autocorrelation tracker
   (To Pitch (ac), standard settings) run twice: once at 75-600 Hz to find each speaker's register, then
   with a per-speaker floor and ceiling of 0.72 x q35 and 1.9 x q65 of the first pass (De Looze & Rauzy,
   2009), which is what keeps octave jumps out. Per word: F0 mean/min/max in semitones relative to that
   speaker's median, intensity, duration, the pause before it and a prominence score. Per utterance:
   speech and articulation rate, pitch range, the final pitch movement, pauses and voice quality (jitter,
   shimmer, HNR). Plus the whole clip's pitch contour at 20 Hz for a timeline row.
   The speaker of an utterance is the hand label where there is one (speaker_labels.json, a person or
   "off-screen"), else the diarized voice's LR-ASD link (speaker_links_lrasd.json), else the diarized
   voice itself; each utterance records which. Words take their utterance's speaker.
2. Speech emotion per utterance (the utterance's own audio, no context):
   audEERING wav2vec2-large-robust-12-ft-emotion-msp-dim (arousal, dominance, valence, about 0..1; the
   model card's regression head, which reproduces the card's dummy-signal output exactly), the Odyssey
   2024 challenge baseline 3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes (WavLM-large + attentive
   statistics pooling; arousal, dominance, valence, about 0..1; stored apart as dimensional_odyssey), and
   emotion2vec+ large via FunASR (9 categories with scores).
   The Odyssey card loads through trust_remote_code, but its pipeline_utils.py imports PretrainedConfig
   from transformers.modeling_utils, which transformers 5 no longer exports. So its classes are copied
   below (MIT; same layers and forward pass) and the checkpoint is loaded into them strictly (every
   tensor present, none left over); input is normalised with the card's config mean/std as the card does.
3. Style captions per utterance, from the utterance with CONTEXT_S of audio on each side:
   - gpt: gpt-audio-1.5 through the OpenAI API (the only family on this key that accepts audio input;
     gpt-6.1-sol and gpt-5.5 reject it). It writes a short description and picks tags from STYLE_TAGS,
     returned through a strict function call (the audio models do not take json_schema).
   - qwen: Qwen3-Omni-30B-A3B-Captioner, run locally in bf16 with transformers. It takes audio only, no
     prompt, so it cannot be told which part is the target or asked for tags. It describes the whole
     snippet, context included.
   Both free-text descriptions are then mapped onto STYLE_TAGS by the same text-only model (TAGGER), so
   the two sources can be compared tag by tag without the tag-picking step differing between them.

Only per-utterance snippets are sent to OpenAI, never the clip. API keys come from
~/.config/hyperstudy/bakeoff.env and are never printed.

Each part merges into runs/<name>/prosody.json, so parts can run one per process (the Qwen captioner
needs ~65 GB and must not share the GPU with anything else).

Usage (from prosody/): uv run python prosody.py ../runs/<name> [--device mps|cpu] [--captions gpt|qwen|none]
           [--parts measure,emotion,captions] [--emotion-models dimensional_odyssey (default); also dimensional,categorical]
           [--verify-cpu 6]
       uv run python prosody.py ../runs/<name> --parts emotion --emotion-models dimensional_odyssey
           (adds one emotion model; the other models' outputs, measure and captions are kept as stored)
       uv run python prosody.py ../runs/<name> --parts captions --captions qwen --device cpu --verify-cpu 3
           (re-captions 3 utterances on CPU and stores the comparison with the MPS captions)
"""

import argparse
import base64
import io
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BAKEOFF = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BAKEOFF))
from extract import load_env  # noqa: E402
from label_page import utterances  # noqa: E402  the annotator's utterance units

SR = 16000                     # audio.wav is 16 kHz mono, which all models here expect

# --- measured prosody ----------------------------------------------------------
PITCH_STEP = 0.01              # Praat's default for these pitch ranges
FIRST_PASS = (75.0, 600.0)     # Praat's defaults: wide enough for any adult voice
RANGE_FACTORS = (0.72, 1.9)    # floor = 0.72 x q35, ceiling = 1.9 x q65 of the first pass (De Looze & Rauzy 2009)
MIN_VOICED_FRAMES = 50         # 0.5 s of voiced speech before a speaker gets their own pitch range
INTENSITY_MIN_PITCH = 75.0     # one intensity analysis for all speakers, so levels are comparable
CONTOUR_STEP = 0.05            # 20 Hz contour for the timeline
FINAL_WINDOW = 0.3             # final pitch movement: the last 300 ms of voiced frames in the utterance
FINAL_MIN_FRAMES = 6
FINAL_THRESHOLD_ST = 1.5       # fitted change over the window that counts as a rise or a fall
# Parakeet stretches a sentence-final word over the silence or music after it (NES: "now." lasts 49 s; 99th
# percentile of OOO words is 0.88 s). Every analysis (pitch, rates, emotion, captions) stops a word this long
# after its onset; the word keeps its transcript times and is flagged.
WORD_MAX_S = 1.2
PAUSE_MIN = 0.1                # gaps between words shorter than this are not counted as pauses
MIN_PROMINENCE_WORDS = 10      # fewer words than this: z-scores against the whole clip instead of the speaker
OCTAVE_ST = 12.0

# --- speech emotion ------------------------------------------------------------
DIMENSIONAL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
CATEGORICAL = "emotion2vec/emotion2vec_plus_large"
ODYSSEY = "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes"
ODYSSEY_REVISION = "00d0e12ba9bf957f5aeea36e8663c8c61cb50ac9"  # main on 2026-10-23 (last change 2024-06-12)
WAVLM_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"    # microsoft/wavlm-large, the card's ssl_type
# One-off, from a scratch env: the card's own trust_remote_code under transformers 4.57.6 (CPU) against load_odyssey
ODYSSEY_CARD_CHECK = ("card code under transformers 4.57.6 vs this loader under 5.20.2, CPU, OOO utterances "
                      "0,17,34,51,68,85: max abs diff 2.2e-06")
EMOTION_MODELS = ("dimensional", "dimensional_odyssey", "categorical")
# Run by default: the Odyssey baseline alone. On OOO it agreed with audEERING (arousal r 0.88, valence 0.63) and the
# user judged them equivalent in the annotator, so the more permissive one is kept (MIT weights vs audEERING's
# CC-BY-NC-SA; Odyssey's MSP-Podcast training data is still academic-licensed). emotion2vec+ categories were judged
# unusable in review (2026-10-23). Both stay available through --emotion-models.
DEFAULT_EMOTION_MODELS = ("dimensional_odyssey",)
DEFAULT_DIMENSIONAL = "dimensional_odyssey"   # recorded in prosody.json; the annotator draws this one

# --- style captions ------------------------------------------------------------
CONTEXT_S = 1.0
GPT_MODEL = "gpt-audio-1.5"
TAGGER = "gpt-5.4-mini"
QWEN = "Qwen/Qwen3-Omni-30B-A3B-Captioner"
# The captioner writes ~400-450 tokens per snippet (delivery first, then room acoustics, accent and a summary) at
# ~9.5 tokens/s on MPS: ~65 min for the 86 OOO utterances. 200 tokens keeps the delivery paragraph and ~32 min.
QWEN_MAX_NEW_TOKENS = 200
API_WORKERS = 8
# USD per 1M tokens, from developers.openai.com/api/docs/models (checked 2026-10-23).
PRICES = {GPT_MODEL: {"text_in": 2.50, "audio_in": 32.00, "out": 10.00},
          TAGGER: {"text_in": 0.75, "audio_in": None, "out": 4.50}}

# Fixed vocabulary, after SpeechCraft's (pitch, speed, energy, emotion) and ParaSpeechCaps' (situational and
# voice-quality tags) attribute sets, plus the conversational attitudes and laughter this footage needs.
STYLE_TAGS = {
    "pitch": ["high-pitch", "low-pitch", "wide-pitch-range", "monotone", "rising-end", "falling-end"],
    "pace": ["fast", "slow", "rushed", "drawn-out"],
    "loudness": ["loud", "soft", "whispered", "shouting"],
    "emphasis": ["emphatic", "stressed-words"],
    "voice_quality": ["breathy", "creaky", "tense", "nasal", "husky", "clear", "shaky"],
    "emotion": ["neutral", "happy", "amused", "excited", "calm", "sad", "angry", "annoyed", "surprised",
                "fearful", "anxious", "confused", "bored", "confident"],
    "attitude": ["sarcastic", "teasing", "playful", "sincere", "questioning", "assertive", "doubtful",
                 "enthusiastic"],
    "hesitation": ["hesitant", "filled-pause", "stammering", "trailing-off"],
    "laughter": ["laughing-while-speaking", "laughter-nearby", "smiling-voice"],
    "delivery": ["conversational", "narration", "dramatic", "enunciated", "mumbled"],
}
ALL_TAGS = [t for group in STYLE_TAGS.values() for t in group]
TAG_GLOSSARY = ("rising-end/falling-end: final pitch movement of the target. stressed-words: one or more words "
                "clearly stressed. laughter-nearby: laughter just before/after the target or by other people "
                "during it. smiling-voice: audible smile without laughter. narration: voice-over or storytelling "
                "delivery.")

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string", "enum": ALL_TAGS}},
        "emphasized_words": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["description", "tags", "emphasized_words"],
    "additionalProperties": False,
}
TAG_SCHEMA = {
    "type": "object",
    "properties": {"tags": {"type": "array", "items": {"type": "string", "enum": ALL_TAGS}}},
    "required": ["tags"],
    "additionalProperties": False,
}

GPT_INSTRUCTIONS = """You are an expert in speech prosody and paralinguistics. You will hear a short audio \
snippet from a video. It contains one target utterance with up to 1 s of surrounding audio before and after it.

Describe HOW the target utterance is said, judged from the voice, not from the words alone: pitch (level, \
range, the movement at the end), pace, loudness, emphasis (which words are stressed), voice quality, emotion, \
attitude (e.g. sarcasm, teasing, sincerity), hesitation, and laughter in or around the speech. Mention the \
context only where it matters (e.g. others laughing). Do not invent what you cannot hear; if the target is \
too short or masked to judge something, say so.

Return through the function: a 1-3 sentence description, the tags from the fixed list that clearly apply \
(only those you can hear; "neutral" only if no emotion is audible), and the words of the target that are \
audibly emphasized (copied from the transcript; empty if none). Tag notes: """ + TAG_GLOSSARY

TAGGER_INSTRUCTIONS = """You map a free-text description of how someone speaks onto a fixed tag list. \
Choose only tags the description states or clearly implies about the main/target speech. Do not infer \
from your own guesses about the situation. If the description covers several speakers or sounds, tag what \
it says about the speech, and use laughter-nearby for laughter by others or around the speech. Tag notes: """ \
    + TAG_GLOSSARY

LICENCES = {
    "measure": {"model": f"Parselmouth {{version}} (Praat {{praat}})", "licence": "GPL-3.0 (Parselmouth and Praat)",
                "licence_mode": "needs-review",
                "note": "GPL-3 code, used as a separate tool; outputs are not covered, linking it into "
                        "distributed software would be."},
    "dimensional": {"model": DIMENSIONAL, "licence": "CC-BY-NC-SA-4.0 weights (audEERING; research use only)",
                    "licence_mode": "academic-only"},
    "dimensional_odyssey": {"model": f"{ODYSSEY}@{ODYSSEY_REVISION[:8]}",
                            "licence": "MIT (code and weights, MSP-UTD / 3loi); WavLM-large backbone MIT (Microsoft)",
                            "licence_mode": "needs-review",
                            "note": "Trained on MSP-Podcast, whose data licence is academic-only; whether that "
                                    "carries over to MIT-licensed weights is unresolved."},
    "categorical": {"model": CATEGORICAL,
                    "licence": "FunASR Model Open Source License v1.1 (weights; use, modify and share with "
                               "attribution, says 'for reference and learning purposes only'); FunASR code MIT",
                    "licence_mode": "needs-review"},
    "gpt": {"model": GPT_MODEL, "licence": "OpenAI API terms (outputs owned by the user; audio sent to OpenAI)",
            "licence_mode": "commercial-ok"},
    "qwen": {"model": QWEN, "licence": "Apache-2.0 (weights and code)", "licence_mode": "commercial-ok"},
    "tagger": {"model": TAGGER, "licence": "OpenAI API terms (text only)", "licence_mode": "commercial-ok"},
}


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def rounded(x, n=2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


# --- inputs --------------------------------------------------------------------

def load_units(run_dir):
    """The annotator's utterances, each with its words, voice, speaker and the speaker's source."""
    def load(name):
        path = run_dir / name
        return json.loads(path.read_text()) if path.exists() else None

    extract, transcript = load("extract.json"), load("transcript_parakeet.json")
    labels, links = load("speaker_labels.json"), load("speaker_links_lrasd.json")
    words = [w for segment in transcript["segments"] for w in segment["words"]]
    units = utterances(transcript, extract["shots"])

    if labels and len(labels["units"]) != len(units):
        sys.exit("speaker_labels.json has a different number of utterances than label_page.utterances()")
    labelled = {u["id"]: u for u in (labels or {"units": []})["units"]}

    i = 0
    for unit in units:
        # units partition the word list in order, so take the next n words and check they are this unit
        n = len(unit["text"].split(" "))
        unit_words = words[i:i + n]
        if (" ".join(w["word"] for w in unit_words) != unit["text"] or unit_words[0]["start"] != unit["start"]
                or unit_words[-1]["end"] != unit["end"]):
            sys.exit(f"could not recover the words of utterance {unit['id']}")
        unit["word_index"] = list(range(i, i + n))
        unit["analysed_end"] = round(word_end(unit_words[-1]), 3)
        i += n
        # the annotator's rule: the diarized voice of most of the unit's words
        voice = Counter(w.get("speaker") for w in unit_words).most_common(1)[0][0]
        unit["voice"] = voice

        hand = labelled.get(unit["id"])
        if hand and (hand["start"], hand["text"]) != (unit["start"], unit["text"]):
            sys.exit(f"speaker_labels.json unit {unit['id']} does not match this transcript")
        label = hand["label"] if hand else None
        if label and label["kind"] == "face" and label.get("person") not in (None, "Unknown"):
            unit["speaker"], unit["speaker_source"] = label["person"], "hand_label"
        elif label and label["kind"] == "offscreen":
            # the labeller heard an off-screen voice but could not name it: keep voices apart by diarization
            unit["speaker"] = f"off-screen {label.get('voice') or voice}"
            unit["speaker_source"] = "hand_label_offscreen+diarized_voice"
        elif links and voice in links["links"]:
            unit["speaker"], unit["speaker_source"] = links["links"][voice]["person"], "lrasd_link"
        else:
            unit["speaker"], unit["speaker_source"] = voice, "diarized_voice"
    return extract, words, units, bool(labels), bool(links)


# --- measured prosody ----------------------------------------------------------

def syllables(word):
    """Orthographic syllable estimate: vowel groups, a silent final e dropped. Numbers count as 2."""
    w = re.sub(r"[^a-z0-9']", "", word.lower())
    if any(c.isdigit() for c in w):
        return 2
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups) - (1 if w.endswith("e") and not w.endswith(("le", "ee")) and len(groups) > 1 else 0)
    return max(1, n)


def pitch_frames(sound, floor, ceiling):
    pitch = sound.to_pitch_ac(time_step=PITCH_STEP, pitch_floor=floor, pitch_ceiling=ceiling)
    f0 = pitch.selected_array["frequency"].copy()
    f0[f0 == 0] = np.nan
    return pitch.xs(), f0


def word_end(w):
    """End of the part of a word that is analysed (see WORD_MAX_S)."""
    return min(w["end"], w["start"] + WORD_MAX_S)


def in_span(t, start, end):
    return (t >= start) & (t <= end)


def octave_jump_share(t, f0, spans):
    """Share of 10 ms steps between two voiced frames inside the spans that jump more than an octave."""
    jumps = total = 0
    for start, end in spans:
        st = 12 * np.log2(f0[in_span(t, start, end)])
        d = np.abs(np.diff(st))
        d = d[np.isfinite(d)]
        jumps += int((d > OCTAVE_ST).sum())
        total += len(d)
    return {"pairs": total, "share_gt_12st": rounded(jumps / total if total else np.nan, 4), "n_gt_12st": jumps}


def zscores(values):
    v = np.array([np.nan if x is None else x for x in values], float)
    sd = np.nanstd(v)
    return (v - np.nanmean(v)) / sd if sd > 0 else np.full_like(v, np.nan)


def measure(run_dir, wave, words, units):
    import parselmouth
    from parselmouth.praat import call

    t_start = time.time()
    sound = parselmouth.Sound(wave.astype(np.float64), sampling_frequency=SR)
    speakers = sorted({u["speaker"] for u in units})
    spans = {s: [(w["start"], word_end(w)) for u in units if u["speaker"] == s for w in (words[i] for i in u["word_index"])]
             for s in speakers}

    # pass 1: Praat's default range, to find each speaker's register
    t1, f1 = pitch_frames(sound, *FIRST_PASS)
    ranges, tracks, info = {}, {}, {}
    for s in speakers:
        mask = np.zeros_like(t1, bool)
        for a, b in spans[s]:
            mask |= in_span(t1, a, b)
        voiced = f1[mask & np.isfinite(f1)]
        if len(voiced) >= MIN_VOICED_FRAMES:
            q35, q65 = np.percentile(voiced, [35, 65])
            ranges[s] = (round(RANGE_FACTORS[0] * q35, 1), round(RANGE_FACTORS[1] * q65, 1)), "speaker"
        else:
            ranges[s] = FIRST_PASS, "default (too little voiced speech)"
    # pass 2: per-speaker range
    for s in speakers:
        tracks[s] = pitch_frames(sound, *ranges[s][0])
        t, f0 = tracks[s]
        mask = np.zeros_like(t, bool)
        for a, b in spans[s]:
            mask |= in_span(t, a, b)
        voiced = f0[mask & np.isfinite(f0)]
        info[s] = {"floor_hz": ranges[s][0][0], "ceiling_hz": ranges[s][0][1], "range_source": ranges[s][1],
                   "median_hz": rounded(np.median(voiced) if len(voiced) else np.nan, 1),
                   "voiced_s": round(len(voiced) * PITCH_STEP, 2),
                   "utterances": sum(u["speaker"] == s for u in units),
                   "octave_jumps": octave_jump_share(t, f0, spans[s]),
                   "octave_jumps_first_pass": octave_jump_share(t1, f1, spans[s])}

    intensity = sound.to_intensity(minimum_pitch=INTENSITY_MIN_PITCH, time_step=PITCH_STEP)
    ti, db = intensity.xs(), intensity.values[0]

    def st(f, s):
        return 12 * np.log2(f / info[s]["median_hz"])

    # words
    word_rows = {}
    for u in units:
        s = u["speaker"]
        t, f0 = tracks[s]
        for i in u["word_index"]:
            w = words[i]
            seg = f0[in_span(t, w["start"], word_end(w))]
            voiced = seg[np.isfinite(seg)]
            dbs = db[in_span(ti, w["start"], word_end(w))]
            level = 10 * np.log10(np.mean(10 ** (dbs / 10))) if len(dbs) else np.nan  # energy mean, as Praat
            word_rows[i] = {
                "i": i, "word": w["word"], "start": w["start"], "end": w["end"],
                "duration": round(word_end(w) - w["start"], 3),
                "timing_clamped": w["end"] > word_end(w),
                "pause_before": round(w["start"] - word_end(words[i - 1]), 3) if i > 0 else None,
                "syllables": syllables(w["word"]),
                "voiced_frac": round(len(voiced) / len(seg), 2) if len(seg) else 0.0,
                "f0_mean_st": rounded(st(np.mean(voiced), s)) if len(voiced) else None,
                "f0_min_st": rounded(st(np.min(voiced), s)) if len(voiced) else None,
                "f0_max_st": rounded(st(np.max(voiced), s)) if len(voiced) else None,
                "intensity_db": rounded(level, 1),
            }
    # speaker-relative intensity, and prominence = mean z of F0 peak, loudness and duration per syllable
    speaker_words = {s: [i for u in units if u["speaker"] == s for i in u["word_index"]] for s in speakers}
    for s, idx in speaker_words.items():
        med = np.nanmedian([word_rows[i]["intensity_db"] or np.nan for i in idx])
        for i in idx:
            word_rows[i]["intensity_rel_db"] = rounded(word_rows[i]["intensity_db"] - med, 1) \
                if word_rows[i]["intensity_db"] is not None else None
    for s, idx in speaker_words.items():
        pool = idx if len(idx) >= MIN_PROMINENCE_WORDS else sorted(word_rows)
        z = {"f0": zscores([word_rows[i]["f0_max_st"] for i in pool]),
             "intensity": zscores([word_rows[i]["intensity_rel_db"] for i in pool]),
             "duration": zscores([np.log(word_rows[i]["duration"] / word_rows[i]["syllables"]) for i in pool])}
        at = {i: k for k, i in enumerate(pool)}
        for i in idx:
            parts = {k: rounded(v[at[i]]) for k, v in z.items()}
            got = [v for v in parts.values() if v is not None]
            word_rows[i]["prominence"] = rounded(np.mean(got)) if got else None
            word_rows[i]["prominence_parts"] = parts
            word_rows[i]["prominence_norm"] = "speaker" if pool is idx else "clip (speaker has few words)"

    # utterances
    out_units = []
    for k, u in enumerate(units):
        s = u["speaker"]
        t, f0 = tracks[s]
        ws = [word_rows[i] for i in u["word_index"]]
        mask = np.zeros_like(t, bool)
        for w in ws:
            mask |= in_span(t, w["start"], word_end(w))
        tv, fv = t[mask & np.isfinite(f0)], f0[mask & np.isfinite(f0)]
        sv = st(fv, s)
        syl = sum(w["syllables"] for w in ws)
        dur = u["analysed_end"] - u["start"]
        gaps = [w["pause_before"] for w in ws[1:]]
        final = {"final_slope_st_s": None, "final_delta_st": None, "final_movement": None, "final_window_s": None}
        if len(tv):
            sel = tv >= tv[-1] - FINAL_WINDOW
            if sel.sum() >= FINAL_MIN_FRAMES:
                slope, _ = np.polyfit(tv[sel], sv[sel], 1)
                span = tv[sel][-1] - tv[sel][0]
                delta = slope * span
                final = {"final_slope_st_s": rounded(slope, 1), "final_delta_st": rounded(delta),
                         "final_movement": "rise" if delta >= FINAL_THRESHOLD_ST else
                         "fall" if delta <= -FINAL_THRESHOLD_ST else "level",
                         "final_window_s": rounded(span)}
        part = sound.extract_part(from_time=u["start"], to_time=u["analysed_end"], preserve_times=True)
        floor, ceiling = info[s]["floor_hz"], info[s]["ceiling_hz"]
        points = call(part, "To PointProcess (periodic, cc)", floor, ceiling)
        n_periods = call(points, "Get number of periods", 0, 0, 0.0001, 0.02, 1.3)
        jitter = call(points, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) if n_periods >= 3 else np.nan
        shimmer = call([part, points], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) if n_periods >= 3 else np.nan
        hnr = call(part.to_harmonicity_cc(time_step=PITCH_STEP, minimum_pitch=floor), "Get mean", 0, 0)
        dbs = db[in_span(ti, u["start"], u["analysed_end"])]
        out_units.append({
            "id": u["id"], "start": u["start"], "end": u["end"], "analysed_end": u["analysed_end"], "text": u["text"],
            "shot": u["shot"],
            "speaker": s, "speaker_source": u["speaker_source"], "voice": u["voice"],
            "prosody": {
                "duration": round(dur, 3), "n_words": len(ws), "syllables": syl,
                "speech_rate_syl_s": rounded(syl / dur) if dur > 0 else None,
                "articulation_rate_syl_s": rounded(syl / sum(w["duration"] for w in ws)),
                "voiced_frac": rounded(len(tv) / mask.sum()) if mask.sum() else 0.0,
                "f0_median_hz": rounded(np.median(fv), 1) if len(fv) else None,
                "f0_median_st": rounded(np.median(sv)) if len(sv) else None,
                "f0_min_st": rounded(np.min(sv)) if len(sv) else None,
                "f0_max_st": rounded(np.max(sv)) if len(sv) else None,
                "f0_range_st": rounded(np.percentile(sv, 90) - np.percentile(sv, 10)) if len(sv) >= 5 else None,
                **final,
                "intensity_db": rounded(10 * np.log10(np.mean(10 ** (dbs / 10))), 1) if len(dbs) else None,
                "pause_before_s": round(u["start"] - units[k - 1]["analysed_end"], 3) if k > 0 else None,
                "pause_after_s": round(units[k + 1]["start"] - u["analysed_end"], 3) if k + 1 < len(units) else None,
                "internal_pauses": sum(g >= PAUSE_MIN for g in gaps),
                "internal_pause_s": round(sum(g for g in gaps if g >= PAUSE_MIN), 3),
                "max_internal_gap_s": round(max(gaps), 3) if gaps else None,
                "jitter_local": rounded(jitter, 4), "shimmer_local": rounded(shimmer, 4),
                "hnr_db": rounded(hnr, 1) if hnr > -199 else None, "n_periods": int(n_periods),
            },
            "words": ws,
        })

    # whole-clip contour at 20 Hz: each bin takes the pitch track of the speaker whose utterance covers it
    n_bins = int(np.ceil(len(wave) / SR / CONTOUR_STEP))
    hz, semis, who = [None] * n_bins, [None] * n_bins, [None] * n_bins
    index = {s: k for k, s in enumerate(speakers)}
    for u in units:
        t, f0 = tracks[u["speaker"]]
        for b in range(int(u["start"] / CONTOUR_STEP), min(n_bins, int(np.ceil(u["analysed_end"] / CONTOUR_STEP)))):
            a = b * CONTOUR_STEP
            seg = f0[(t >= max(a, u["start"])) & (t < min(a + CONTOUR_STEP, u["analysed_end"]))]
            seg = seg[np.isfinite(seg)]
            who[b] = index[u["speaker"]]
            if len(seg):
                hz[b] = rounded(np.median(seg), 1)
                semis[b] = rounded(st(np.median(seg), u["speaker"]))

    import parselmouth as pm
    lic = dict(LICENCES["measure"])
    lic["model"] = lic["model"].format(version=pm.__version__, praat=pm.PRAAT_VERSION)
    return {
        **lic, "pitch": "To Pitch (ac), standard settings, two passes (per-speaker floor/ceiling)",
        "settings": {"pitch_step_s": PITCH_STEP, "first_pass_hz": FIRST_PASS, "range_factors": RANGE_FACTORS,
                     "final_window_s": FINAL_WINDOW, "final_threshold_st": FINAL_THRESHOLD_ST,
                     "pause_min_s": PAUSE_MIN, "intensity_min_pitch_hz": INTENSITY_MIN_PITCH,
                     "prominence": "mean z (within speaker) of word F0 max (st), intensity re speaker median, "
                                   "log duration per syllable; syllables estimated from spelling"},
        "speakers": {s: info[s] for s in speakers},
        "contour": {"step_s": CONTOUR_STEP, "t0": 0.0, "speakers": speakers, "f0_hz": hz, "f0_st": semis,
                    "speaker": who},
        "timings": {"device": "cpu", "total_s": round(time.time() - t_start, 2), "audio_s": round(len(wave) / SR, 2)},
    }, out_units


# --- speech emotion ------------------------------------------------------------

def load_dimensional(device):
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):  # from the model card
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features):
            x = torch.tanh(self.dense(self.dropout(features)))
            return self.out_proj(self.dropout(x))

    class EmotionModel(Wav2Vec2PreTrainedModel):  # from the model card
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return hidden, self.classifier(hidden)

    processor = Wav2Vec2Processor.from_pretrained(DIMENSIONAL)
    model = EmotionModel.from_pretrained(DIMENSIONAL).to(device).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]  # arousal, dominance, valence
    return model, processor, labels


@torch.no_grad()
def run_dimensional(loaded, clips, device):
    model, processor, labels = loaded
    out = []
    for clip in clips:
        x = processor(clip, sampling_rate=SR)["input_values"][0].reshape(1, -1)
        y = model(torch.from_numpy(x).to(device))[1][0].cpu().numpy()
        out.append({k: round(float(v), 6) for k, v in zip(labels, y)})
    return out


def load_odyssey(device):
    """The Odyssey baseline: the card's pipeline_utils.py classes, with the same layers and forward pass.

    The card's SERModel.__init__ calls AutoModel.from_pretrained("microsoft/wavlm-large") and the checkpoint
    then overwrites every backbone tensor; here the backbone is built from that model's config instead, and
    the strict load below checks that the checkpoint does cover every tensor."""
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import WavLMConfig, WavLMModel

    class AttentiveStatisticsPooling(nn.Module):  # from the card's pipeline_utils.py
        def __init__(self, input_size):
            super().__init__()
            self._indim = input_size
            self.sap_linear = nn.Linear(input_size, input_size)
            self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))

        def compute_length_from_mask(self, mask):
            wav_lens = torch.sum(mask, dim=1)
            feat_lens = torch.div(wav_lens - 1, 16000 * 0.02, rounding_mode="floor") + 1
            return feat_lens.int().tolist()

        def forward(self, xs, mask):
            pooled_list = []
            for x, feat_len in zip(xs, self.compute_length_from_mask(mask)):
                x = x[:feat_len].unsqueeze(0)
                h = torch.tanh(self.sap_linear(x))
                w = torch.matmul(h, self.attention).squeeze(dim=2)
                w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
                mu = torch.sum(x * w, dim=1)
                rh = torch.sqrt((torch.sum((x ** 2) * w, dim=1) - mu ** 2).clamp(min=1e-5))
                pooled_list.append(torch.cat((mu, rh), 1).squeeze(0))
            return torch.stack(pooled_list)

    class EmotionRegression(nn.Module):  # from the card's pipeline_utils.py
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.5):
            super().__init__()
            self.fc = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                                   nn.ReLU(), nn.Dropout(dropout))])
            for _ in range(num_layers - 1):
                self.fc.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                             nn.ReLU(), nn.Dropout(dropout)))
            self.out = nn.Sequential(nn.Linear(hidden_dim, output_dim))
            self.inp_drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.inp_drop(x)
            for fc in self.fc:
                h = fc(h)
            return self.out(h)

    class SERModel(nn.Module):  # from the card's pipeline_utils.py
        def __init__(self, config):
            super().__init__()
            self.ssl_model = WavLMModel(WavLMConfig.from_pretrained(config["ssl_type"], revision=WAVLM_REVISION))
            self.pool_model = AttentiveStatisticsPooling(config["hidden_size"])
            self.ser_model = EmotionRegression(config["hidden_size"] * 2, config["hidden_size"],
                                               config["classifier_hidden_layers"], config["num_classes"],
                                               dropout=config["classifier_dropout_prob"])

        def forward(self, x, mask):
            ssl = self.ssl_model(x, attention_mask=mask).last_hidden_state
            return self.ser_model(self.pool_model(ssl, mask))

    config = json.loads(Path(hf_hub_download(ODYSSEY, "config.json", revision=ODYSSEY_REVISION)).read_text())
    model = SERModel(config)
    state = load_file(hf_hub_download(ODYSSEY, "model.safetensors", revision=ODYSSEY_REVISION))
    # the checkpoint was saved by transformers 4.34, whose weight-normed positional conv stored weight_g/weight_v;
    # transformers 5's WavLM keeps them as parametrizations (original0 = g, original1 = v)
    for old, new in (("weight_g", "parametrizations.weight.original0"), ("weight_v", "parametrizations.weight.original1")):
        key = f"ssl_model.encoder.pos_conv_embed.conv.{old}"
        state[key.replace(old, new)] = state.pop(key)
    model.load_state_dict(state, strict=True)
    labels = [config["id2label"][str(i)] for i in range(config["num_classes"])]  # arousal, dominance, valence
    return model.to(device).eval(), config["mean"], config["std"], labels


@torch.no_grad()
def run_odyssey(loaded, clips, device):
    model, mean, std, labels = loaded
    out = []
    for clip in clips:
        norm = (clip - mean) / (std + 0.000001)  # the card's normalisation, epsilon included
        x = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(device)
        y = model(x, torch.ones(1, len(norm), device=device))[0].cpu().numpy()
        out.append({k: round(float(v), 6) for k, v in zip(labels, y)})
    return out


def load_categorical(device):
    from funasr import AutoModel
    model = AutoModel(model=CATEGORICAL, hub="hf", disable_update=True, device=device.type, log_level="ERROR")
    actual = next(model.model.parameters()).device.type
    if actual != device.type:  # FunASR silently falls back to CPU when a device is unavailable
        sys.exit(f"emotion2vec+ loaded on {actual}, not {device.type}")
    return model


def run_categorical(model, clips):
    out = []
    for clip in clips:
        r = model.generate(clip, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
        names = [lab.split("/")[-1].strip("<>") for lab in r["labels"]]  # "开心/happy" -> happy, "<unk>" -> unk
        scores = {n: round(float(s), 6) for n, s in zip(names, r["scores"])}
        top = max(scores, key=scores.get)
        out.append({"top": top, "top_score": scores[top], "scores": scores})
    return out


def emotion(wave, units, device, verify_n, names=EMOTION_MODELS):
    clips = [wave[int(u["start"] * SR):int(u["analysed_end"] * SR)] for u in units]
    audio_s = sum(len(c) for c in clips) / SR
    results, timings, checks = {}, {}, {}
    check_idx = sorted(set(np.linspace(0, len(clips) - 1, verify_n).round().astype(int).tolist())) if verify_n else []

    models = {"dimensional": (load_dimensional, lambda m, c, d: run_dimensional(m, c, d)),
              "dimensional_odyssey": (load_odyssey, lambda m, c, d: run_odyssey(m, c, d)),
              "categorical": (load_categorical, lambda m, c, d: run_categorical(m, c))}
    for name in names:
        load, run = models[name]
        t0 = time.time()
        model = load(device)
        sync(device)
        load_s = time.time() - t0
        t0 = time.time()
        results[name] = run(model, clips, device)
        sync(device)
        infer_s = time.time() - t0
        timings[name] = {"device": device.type, "load_s": round(load_s, 2), "infer_s": round(infer_s, 2),
                         "utterances": len(clips), "audio_s": round(audio_s, 2), "rtf": round(infer_s / audio_s, 4)}
        if device.type == "mps":
            timings[name]["mps_driver_allocated_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        if check_idx and device.type != "cpu":
            cpu = torch.device("cpu")
            ref = run(load(cpu), [clips[i] for i in check_idx], cpu)
            if name != "categorical":
                diff = max(abs(ref[k][d] - results[name][i][d]) for k, i in enumerate(check_idx) for d in ref[k])
                checks[name] = {"utterances": [units[i]["id"] for i in check_idx], "max_abs_diff": float(f"{diff:.2g}")}
            else:
                diff = max(abs(ref[k]["scores"][c] - results[name][i]["scores"][c])
                           for k, i in enumerate(check_idx) for c in ref[k]["scores"])
                same = sum(ref[k]["top"] == results[name][i]["top"] for k, i in enumerate(check_idx))
                checks[name] = {"utterances": [units[i]["id"] for i in check_idx], "max_abs_diff": float(f"{diff:.2g}"),
                                "top_agree": f"{same}/{len(check_idx)}"}
        print(f"{name}: load {timings[name]['load_s']} s, infer {timings[name]['infer_s']} s"
              + (f", vs CPU {checks[name]}" if name in checks else ""))
    return results, timings, checks


# --- style captions ------------------------------------------------------------

def snippet(wave, unit):
    a, b = max(0.0, unit["start"] - CONTEXT_S), min(len(wave) / SR, unit["analysed_end"] + CONTEXT_S)
    return wave[int(a * SR):int(b * SR)], round(unit["start"] - a, 2), round(unit["analysed_end"] - a, 2)


def wav_b64(clip):
    buf = io.BytesIO()
    sf.write(buf, clip, SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def cost(model, usage):
    p, d = PRICES[model], usage.prompt_tokens_details
    audio_in = (d.audio_tokens or 0) if d else 0
    text_in = usage.prompt_tokens - audio_in
    usd = (text_in * p["text_in"] + audio_in * (p["audio_in"] or 0) + usage.completion_tokens * p["out"]) / 1e6
    return {"text_in": text_in, "audio_in": audio_in, "out": usage.completion_tokens, "usd": usd}


def gpt_caption(client, wave, units, k):
    u = units[k]
    clip, a, b = snippet(wave, u)
    prev_text = units[k - 1]["text"] if k > 0 else "(none)"
    next_text = units[k + 1]["text"] if k + 1 < len(units) else "(none)"
    text = (f"Snippet length {len(clip) / SR:.2f} s. The target utterance runs from {a:.2f} s to {b:.2f} s "
            f"of the snippet.\nTarget transcript (automatic, may contain errors): \"{u['text']}\"\n"
            f"Previous line in the conversation: \"{prev_text}\"\nNext line: \"{next_text}\"")
    t0 = time.time()
    r = client.chat.completions.create(
        model=GPT_MODEL, modalities=["text"],
        tools=[{"type": "function", "function": {"name": "record_style", "description": "Record how the target is said",
                                                 "parameters": CAPTION_SCHEMA, "strict": True}}],
        tool_choice={"type": "function", "function": {"name": "record_style"}},
        messages=[{"role": "system", "content": GPT_INSTRUCTIONS},
                  {"role": "user", "content": [{"type": "text", "text": text},
                                               {"type": "input_audio", "input_audio": {"data": wav_b64(clip), "format": "wav"}}]}])
    out = json.loads(r.choices[0].message.tool_calls[0].function.arguments)
    return {**out, "snippet": [round(u["start"] - a, 2), round(u["start"] - a + len(clip) / SR, 2)],
            "seconds": round(time.time() - t0, 2), "usage": cost(GPT_MODEL, r.usage)}


def tag_text(client, description):
    r = client.chat.completions.create(
        model=TAGGER,
        response_format={"type": "json_schema", "json_schema": {"name": "tags", "schema": TAG_SCHEMA, "strict": True}},
        messages=[{"role": "system", "content": TAGGER_INSTRUCTIONS},
                  {"role": "user", "content": f"Description:\n{description}"}])
    return json.loads(r.choices[0].message.content)["tags"], cost(TAGGER, r.usage)


def load_qwen(device):
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN)
    # Load on CPU, then move. transformers 5.20.2 fuses the checkpoint's 128 per-expert matrices into 3D
    # gate_up_proj/down_proj tensors while loading; with device_map="mps" every fused tensor comes out all
    # zeros (router and attention load fine), so every MoE block outputs garbage and the captions are noise.
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(QWEN, dtype=torch.bfloat16, device_map="cpu")
    model = model.to(device).eval()
    experts = model.thinker.model.layers[0].mlp.experts
    if not experts.gate_up_proj.abs().amax() > 0:
        sys.exit("Qwen3-Omni expert weights loaded as zeros")
    return model, processor


@torch.no_grad()
def qwen_caption(loaded, clip, device):
    model, processor = loaded
    conversation = [{"role": "user", "content": [{"type": "audio", "audio": "snippet"}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=[clip], sampling_rate=SR, return_tensors="pt", padding=True)
    inputs = inputs.to(device).to(model.dtype)
    t0 = time.time()
    # greedy, so MPS and CPU runs can be compared and reruns give the same caption
    out = model.generate(**inputs, thinker_max_new_tokens=QWEN_MAX_NEW_TOKENS, thinker_do_sample=False,
                         return_audio=False)
    ids = out[0] if isinstance(out, tuple) else out
    ids = getattr(ids, "sequences", ids)
    new = ids[:, inputs["input_ids"].shape[1]:]
    sync(device)
    caption = processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return {"description": caption.strip(), "tokens": int(new.shape[1]), "seconds": round(time.time() - t0, 2)}


def qwen_cpu_check(wave, units, previous, verify_n):
    """Re-caption a few utterances on CPU and compare with the stored MPS captions.

    Its own process (--captions qwen --device cpu): a CPU copy of the 30B model is another ~60 GB, which must
    not sit next to the MPS copy, and a full CPU run would take hours (~1.8 tokens/s)."""
    idx = sorted(set(np.linspace(0, len(units) - 1, verify_n).round().astype(int).tolist()))
    cpu = torch.device("cpu")
    loaded = load_qwen(cpu)
    rows = []
    for i in idx:
        ref = qwen_caption(loaded, snippet(wave, units[i])[0], cpu)
        mps = previous[i]["description"]
        common = next((k for k, (a, b) in enumerate(zip(ref["description"], mps)) if a != b),
                      min(len(ref["description"]), len(mps)))
        rows.append({"id": units[i]["id"], "identical": ref["description"] == mps, "common_prefix_chars": common,
                     "cpu": ref["description"], "cpu_seconds": ref["seconds"], "cpu_tokens": ref["tokens"]})
        print(f"  {units[i]['id']}: identical {rows[-1]['identical']}, common prefix {common} chars", flush=True)
    return rows


def captions(wave, units, source, device):
    from openai import OpenAI
    client = OpenAI()
    results, timing = [], {}
    if source == "gpt":
        t0 = time.time()
        with ThreadPoolExecutor(API_WORKERS) as pool:
            results = list(pool.map(lambda k: gpt_caption(client, wave, units, k), range(len(units))))
        timing = {"device": "OpenAI API", "wall_s": round(time.time() - t0, 1),
                  "sum_call_s": round(sum(r["seconds"] for r in results), 1), "workers": API_WORKERS}
    else:
        t0 = time.time()
        loaded = load_qwen(device)
        sync(device)
        timing = {"device": device.type, "load_s": round(time.time() - t0, 1)}
        if device.type == "mps":
            timing["mps_driver_allocated_gb_after_load"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
        t0 = time.time()
        for u in units:
            results.append(qwen_caption(loaded, snippet(wave, u)[0], device))
            print(f"  {u['id']} {results[-1]['seconds']} s {results[-1]['tokens']} tok: "
                  f"{results[-1]['description'][:100]!r}", flush=True)
        timing.update({"infer_s": round(time.time() - t0, 1), "tokens": sum(r["tokens"] for r in results)})
        if device.type == "mps":
            timing["mps_driver_allocated_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)

    # free text -> fixed tags, by one text model for both sources
    t0 = time.time()
    with ThreadPoolExecutor(API_WORKERS) as pool:
        tagged = list(pool.map(lambda r: tag_text(client, r["description"]), results))
    for r, (tags, usage) in zip(results, tagged):
        r["tags_from_text"] = tags
        r["tagger_usage"] = usage
    timing["tagger_wall_s"] = round(time.time() - t0, 1)
    usd = {"tagger": round(sum(r["tagger_usage"]["usd"] for r in results), 4)}
    if source == "gpt":
        usd[GPT_MODEL] = round(sum(r["usage"]["usd"] for r in results), 4)
    return results, timing, usd


# --- main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--captions", default="gpt", choices=["gpt", "qwen", "none"])
    parser.add_argument("--parts", default="measure,emotion,captions")
    parser.add_argument("--emotion-models", default=",".join(DEFAULT_EMOTION_MODELS),
                        help="emotion models to (re)run; the others keep their stored outputs")
    parser.add_argument("--verify-cpu", type=int, default=6, help="utterances re-run on CPU to check MPS output")
    args = parser.parse_args()
    parts = set(args.parts.split(","))
    load_env()
    device = torch.device(args.device)

    extract, words, units, has_labels, has_links = load_units(args.run_dir)
    wave, sr = sf.read(args.run_dir / "audio.wav", dtype="float32")
    if sr != SR or wave.ndim != 1:
        sys.exit(f"audio.wav must be {SR} Hz mono, got {sr} Hz, shape {wave.shape}")

    out_path = args.run_dir / "prosody.json"
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    result.update({"clip": extract["video"]["source"], "sha256": extract["video"]["sha256"],
                   "utterance_source": "label_page.utterances (the annotator's units)",
                   "speaker_inputs": {"speaker_labels.json": has_labels, "speaker_links_lrasd.json": has_links}})
    old = {u["id"]: u for u in result.get("utterances", [])}
    if old and [u["id"] for u in units] != sorted(old):
        sys.exit("prosody.json holds different utterances; delete it and rerun all parts")

    if "measure" in parts:
        result["measure"], measured = measure(args.run_dir, wave, words, units)
        for u in measured:
            old[u["id"]] = {**old.get(u["id"], {}), **u}
        print(f"measure: {result['measure']['timings']['total_s']} s")
        for s, v in result["measure"]["speakers"].items():
            print(f"  {s}: {v['utterances']} utts, {v['floor_hz']}-{v['ceiling_hz']} Hz, median {v['median_hz']} Hz, "
                  f"octave jumps {v['octave_jumps']['share_gt_12st']} (first pass {v['octave_jumps_first_pass']['share_gt_12st']})")
    if not old:
        sys.exit("run --parts measure first")

    if "emotion" in parts:
        names = args.emotion_models.split(",")
        if set(names) - set(EMOTION_MODELS):
            sys.exit(f"--emotion-models takes {EMOTION_MODELS}")
        res, timings, checks = emotion(wave, units, device, args.verify_cpu, names)
        outputs = {"dimensional": {"outputs": "arousal, dominance, valence (~0..1)"},
                   "dimensional_odyssey": {"outputs": "arousal, dominance, valence (~0..1 per the card; raw regression "
                                                      "output, not clipped or rescaled)",
                                           "input_norm": "(wave - config.mean) / (config.std + 1e-6), mask of ones",
                                           "card_check": ODYSSEY_CARD_CHECK},
                   "categorical": {}}
        # merge: models not rerun keep their stored metadata and per-utterance outputs
        result.setdefault("emotion", {})["input"] = "the utterance's own audio (start to end), no context"
        result["emotion"]["default_dimensional"] = DEFAULT_DIMENSIONAL
        for name in names:
            result["emotion"][name] = {**LICENCES[name], **outputs[name], "timings": timings[name],
                                       "cpu_check": checks.get(name)}
            for u, r in zip(units, res[name]):
                old[u["id"]].setdefault("emotion", {})[name] = r

    if "captions" in parts and args.captions == "qwen" and args.device == "cpu":
        previous = [old[u["id"]].get("captions", {}).get("qwen") for u in units]
        if not all(previous) or result["captions"]["qwen"]["timings"]["device"] != "mps":
            sys.exit("--captions qwen --device cpu checks stored MPS captions; run it on mps first")
        result["captions"]["qwen"]["cpu_check"] = qwen_cpu_check(wave, units, previous, args.verify_cpu)
    elif "captions" in parts and args.captions != "none":
        key = args.captions
        res, timing, usd = captions(wave, units, key, device)
        result.setdefault("captions", {})
        result["captions"].update({"tags": STYLE_TAGS, "tag_notes": TAG_GLOSSARY, "context_s": CONTEXT_S,
                                   "tagger": {**LICENCES["tagger"], "prices_usd_per_mtok": PRICES[TAGGER]}})
        result["captions"][key] = {**LICENCES[key], "timings": timing, "cost_usd": usd, "cpu_check": None,
                                   **({"prices_usd_per_mtok": PRICES[GPT_MODEL], "instructions": GPT_INSTRUCTIONS}
                                      if key == "gpt" else {"max_new_tokens": QWEN_MAX_NEW_TOKENS, "decoding": "greedy",
                                                            "input": "utterance plus context, no prompt"})}
        for u, r in zip(units, res):
            old[u["id"]].setdefault("captions", {})[key] = r
        print(f"captions {key}: {timing}, cost {usd}")

    result["utterances"] = [old[u["id"]] for u in units]
    out_path.write_text(json.dumps(result))
    print(f"wrote {out_path} ({len(units)} utterances)")


if __name__ == "__main__":
    main()
