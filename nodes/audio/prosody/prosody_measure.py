"""Measured prosody, as the bake-off's prosody/prosody.py measure() computed it, from Praat's analyses. numpy only:
the Praat calls are behind a small interface (`praat`, praat.py's Praat in the node, a stand-in in the tests).

Pitch is Praat's autocorrelation tracker (To Pitch (ac), standard settings) run twice: once at 75-600 Hz to find each
speaker's register, then with a per-speaker floor and ceiling of 0.72 x q35 and 1.9 x q65 of the first pass (De Looze
& Rauzy, 2009), which keeps octave jumps out. A speaker with under 0.5 s of voiced speech keeps the first pass's range.
Per word: F0 mean/min/max in semitones relative to that speaker's median, intensity (energy mean, as Praat), duration,
the pause before it and a prominence score (mean z of F0 peak, intensity re the speaker's median and log duration per
syllable; within the speaker, or the whole clip for a speaker with under 10 words). Per utterance: speech and
articulation rate, pitch range, the final pitch movement (a line fitted to the last 300 ms of voiced frames: a rise or
fall at 1.5 st or more), pauses and voice quality (jitter, shimmer, HNR, from Praat on the utterance's own audio). Plus
the whole clip's pitch contour at 20 Hz: each bin the median of the pitch track of the speaker whose utterance covers
it, in semitones re that speaker's median (the layer) and in Hz (the speakers side file).

Every analysis stops a word WORD_MAX_S after its onset (prosody_units.word_end). Rounding is the bake-off's, so the
same frames give the same numbers. Three changes: a word of zero duration has no duration z of its own (the bake-off's
log 0 took every word's in the pool); a speaker with voiced frames in an utterance's pauses but none in its
words has no median, and its contour bins there are null (the bake-off would have failed on it); and a 50 ms bin two
utterances share takes the later one's speaker and pitch only where the later one has voiced frames in it, so its
speaker and pitch come from one utterance (the bake-off gave such a bin the later speaker with the earlier one's pitch:
6 of OOO clip 1's 2381 bins).

The interface, all times in seconds from the audio's first sample:
- praat.pitch(floor, ceiling) -> (t, f0, strength): every 10 ms frame of the clip; f0 is 0 where unvoiced (Praat's).
- praat.intensity() -> (t, db): every 10 ms frame of the clip.
- praat.voice_quality(start, end, floor, ceiling) -> {jitter, shimmer, hnr (nan where undefined), n_periods,
  harmonicity: (t, db), pulses: t}: Praat on the part of the audio from start to end.
"""

from __future__ import annotations

import re

import numpy as np

from prosody_units import WORD_MAX_S, word_end

PITCH_STEP = 0.01              # Praat's default for these pitch ranges
FIRST_PASS = (75.0, 600.0)     # Praat's defaults: wide enough for any adult voice
RANGE_FACTORS = (0.72, 1.9)    # floor = 0.72 x q35, ceiling = 1.9 x q65 of the first pass (De Looze & Rauzy 2009)
MIN_VOICED_FRAMES = 50         # 0.5 s of voiced speech before a speaker gets their own pitch range
INTENSITY_MIN_PITCH = 75.0     # one intensity analysis for all speakers, so levels are comparable
CONTOUR_STEP = 0.05            # 20 Hz contour for the timeline
FINAL_WINDOW = 0.3             # final pitch movement: the last 300 ms of voiced frames in the utterance
FINAL_MIN_FRAMES = 6
FINAL_THRESHOLD_ST = 1.5       # fitted change over the window that counts as a rise or a fall
PAUSE_MIN = 0.1                # gaps between words shorter than this are not counted as pauses
MIN_PROMINENCE_WORDS = 10      # fewer words than this: z-scores against the whole clip instead of the speaker
OCTAVE_ST = 12.0
SAMPLE_RATE = 16000

SETTINGS = {"pitch_step_s": PITCH_STEP, "first_pass_hz": list(FIRST_PASS), "range_factors": list(RANGE_FACTORS),
            "min_voiced_frames": MIN_VOICED_FRAMES, "final_window_s": FINAL_WINDOW, "final_min_frames": FINAL_MIN_FRAMES,
            "final_threshold_st": FINAL_THRESHOLD_ST, "pause_min_s": PAUSE_MIN, "intensity_min_pitch_hz": INTENSITY_MIN_PITCH,
            "contour_step_s": CONTOUR_STEP, "word_max_s": WORD_MAX_S, "min_prominence_words": MIN_PROMINENCE_WORDS,
            "prominence": "mean z (within speaker) of word F0 max (st), intensity re speaker median, log duration per "
                          "syllable; syllables estimated from spelling"}


def rounded(x, n=2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def syllables(word: str) -> int:
    """Orthographic syllable estimate: vowel groups, a silent final e dropped. Numbers count as 2."""
    w = re.sub(r"[^a-z0-9']", "", word.lower())
    if any(c.isdigit() for c in w):
        return 2
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups) - (1 if w.endswith("e") and not w.endswith(("le", "ee")) and len(groups) > 1 else 0)
    return max(1, n)


def in_span(t, start, end):
    return (t >= start) & (t <= end)


def voiced(f0):
    """Praat's f0 (0 where unvoiced) with nan where unvoiced, as the measures read it."""
    f = np.asarray(f0, dtype=np.float64).copy()
    f[f == 0] = np.nan
    return f


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


def span_mask(t, spans):
    mask = np.zeros_like(t, bool)
    for a, b in spans:
        mask |= in_span(t, a, b)
    return mask


def speaker_range(t1, f1, spans):
    """((floor, ceiling), source) for a speaker from the first pass's voiced frames in their words."""
    v = f1[span_mask(t1, spans) & np.isfinite(f1)]
    if len(v) >= MIN_VOICED_FRAMES:
        q35, q65 = np.percentile(v, [35, 65])
        return (round(RANGE_FACTORS[0] * q35, 1), round(RANGE_FACTORS[1] * q65, 1)), "speaker"
    return FIRST_PASS, "default (too little voiced speech)"


def semitones(f, median_hz):
    return 12 * np.log2(f / median_hz)


def measure(units: list[dict], words: list[dict], n_samples: int, praat) -> dict:
    """Everything measured: {speakers: {name: info}, contour: {step, t0, speakers, f0_st, f0_hz, speaker},
    utterances: [{measures, words}] (in `units` order), tracks: the Praat frames they come from (prosody_tracks.py)}.
    `units` as prosody_units.units() gives them; `words` the transcript layer's."""
    names = sorted({u["speaker"] for u in units})
    spans = {s: [(words[i]["start"], word_end(words[i])) for u in units if u["speaker"] == s for i in u["words"]]
             for s in names}

    # pass 1: Praat's default range, to find each speaker's register
    t1, f1_raw, s1 = praat.pitch(*FIRST_PASS)
    f1 = voiced(f1_raw)
    ranges = {s: speaker_range(t1, f1, spans[s]) for s in names}
    # pass 2: per-speaker range
    tracks, raw_tracks, info = {}, {}, {}
    for s in names:
        t, f_raw, strength = praat.pitch(*ranges[s][0])
        raw_tracks[s] = (t, np.asarray(f_raw, np.float64), np.asarray(strength, np.float64))
        tracks[s] = (t, voiced(f_raw))
        f0 = tracks[s][1]
        v = f0[span_mask(t, spans[s]) & np.isfinite(f0)]
        info[s] = {"floor_hz": ranges[s][0][0], "ceiling_hz": ranges[s][0][1], "range_source": ranges[s][1],
                   "median_hz": rounded(np.median(v) if len(v) else np.nan, 1),
                   "voiced_s": round(len(v) * PITCH_STEP, 2),
                   "utterances": sum(u["speaker"] == s for u in units),
                   "octave_jumps": octave_jump_share(t, f0, spans[s]),
                   "octave_jumps_first_pass": octave_jump_share(t1, f1, spans[s])}

    ti, db = praat.intensity()
    ti, db = np.asarray(ti, np.float64), np.asarray(db, np.float64)

    def st(f, s):
        return semitones(f, info[s]["median_hz"])

    # words
    rows = {}
    for u in units:
        s = u["speaker"]
        t, f0 = tracks[s]
        for i in u["words"]:
            w = words[i]
            seg = f0[in_span(t, w["start"], word_end(w))]
            vs = seg[np.isfinite(seg)]
            dbs = db[in_span(ti, w["start"], word_end(w))]
            level = 10 * np.log10(np.mean(10 ** (dbs / 10))) if len(dbs) else np.nan   # energy mean, as Praat
            rows[i] = {
                "i": i, "word": w["w"], "start": w["start"], "end": w["end"],
                "duration": round(word_end(w) - w["start"], 3),
                "timing_clamped": w["end"] > word_end(w),
                "pause_before": round(w["start"] - word_end(words[i - 1]), 3) if i > 0 else None,
                "syllables": syllables(w["w"]),
                "voiced_frac": round(len(vs) / len(seg), 2) if len(seg) else 0.0,
                "f0_mean_st": rounded(st(np.mean(vs), s)) if len(vs) else None,
                "f0_min_st": rounded(st(np.min(vs), s)) if len(vs) else None,
                "f0_max_st": rounded(st(np.max(vs), s)) if len(vs) else None,
                "intensity_db": rounded(level, 1),
            }
    # speaker-relative intensity, and prominence = mean z of F0 peak, loudness and duration per syllable
    speaker_words = {s: [i for u in units if u["speaker"] == s for i in u["words"]] for s in names}
    for s, idx in speaker_words.items():
        med = np.nanmedian([np.nan if rows[i]["intensity_db"] is None else rows[i]["intensity_db"] for i in idx])
        for i in idx:
            rows[i]["intensity_rel_db"] = rounded(rows[i]["intensity_db"] - med, 1) if rows[i]["intensity_db"] is not None else None
    for s, idx in speaker_words.items():
        own = len(idx) >= MIN_PROMINENCE_WORDS
        pool = idx if own else sorted(rows)
        # a word of zero duration has no log duration (None, as a missing F0): only its own duration z is missing (the
        # bake-off's -inf made the pool's mean -inf, and every word's duration z nan)
        durations = [np.log(rows[i]["duration"] / rows[i]["syllables"]) if rows[i]["duration"] > 0 else None for i in pool]
        z = {"f0": zscores([rows[i]["f0_max_st"] for i in pool]),
             "intensity": zscores([rows[i]["intensity_rel_db"] for i in pool]),
             "duration": zscores(durations)}
        at = {i: k for k, i in enumerate(pool)}
        for i in idx:
            parts = {k: rounded(v[at[i]]) for k, v in z.items()}
            got = [v for v in parts.values() if v is not None]
            rows[i]["prominence"] = rounded(np.mean(got)) if got else None
            rows[i]["prominence_parts"] = parts
            rows[i]["prominence_norm"] = "speaker" if own else "clip (speaker has few words)"

    # utterances
    out, per_utt = [], []
    for k, u in enumerate(units):
        s = u["speaker"]
        t, f0 = tracks[s]
        ws = [rows[i] for i in u["words"]]
        mask = span_mask(t, [(w["start"], word_end(words[w["i"]])) for w in ws])
        tv, fv = t[mask & np.isfinite(f0)], f0[mask & np.isfinite(f0)]
        sv = st(fv, s) if len(fv) else fv
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
        vq = praat.voice_quality(u["start"], u["analysed_end"], info[s]["floor_hz"], info[s]["ceiling_hz"])
        dbs = db[in_span(ti, u["start"], u["analysed_end"])]
        hnr = vq["hnr"]
        out.append({
            "measures": {
                "duration": round(dur, 3), "n_words": len(ws), "syllables": syl,
                "speech_rate_syl_s": rounded(syl / dur) if dur > 0 else None,
                "articulation_rate_syl_s": rounded(syl / sum(w["duration"] for w in ws)) if sum(w["duration"] for w in ws) > 0 else None,
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
                "jitter_local": rounded(vq["jitter"], 4), "shimmer_local": rounded(vq["shimmer"], 4),
                "hnr_db": rounded(hnr, 1) if np.isfinite(hnr) and hnr > -199 else None, "n_periods": int(vq["n_periods"]),
            },
            "words": ws,
        })
        # the frames this utterance's measures and contour bins read: its speaker's pass-2 track over its span, and
        # Praat's harmonicity and glottal pulses on its audio
        t2, f2, s2 = raw_tracks[s]
        inside = in_span(t2, u["start"], u["analysed_end"])
        per_utt.append({"pitch": (t2[inside], f2[inside], s2[inside]), "harmonicity": vq["harmonicity"],
                        "pulses": np.asarray(vq["pulses"], np.float64)})

    # whole-clip contour at 20 Hz: each bin takes the pitch track of the speaker whose utterance covers it
    n_bins = int(np.ceil(n_samples / SAMPLE_RATE / CONTOUR_STEP))
    hz, semis, who = [None] * n_bins, [None] * n_bins, [None] * n_bins
    index = {s: k for k, s in enumerate(names)}
    for u in units:
        t, f0 = tracks[u["speaker"]]
        median = info[u["speaker"]]["median_hz"]
        for b in range(int(u["start"] / CONTOUR_STEP), min(n_bins, int(np.ceil(u["analysed_end"] / CONTOUR_STEP)))):
            a = b * CONTOUR_STEP
            seg = f0[(t >= max(a, u["start"])) & (t < min(a + CONTOUR_STEP, u["analysed_end"]))]
            seg = seg[np.isfinite(seg)]
            # a bin two utterances share is the later one's if it has voiced frames there, else stays the earlier's:
            # its speaker and pitch always come from the same utterance
            if len(seg):
                who[b] = index[u["speaker"]]
                hz[b] = rounded(np.median(seg), 1)
                semis[b] = rounded(semitones(np.median(seg), median)) if median else None
            elif hz[b] is None:
                who[b] = index[u["speaker"]]

    return {"speakers": {s: info[s] for s in names},
            "contour": {"step": CONTOUR_STEP, "t0": 0.0, "speakers": names, "f0_st": semis, "f0_hz": hz, "speaker": who},
            "utterances": out,
            "tracks": {"first_pass": (t1, np.asarray(f1_raw, np.float64), np.asarray(s1, np.float64)),
                       "intensity": (ti, db), "speaker_index": index, "utterances": per_utt}}
