"""Measured prosody from Praat's frames (prosody_measure.py), with a stand-in for Praat that serves synthetic tracks.
On real audio the node's measures were checked against the bake-off's prosody.json for OOO clip 1, with the bake-off's
speakers: every measure, word and contour bin the same (the pull request says how)."""

import numpy as np
import pytest

from prosody_measure import FIRST_PASS, RANGE_FACTORS, measure, syllables

STEP = 0.01
RATE = 16000


class FakePraat:
    """Frames every 10 ms from 0.02 s; f0(t) where it lies within the pass's floor and ceiling (else unvoiced: 0)."""

    def __init__(self, f0, seconds=4.0, db=70.0):
        self.t = 0.02 + STEP * np.arange(int((seconds - 0.04) / STEP) + 1)
        self.f0, self.db = f0, db
        self.passes = []

    def pitch(self, floor, ceiling):
        self.passes.append((floor, ceiling))
        f = np.array([self.f0(t) for t in self.t], float)
        f[(f < floor) | (f > ceiling)] = 0.0
        return self.t, f, np.where(f > 0, 0.9, 0.3)

    def intensity(self):
        return self.t, np.full_like(self.t, self.db)

    def voice_quality(self, start, end, floor, ceiling):
        inside = self.t[(self.t >= start) & (self.t <= end)]
        return {"jitter": 0.0123, "shimmer": 0.0456, "hnr": 15.04, "n_periods": 42,
                "harmonicity": (inside, np.full_like(inside, 15.0)), "pulses": np.arange(start, end, 0.005)}


def unit(n, words, speaker, analysed_end=None):
    return {"id": f"utt:{n}", "n": n, "start": words[0]["start"], "end": words[-1]["end"],
            "words": [w["i"] for w in words], "analysed_end": analysed_end or min(words[-1]["end"], words[-1]["start"] + 1.2),
            "speaker": speaker, "speaker_source": "asd_face", "voice": "SPEAKER_00"}


def word(i, w, start, end):
    return {"i": i, "w": w, "start": start, "end": end}


def two_speakers(rise=False):
    """A speaks 0.1-1.5 s at 200 Hz (rising 4 semitones over its last 0.3 s with rise=True), B 2.0-3.4 s at 110 Hz."""
    def f0(t):
        if 0.1 <= t <= 1.5:
            return 200.0 * (2 ** (4 / 12 * (t - 1.2) / 0.3) if rise and t > 1.2 else 1.0)
        if 2.0 <= t <= 3.4:
            return 110.0
        return 0.0
    words = [word(0, "hello", 0.1, 0.7), word(1, "there", 0.8, 1.5), word(2, "fine", 2.0, 2.6), word(3, "thanks.", 2.7, 3.4)]
    units = [unit(0, words[:2], "A"), unit(1, words[2:], "B")]
    return f0, [{k: v for k, v in w.items() if k != "i"} for w in words], units


def test_each_speaker_gets_their_own_range_from_the_first_pass_and_semitones_re_their_median():
    f0, words, units = two_speakers()
    praat = FakePraat(f0)
    m = measure(units, words, int(4.0 * RATE), praat)
    assert praat.passes[0] == FIRST_PASS
    a, b = m["speakers"]["A"], m["speakers"]["B"]
    assert (a["floor_hz"], a["ceiling_hz"]) == (round(RANGE_FACTORS[0] * 200, 1), round(RANGE_FACTORS[1] * 200, 1))
    assert (b["floor_hz"], b["ceiling_hz"]) == (round(RANGE_FACTORS[0] * 110, 1), round(RANGE_FACTORS[1] * 110, 1))
    assert praat.passes[1:] == [(144.0, 380.0), (79.2, 209.0)]   # one second pass per speaker, in name order
    assert a["median_hz"] == 200.0 and b["median_hz"] == 110.0 and a["range_source"] == "speaker"
    assert a["octave_jumps"]["n_gt_12st"] == 0 and a["octave_jumps"]["share_gt_12st"] == 0.0 and a["octave_jumps"]["pairs"] > 120
    w = m["utterances"][0]["words"][0]
    assert (w["f0_mean_st"], w["f0_min_st"], w["f0_max_st"]) == (0.0, 0.0, 0.0) and w["voiced_frac"] == 1.0
    assert w["syllables"] == 2 and w["pause_before"] is None and m["utterances"][1]["words"][0]["pause_before"] == 0.5
    u = m["utterances"][1]["measures"]
    assert u["f0_median_hz"] == 110.0 and u["f0_range_st"] == 0.0 and u["final_movement"] == "level"
    assert u["jitter_local"] == 0.0123 and u["shimmer_local"] == 0.0456 and u["hnr_db"] == 15.0 and u["n_periods"] == 42
    assert u["pause_before_s"] == 0.5 and m["utterances"][0]["measures"]["pause_after_s"] == 0.5
    assert u["intensity_db"] == 70.0 and m["utterances"][0]["words"][0]["intensity_rel_db"] == 0.0


def test_a_final_rise_of_four_semitones_is_a_rise():
    f0, words, units = two_speakers(rise=True)
    u = measure(units, words, int(4.0 * RATE), FakePraat(f0))["utterances"][0]["measures"]
    assert u["final_movement"] == "rise" and u["final_delta_st"] == pytest.approx(4.0, abs=0.15) and u["final_window_s"] == 0.3


def test_the_contour_takes_each_bins_speaker_and_their_median():
    f0, words, units = two_speakers()
    c = measure(units, words, int(4.0 * RATE), FakePraat(f0))["contour"]
    assert c["step"] == 0.05 and c["t0"] == 0.0 and c["speakers"] == ["A", "B"] and len(c["f0_st"]) == 80
    assert c["speaker"][2:30] == [0] * 28 and c["speaker"][40:68] == [1] * 28 and c["speaker"][35] is None
    assert c["f0_hz"][10] == 200.0 and c["f0_st"][10] == 0.0 and c["f0_hz"][50] == 110.0 and c["f0_st"][50] == 0.0
    assert c["f0_hz"][0] is None and c["speaker"][0] is None


def test_a_bin_two_utterances_share_has_the_speaker_its_pitch_is_from():
    """A cut between speakers inside a 50 ms bin: B's utterance starts at 1.52 s, unvoiced until 1.60 s, in the bin
    [1.50, 1.55) that A's ends in (A at 300 Hz, above B's ceiling: B's track is unvoiced there). The bake-off gave the
    bin B's index with A's pitch (in semitones re A's median): the app would colour A's pitch as B's. The bin keeps A,
    whose frames it holds."""
    def f0(t):
        return 300.0 if 0.1 <= t <= 1.53 else 110.0 if 1.60 <= t <= 3.0 else 0.0
    words = [word(0, "hello", 0.1, 0.7), word(1, "there", 0.8, 1.51), word(2, "fine", 1.52, 2.2), word(3, "thanks.", 2.3, 3.0)]
    units = [unit(0, words[:2], "A"), unit(1, words[2:], "B")]
    c = measure(units, [{k: v for k, v in w.items() if k != "i"} for w in words], int(4.0 * RATE), FakePraat(f0))["contour"]
    assert c["speaker"][30] == 0 and c["f0_hz"][30] == 300.0 and c["f0_st"][30] == 0.0
    assert c["speaker"][32] == 1 and c["f0_hz"][32] == 110.0


def test_a_speaker_with_little_voiced_speech_keeps_the_first_passs_range_and_has_clip_prominence():
    f0, words, units = two_speakers()
    short = lambda t: f0(t) if t < 1.6 or 2.0 <= t <= 2.3 else 0.0   # B voiced 0.3 s: 31 frames, under 50
    m = measure(units, words, int(4.0 * RATE), FakePraat(short))
    assert m["speakers"]["B"]["range_source"] == "default (too little voiced speech)"
    assert (m["speakers"]["B"]["floor_hz"], m["speakers"]["B"]["ceiling_hz"]) == FIRST_PASS
    assert m["utterances"][1]["words"][0]["prominence_norm"] == "clip (speaker has few words)"


def test_voiced_frames_only_between_a_speakers_words_give_no_median_and_null_contour_bins():
    """The bake-off would divide by a missing median here; the measures have no voiced frame to need it."""
    words = [word(0, "a", 0.1, 0.4), word(1, "b", 1.0, 1.3)]
    units = [unit(0, words, "A")]
    gap = lambda t: 150.0 if 0.5 <= t <= 0.9 else 0.0
    m = measure(units, [{k: v for k, v in w.items() if k != "i"} for w in words], int(2.0 * RATE), FakePraat(gap, seconds=2.0))
    assert m["speakers"]["A"]["median_hz"] is None and m["utterances"][0]["measures"]["f0_median_hz"] is None
    assert m["contour"]["f0_hz"][14] == 150.0 and m["contour"]["f0_st"][14] is None


def test_the_tracks_hold_every_frame_the_measures_read():
    f0, words, units = two_speakers()
    m = measure(units, words, int(4.0 * RATE), FakePraat(f0))
    tr = m["tracks"]
    assert len(tr["first_pass"][0]) == len(tr["intensity"][0]) == 397
    t, f, s = tr["utterances"][0]["pitch"]
    assert t[0] == pytest.approx(0.1) and t[-1] == pytest.approx(1.5) and set(f) == {200.0} and set(s) == {0.9}
    assert tr["speaker_index"] == {"A": 0, "B": 1}


def test_syllables_are_vowel_groups_less_a_silent_e():
    assert [syllables(w) for w in ("hello", "there", "fine", "thanks.", "table", "free", "1999", "rhythm")] == [2, 1, 1, 1, 2, 1, 2, 1]
