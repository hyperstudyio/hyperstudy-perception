"""Praat's analyses through Parselmouth (praat-parselmouth 0.4.7, Praat 6.1.38; GPL-3), the interface
prosody_measure.measure() reads. The calls and settings are the bake-off's (prosody/prosody.py): To Pitch (ac) at a
10 ms step, To Intensity at 75 Hz, and on each utterance's part of the audio (times preserved) To PointProcess
(periodic, cc), its local jitter and shimmer (period 0.1-20 ms, factor 1.3, amplitude factor 1.6) and the mean of
To Harmonicity (cc). Praat runs on the CPU, deterministically.
"""

from __future__ import annotations

import numpy as np

from prosody_measure import INTENSITY_MIN_PITCH, PITCH_STEP

LICENCE = "GPL-3.0 (Parselmouth and Praat)"
PERIOD_ARGS = (0.0001, 0.02, 1.3)   # shortest and longest period, maximum period factor
AMPLITUDE_FACTOR = 1.6


def versions() -> tuple[str, str]:
    import parselmouth
    return parselmouth.__version__, parselmouth.PRAAT_VERSION


class Praat:
    def __init__(self, samples: np.ndarray, rate: int):
        import parselmouth
        self.sound = parselmouth.Sound(np.asarray(samples, dtype=np.float64), sampling_frequency=rate)

    def pitch(self, floor: float, ceiling: float):
        p = self.sound.to_pitch_ac(time_step=PITCH_STEP, pitch_floor=floor, pitch_ceiling=ceiling)
        a = p.selected_array
        return p.xs(), a["frequency"].copy(), a["strength"].copy()

    def intensity(self):
        i = self.sound.to_intensity(minimum_pitch=INTENSITY_MIN_PITCH, time_step=PITCH_STEP)
        return i.xs(), i.values[0].copy()

    def voice_quality(self, start: float, end: float, floor: float, ceiling: float) -> dict:
        from parselmouth.praat import call
        part = self.sound.extract_part(from_time=start, to_time=end, preserve_times=True)
        points = call(part, "To PointProcess (periodic, cc)", floor, ceiling)
        n_periods = call(points, "Get number of periods", 0, 0, *PERIOD_ARGS)
        jitter = call(points, "Get jitter (local)", 0, 0, *PERIOD_ARGS) if n_periods >= 3 else np.nan
        shimmer = call([part, points], "Get shimmer (local)", 0, 0, *PERIOD_ARGS, AMPLITUDE_FACTOR) if n_periods >= 3 else np.nan
        harmonicity = part.to_harmonicity_cc(time_step=PITCH_STEP, minimum_pitch=floor)
        hnr = call(harmonicity, "Get mean", 0, 0)
        pulses = [call(points, "Get time from index", k) for k in range(1, call(points, "Get number of points") + 1)]
        return {"jitter": jitter, "shimmer": shimmer, "hnr": np.nan if hnr is None else hnr, "n_periods": n_periods,
                "harmonicity": (harmonicity.xs(), harmonicity.values[0].copy()), "pulses": np.array(pulses, np.float64)}
