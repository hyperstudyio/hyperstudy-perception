"""audio.waveform: the audio's loudness over time, as the bundle's waveform layer (the annotator's waveform plot).

The peak absolute amplitude of decode's 16 kHz mono audio per 1/50 s, scaled 0-100 relative to the loudest bin
(waveform_peaks.py), as the bake-off's annotator/build.py waveform() computed it from its own audio.wav.
"""

import time

from perception_node import Node
from waveform_peaks import HZ, waveform

with Node() as node:
    t0 = time.time()
    peaks = waveform(node.input("audio"))
    seconds = round(time.time() - t0, 2)
    node.model("peak absolute amplitude per bin", licence_mode="commercial-ok", licence="this repo")
    node.write_json("waveform", "waveform.json", {
        "layer": "waveform", "schema_version": "0.1.0",
        "producer": node.producer(timings={"peaks_s": seconds}, device="cpu"),
        "hz": HZ, "peaks": peaks})
    node.usage(bins=len(peaks), audio_seconds=round(len(peaks) / HZ, 1))
