"""audio.sounds: sound events, as the bundle's sounds layer.

The bake-off's choice (audio/audio_events.py, and the annotator's tagger_events and peaframe_events): two AudioSet
taggers, PANNs Cnn14_DecisionLevelMax and EfficientAT mn40, label every second of the soundtrack, and PE-A-Frame large
finds when each of a fixed list of sounds (sound_events.QUERIES) is heard, with whether a tagger agrees. models.py runs
the models; sound_events.py turns their scores into events.

Long soundtracks are cut into chunks (sound_events.plan_chunks, and its constants for why these sizes): a soundtrack of
up to 140 s is one piece for every model, as the bake-off ran its clips. mn40 scores independent 5 s windows,
batch_size at a time.

Beside the layer, two side files keep every score the models computed, which the layer thresholds into events
(sound_scores.py): layers/sounds_tags.parquet, PANNs' scores for the 527 AudioSet classes every 0.32 s segment and
mn40's every 1 s step, and layers/sounds_queries.parquet, PE-A-Frame's probability for each query every 40 ms. The
layer points at them with raw.tags and raw.queries.

Licences: PANNs code MIT and weights CC-BY-4.0, EfficientAT code and weights MIT (both trained on AudioSet, whose labels
are CC-BY-4.0), PE-A-Frame Apache-2.0: commercial-ok, as PLAN.md §4 lists them.
"""

import time

import numpy as np

from perception_node import Node, NodeError
from models import PEAFRAME, PEAFRAME_SR, TAGGER_SR, WEIGHTS, device, load_mn40, load_panns, load_peaframe, run_mn40, run_panns, run_peaframe, sync
from sound_events import (PANNS_CONTEXT_S, PANNS_CORE_S, PANNS_FPS, PE_CONTEXT_S, PE_CORE_S, PE_HOP, QUERIES, audioset_labels,
                          padded_for_windows, plan_chunks, sound_events, spans, stitch, step_max, tag_rows, tag_steps, windows)
from soundtrack import samples
import sound_scores

MIN_SECONDS = 1.0   # one tagger step


def timed(timings: dict, key: str, dev, fn, *args):
    t0 = time.time()
    out = fn(*args)
    sync(dev)
    timings[key] = round(timings.get(key, 0) + time.time() - t0, 2)
    return out


with Node() as node:
    batch_size = node.config["batch_size"]
    dev = device(node.accelerator)
    video = node.input("video")
    timings = {}

    t0 = time.time()
    wave32, wave48 = samples(video, TAGGER_SR), samples(video, PEAFRAME_SR)
    timings["decode_s"] = round(time.time() - t0, 2)
    seconds = len(wave32) / TAGGER_SR
    if seconds < MIN_SECONDS:
        raise NodeError("input", f"the soundtrack is {seconds:.2f} s long: sound events need at least {MIN_SECONDS:.0f} s")
    labels = audioset_labels()

    # PANNs: frame-wise scores, max-pooled into 1 s steps
    panns = timed(timings, "load_panns_s", dev, load_panns, dev)
    hop = TAGGER_SR // PANNS_FPS
    pieces = [(c, timed(timings, "panns_s", dev, run_panns, panns, wave32[c.start:c.end], dev))
              for c in plan_chunks(len(wave32), round(PANNS_CORE_S * TAGGER_SR), round(PANNS_CONTEXT_S * TAGGER_SR))]
    framewise = stitch(pieces, hop, axis=0)
    steps = {"panns": tag_rows(step_max(framewise, tag_steps(len(wave32), TAGGER_SR)), labels)}
    try:
        panns_segments = sound_scores.panns_segments(framewise)
    except ValueError as e:
        raise NodeError("internal", str(e)) from None
    panns_frames = len(framewise)
    del panns, pieces, framewise

    # EfficientAT mn40: one 5 s window per 1 s step, batch_size windows at a time
    mn40 = timed(timings, "load_mn40_s", dev, load_mn40, dev)
    padded, n_steps = padded_for_windows(wave32, TAGGER_SR), tag_steps(len(wave32), TAGGER_SR)
    scores = [timed(timings, "mn40_s", dev, run_mn40, mn40, windows(padded, TAGGER_SR, i, min(i + batch_size, n_steps)), dev)
              for i in range(0, n_steps, batch_size)]
    mn40_scores = np.concatenate(scores)
    steps["mn40"] = tag_rows(mn40_scores, labels)
    del mn40, padded, scores

    # PE-A-Frame: each query's probability per 40 ms frame, then spans above its threshold
    pe = timed(timings, "load_peaframe_s", dev, load_peaframe, dev)
    pieces = [(c, timed(timings, "peaframe_s", dev, run_peaframe, pe, wave48[c.start:c.end], QUERIES, dev))
              for c in plan_chunks(len(wave48), round(PE_CORE_S * PEAFRAME_SR), round(PE_CONTEXT_S * PEAFRAME_SR))]
    probs = stitch(pieces, PE_HOP, axis=1)
    query_spans = {q: spans(p, PEAFRAME_SR) for q, p in zip(QUERIES, probs)}
    del pe, pieces

    node.model("PANNs Cnn14_DecisionLevelMax", licence_mode="commercial-ok", source=WEIGHTS["panns"]["url"],
               revision=WEIGHTS["panns"]["sha256"], licence="MIT code, CC-BY-4.0 weights (trained on AudioSet)")
    node.model("EfficientAT mn40_as", licence_mode="commercial-ok", source=WEIGHTS["mn40"]["url"],
               revision=WEIGHTS["mn40"]["sha256"], licence="MIT code and weights (trained on AudioSet)")
    node.model("Meta PE-A-Frame large", licence_mode="commercial-ok", source=f"hf://{PEAFRAME['repo']}@{PEAFRAME['revision']}",
               revision=PEAFRAME["revision"], licence="Apache-2.0")
    events = sound_events(steps, query_spans)
    try:
        sound_scores.write_tags(node.path("tags", "sounds_tags.parquet"), panns_segments, panns_frames, mn40_scores, labels)
        sound_scores.write_queries(node.path("queries", "sounds_queries.parquet"), probs, QUERIES)
    except ValueError as e:
        raise NodeError("internal", f"the models' scores: {e}") from None
    node.write_json("sounds", "sounds.json", {
        "layer": "sounds", "schema_version": "0.1.0",
        "producer": node.producer(timings={**timings, "total_s": round(time.time() - t0, 2)}, device=dev.type),
        "events": events,
        "raw": {"tags": {"file": sound_scores.TAGS_FILE, "format": "parquet", "description": sound_scores.TAGS_DESCRIPTION},
                "queries": {"file": sound_scores.QUERIES_FILE, "format": "parquet",
                            "description": sound_scores.QUERIES_DESCRIPTION}}})
    node.usage(audio_seconds=round(seconds, 2), events=len(events),
               tagger_events=sum(1 for e in events if e["agreement"] is None),
               query_events=sum(1 for e in events if e["agreement"] is not None),
               realtime_factor=round((time.time() - t0) / seconds, 3))
