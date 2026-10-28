"""The prosody layer's raw side file, layers/prosody_tracks.parquet: Praat's per-frame output the layer's measures and
20 Hz contour are computed from, of which the layer keeps per-word and per-utterance summaries and the contour in
semitones (schema: #/$defs/parquet/prosody_tracks).

One row per frame (or glottal pulse), in this order:
- pitch_first_pass: every 10 ms frame of the clip, pitch at 75-600 Hz (the pass that sets each speaker's range);
- intensity: every 10 ms frame of the clip (dB, 75 Hz minimum pitch; Praat floors silence at -300 dB);
- then per utterance, in the layer's order:
  - pitch: the frames of its speaker's second-pass track (that speaker's floor and ceiling) from its start to its
    analysed end, both included: every frame its words' and its own pitch measures and its contour bins read;
  - harmonicity: Praat's To Harmonicity (cc) frames on its audio (dB; -200 where Praat finds no periodicity), whose
    mean over the frames above -200 is its hnr_db;
  - pulses: the glottal pulses of To PointProcess (periodic, cc) on its audio, from which jitter is computed (value 0).
Columns: track, utt (the utterance's number, utt:N; -1 for the clip-wide tracks), speaker (its index in the layer's
contour.speakers; -1 for the clip-wide tracks), t (seconds from the audio's first sample), value (Hz with 0 where
unvoiced, as Praat gives it, for pitch; dB for intensity and harmonicity; 0 for pulses) and strength (the selected pitch
candidate's strength, 0 for the other tracks). Unrounded. No column is nullable and every value is finite.

Deterministic: identical rows give identical bytes. Every writer option that shapes the file is fixed below, and the
file carries no time; the pyarrow version (envs/prosody's lock, part of the node's cache key) names itself in the
footer. The rows are put together in memory (the Praat analyses they come from already hold every frame of the clip:
about 1.3 million rows, ~50 MB, an hour) and written in row groups of ROW_GROUP_ROWS.
"""

from __future__ import annotations

import numpy as np

FILE = "layers/prosody_tracks.parquet"
DESCRIPTION = ("Praat's frames: first-pass pitch and intensity over the clip, and per utterance its speaker's pitch, "
               "harmonicity and glottal pulses, unrounded, a row per frame")
TRACKS = ("pitch_first_pass", "intensity", "pitch", "harmonicity", "pulses")
ROW_GROUP_ROWS = 65536
ZSTD_LEVEL = 19


def schema():
    import pyarrow as pa
    return pa.schema([pa.field("track", pa.string(), nullable=False), pa.field("utt", pa.int32(), nullable=False),
                      pa.field("speaker", pa.int32(), nullable=False), pa.field("t", pa.float64(), nullable=False),
                      pa.field("value", pa.float64(), nullable=False), pa.field("strength", pa.float64(), nullable=False)],
                     metadata={"side": "prosody_tracks",
                               "value": "pitch: Hz, 0 unvoiced; intensity, harmonicity: dB; pulses: 0"})


def blocks(tracks: dict, units: list[dict]):
    """(track, utt, speaker, t, value, strength) column blocks in file order from prosody_measure.measure()'s tracks."""
    def block(name, utt, speaker, t, value, strength=None):
        t = np.asarray(t, np.float64)
        value = np.asarray(value, np.float64)
        strength = np.zeros_like(t) if strength is None else np.asarray(strength, np.float64)
        if not (len(t) == len(value) == len(strength)):
            raise ValueError(f"{name} of utterance {utt}: {len(t)} times, {len(value)} values, {len(strength)} strengths")
        for col, arr in (("t", t), ("value", value), ("strength", strength)):
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} of utterance {utt}: a {col} that is not finite")
        return name, utt, speaker, t, value, strength

    yield block("pitch_first_pass", -1, -1, *tracks["first_pass"])
    yield block("intensity", -1, -1, *tracks["intensity"])
    index = tracks["speaker_index"]
    if len(units) != len(tracks["utterances"]):
        raise ValueError(f"{len(tracks['utterances'])} utterances' tracks for {len(units)} utterances")
    for u, per in zip(units, tracks["utterances"]):
        k = index[u["speaker"]]
        yield block("pitch", u["n"], k, *per["pitch"])
        yield block("harmonicity", u["n"], k, *per["harmonicity"])
        yield block("pulses", u["n"], k, per["pulses"], np.zeros(len(per["pulses"])))


def write(path, tracks: dict, units: list[dict]) -> int:
    """Write the file; returns its number of rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    s = schema()
    columns = {name: [] for name in s.names}
    rows = 0
    writer = pq.ParquetWriter(path, s, compression="zstd", compression_level=ZSTD_LEVEL, use_dictionary=["track"],
                              write_statistics=["track", "utt", "t"], version="2.6", data_page_version="1.0",
                              store_schema=True, write_page_index=False, use_byte_stream_split=False)
    try:
        for name, utt, speaker, t, value, strength in blocks(tracks, units):
            n = len(t)
            columns["track"].append(pa.array([name] * n, pa.string()))
            columns["utt"].append(pa.array(np.full(n, utt, np.int32)))
            columns["speaker"].append(pa.array(np.full(n, speaker, np.int32)))
            columns["t"].append(pa.array(t))
            columns["value"].append(pa.array(value))
            columns["strength"].append(pa.array(strength))
            rows += n
        table = pa.table({name: pa.chunked_array(parts, s.field(name).type) for name, parts in columns.items()}, schema=s)
        writer.write_table(table.combine_chunks(), row_group_size=ROW_GROUP_ROWS)
    finally:
        writer.close()
    return rows


def read(path) -> dict:
    """The file back, as {track: {utt: (speaker, t, value, strength)}} (numpy arrays): what tests and checks read."""
    import pyarrow.parquet as pq
    table = pq.read_table(path).to_pydict()
    out: dict = {}
    for track, utt, speaker, t, value, strength in zip(*(table[c] for c in ("track", "utt", "speaker", "t", "value", "strength"))):
        out.setdefault(track, {}).setdefault(utt, (speaker, [], [], []))
        _, ts, vs, ss = out[track][utt]
        ts.append(t), vs.append(value), ss.append(strength)
    return {track: {utt: (sp, np.array(ts), np.array(vs), np.array(ss)) for utt, (sp, ts, vs, ss) in per.items()}
            for track, per in out.items()}
