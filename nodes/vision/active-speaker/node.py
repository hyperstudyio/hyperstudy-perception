"""vision.active-speaker: who on screen is speaking, as the bundle's speakers layer: per utterance of the diarized
transcript, the face tracks of its shot ranked by LR-ASD's score over it (the top one, if positive, is the speaker),
else the person its voice is linked to, else an off-screen voice; and the voice -> person links.

Every face track of 0.4 s or more is scored: its crops at 25 fps, cut in one pass over the video (video25.py, as
asd_crops.py plans them), and the MFCCs of decode's audio under it, through LR-ASD (models.py), each track as soon as
its last frame has gone by. The layer: speaker_links.py.

Beside the layer, two side files keep what it drops (asd_raw.py): layers/speakers_scores.parquet, LR-ASD's score for
every frame of every scored track (per window length, and their mean), and layers/speakers_votes.json, the word votes
behind the voice links, so candidate rankings and the voice-link gate can be revisited. The layer is made from them:
its scored tracks are read back from the scores file once written, and its links come from the votes. It points at
them with raw.scores and raw.votes.

Times: a frame's time counts from the video stream's first frame (as face-track's samples do) and a sample's from
the audio's first (as the transcript's words do), as every node of the graph counts them.

Licence: LR-ASD's weights state no licence (needs-review); the face tracks are py-feat's (the faces layer,
academic-only), and export gives the layer the stricter of the two.
"""

import time
import wave

import numpy as np
from scipy.optimize import linear_sum_assignment

from perception_node import Node, NodeError
import asd_raw
from asd_crops import AUDIO_RATE, audio_span, plans_for
from models import LRASD_ID, LRASD_LICENCE, WEIGHTS, ASDModel, mfcc
from speaker_links import layer_content, require_diarized, speaking_intervals, voice_votes
from video25 import face_clips


def read_samples(path) -> np.ndarray:
    """Decode's audio (16 kHz mono 16-bit WAV) as int16 samples, as the bake-off gave them to python_speech_features."""
    with wave.open(str(path)) as w:
        got = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        if got != (1, 2, AUDIO_RATE):
            raise NodeError("input", f"the audio is {got[0]} channel(s), {8 * got[1]}-bit, {got[2]} Hz; "
                                     f"decode writes 1 channel, 16-bit, {AUDIO_RATE} Hz")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


with Node() as node:
    min_share = node.config["voice_link_min_share"]
    video, shots, faces = node.input("video"), node.input_json("shots"), node.input_json("faces")
    transcript = node.input_json("transcript")
    samples = read_samples(node.input("audio"))
    try:
        require_diarized(transcript)
        plans = plans_for(faces["tracks"], {s["id"]: s for s in shots["shots"]})
    except ValueError as e:
        raise NodeError("input", str(e)) from None

    t0 = time.time()
    model = ASDModel(node.accelerator)
    load_s = round(time.time() - t0, 1)
    node.model(LRASD_ID, licence_mode="needs-review", source=WEIGHTS["lrasd"]["url"], revision=WEIGHTS["lrasd"]["sha256"],
               licence=LRASD_LICENCE)

    # each track is scored as soon as its last frame is cut: only the clips of tracks on screen at once are held
    scores, windows, score_s, t_all = {}, {}, 0.0, time.time()
    for slot, clip in face_clips(video, plans):
        a0, a1 = audio_span(plans[slot][1])
        tick = time.time()
        scores[slot], windows[slot] = model.score(clip, mfcc(samples[a0:a1]))
        score_s += time.time() - tick
    crop_s, score_s = round(time.time() - t_all - score_s, 1), round(score_s, 1)

    # in track order, as the layer's ties and votes read them
    tracks = [(faces["tracks"][i]["id"], frames, scores[slot], windows[slot]) for slot, (i, frames, _) in enumerate(plans)]
    scores_path = node.path("scores", "speakers_scores.parquet")
    try:
        face_frames = asd_raw.write_scores(scores_path, tracks, LRASD_ID)
    except ValueError as e:
        raise NodeError("internal", f"LR-ASD's output: {e}") from None
    # the layer is made from the file as written: its scored tracks are read back from it, as a reader rebuilds them
    scored = asd_raw.scored_tracks(*asd_raw.read_scores(scores_path), faces)
    voices = voice_votes(speaking_intervals(transcript["words"]), scored)
    node.write_json("votes", "speakers_votes.json", asd_raw.votes_doc([p["id"] for p in faces["people"]], voices))

    content = layer_content(transcript, faces, scored, min_share, linear_sum_assignment, voices)
    node.write_json("speakers", "speakers.json", {
        "layer": "speakers", "schema_version": "0.1.0",
        "producer": node.producer(timings={"crop_s": crop_s, "load_s": load_s, "score_s": score_s}, device=str(model.device)),
        **content,
        "raw": {"scores": {"file": asd_raw.SCORES_FILE, "format": "parquet", "description": asd_raw.SCORES_DESCRIPTION},
                "votes": {"file": asd_raw.VOTES_FILE, "format": "json", "description": asd_raw.VOTES_DESCRIPTION}}})
    rows = content["utterances"]
    node.usage(tracks_scored=len(plans), tracks_too_short=len(faces["tracks"]) - len(plans), face_frames=face_frames,
               speaking_frames=sum(1 for t in scored for s in t["scores"] if s > 0),
               via={via: sum(r["predicted"]["via"] == via for r in rows) for via in ("asd", "voice_link", "none")},
               realtime_factor=round((crop_s + score_s) / (len(samples) / AUDIO_RATE), 3) if len(samples) else None)
