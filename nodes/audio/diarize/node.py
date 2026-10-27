"""audio.diarize: who speaks when, as voices on the bundle's transcript layer.

pyannote speaker-diarization-community-1 on decode's audio (diarizer.py), its speaker turns then assigned to the words
and utterances of the asr node's transcript (voices.py), as the bake-off did (transcribe_parakeet.py and
convert_bakeoff.py). It refines asr's transcript layer, which has no voices: it writes the transcript layer again, the
words, times and utterances unchanged, with `voices` and each word's and utterance's voice filled in.

Licence: the pipeline's weights are CC-BY-4.0 (gated on Hugging Face), pyannote.audio is MIT; the words are Parakeet's
(CC-BY-4.0, recorded as the asr node recorded it).
"""

import time

from perception_node import Node
from diarizer import LICENCE, REPO, REVISION, load, turns
from pcm16 import RATE, read_waveform
from voices import diarize

with Node() as node:
    transcript = node.input_json("transcript")
    waveform = read_waveform(node.input("audio"))

    t0 = time.time()
    pipeline, device = load(node.secret("HF_TOKEN"), node.accelerator)
    load_s = round(time.time() - t0, 1)
    t0 = time.time()
    found = turns(pipeline, waveform)
    diarize_s = round(time.time() - t0, 1)

    for m in transcript["producer"]["models"]:   # the words' model, as asr recorded it (the manifest declares it)
        node.model(m["id"], licence_mode=m["licence_mode"], source=m.get("source"), revision=m.get("revision"), licence=m.get("licence"))
    node.model(REPO, licence_mode="commercial-ok", source=f"hf://{REPO}@{REVISION}", revision=REVISION, licence=LICENCE)

    layer = diarize(transcript, found)
    node.write_json("transcript", "transcript.json", {
        **layer, "producer": node.producer(timings={"load_s": load_s, "diarize_s": diarize_s}, device=device)})
    audio_s = len(waveform) / RATE
    node.usage(audio_seconds=round(audio_s, 1), turns=len(found), voices=len(layer["voices"]),
               words_without_voice=sum(1 for w in layer["words"] if w["voice"] is None),
               realtime_factor=round(diarize_s / audio_s, 3) if audio_s else None)
