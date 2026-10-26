"""core.decode: what every other node starts from.

- media: the video's identity and timing (sha256, duration, size, frame rates, the stream's PTS time base, whether
  the frame rate varies), so later nodes and the integer-microsecond mapping never need to re-probe the file;
- shots: cuts by PySceneDetect's AdaptiveDetector, as the bundle's shots layer;
- keyframes: three frames per shot (at 15%, 50%, 85% of it), referenced from the shots layer as assets/keyframes/...;
- audio: 16 kHz mono 16-bit WAV for the audio nodes.

From pipeline/bakeoff/extract.py (shots and keyframes unchanged), without its transcription.
"""

import time
from pathlib import Path

from perception_node import Node, NodeError
from media import audio, keyframes, probe


def detect_shots(video: Path, duration: float, threshold: float) -> list[dict]:
    from scenedetect import AdaptiveDetector, detect

    scenes = detect(str(video), AdaptiveDetector(adaptive_threshold=threshold))
    if not scenes:
        return [{"index": 0, "start": 0.0, "end": round(duration, 3)}]
    return [{"index": i, "start": round(s.get_seconds(), 3), "end": round(e.get_seconds(), 3)} for i, (s, e) in enumerate(scenes)]


with Node() as node:
    video = node.input("video")
    c = node.config
    media = {"sha256": node.request["inputs"]["video"]["sha256"], "filename": video.name, **probe(video)}
    node.write_json("media", "media.json", media)

    timings = {}
    t0 = time.time()
    shots = detect_shots(video, media["duration"], c["shot_threshold"])
    timings["shots_s"] = round(time.time() - t0, 1)
    node.model("PySceneDetect AdaptiveDetector", licence_mode="commercial-ok", licence="BSD-3-Clause")

    t0 = time.time()
    assets = node.path("keyframes", "assets")
    keyframes(video, shots, assets / "keyframes", c["keyframe_width"])
    timings["keyframes_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    wav = node.path("audio", "audio.wav")
    if media["audio"] is None:
        raise NodeError("input", "the video has no audio stream")
    audio(video, wav)
    timings["audio_s"] = round(time.time() - t0, 1)

    layer = [{"id": f"shot:{s['index']}", "index": s["index"], "start": s["start"], "end": s["end"],
              "keyframes": s["keyframes"], "caption": None} for s in shots]
    node.write_json("shots", "shots.json", {"layer": "shots", "schema_version": "0.1.0",
                                            "producer": node.producer(timings=timings, device="cpu"), "shots": layer})
    node.usage(shots=len(shots))
