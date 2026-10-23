"""core.decode: what every other node starts from.

- media: the video's identity and timing (sha256, duration, size, frame rates, the stream's PTS time base, whether
  the frame rate varies), so later nodes and the integer-microsecond mapping never need to re-probe the file;
- shots: cuts by PySceneDetect's AdaptiveDetector, as the bundle's shots layer;
- keyframes: three frames per shot (at 15%, 50%, 85% of it), referenced from the shots layer as assets/keyframes/...;
- audio: 16 kHz mono 16-bit WAV for the audio nodes.

From pipeline/bakeoff/extract.py (shots and keyframes unchanged), without its transcription.
"""

import json
import subprocess
import time
from fractions import Fraction
from pathlib import Path

from perception_node import Node, NodeError

KEYFRAME_POSITIONS = (0.15, 0.5, 0.85)


def probe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration:stream=codec_type,width,height,r_frame_rate,avg_frame_rate,time_base,start_pts,sample_rate,channels",
                          "-of", "json", str(video)], capture_output=True, text=True)
    if out.returncode:
        raise NodeError("input", f"ffprobe cannot read the video: {out.stderr.strip()[:300]}")
    info = json.loads(out.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise NodeError("input", "the file has no video stream")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    r, avg = Fraction(v["r_frame_rate"]), Fraction(v.get("avg_frame_rate") or v["r_frame_rate"])
    return {"duration": float(info["format"]["duration"]), "width": v["width"], "height": v["height"],
            "fps": float(avg or r), "r_frame_rate": v["r_frame_rate"], "avg_frame_rate": v.get("avg_frame_rate"),
            "time_base": v["time_base"], "start_pts": v.get("start_pts"),
            # containers report r_frame_rate as the base rate and avg_frame_rate from the frame count; they differ
            # when frame durations vary
            "variable_frame_rate": bool(avg) and abs(float(r) - float(avg)) > 1e-3,
            "audio": {"sample_rate": int(a["sample_rate"]), "channels": a["channels"]} if a else None}


def detect_shots(video: Path, duration: float, threshold: float) -> list[dict]:
    from scenedetect import AdaptiveDetector, detect

    scenes = detect(str(video), AdaptiveDetector(adaptive_threshold=threshold))
    if not scenes:
        return [{"index": 0, "start": 0.0, "end": round(duration, 3)}]
    return [{"index": i, "start": round(s.get_seconds(), 3), "end": round(e.get_seconds(), 3)} for i, (s, e) in enumerate(scenes)]


def keyframes(video: Path, shots: list[dict], out_dir: Path, width: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        shot["keyframes"] = []
        for pos in KEYFRAME_POSITIONS:
            t = shot["start"] + pos * (shot["end"] - shot["start"])
            name = f"shot{shot['index']:04d}_{int(pos * 100):02d}.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                            "-vf", f"scale={width}:-2", "-q:v", "3", str(out_dir / name)], check=True)
            shot["keyframes"].append({"t": round(t, 3), "file": f"assets/keyframes/{name}"})


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
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(wav)], check=True)
    timings["audio_s"] = round(time.time() - t0, 1)

    layer = [{"id": f"shot:{s['index']}", "index": s["index"], "start": s["start"], "end": s["end"],
              "keyframes": s["keyframes"], "caption": None} for s in shots]
    node.write_json("shots", "shots.json", {"layer": "shots", "schema_version": "0.1.0",
                                            "producer": node.producer(timings=timings, device="cpu"), "shots": layer})
    node.usage(shots=len(shots))
