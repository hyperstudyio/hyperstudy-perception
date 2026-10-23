"""Stage 1 of the bake-off: the deterministic specialist tools.

Produces, for one video, everything every captioning model will share:
shot boundaries (PySceneDetect), three keyframes per shot (ffmpeg), and a
word-timed, speaker-labelled transcript (WhisperX + pyannote).

Usage: uv run python extract.py VIDEO.mp4 runs/<name> [--no-diarize]
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

KEYFRAME_POSITIONS = (0.15, 0.5, 0.85)  # fraction of the way through each shot
KEYFRAME_WIDTH = 768
WHISPER_MODEL = "large-v3-turbo"


def load_env(path=Path.home() / ".config/hyperstudy/bakeoff.env"):
    for line in path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate",
         "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    )
    info = json.loads(out.stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = v["r_frame_rate"].split("/")
    return {"duration": float(info["format"]["duration"]), "width": v["width"], "height": v["height"],
            "fps": float(num) / float(den)}


def detect_shots(video, duration):
    from scenedetect import AdaptiveDetector, detect

    scenes = detect(str(video), AdaptiveDetector())
    if not scenes:
        return [{"index": 0, "start": 0.0, "end": duration}]
    return [{"index": i, "start": round(s.get_seconds(), 3), "end": round(e.get_seconds(), 3)}
            for i, (s, e) in enumerate(scenes)]


def extract_keyframes(video, shots, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        shot["keyframes"] = []
        for pos in KEYFRAME_POSITIONS:
            t = shot["start"] + pos * (shot["end"] - shot["start"])
            name = f"shot{shot['index']:03d}_{int(pos * 100):02d}.jpg"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                 "-vf", f"scale={KEYFRAME_WIDTH}:-2", "-q:v", "3", str(out_dir / name)],
                check=True,
            )
            shot["keyframes"].append({"t": round(t, 3), "file": f"keyframes/{name}"})


def transcribe(video, out_dir, diarize_speakers):
    import whisperx

    wav = out_dir / "audio.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-ac", "1", "-ar", "16000", str(wav)], check=True)

    audio = whisperx.load_audio(str(wav))
    model = whisperx.load_model(WHISPER_MODEL, "cpu", compute_type="int8")
    result = model.transcribe(audio, batch_size=8)
    language = result["language"]
    align_model, metadata = whisperx.load_align_model(language_code=language, device="cpu")
    result = whisperx.align(result["segments"], align_model, metadata, audio, "cpu")

    diarized = False
    if diarize_speakers:
        from whisperx.diarize import DiarizationPipeline

        diarize = DiarizationPipeline(token=os.environ["HF_TOKEN"], device="cpu")
        result = whisperx.assign_word_speakers(diarize(audio), result)
        diarized = True

    segments = [{"start": round(s["start"], 3), "end": round(s["end"], 3), "speaker": s.get("speaker"),
                 "text": s["text"].strip(),
                 "words": [{"word": w["word"], "start": w.get("start"), "end": w.get("end"), "speaker": w.get("speaker")}
                           for w in s.get("words", [])]}
                for s in result["segments"]]
    return {"language": language, "model": WHISPER_MODEL, "diarized": diarized, "segments": segments}


def main():
    video, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
    run_dir.mkdir(parents=True, exist_ok=True)
    load_env()

    timings = {}
    t0 = time.time()
    meta = {"source": str(video), "sha256": sha256(video), **probe(video)}
    shots = detect_shots(video, meta["duration"])
    timings["shots_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    extract_keyframes(video, shots, run_dir / "keyframes")
    timings["keyframes_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    transcript = transcribe(video, run_dir, "--no-diarize" not in sys.argv)
    timings["transcript_s"] = round(time.time() - t0, 1)

    (run_dir / "extract.json").write_text(json.dumps(
        {"video": meta, "shots": shots, "transcript": transcript, "timings": timings}, indent=2))
    print(f"{len(shots)} shots, {len(transcript['segments'])} transcript segments, "
          f"diarized={transcript['diarized']}, timings={timings}")


if __name__ == "__main__":
    main()
