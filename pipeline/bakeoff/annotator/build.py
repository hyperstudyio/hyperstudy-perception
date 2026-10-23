"""Perception track annotator (local prototype): see HyperStudy's own design notes.

Assembles the bake-off outputs for one clip into a machine track and writes
runs/<name>/annotator.html, a local page that plays the clip from disk (nothing is uploaded).
Corrections made on the page are an op log on top of this track; the track itself is never edited.

Inputs (runs/<name>/): extract.json, transcript_parakeet.json, faces.json, asd_lrasd.json
(primary speaker evidence), asd_talknet.json (second opinion), audio_events.json,
captions/<model>.json, identity_similarity.json (merge suggestions, from faces/identity_similarity.py).

Serve with byte ranges so seeking works, from tools/perception-bakeoff:
  npx http-server -p 8766 -c-1 .   then open http://127.0.0.1:8766/runs/<name>/annotator.html

Usage: uv run python annotator/build.py runs/<name> [--captions claude-opus-5-5+faces]
"""

import argparse
import json
import math
import subprocess
import sys
import wave
from array import array
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from label_page import utterances  # noqa: E402  same utterance units as the labelling page

TAGGERS = ("panns", "mn40")
IGNORED_TAGS = {"Speech", "Silence"}  # speech is the transcript's job; silence is not an event
# PE-A-Frame query -> AudioSet labels that count as a tagger agreeing with it
AGREEING_TAGS = {
    "laughter": {"Laughter", "Giggle", "Chuckle, chortle", "Belly laugh", "Snicker", "Baby laughter"},
    "music": {"Music", "Musical instrument", "Background music"},
    "applause": {"Applause", "Clapping", "Cheering"},
    "dog barking": {"Dog", "Bark", "Bow-wow", "Yip"},
    "gunshot": {"Gunshot, gunfire", "Machine gun", "Cap gun", "Explosion"},
    "glass breaking": {"Glass", "Shatter", "Breaking"},
    "car horn": {"Vehicle horn, car horn, honking", "Car alarm", "Honk"},
    "baby crying": {"Baby cry, infant cry", "Crying, sobbing"},
}
PEAFRAME_SKIP = {"speech"}  # covered by the transcript
WAVEFORM_HZ = 50             # waveform peaks per second of audio
CROP_PAD = 0.4               # padding around a face box in the pop-out crops, as a fraction of box size
CROP_WIDTH = 320
LOOKING_AT_CAMERA = math.sin(math.radians(12))  # gaze this close to the camera axis counts as looking into it
GAZE_CONE = math.radians(15)  # a face within this angle of the projected gaze ray is the estimated target


def mean_score(track, start, end):
    i0, i1 = int((start - track["start"]) * track["fps"]), int((end - track["start"]) * track["fps"]) + 1
    window = track["scores"][max(0, i0):max(0, i1)]
    return sum(window) / len(window) if window else None


def asd_candidates(asd, unit):
    scored = [(mean_score(t, unit["start"], unit["end"]), t["track"]) for t in asd["tracks"] if t["shot"] == unit["shot"]]
    return sorted(((s, i) for s, i in scored if s is not None), reverse=True)


def tagger_events(audio):
    """Consecutive 1 s steps with the same label become one event; panns and mn40 merge when they overlap."""
    spans = []
    for model in TAGGERS:
        if model not in audio["models"]:
            continue
        open_ = {}
        for step in audio["models"][model]["steps"]:
            labels = {name: p for name, p in step["labels"] if name not in IGNORED_TAGS}
            for name in list(open_):
                if name not in labels:
                    spans.append(open_.pop(name))
            for name, p in labels.items():
                if name in open_ and abs(open_[name]["end"] - step["start"]) < 1e-6:
                    open_[name]["end"] = step["end"]
                    open_[name]["score"] = max(open_[name]["score"], p)
                else:
                    open_[name] = {"label": name, "start": step["start"], "end": step["end"], "score": p, "sources": [model]}
        spans.extend(open_.values())
    spans.sort(key=lambda s: (s["label"], s["start"]))
    merged = []
    for s in spans:
        last = merged[-1] if merged else None
        if last and last["label"] == s["label"] and s["start"] <= last["end"]:
            last["end"] = max(last["end"], s["end"])
            last["score"] = max(last["score"], s["score"])
            last["sources"] = sorted(set(last["sources"]) | set(s["sources"]))
        else:
            merged.append(dict(s))
    return merged


def peaframe_events(audio, taggers):
    model = audio["models"].get("peaframe")
    if not model:
        return []
    events = []
    for query, result in model["queries"].items():
        if query in PEAFRAME_SKIP:
            continue
        agreeing = AGREEING_TAGS.get(query, set())
        for start, end in result["spans"]:
            agree = any(t["label"] in agreeing and t["start"] < end and t["end"] > start for t in taggers)
            events.append({"label": query, "start": start, "end": end, "score": None,
                           "sources": ["pe-a-frame"], "agreement": agree})
    return events


def waveform(path):
    """Peak absolute amplitude per 1/WAVEFORM_HZ s, scaled 0-100 (16-bit mono WAV from extract.py)."""
    with wave.open(str(path)) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit mono audio")
        samples = array("h", w.readframes(w.getnframes()))
        step = w.getframerate() // WAVEFORM_HZ
    peaks = [max(map(abs, samples[i:i + step]), default=0) for i in range(0, len(samples), step)]
    top = max(peaks) or 1
    return [round(100 * p / top) for p in peaks]


def track_crops(video, tracks, frame_w, frame_h, out_dir):
    """One padded face crop per track, from its largest detection, for the inspector's pop-out view."""
    out_dir.mkdir(exist_ok=True)
    jobs = []
    for track in tracks:
        out = out_dir / f"track_{track['id']}.jpg"
        if out.exists():
            continue
        i = max(range(len(track["box"])), key=lambda k: track["box"][k][2] * track["box"][k][3])
        x, y, w, h = track["box"][i]
        x0, y0 = max(0, x - CROP_PAD * w), max(0, y - CROP_PAD * h)
        x1, y1 = min(frame_w, x + (1 + CROP_PAD) * w), min(frame_h, y + (1 + CROP_PAD) * h)
        crop = f"crop={int(x1 - x0)}:{int(y1 - y0)}:{int(x0)}:{int(y0)},scale={CROP_WIDTH}:-2"
        jobs.append(["ffmpeg", "-v", "error", "-y", "-ss", f"{track['t'][i]:.3f}", "-i", str(video),
                     "-frames:v", "1", "-vf", crop, "-q:v", "3", str(out)])
    with ThreadPoolExecutor(8) as pool:
        for result in pool.map(lambda cmd: subprocess.run(cmd, capture_output=True, text=True), jobs):
            if result.returncode:
                raise SystemExit(f"ffmpeg crop failed: {result.stderr.strip()}")
    return len(jobs)


def box_at(track, t):
    ts = track["t"]
    if t < ts[0] or t > ts[-1]:
        return None
    i = next(k for k in range(len(ts)) if ts[k] >= t)
    if ts[i] == t or i == 0:
        return track["box"][i]
    f = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return [a + f * (b - a) for a, b in zip(track["box"][i - 1], track["box"][i])]


def gaze_target(track, sample, candidates):
    """Estimated 2D gaze target: 'camera', a face track id (int), an object track id ('obj:N'), or None.

    py-feat gaze is in camera coordinates (radians; yaw + = image left, pitch + = up). Projected onto the
    image it gives a ray from the eyes; the face or object (from objects.json) whose box centre lies
    closest to that ray, within GAZE_CONE, is the estimate. Depth is ignored, so this is a heuristic for
    review, not a measurement.
    """
    yaw, pitch = sample["gaze"]
    dx, dy = -math.sin(yaw) * math.cos(pitch), -math.sin(pitch)
    length = math.hypot(dx, dy)
    if length < LOOKING_AT_CAMERA:
        return "camera"
    ux, uy = dx / length, dy / length
    lmk = sample["lmk"]
    ex, ey = sum(lmk[36:48]) / 12, sum(lmk[68 + 36:68 + 48]) / 12  # mean of the 12 eye landmarks
    best, best_angle = None, GAZE_CONE
    for other in candidates:
        if other["id"] == track["id"]:
            continue
        box = box_at(other, sample["t"])
        if not box:
            continue
        vx, vy = box[0] + box[2] / 2 - ex, box[1] + box[3] / 2 - ey
        dist = math.hypot(vx, vy)
        if dist == 0 or vx * ux + vy * uy <= 0:
            continue
        angle = math.acos(max(-1.0, min(1.0, (vx * ux + vy * uy) / dist)))
        if angle < best_angle:
            best, best_angle = other["id"], angle
    return best


def pose_summary(path):
    """What the annotator needs to offer a pose file in its skeleton picker; the samples are fetched on demand."""
    pose = json.loads(path.read_text())
    return {"file": path.name, "model": pose["model"], "licence_mode": pose["licence_mode"], "fps": pose["fps"],
            "keypoints": len(pose["keypoints"]), "tracks": len(pose["tracks"]), "has_3d": any("facing" in s for t in pose["tracks"][:5] for s in t["samples"][:1])}


def hand_summary(path):
    """What the annotator needs to offer a hands file in its picker; the samples are fetched on demand."""
    hands = json.loads(path.read_text())
    return {"file": path.name, "model": hands["model"], "licence_mode": hands["licence_mode"], "fps": hands["fps"], "hands": len(hands["hands"]),
            "linked": sum(1 for x in hands["hands"] if x["person"]), "from_detector": sum(1 for x in hands["hands"] if x["box_source"] == "detector")}


def manifest(run, data, extract, transcript, faces, lrasd, talknet, audio, captions, detail):
    """What was extracted for this clip, stage by stage: model, licence mode, file, run time, feature shapes.

    In the product this is a versioned record per pipeline run; here it is read from one run directory.
    Licence mode follows the per-component field proposed in the research reports.
    """
    def stat(name):
        path = run / name
        st = path.stat()
        return {"file": name, "bytes": st.st_size, "run_at": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")}

    n_samples = sum(len(t["series"]) for t in faces["tracks"])
    words = sum(len(s["words"]) for s in transcript["segments"])
    speakers = len({s.get("speaker") for s in transcript["segments"] if s.get("speaker")})
    stages = [
        {"stage": "Shots and keyframes", "model": "PySceneDetect AdaptiveDetector; ffmpeg keyframes (3 per shot, 768 px)",
         "licence_mode": "commercial-ok", "licence": "BSD-3-Clause", **stat("extract.json"), "seconds": extract["timings"]["shots_s"] + extract["timings"]["keyframes_s"],
         "features": [{"name": "shot boundaries", "shape": f"{len(extract['shots'])} shots"},
                      {"name": "keyframes (JPEG)", "shape": f"{3 * len(extract['shots'])} images"}]},
        {"stage": "Transcript, primary", "model": f"{transcript['model']} + pyannote speaker-diarization-community-1",
         "licence_mode": "commercial-ok", "licence": "CC-BY-4.0 (Parakeet, pyannote; pyannote gated)", **stat("transcript_parakeet.json"), "seconds": transcript.get("seconds"),
         "features": [{"name": "words with start/end", "shape": f"{words} words"}, {"name": "diarized voices", "shape": f"{speakers} voices"}],
         "note": "parakeet-mlx runs on Apple silicon only; production needs the NeMo build on CUDA."},
        {"stage": "Transcript, second", "model": f"WhisperX {extract['transcript']['model']} + pyannote",
         "licence_mode": "commercial-ok", "licence": "BSD-2 / MIT / CC-BY-4.0", **stat("extract.json"), "seconds": extract["timings"]["transcript_s"],
         "features": [{"name": "segments with words", "shape": f"{len(extract['transcript']['segments'])} segments"}]},
        {"stage": "Faces", "model": "py-feat v2: " + ", ".join(f"{k} {v}" for k, v in faces["model"].items()),
         "licence_mode": "academic-only", "licence": faces["licence"], **stat("faces.json"),
         "features": [{"name": "face boxes", "shape": f"{n_samples} samples × 4 at {faces['sample_fps']:g} fps"},
                      {"name": "tracks / identities", "shape": f"{len(faces['tracks'])} tracks, {len(faces['people'])} people"},
                      {"name": "ArcFace identity embedding", "shape": "512-d per sample (cache only)"},
                      {"name": "mouth openness (AU25/26)", "shape": "1 per sample"}],
         "note": f"Every py-feat column is in faces_detections_{faces['sample_fps']:g}fps.pkl, the stage's cache; production must store all of it, not just what the annotator draws."},
    ]
    if detail:
        stages.append({"stage": "Face detail", "model": "py-feat v2 (same run)", "licence_mode": "academic-only", "licence": faces["licence"], **stat("face_detail.json"),
            "features": [{"name": "landmarks", "shape": "68 × 2 per sample"}, {"name": "action units", "shape": f"{len(detail['au_names'])} per sample"},
                         {"name": "emotions", "shape": f"{len(detail['emotion_names'])} per sample"}, {"name": "valence, arousal", "shape": "2 per sample"},
                         {"name": "head pose (yaw, pitch, roll)", "shape": "3 per sample"}, {"name": "gaze (yaw, pitch)", "shape": "2 per sample"},
                         {"name": "estimated gaze target", "shape": "1 per sample (2D heuristic)"},
                         {"name": "face mesh", "shape": "478 × 2 per sample (face_mesh.json); z and 52 blendshapes in cache only"}]})
    for name, result in (("LR-ASD", lrasd), ("TalkNet", talknet)):
        if result:
            stages.append({"stage": f"Active speaker ({name})", "model": result["model"], "licence_mode": "unverified",
                "licence": "MIT code; training data (VoxCeleb2, LRS3) terms unverified", **stat(f"asd_{'lrasd' if name == 'LR-ASD' else 'talknet'}.json"),
                "seconds": result["seconds"]["crop"] + result["seconds"]["score"],
                "features": [{"name": "speaking score per face", "shape": f"{len(result['tracks'])} tracks × {result['fps']} fps"}]})
    if audio:
        for key, model in audio["models"].items():
            if "steps" in model:
                feat = {"name": "AudioSet labels (top-5 over threshold)", "shape": f"{len(model['steps'])} steps × {model['step_s']:g} s"}
            else:
                feat = {"name": "text-query spans", "shape": f"{len(model['queries'])} queries at {model['frame_s'] * 1000:g} ms"}
            stages.append({"stage": f"Sound events ({key})", "model": model["weights"],
                "licence_mode": "commercial-ok", "licence": "Apache-2.0" if key.startswith("peaframe") else "MIT / CC-BY-4.0",
                **stat("audio_events.json"), "seconds": audio["timings"].get(f"{key}@{model['device']}", {}).get("infer_warm_s"), "features": [feat]})
    if captions:
        totals = captions["totals"]
        stages.append({"stage": "Shot captions", "model": f"{captions['config']['provider']} {captions['config']['model']} (3 keyframes per shot, face roster)",
            "licence_mode": "commercial-ok", "licence": "provider API terms (local Qwen: Apache-2.0)",
            **stat(f"captions/{captions['model']}.json"), "seconds": totals["seconds"],
            "features": [{"name": "description, setting, people, on-screen text, mood", "shape": f"{len(captions['shots'])} shots"},
                         {"name": "cost", "shape": f"${totals['cost_usd']:.2f}, {totals['input_tokens']:,} in / {totals['output_tokens']:,} out tokens"}]})
    if (run / "identity_similarity.json").exists():
        stages.append({"stage": "Identity similarity", "model": "ArcFace centroid cosine distance (faces/identity_similarity.py)", "licence_mode": "academic-only",
            "licence": "derived from ArcFace embeddings", **stat("identity_similarity.json"),
            "features": [{"name": "person pairs", "shape": f"{len(data['similar'])} never on screen together"}]})
    objects = data["objects"]
    if objects:
        stages.append({"stage": "Objects", "model": f"{objects['segmenter']}" + (f" + {objects['detector']}" if objects["detector"] else "") + f"; identities by {objects['embedder']}",
            "licence_mode": objects["licence_mode"], "licence": "see objects.json", **stat("objects.json"),
            "features": [{"name": "object tracks (mask outline + box)", "shape": f"{len(objects['tracks'])} tracks at {objects['fps']} fps"},
                         {"name": "object identities across shots", "shape": f"{len(objects['identities'])} identities"},
                         {"name": "per-shot vocabulary", "shape": "object_vocabulary.json"}]})
    for pose in data["pose_models"]:
        stages.append({"stage": f"Body pose ({pose['model']})", "model": pose["model"], "licence_mode": pose["licence_mode"], "licence": "see the pose file",
            **stat(pose["file"]), "features": [{"name": "keypoints per body", "shape": f"{pose['keypoints']} × (x, y, score) at {pose['fps']} fps"},
                                                 {"name": "body tracks", "shape": f"{pose['tracks']} tracks"}]
            + ([{"name": "3D facing direction", "shape": "yaw, pitch per sample"}] if pose["has_3d"] else [])})
    for hand in data["hand_models"]:
        stages.append({"stage": f"Hands ({hand['model']})", "model": hand["model"], "licence_mode": hand["licence_mode"], "licence": "see the hands file",
            **stat(hand["file"]), "features": [{"name": "hand keypoints", "shape": f"21 × (x, y, score) at {hand['fps']} fps"},
                                                 {"name": "hand tracks", "shape": f"{hand['hands']} ({hand['linked']} linked to a person, {hand['from_detector']} from the hand detector)"}]})
    if data["hand_events"]:
        kinds = Counter(e["kind"] for e in data["hand_events"]["events"])
        stages.append({"stage": "Gestures", "model": f"rules over {data['hand_events']['source']} and py-feat landmarks", "licence_mode": "academic-only", "licence": "derived",
            **stat("hand_events.json"), "features": [{"name": k.replace("_", " "), "shape": f"{n} events"} for k, n in kinds.items()]})
    for key, label in (("gaze_lle", "Gaze targets (Gaze-LLE)"), ("gaze_page", "Gaze targets (PaGE)"), ("emblems", "Emblems"), ("grw", "Semantic gestures (GRW)")):
        src = data["event_sources"].get(key)
        if src:
            kinds = Counter(e["kind"] for e in src["events"])
            stages.append({"stage": label, "model": src["model"], "licence_mode": src["licence_mode"], "licence": f"see {key}.json", **stat(f"{key}.json"),
                           "features": [{"name": k.replace("_", " "), "shape": f"{n} events"} for k, n in kinds.items()]})
    stages.append({"stage": "Waveform", "model": "peak amplitude of audio.wav", "licence_mode": "commercial-ok", "licence": "computed", **stat("audio.wav"),
        "features": [{"name": "peaks", "shape": f"{len(data['waveform'])} at {WAVEFORM_HZ} Hz"}]})
    return {"stages": stages}


PROSODY_MEASURES = ("final_movement", "final_delta_st", "f0_median_hz", "f0_range_st", "speech_rate_syl_s",
                    "intensity_db", "pause_before_s", "internal_pauses", "hnr_db", "jitter_local", "shimmer_local")


def prosody_data(prosody, utts):
    """The page's view of prosody.json: the whole-clip pitch contour, the models that produced each layer, and
    per-utterance measures, emotion and caption tags keyed by utterance id. Every dimensional emotion model in
    prosody.json ("dimensional", "dimensional_<name>") becomes its own pair of rows, so a newly added model shows
    up without page changes. Utterances must match the page's (same label_page.utterances units)."""
    by_id = {u["id"]: u for u in prosody["utterances"]}
    for u in utts:
        p = by_id.get(u["id"])
        if p is None or abs(p["start"] - u["start"]) > 1e-3 or abs(p["end"] - u["end"]) > 1e-3:
            raise SystemExit(f"prosody.json utterance {u['id']} does not match the page's; re-run prosody/prosody.py")
    contour = prosody["measure"]["contour"]
    dims = {k: {"model": v["model"], "licence_mode": v.get("licence_mode")}
            for k, v in prosody["emotion"].items() if k.startswith("dimensional") and isinstance(v, dict) and "model" in v}
    # emotion categories (emotion2vec+) stay in prosody.json but are not shown: judged unusable in review (2026-10-23)
    # caption sources are the ones with per-utterance captions ("tagger" only maps their text onto tags)
    sources = {k for p in prosody["utterances"] for k in p.get("captions", {})}
    captions = {k: {"model": v["model"], "licence_mode": v.get("licence_mode")}
                for k, v in prosody["captions"].items() if k in sources}
    per_utt = {}
    for p in prosody["utterances"]:
        emo, cap = p.get("emotion", {}), p.get("captions", {})
        per_utt[p["id"]] = {
            "speaker": p.get("speaker"), "speaker_source": p.get("speaker_source"),
            "measure": {k: p["prosody"].get(k) for k in PROSODY_MEASURES},
            "dims": {k: emo[k] for k in dims if k in emo},
            "captions": {k: {"description": c.get("description"), "tags": c.get("tags") or c.get("tags_from_text") or [],
                             "emphasized": c.get("emphasized_words") or []} for k, c in cap.items()},
            # per-word prominence: z-score against the speaker (f0, intensity, duration); the page highlights high ones
            "words": [[w["word"], None if w.get("prominence") is None else round(w["prominence"], 2)] for w in p.get("words", [])],
        }
    return {"step": contour["step_s"], "t0": contour.get("t0", 0.0), "speakers": contour["speakers"],
            "f0_st": [None if v is None else round(v, 1) for v in contour["f0_st"]], "speaker": contour["speaker"],
            "dims": dims,
            # the timeline draws the stage's default model; the inspector lists every stored one. Older prosody.json
            # files have no default recorded, and then every model is drawn.
            "default_dim": prosody["emotion"].get("default_dimensional"),
            "captions": captions, "pitch_model": prosody["measure"]["model"], "utts": per_utt}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--captions", default="claude-opus-5-5+faces")
    args = parser.parse_args()
    run = args.run_dir

    def load(name):
        path = run / name
        return json.loads(path.read_text()) if path.exists() else None

    extract, transcript, faces = load("extract.json"), load("transcript_parakeet.json"), load("faces.json")
    lrasd, talknet, audio = load("asd_lrasd.json"), load("asd_talknet.json"), load("audio_events.json")
    captions, similarity = load(f"captions/{args.captions}.json"), load("identity_similarity.json")
    detail, objects = load("face_detail.json"), load("objects.json")
    pose_files = sorted(run.glob("pose_*.json"))
    # hands_body_*.json is a body-pose stand-in the hand stage writes when pose_*.json is missing, not a hand file
    hand_files = sorted(f for f in run.glob("hands_*.json") if not f.name.startswith("hands_body_"))
    hand_events = load("hand_events.json")
    facing = {f.name: json.loads(f.read_text()) for f in sorted(run.glob("facing_*.json"))}
    gaze_lle, gaze_page, emblems, grw = load("gaze_lle.json"), load("gaze_page.json"), load("emblems.json"), load("grw.json")
    prosody = load("prosody.json")
    video = Path(extract["video"]["source"])

    tracks = [{"id": i, "person": t["person"], "shot": t["shot"], "start": t["start"], "end": t["end"],
               "t": [s["t"] for s in t["series"]], "box": [s["box"] for s in t["series"]],
               "emotion": t["dominant_emotion"], "head_yaw": t["head_yaw"]}
              for i, t in enumerate(faces["tracks"])]
    unknown = sum(1 for t in tracks if t["person"] == "Unknown")
    video_path = video if video.is_absolute() else run.parent.parent / video
    made = track_crops(video_path, tracks, extract["video"]["width"], extract["video"]["height"], run / "annotator_crops")
    object_boxes = {}  # shot -> object tracks shaped like face tracks, as gaze-target candidates
    for o in (objects or {}).get("tracks", []):
        object_boxes.setdefault(o["shot"], []).append({"id": o["id"], "t": [x["t"] for x in o["samples"]], "box": [x["box"] for x in o["samples"]]})
    if detail:
        for track, samples in zip(tracks, detail["tracks"]):
            if len(samples) != len(track["t"]):
                raise SystemExit("face_detail.json does not match faces.json; re-run faces/face_detail.py")
            candidates = [t for t in tracks if t["shot"] == track["shot"]] + object_boxes.get(track["shot"], [])
            track["detail"] = [{**s, "look": gaze_target(track, s, candidates)} for s in samples]
    people = [{"id": p["label"], "crop": p["crop"], "first_seen": p["first_seen"], "seconds": p["seconds_on_screen"],
               "tracks": p["tracks"]} for p in faces["people"]]
    people.append({"id": "Unknown", "crop": None, "first_seen": None, "seconds": None, "tracks": unknown})

    words = [w for s in transcript["segments"] for w in s["words"]]
    utts = []
    for unit in utterances(transcript, extract["shots"]):
        voices = Counter(w.get("speaker") for w in words if unit["start"] <= w["start"] <= unit["end"])
        voice = voices.most_common(1)[0][0] if voices else None
        lr = asd_candidates(lrasd, unit) if lrasd else []  # no ASD stage for this run: speakers are diarized voices only
        tn = asd_candidates(talknet, unit) if talknet else []
        candidates = [{"track": i, "score": round(s, 2)} for s, i in lr]
        if lr and lr[0][0] > 0:
            predicted = {"kind": "face", "track": lr[0][1]}
            margin = lr[0][0] - (lr[1][0] if len(lr) > 1 else 0.0)
        else:
            predicted = {"kind": "offscreen", "voice": voice}
            margin = -lr[0][0] if lr else None  # how far the best visible face is below "speaking"
        talknet_top = tn[0][1] if tn and tn[0][0] > 0 else None
        agree = (talknet_top == predicted.get("track")) if talknet else None
        utts.append({**unit, "voice": voice, "candidates": candidates, "predicted": predicted,
                     "margin": None if margin is None else round(margin, 2), "talknet_agrees": agree})

    taggers = tagger_events(audio) if audio else []
    sounds = [{"id": f"sound:{i}", **e} for i, e in enumerate(
        sorted([{**t, "agreement": None} for t in taggers] + peaframe_events(audio or {"models": {}}, taggers),
               key=lambda e: e["start"]))]

    shots, texts = [], []
    caption_shots = {s["index"]: s for s in captions["shots"]} if captions else {}
    for shot in extract["shots"]:
        cap = (caption_shots.get(shot["index"]) or {}).get("caption") or {}
        keyframes = shot.get("keyframes") or []
        shots.append({"index": shot["index"], "start": shot["start"], "end": shot["end"],
                      "thumb": keyframes[len(keyframes) // 2]["file"] if keyframes else None,
                      "description": cap.get("description"), "setting": cap.get("setting"), "mood": cap.get("mood")})
        for string in cap.get("onscreen_text") or []:
            # the same string in consecutive shots (a logo, a name caption) is one text item
            same = next((t for t in reversed(texts) if t["text"] == string and t["end"] == shot["start"]), None)
            if same:
                same["end"] = shot["end"]
            else:
                texts.append({"id": f"text:{len(texts)}", "text": string, "start": shot["start"], "end": shot["end"]})

    data = {
        "clip": video.name, "sha256": extract["video"]["sha256"], "duration": extract["video"]["duration"],
        "width": extract["video"]["width"], "height": extract["video"]["height"], "face_fps": faces["sample_fps"],
        "sources": {"transcript": transcript["model"], "faces": "py-feat v2 + ArcFace (academic)",
                    "speaker": lrasd["model"] if lrasd else "not run (diarized voices only)", "speaker_second": talknet["model"] if talknet else None,
                    "sounds": "PANNs + EfficientAT mn40 (taggers), PE-A-Frame large (queries)",
                    "captions": captions["model"] if captions else None,
                    "text": f"on-screen text read by {captions['model']} (no OCR stage yet)" if captions else None},
        "shots": shots, "people": people, "tracks": tracks, "utterances": utts, "sounds": sounds, "texts": texts,
        "similar": [p for p in (similarity or {}).get("pairs", []) if not p["co_visible"]],
        "objects": {k: objects[k] for k in ("segmenter", "detector", "embedder", "fps", "licence_mode", "tracks", "identities")} if objects else None,
        "pose_models": [pose_summary(f) for f in pose_files],
        "hand_models": [hand_summary(f) for f in hand_files],
        "hand_events": hand_events,
        "facing": {name: {"source": f["source"], "events": f["events"]} for name, f in facing.items()},
        # model outputs whose events become timeline rows; gaze_lle also carries per-sample gaze peaks for the overlay
        "event_sources": {k: {"model": v["model"], "licence_mode": v["licence_mode"], "events": v["events"]}
                          for k, v in (("gaze_lle", gaze_lle), ("gaze_page", gaze_page), ("emblems", emblems), ("grw", grw)) if v},
        # gaze-target models (same schema), keyed by event source, with per-sample peaks for the overlay
        "gaze_models": {k: {"model": v["model"], "fps": v["fps"], "tracks": {str(t["face_track"]): t["samples"] for t in v["tracks"]}}
                        for k, v in (("gaze_lle", gaze_lle), ("gaze_page", gaze_page)) if v},
        "waveform": waveform(run / "audio.wav"), "waveform_hz": WAVEFORM_HZ,
        "face_detail": {"au_names": detail["au_names"], "emotion_names": detail["emotion_names"], "angles": detail["angles"],
                        "mesh": (run / "face_mesh.json").exists()} if detail else None,
    }

    data["prosody"] = prosody_data(prosody, utts) if prosody else None
    data["manifest"] = manifest(run, data, extract, transcript, faces, lrasd, talknet, audio, captions, detail)
    if prosody:
        pd = data["prosody"]
        st = (run / "prosody.json").stat()
        data["manifest"]["stages"].append({
            "stage": "Prosody", "model": "; ".join([pd["pitch_model"]] + [m["model"] for m in pd["dims"].values()]
                                                   + [m["model"] for m in pd["captions"].values()]),
            "licence_mode": "academic-only" if any(m["licence_mode"] == "academic-only" for m in pd["dims"].values()) else "needs-review",
            "licence": "see prosody.json", "file": "prosody.json", "bytes": st.st_size,
            "run_at": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "features": [{"name": "pitch contour (semitones re speaker)", "shape": f"{len(pd['f0_st'])} at {round(1 / pd['step'])} Hz"},
                         {"name": "per-utterance measures", "shape": f"{len(PROSODY_MEASURES)} per utterance"},
                         {"name": "dimensional emotion (arousal, valence, dominance)", "shape": f"{len(pd['dims'])} model(s)"},
                         {"name": "delivery captions and tags", "shape": ", ".join(pd["captions"]) or "none"}]})

    template = (Path(__file__).parent / "template.html").read_text()
    page = (template.replace("__VIDEO_SRC__", str(Path("..") / ".." / video))
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    out = run / "annotator.html"
    out.write_text(page)
    flagged = sum(1 for u in utts if u["talknet_agrees"] is False)
    print(f"{out}: {len(shots)} shots, {len(people)} people, {len(tracks)} tracks, {len(utts)} utterances "
          f"({flagged} where TalkNet disagrees), {len(sounds)} sound events, {len(texts)} text items, "
          f"{len(data['similar'])} identity pairs; {made} new track crops")


if __name__ == "__main__":
    main()
