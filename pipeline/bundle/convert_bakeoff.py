"""Convert a bake-off run directory into a track bundle (schema/bundle-0.1.schema.json).

The bake-off stages wrote one file each, in shapes that grew stage by stage. This maps them onto the bundle's
conventions (string ids, seconds, [x, y, w, h] boxes, a producer and licence mode per layer) without re-running
anything. Where the bake-off's own code already derives what a layer needs (utterance units, speaker candidates,
sound events), it is imported from pipeline/bakeoff rather than re-implemented, so the bundle matches what the
annotator prototype showed.

Model and licence facts the stage files do not record are declared in STAGES below, in one place. A layer's
licence mode in the manifest is the most restrictive of its own models and of the layers it refers to
(LAYER_DEPS): facing events use MediaPipe (commercial-ok) but point at py-feat face tracks (academic-only).

Usage (from pipeline/bundle/): uv run python convert_bakeoff.py RUN_DIR OUT_DIR
   e.g. RUN_DIR = .../tools/perception-bakeoff/runs/ooo-genz-clip1
Then: uv run python validate.py OUT_DIR
"""

import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BAKEOFF = REPO / "pipeline" / "bakeoff"
sys.path.insert(0, str(BAKEOFF))
sys.path.insert(0, str(BAKEOFF / "annotator"))
from label_page import utterances  # noqa: E402  the annotator's utterance units
import build  # noqa: E402  annotator/build.py: asd_candidates, tagger_events, peaframe_events

SCHEMA_VERSION = "0.1.0"
CAPTIONS = "claude-opus-5-5+faces"     # the caption run the annotator prototype shows
VOICE_LINK_MIN_SHARE = 0.3             # trusted voice links (faces/evaluate_identity.py, OOO: 72% vs 65% ungated)
LICENCE_ORDER = ["commercial-ok", "api-terms", "needs-review", "academic-only"]   # least to most restrictive


def model(id, licence_mode, source=None, revision=None, licence=None):
    return {"id": id, "source": source, "revision": revision, "licence": licence, "licence_mode": licence_mode}


# Models per stage, for stage files that do not record them. Facts from the bake-off notes
# (docs/research/.../measured_bakeoff_results.md and the licence audit); "needs-review" where unverified.
STAGES = {
    "shots": [model("PySceneDetect AdaptiveDetector", "commercial-ok", licence="BSD-3-Clause")],
    "captions": [model("claude-opus-5-5", "api-terms", licence="Anthropic API terms")],
    "transcript": [
        model("nvidia/parakeet-tdt-0.6b-v3 (mlx-community conversion)", "commercial-ok",
              source="hf://mlx-community/parakeet-tdt-0.6b-v3", licence="CC-BY-4.0"),
        model("pyannote speaker diarization (via WhisperX DiarizationPipeline)", "needs-review",
              licence="pipeline id not recorded by the bake-off; pyannote community-1 is CC-BY-4.0 (gated)")],
    "speakers": [
        model("LR-ASD (TalkSet finetune)", "needs-review", source="https://github.com/Junhua-Liao/LR-ASD",
              licence="MIT code; weights have no stated licence; trained on VoxCeleb2/LRS3-derived TalkSet"),
        model("TalkNet-ASD (second opinion)", "needs-review", source="https://github.com/TaoRuijie/TalkNet-ASD",
              licence="MIT code; weights trained on AVA-ActiveSpeaker/TalkSet")],
    "faces": [
        model("py-feat Detectorv2 (RetinaFace + multitask)", "academic-only", source="hf://py-feat",
              licence="py-feat code MIT; v2 weights non-commercial"),
        model("ArcFace (InsightFace, via py-feat)", "academic-only", licence="research-only weights")],
    "facing": [model("MediaPipe Pose Landmarker heavy", "commercial-ok",
                     source="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
                     licence="Apache-2.0")],
    "gestures": [model("rules over hand keypoints and face landmarks", "commercial-ok", licence="this repo")],
    "sounds": [
        model("PANNs Cnn14_DecisionLevelMax", "commercial-ok", source="https://zenodo.org/record/3987831", licence="MIT code, CC-BY-4.0 weights"),
        model("EfficientAT mn40_as", "commercial-ok", source="https://github.com/fschmid56/EfficientAT", licence="MIT"),
        model("Meta PE-A-Frame large", "commercial-ok", licence="Apache-2.0")],
}

# layer -> layers whose ids it refers to (its licence cannot be looser than theirs)
LAYER_DEPS = {"speakers": ["transcript", "faces"], "bodies": ["faces"], "hands": ["bodies", "faces"],
              "gestures": ["hands", "faces"], "facing": ["faces"], "gaze": ["faces", "objects"],
              "objects": ["faces"], "prosody": ["transcript"], "text": ["shots"]}


def strictest(modes):
    return max(modes, key=LICENCE_ORDER.index)


def load(run, name):
    path = run / name
    return json.loads(path.read_text()) if path.exists() else None


def producer(stage, models, config=None, timings=None, device=None, run_at=None):
    return {"stage": stage, "stage_version": "bakeoff-2026-10-23", "models": models, "config": config or {},
            "device": device, "run_at": run_at, "timings": timings or {}}


def layer(name, prod, **payload):
    return {"layer": name, "schema_version": SCHEMA_VERSION, "producer": prod, **payload}


def mtime(path):
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds") if path.exists() else None


def face_id(i):
    return None if i is None else f"face:{i}"


def person_id(label):
    return None if label in (None, "Unknown") else f"person:{label}"


class Assets:
    """Small images copied into the bundle's assets/ (person, face-track and object crops), with checksums for the
    manifest. Sources are the bake-off's own crops; a missing source is no asset (the reader shows a swatch)."""

    def __init__(self, run, out):
        self.run, self.out, self.files = run, out, []

    def add(self, src_rel, dest_rel):
        if not src_rel or not (self.run / src_rel).exists():
            return None
        dest = self.out / "assets" / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.run / src_rel, dest)
        data = dest.read_bytes()
        rel = str(dest.relative_to(self.out))
        self.files.append({"file": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        return rel


def slug(text):
    return "".join(c if c.isalnum() else "_" for c in text).strip("_")


def num_id(prefixed, kind):
    """'pose:12' -> 'body:12', 'hand:3' -> 'hand:3'."""
    return None if prefixed is None else f"{kind}:{str(prefixed).split(':', 1)[1]}"


# --- layers ---------------------------------------------------------------------------------------

def shots_layer(run, extract, captions):
    by_index = {s["index"]: s for s in captions["shots"]} if captions else {}
    shots = []
    for s in extract["shots"]:
        cap = (by_index.get(s["index"]) or {}).get("caption")
        shots.append({"id": f"shot:{s['index']}", "index": s["index"], "start": s["start"], "end": s["end"],
                      "keyframes": [{"t": k["t"], "file": None} for k in s.get("keyframes", [])],
                      "caption": None if not cap else {"description": cap.get("description"), "setting": cap.get("setting"),
                                                       "mood": cap.get("mood"), "people": cap.get("people", []),
                                                       "model": captions["model"]}})
    models = STAGES["shots"] + (STAGES["captions"] if captions else [])
    return layer("shots", producer("extract+caption", models, {"keyframe_positions": [0.15, 0.5, 0.85], "captions": CAPTIONS if captions else None},
                                   extract.get("timings"), run_at=mtime(run / "extract.json")), shots=shots)


def transcript_layer(run, transcript, extract):
    words = [w for seg in transcript["segments"] for w in seg["words"]]
    out_words = [{"w": w["word"], "start": w["start"], "end": w["end"],
                  "voice": f"voice:{w['speaker']}" if w.get("speaker") else None} for w in words]
    units = utterances(transcript, extract["shots"])
    utts = []
    for u in units:
        span = [i for i, w in enumerate(words) if u["start"] <= w["start"] and w["end"] <= u["end"]]
        if not span:
            raise SystemExit(f"utterance {u['id']} matches no words")
        voices = Counter(words[i].get("speaker") for i in span)
        voice = voices.most_common(1)[0][0]
        u["voice"] = voice   # the unit's majority voice, as the annotator computes it; the speakers layer reads it
        utts.append({"id": f"utt:{u['id']}", "start": u["start"], "end": u["end"], "text": u["text"],
                     "shot": f"shot:{u['shot']}", "words": [span[0], span[-1]], "voice": f"voice:{voice}" if voice else None})
    voices = sorted({w["voice"] for w in out_words if w["voice"]})
    return layer("transcript", producer("transcribe", STAGES["transcript"], {"utterance_split": "pause > 0.25 s, sentence end, or cut"},
                                        {"seconds": transcript.get("seconds")}, run_at=mtime(run / "transcript_parakeet.json")),
                 voices=[{"id": v} for v in voices], words=out_words, utterances=utts), units


def speakers_layer(run, units, faces, lrasd, talknet, links_file):
    person_of_track = {i: t["person"] for i, t in enumerate(faces["tracks"])}
    links = (links_file or {}).get("links", {})
    rows = []
    for u in units:
        lr = build.asd_candidates(lrasd, u) if lrasd else []
        tn = build.asd_candidates(talknet, u) if talknet else []
        if lr and lr[0][0] > 0:
            track = lr[0][1]
            predicted = {"kind": "face", "face": face_id(track), "via": "asd"}
            if person_id(person_of_track.get(track)):
                predicted["person"] = person_id(person_of_track[track])
            margin = lr[0][0] - (lr[1][0] if len(lr) > 1 else 0.0)
        else:
            margin = -lr[0][0] if lr else None
            words_voice = u.get("voice")
            link = links.get(words_voice or "", {})
            if link and link.get("word_share", 0) >= VOICE_LINK_MIN_SHARE and person_id(link.get("person")):
                predicted = {"kind": "person", "person": person_id(link["person"]), "voice": f"voice:{words_voice}", "via": "voice_link"}
            else:
                predicted = {"kind": "offscreen", "voice": f"voice:{words_voice}" if words_voice else None, "via": "none"}
        talknet_top = tn[0][1] if tn and tn[0][0] > 0 else None
        rows.append({"utt": f"utt:{u['id']}", "predicted": predicted,
                     "candidates": [{"face": face_id(i), "score": round(s, 3)} for s, i in lr],
                     "margin": None if margin is None else round(margin, 3),
                     "second_opinion_agrees": (talknet_top == (lr[0][1] if lr and lr[0][0] > 0 else None)) if talknet else None})
    voice_links = [{"voice": f"voice:{v}", "person": person_id(l["person"]), "word_share": l["word_share"],
                    "trusted": l["word_share"] >= VOICE_LINK_MIN_SHARE} for v, l in sorted(links.items()) if person_id(l.get("person"))]
    return layer("speakers", producer("speakers", STAGES["speakers"],
                                      {"asd": lrasd["model"] if lrasd else None, "voice_link_min_share": VOICE_LINK_MIN_SHARE,
                                       "voice_link_method": (links_file or {}).get("method")},
                                      {"lrasd_s": (lrasd or {}).get("seconds")}, run_at=mtime(run / "asd_lrasd.json")),
                 utterances=rows, voice_links=voice_links)


def faces_layer(run, faces, detail, similarity, pose, out_dir, assets):
    linked = {t["face_track"] for t in (pose or {}).get("tracks", []) if t.get("face_track") is not None}
    tracks = []
    for i, t in enumerate(faces["tracks"]):
        tracks.append({"id": face_id(i), "person": person_id(t["person"]), "shot": f"shot:{t['shot']}",
                       "start": t["start"], "end": t["end"], "linked_body": (i in linked) if pose else None,
                       "crop": assets.add(f"annotator_crops/track_{i}.jpg", f"crops/face_{i}.jpg"),
                       "samples": [{"t": s["t"], "box": s["box"], "mouth": s.get("mouth"),
                                    "gaze": [s["gaze_yaw"], s["gaze_pitch"]] if s.get("gaze_yaw") is not None else None}
                                   for s in t["series"]],
                       "summary": {k: t.get(k) for k in ("head_yaw", "gaze_yaw", "gaze_pitch", "dominant_emotion", "emotions", "mean_box")}})
    people = [{"id": person_id(p["label"]), "first_seen": p.get("first_seen"), "seconds_on_screen": p.get("seconds_on_screen"),
               "crop": assets.add(p.get("crop"), f"crops/person_{slug(p['label'])}.jpg"),
               "shots": [f"shot:{s}" for s in p.get("shots", [])]}
              for p in faces["people"] if person_id(p["label"])]
    pairs = [{"a": person_id(p["a"]), "b": person_id(p["b"]), "distance": p["distance"], "co_visible": p["co_visible"]}
             for p in (similarity or {}).get("pairs", []) if person_id(p["a"]) and person_id(p["b"])]
    side = None
    if detail:
        if len(detail["tracks"]) != len(faces["tracks"]):
            raise SystemExit("face_detail.json does not match faces.json")
        side_doc = {"au_names": detail["au_names"], "emotion_names": detail["emotion_names"], "angles": detail["angles"],
                    "tracks": {face_id(i): samples for i, samples in enumerate(detail["tracks"])}}
        side = write_json(out_dir / "layers" / "faces_detail.json", side_doc)
    prod = producer("faces", STAGES["faces"], {"cluster_threshold": faces.get("threshold"), "frontal_yaw": faces.get("frontal_yaw"),
                                               "sample_fps": faces["sample_fps"]}, run_at=mtime(run / "faces.json"))
    extra = {"angles": detail["angles"]} if detail else {}   # the units of head pose and gaze; only with face detail
    doc = layer("faces", prod, sample_fps=faces["sample_fps"], **extra, people=people, tracks=tracks, person_pairs=pairs,
                detail={"file": "layers/faces_detail.json", "fields": ["lmk", "au", "emo", "va", "head", "gaze"]} if detail else None)
    return doc, ([side] if side else [])


def file_models(doc, stage):
    """A model entry from a stage file that records its own weights and licence."""
    return [model(doc.get("model") or stage, doc.get("licence_mode", "needs-review"),
                  revision=doc.get("weights") or doc.get("checkpoint"), licence=doc.get("licence"))]


def bodies_layer(run, pose):
    tracks = [{"id": num_id(t["id"], "body"), "shot": f"shot:{t['shot']}", "face": face_id(t.get("face_track")),
               "link": t.get("link"), "start": t["start"], "end": t["end"],
               "samples": [{"t": s["t"], "box": s["box"], "kpts": s["kpts"]} for s in t["samples"]]} for t in pose["tracks"]]
    return layer("bodies", producer("bodies", file_models(pose, "rtmw"), {"groups": pose.get("groups")}, pose.get("timings"),
                                    pose.get("device"), mtime(run / "pose_rtmw.json")),
                 keypoint_names=pose["keypoints"], skeleton=pose["skeleton"], score_threshold=pose["score_threshold"],
                 fps=pose["fps"], tracks=tracks)


def hands_layer(run, hands):
    out = [{"id": h["id"], "shot": f"shot:{h['shot']}", "side": h["side"],
            "body": num_id((h["person"] or {}).get("pose_track"), "body") if h["person"] else None,
            "face": face_id((h["person"] or {}).get("face_track")) if h["person"] else None,
            "box_source": h.get("box_source"), "start": h["start"], "end": h["end"],
            "samples": [{"t": s["t"], "box": s["box"], "kpts": s["kpts"]} for s in h["samples"]]} for h in hands["hands"]]
    return layer("hands", producer("hands", file_models(hands, "rtmpose_hand"), {}, hands.get("timings"), hands.get("device"),
                                   mtime(run / "hands_rtmpose_hand.json")),
                 keypoint_names=hands["keypoints"], skeleton=hands["skeleton"], score_threshold=hands.get("score_threshold"),
                 fps=hands["fps"], hands=out)


def gestures_layer(run, events):
    out = [{"id": f"gesture:{i}", "kind": e["kind"], "start": e["start"], "end": e["end"],
            "face": face_id(e.get("face_track")), "hand": e.get("hand"), "score": e.get("score"), "detail": e.get("detail") or {}}
           for i, e in enumerate(events["events"])]
    return layer("gestures", producer("gestures", STAGES["gestures"], {"source": events.get("source")}, run_at=mtime(run / "hand_events.json")),
                 events=out)


def facing_layer(run, facing):
    out = [{"id": f"facing:{i}", "from": face_id(e["from_face_track"]), "to": face_id(e["to_face_track"]),
            "start": e["start"], "end": e["end"], "mutual": e.get("mutual", False), "score": e.get("score"), "detail": e.get("detail") or {}}
           for i, e in enumerate(facing["events"])]
    return layer("facing", producer("facing", STAGES["facing"], {k: facing.get(k) for k in ("source", "cone_deg", "min_facing_s", "max_gap_s")},
                                    run_at=mtime(run / "facing_mediapipe.json")), events=out)


def gaze_target(t):
    if t is None or t in ("out", "camera"):
        return t
    if isinstance(t, int):
        return face_id(t)
    if isinstance(t, str) and t.startswith("obj:"):
        return t
    raise SystemExit(f"unknown gaze target {t!r}")


def gaze_layer(run, gaze):
    tracks = [{"face": face_id(t["face_track"]),
               "samples": [{"t": s["t"], "peak": s.get("peak"), "in_frame": s.get("inout"), "target": gaze_target(s.get("target")),
                            "target_score": s.get("target_score")} for s in t["samples"]]} for t in gaze["tracks"]]
    events = [{"id": f"gaze:{i}", "kind": e["kind"], "from": face_id(e["from_face_track"]), "to": face_id(e.get("to_face_track")),
               "object": e.get("object"), "also": face_id(e.get("also_face_track")),
               "start": e["start"], "end": e["end"], "score": e.get("score")} for i, e in enumerate(gaze["events"])]
    return layer("gaze", producer("gaze", file_models(gaze, "gaze_lle"), {"head_box_scale": gaze.get("head_box_scale")},
                                  gaze.get("timings"), gaze.get("device"), mtime(run / "gaze_lle.json")),
                 fps=gaze["fps"], tracks=tracks, events=events)


def objects_layer(run, objects, assets):
    def sample(s):
        return {"t": s["t"], "box": s["box"], "poly": s.get("poly", []), "score": s.get("score")}
    tracks = [{"id": t["id"], "label": t["label"], "identity": f"objid:{t['identity']}", "shot": f"shot:{t['shot']}",
               "start": t["start"], "end": t["end"], "wearer": person_id(t.get("wearer")), "verdict": t.get("verdict"),
               "samples": [sample(s) for s in t["samples"]]} for t in objects["tracks"]]
    identities = [{"id": f"objid:{i['id']}", "label": i["label"], "family": i.get("family"),
                   "worn_by": [person_id(p) for p in i.get("worn_by", []) if person_id(p)], "tracks": i["tracks"],
               "crop": assets.add(i.get("crop"), f"crops/{slug(i['id'].split(':', 1)[-1])}.jpg")}
                  for i in objects["identities"]]
    rejected = [{"id": t["id"], "label": t["label"], "verdict": t["verdict"], "shot": f"shot:{t['shot']}",
                 "start": t["start"], "end": t["end"], "samples": [sample(s) for s in t["samples"]]} for t in objects.get("rejected_tracks", [])]
    models = [model(objects["segmenter"], "needs-review", source="hf://facebook/sam3", licence="Meta SAM licence (gated)"),
              model(objects["embedder"], "needs-review", source=f"hf://{objects['embedder']}", licence="Meta DINOv3 licence (gated)"),
              model("claude-haiku-5-5 (object names, name families, crop check)", "api-terms", licence="Anthropic API terms")]
    config = {k: objects.get(k) for k in ("fps", "link_threshold", "cross_name_margin", "same_wearer_bonus", "min_link_px", "different_wearer_rule")}
    return layer("objects", producer("objects", models, config, objects.get("timings"), run_at=mtime(run / "objects.json")),
                 fps=objects["fps"], tracks=tracks, identities=identities, rejected_tracks=rejected)


def sounds_layer(run, audio):
    taggers = build.tagger_events(audio)
    events = sorted([{**t, "agreement": None} for t in taggers] + build.peaframe_events(audio, taggers), key=lambda e: e["start"])
    out = [{"id": f"sound:{i}", "label": e["label"], "start": e["start"], "end": e["end"], "sources": e["sources"],
            "agreement": e.get("agreement"), "score": e.get("score")} for i, e in enumerate(events)]
    return layer("sounds", producer("sounds", STAGES["sounds"], {"taggers": list(build.TAGGERS), "query_skip": sorted(build.PEAFRAME_SKIP)},
                                    audio.get("timings"), run_at=mtime(run / "audio_events.json")), events=out)


def text_layer(run, extract, captions):
    by_index = {s["index"]: s for s in captions["shots"]}
    items = []
    for shot in extract["shots"]:
        for string in ((by_index.get(shot["index"]) or {}).get("caption") or {}).get("onscreen_text") or []:
            same = next((t for t in reversed(items) if t["text"] == string and t["end"] == shot["start"]), None)
            if same:
                same["end"] = shot["end"]   # the same string in consecutive shots is one item
            else:
                items.append({"id": f"text:{len(items)}", "text": string, "start": shot["start"], "end": shot["end"], "box": None})
    return layer("text", producer("text", STAGES["captions"], {"source": "on-screen text read by the shot captioner; no OCR stage yet"},
                                  run_at=mtime(run / f"captions/{CAPTIONS}.json")), items=items)


def prosody_layer(run, prosody):
    c = prosody["measure"]["contour"]
    models = [model(prosody["measure"]["model"], prosody["measure"].get("licence_mode", "needs-review"), licence=prosody["measure"].get("licence"))]
    for key, v in prosody["emotion"].items():
        if isinstance(v, dict) and "model" in v:
            models.append(model(v["model"], v.get("licence_mode", "needs-review"), licence=v.get("licence")))
    for key, v in prosody["captions"].items():
        if isinstance(v, dict) and "model" in v:
            models.append(model(v["model"], v.get("licence_mode", "needs-review"), licence=v.get("licence")))
    # which model wrote each per-utterance key (emotion.dimensional, captions.gpt, ...): readers name and licence them
    def keyed(section, kind=None):
        return {key: {"model": v["model"], "licence_mode": v.get("licence_mode", "needs-review"), **({"kind": kind(key)} if kind else {})}
                for key, v in section.items() if isinstance(v, dict) and "model" in v and key != "tagger"}
    emotion_models = keyed(prosody["emotion"], lambda key: "categorical" if key.startswith("categorical") else "dimensional")
    caption_models = keyed(prosody["captions"])
    utts = [{"utt": f"utt:{u['id']}", "speaker": u.get("speaker"), "speaker_source": u.get("speaker_source"),
             "measures": u.get("prosody", {}), "words": u.get("words", []), "emotion": u.get("emotion", {}),
             "captions": u.get("captions", {})} for u in prosody["utterances"]]
    return layer("prosody", producer("prosody", models, {"utterance_source": prosody.get("utterance_source")},
                                     run_at=mtime(run / "prosody.json")),
                 contour={"step": c["step_s"], "t0": c.get("t0", 0.0), "speakers": c["speakers"], "f0_st": c["f0_st"], "speaker": c["speaker"]},
                 default_dimensional=prosody["emotion"].get("default_dimensional"), emotion_models=emotion_models,
                 caption_models=caption_models, utterances=utts)


# --- writing -------------------------------------------------------------------------------------

def waveform_layer(run):
    path = run / "audio.wav"
    if not path.exists():
        return None
    peaks = build.waveform(path)
    return layer("waveform", producer("extract", [model("peak absolute amplitude per bin", "commercial-ok", licence="this repo")],
                                      {"scale": "0-100, relative to the loudest bin"}, run_at=mtime(path)),
                 hz=build.WAVEFORM_HZ, peaks=peaks)


def write_json(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(doc, separators=(",", ":")).encode()
    path.write_bytes(data)
    return {"file": str(path.relative_to(path.parent.parent)), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def main():
    run, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} already holds a bundle; remove it first")
    extract, transcript, faces = load(run, "extract.json"), load(run, "transcript_parakeet.json"), load(run, "faces.json")
    if not (extract and transcript and faces):
        raise SystemExit("a bake-off run needs at least extract.json, transcript_parakeet.json and faces.json")
    captions = load(run, f"captions/{CAPTIONS}.json")
    pose = load(run, "pose_rtmw.json")

    docs, sides, assets = {}, {}, Assets(run, out)
    docs["shots"] = shots_layer(run, extract, captions)
    docs["transcript"], units = transcript_layer(run, transcript, extract)
    docs["faces"], sides["faces"] = faces_layer(run, faces, load(run, "face_detail.json"), load(run, "identity_similarity.json"), pose, out, assets)
    lrasd = load(run, "asd_lrasd.json")
    if lrasd:
        docs["speakers"] = speakers_layer(run, units, faces, lrasd, load(run, "asd_talknet.json"), load(run, "speaker_links_lrasd.json"))
    if pose:
        docs["bodies"] = bodies_layer(run, pose)
    optional = {"hands": ("hands_rtmpose_hand.json", hands_layer), "gestures": ("hand_events.json", gestures_layer),
                "facing": ("facing_mediapipe.json", facing_layer), "gaze": ("gaze_lle.json", gaze_layer),
                "objects": ("objects.json", lambda r, d: objects_layer(r, d, assets)), "sounds": ("audio_events.json", sounds_layer),
                "prosody": ("prosody.json", prosody_layer)}
    for name, (file, make) in optional.items():
        doc = load(run, file)
        if doc:
            docs[name] = make(run, doc)
    if captions:
        docs["text"] = text_layer(run, extract, captions)
    wave = waveform_layer(run)
    if wave:
        docs["waveform"] = wave

    own = {name: strictest([m["licence_mode"] for m in d["producer"]["models"]]) for name, d in docs.items()}
    effective = {}

    def resolve(name):
        if name not in effective:
            effective[name] = strictest([own[name]] + [resolve(dep) for dep in LAYER_DEPS.get(name, []) if dep in docs])
        return effective[name]

    layers = {}
    for name, doc in docs.items():
        entry = write_json(out / "layers" / f"{name}.json", doc)
        layers[name] = {**entry, "licence_mode": resolve(name), "stage": doc["producer"]["stage"]}
        if sides.get(name):
            layers[name]["side_files"] = sides[name]

    video = extract["video"]
    manifest = {"format": "hyperstudy-perception-bundle", "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "video": {"sha256": video["sha256"], "duration": video["duration"], "fps": video["fps"],
                          "width": video["width"], "height": video["height"], "filename": Path(video["source"]).name},
                "pipeline": {"name": "hyperstudy-perception", "version": "0.0.0-bakeoff", "git_commit": git_commit(),
                             "source": f"converted from bake-off run {run.name}"},
                "layers": layers, "assets": assets.files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(l["bytes"] + sum(s["bytes"] for s in l.get("side_files", [])) for l in layers.values())
    print(f"{out}: {len(layers)} layers, {total / 1e6:.1f} MB; {len(assets.files)} assets, {sum(a['bytes'] for a in assets.files) / 1e6:.1f} MB")
    for name, l in layers.items():
        print(f"  {name:10s} {l['bytes'] / 1e6:6.2f} MB  {l['licence_mode']}" + (f"  (own: {own[name]})" if own[name] != l["licence_mode"] else ""))


if __name__ == "__main__":
    main()
