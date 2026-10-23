"""Hand emblem events (thumbs up, peace, OK, call me, ...) from the hand stage's 21-keypoint hands.

Models (--model):
  stgcnpp    PYSKL's HaGRID gesture model (demo/stgcnpp_gesture.py + demo/hagrid.pth, a light ST-GCN++), loaded in plain
             PyTorch by stgcnpp.py; its outputs match PYSKL's own pipeline on the same input to 4e-7 (see the README
             section of the report / verify notes). Input: the hand file's keypoints, no image.
  mediapipe  comparator: MediaPipe Gesture Recognizer (float16 v1, 7 gestures + None) in IMAGE mode on a crop around
             each hand box; it finds its own hand landmarks in the crop. Always the CPU delegate (as hands.py).

What hagrid.pth is (measured, not documented by PYSKL):
  - Its head has 40 outputs, PYSKL's GestureDataset label space: 20BN-Jester's 27 classes (with Thumb Down/Up renamed
    Dislike/Like) + 13 HaGRID ones. Only 16 were trained: the 24 Jester-only rows of the classifier all have the same
    weight norm (0.218) and the same bias (-0.051), i.e. never saw a positive example, while the 16 HaGRID rows differ
    (norms 1.39-1.75). So it is a HaGRID-only model; no Jester data, whose licence is non-commercial.
  - The 16 trained classes, mapped to HaGRID names by running the model on HaGRID v2 test-split MediaPipe landmarks
    (600 hands per class, 99.4% top-1; v2 resplit users, so some may have been in PYSKL's training split):
    call, dislike, fist, four, like, mute, ok, one, palm, peace (+ peace_inverted), rock, stop (+ stop_inverted,
    "Stop Sign"), three ("Three-Middle"), three2 ("Three-Left"), two_up (+ two_up_inverted), no_gesture.
  - Input coordinates: x / image width, y / image height in [0, 1] (MediaPipe normalised landmarks), as the demo feeds
    it; the input BatchNorm's running mean is 0.49 / 0.50 for every joint, which only [0, 1] coordinates produce.
    PYSKL's PreNormalize2D('auto') is a no-op on such inputs (it rescales only extents > 10). On the HaGRID test
    landmarks the model is insensitive to the hand's position, scale (x0.5, x2), mirroring and a 1.78 aspect stretch
    (top-1 0.983-0.994), so our 1920x1080 landscape frames are fed the same way (x / 1920, y / 1080).
  - HaGRID is single images: each training hand is one frame, which PYSKL's UniformSample repeats to the clip's 10
    frames. So each sample is classified on its own (one frame repeated 10 times) and smoothing is done over time here.
  - Keypoint order: our 21 points (COCO-WholeBody hand / RTMPose Hand5) and MediaPipe's landmarks are the same order;
    checked on 307 OOO hands found by both RTMPose-hand and MediaPipe (every RTMPose index's nearest MediaPipe landmark,
    by median distance, is the same index; 0.02-0.06 of the hand box). No remapping.

Per hand (only hands linked to a person by default: unlinked detector-only hands are often faces or logos):
  1. stgcnpp: a sample is classified when at least MIN_VALID_KPTS of its 21 keypoints pass the hand stage's score
     threshold (KPT_SCORE_THRESHOLD, the values hand_events.py calibrated). mediapipe: every sample is cropped and
     classified when the recognizer finds a hand centred on the box (CENTRE_GATE).
  2. Class probabilities are averaged over the hand's classified samples within SMOOTH_HALF_S of each sample.
  3. A sample is positive for its smoothed top class when that class is an emblem (not no_gesture / None), its
     smoothed probability is at least MIN_SCORE, and the hand is not on the face at that time (hand_events.json's
     hand_over_mouth / self_touch_face events of the same hand), unless the class is made at the face (FACE_EMBLEMS).
  4. Positives of one label are merged when at most MAX_EVENT_GAP_S apart; runs shorter than MIN_EVENT_S are dropped.
     score = mean smoothed probability of the label over the run.

Needs runs/<name>/hand_events.json from the same --hands file (run ../hand_events.py first).
Usage: uv run python emblems.py ../../runs/<name> [--model stgcnpp|mediapipe] [--hands hands_rtmpose_hand.json] [--include-unlinked]
Writes runs/<name>/emblems.json (stgcnpp) or emblems_mediapipe.json.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import stgcnpp

HERE = Path(__file__).parent
BAKEOFF = HERE.parent.parent  # extract.json's video.source is relative to this
CHECKPOINT = HERE / "vendor/pyskl/demo/hagrid.pth"  # git clone https://github.com/kennymckormick/pyskl vendor/pyskl (f2bf3a6; sha256 14d3e55e...07e7)
GESTURE_TASK = HERE / "vendor/gesture_recognizer.task"  # curl -L -o vendor/gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task (sha256 97952348...0482)

# hagrid.pth output index -> HaGRID class name, for the 16 trained outputs (mapping measured on HaGRID test landmarks,
# see the docstring). Every other output index is untrained and is folded into no_gesture.
STGCNPP_CLASSES = {27: "call", 19: "dislike", 28: "fist", 29: "four", 20: "like", 30: "mute", 31: "ok", 32: "one",
                   33: "palm", 34: "peace", 35: "rock", 14: "stop", 36: "three", 37: "three2", 38: "two_up",
                   39: "no_gesture"}
MEDIAPIPE_CLASSES = ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]
NO_GESTURE = {"stgcnpp": "no_gesture", "mediapipe": "None"}

# Same per-model keypoint thresholds as hand_events.py (calibrated there against MediaPipe keypoint agreement).
KPT_SCORE_THRESHOLD = {"rtmw": 5.0, "rtmpose_hand": 0.3, "mediapipe": 0.5}
# A gesture class is defined by which fingers are extended, so the whole hand must be seen. At most 3 of 21 points may
# fall below the threshold (one finger's worth, e.g. a curled finger hidden behind the palm in a fist or a thumbs up).
# On OOO, 723 of 3658 RTMPose-hand samples pass at 18, 392 at 21; most of the rest have fewer than 6 valid points.
MIN_VALID_KPTS = 18
# +-0.2 s takes the sample and its neighbours at 8 fps (0.125 s apart): a 3-sample moving average, enough to suppress
# one-frame keypoint glitches without blurring a 0.5 s emblem.
SMOOTH_HALF_S = 0.2
# Smoothed probability a label needs. On 2225 HaGRID test hands (15 gesture classes) the model's top-1 probability has
# median 0.998 and is below 0.8 for 0.9% of them, so a top class under 0.8 means the hand is unlike its training data. On the two clips no threshold separated right from wrong events (see
# the evaluation); 0.8 is kept to cut volume, since precision matters more than recall here.
MIN_SCORE = 0.8
# HaGRID has no hand-on-face class: a flat hand over the mouth or cheek is read as "stop" with probability ~1.0 (all six
# OOO "stop" events at 1.0 before this rule were hands over the mouth, each matching a hand_events.json hand_over_mouth
# event of the same hand). So samples inside the same hand's hand_over_mouth / self_touch_face events are not emblems,
# except for the emblems that are made at the face ("mute": index finger at the lips; "call": thumb and pinky at the ear).
FACE_EVENT_KINDS = {"hand_over_mouth", "self_touch_face"}
FACE_EMBLEMS = {"mute", "call"}
MAX_EVENT_GAP_S = 0.3  # as hand_events.py: bridges one missed sample at 8 fps
MIN_EVENT_S = 0.25     # as hand_events.py: at least 3 consecutive samples at 8 fps; deliberate emblems are held longer
CLASSIFY_BATCH = 512   # samples per forward pass (the model has 197k parameters; batching only saves Python overhead)

# MediaPipe comparator crops: the same crop rule hands.py uses for its MediaPipe hand landmarker (3x the hand box,
# resized to 256 px), and the same centre gate to keep the hand the box is about.
CROP_SCALE = 3.0
CROP_PX = 256
CENTRE_GATE = 0.5

LICENCES = {
    "stgcnpp": {
        "model": "PYSKL ST-GCN++ gesture (demo/stgcnpp_gesture.py), HaGRID-trained",
        "checkpoint": "pyskl/demo/hagrid.pth (commit f2bf3a6, added 2023-02-10, sha256 14d3e55e7c29ee73d22a2306050c7156d516bdeee3ab47547174e38256f807e7)",
        "licence_mode": "academic-only",
        "licence": ("Classifier: PYSKL code Apache-2.0 (repo LICENSE; the checkpoint ships in the repo with no separate "
                    "terms); weights trained on HaGRID (v1 era), whose licence (github.com/hukenovs/hagrid/license/en_us.pdf) "
                    "is a non-Creative-Commons reworking of CC BY-SA 4.0: attribution, share-alike for Adapted Material, "
                    "no non-commercial clause, publicity/privacy rights not granted. Whether trained weights are Adapted "
                    "Material is unsettled (commercial use needs review). The mode is academic-only because the input "
                    "keypoints come from RTMPose-hand on RTMW boxes (academic-only, see hands_rtmpose_hand.json)."),
    },
    "mediapipe": {
        "model": "MediaPipe Gesture Recognizer (canned 7 gestures + None)",
        "checkpoint": "gesture_recognizer/float16/1/gesture_recognizer.task (sha256 97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482)",
        "licence_mode": "commercial-review",
        "licence": ("MediaPipe code Apache-2.0; the gesture classifier's model card states no licence and no training "
                    "data provenance (only 'hand landmarks from images with various lighting, noise and motion conditions'); "
                    "its bundled hand landmarker's card says Apache-2.0. Hand boxes come from RTMW, so this pipeline as "
                    "run is academic-only; MediaPipe alone on full frames would need only the weight-terms review."),
    },
}


# ---------------------------------------------------------------- classifiers

class StgcnppClassifier:
    """{(hand index, t): probabilities over the 16 trained classes} from keypoints alone."""

    classes = list(STGCNPP_CLASSES.values())

    def __init__(self, width, height, threshold):
        torch.set_num_threads(2)  # tiny model; leave the cores to the other bake-off stages
        self.model = stgcnpp.load(CHECKPOINT)
        self.scale = np.array([width, height], dtype=np.float32)
        self.width, self.height, self.threshold = width, height, threshold
        self.index = np.array(list(STGCNPP_CLASSES))

    def classify(self, hands, stats):
        keys, inputs = [], []
        for i, hand in enumerate(hands):
            for s in hand["samples"]:
                kpts = np.array(s["kpts"], dtype=np.float32)
                valid = ((kpts[:, 2] >= self.threshold) & (kpts[:, 0] >= 0) & (kpts[:, 0] < self.width)
                         & (kpts[:, 1] >= 0) & (kpts[:, 1] < self.height))
                if valid.sum() < MIN_VALID_KPTS:
                    stats["too_few_kpts"] += 1
                    continue
                keys.append((i, s["t"]))
                inputs.append(stgcnpp.preprocess(kpts[None, :, :2] / self.scale))
        probs = []
        for j in range(0, len(inputs), CLASSIFY_BATCH):
            probs.append(self.model.predict(torch.from_numpy(np.stack(inputs[j:j + CLASSIFY_BATCH]))).numpy())
        probs = np.concatenate(probs) if probs else np.zeros((0, 40), np.float32)
        stats["untrained_argmax"] = int((~np.isin(probs.argmax(1), self.index)).sum()) if len(probs) else 0
        out = probs[:, self.index]
        out[:, -1] += 1.0 - out.sum(1)  # the untrained outputs' (tiny) mass goes to no_gesture
        stats["classified"] = len(keys)
        return dict(zip(keys, out))


class MediapipeClassifier:
    """{(hand index, t): probabilities over None + 7 gestures} from MediaPipe Gesture Recognizer on hand-box crops."""

    classes = MEDIAPIPE_CLASSES

    def __init__(self, video, fps, step):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        from mediapipe.tasks.python.components.processors import ClassifierOptions
        self.mp = mp
        self.video, self.fps, self.step = video, fps, step
        self.recognizer = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(GESTURE_TASK), delegate=BaseOptions.Delegate.CPU),
            num_hands=2, canned_gesture_classifier_options=ClassifierOptions(max_results=-1, score_threshold=0.0)))

    def crop(self, frame, box):
        side = box[2] * CROP_SCALE
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        x0, y0, s = int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))
        crop = np.zeros((s, s, 3), np.uint8)  # black outside the frame
        fx0, fy0 = max(x0, 0), max(y0, 0)
        fx1, fy1 = min(x0 + s, frame.shape[1]), min(y0 + s, frame.shape[0])
        crop[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] = frame[fy0:fy1, fx0:fx1]
        return cv2.resize(crop, (CROP_PX, CROP_PX), interpolation=cv2.INTER_LINEAR), s / CROP_PX, x0, y0

    def classify(self, hands, stats):
        wanted = {}
        for i, hand in enumerate(hands):
            for s in hand["samples"]:
                wanted.setdefault(round(s["t"], 3), []).append((i, s))
        out = {}
        for t, frame in sample_frames(BAKEOFF / self.video, self.fps, self.step, set(wanted)):
            for i, s in wanted[t]:
                image, scale, x0, y0 = self.crop(frame, s["box"])
                rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                result = self.recognizer.recognize(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
                centre = np.array([s["box"][0] + s["box"][2] / 2, s["box"][1] + s["box"][3] / 2])
                best, best_d = None, CENTRE_GATE * s["box"][2]
                for lms, gestures in zip(result.hand_landmarks, result.gestures):
                    xy = np.array([[p.x * CROP_PX, p.y * CROP_PX] for p in lms]) * scale + [x0, y0]
                    d = float(np.hypot(*(xy.mean(0) - centre)))
                    if d <= best_d:
                        best, best_d = gestures, d
                if best is None:
                    stats["no_hand_in_crop"] += 1
                    continue
                p = np.zeros(len(MEDIAPIPE_CLASSES), np.float32)
                for c in best:
                    p[MEDIAPIPE_CLASSES.index(c.category_name)] = c.score
                out[(i, s["t"])] = p / p.sum()
        stats["classified"] = len(out)
        return out


def sample_frames(video_path, fps, step, times):
    """(t, BGR frame) for every step-th frame whose timestamp is in times, decoded sequentially (hands.py's sampler)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    index = 0
    while True:
        if index % step:
            if not capture.grab():
                break
        else:
            ok, frame = capture.read()
            if not ok:
                break
            t = round(index / fps, 3)
            if t in times:
                yield t, frame
        index += 1
    capture.release()


# ---------------------------------------------------------------- events

def merge(times, gap):
    """Runs of sorted times whose neighbours are at most gap apart."""
    runs = []
    for t in times:
        if runs and t - runs[-1][-1] <= gap + 1e-6:
            runs[-1].append(t)
        else:
            runs.append([t])
    return runs


def hand_events(hand, probs, classes, no_gesture, face_spans):
    """Events of one hand from {t: probabilities} of its classified samples; face_spans are its (start, end) face contacts."""
    times = sorted(probs)
    if not times:
        return []
    arr = np.array([probs[t] for t in times])
    tt = np.array(times)
    positives = {}
    for k, t in enumerate(times):
        smooth = arr[np.abs(tt - t) <= SMOOTH_HALF_S + 1e-6].mean(0)
        c = int(smooth.argmax())
        on_face = any(a - 1e-6 <= t <= b + 1e-6 for a, b in face_spans)
        if classes[c] != no_gesture and smooth[c] >= MIN_SCORE and (classes[c] in FACE_EMBLEMS or not on_face):
            positives.setdefault(classes[c], {})[t] = float(smooth[c])
    events = []
    person = hand["person"]
    for label, samples in positives.items():
        for run in merge(sorted(samples), MAX_EVENT_GAP_S):
            if run[-1] - run[0] < MIN_EVENT_S:
                continue
            events.append({"kind": "emblem", "label": label, "hand": hand["id"], "side": hand["side"],
                           "face_track": person["face_track"] if person else None, "start": run[0], "end": run[-1],
                           "score": round(float(np.mean([samples[t] for t in run])), 2)})
    return events


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", choices=["stgcnpp", "mediapipe"], default="stgcnpp")
    parser.add_argument("--hands", default="hands_rtmpose_hand.json")
    parser.add_argument("--include-unlinked", action="store_true", help="also classify detector-only hands with no person")
    args = parser.parse_args()

    start = time.perf_counter()
    hands_file = json.loads((args.run_dir / args.hands).read_text())
    extract = json.loads((args.run_dir / "extract.json").read_text())
    video = extract["video"]
    hands = [h for h in hands_file["hands"] if args.include_unlinked or h["person"] is not None]
    stats = {"hands": len(hands), "samples": sum(len(h["samples"]) for h in hands), "too_few_kpts": 0, "no_hand_in_crop": 0}
    if args.model == "stgcnpp":
        classifier = StgcnppClassifier(video["width"], video["height"], KPT_SCORE_THRESHOLD[hands_file["model"]])
    else:
        step = max(1, round(video["fps"] / hands_file["fps"]))
        classifier = MediapipeClassifier(video["source"], video["fps"], step)
    probs = classifier.classify(hands, stats)

    # Face contacts from hand_events.json, which must have been built from the same hand file (same hand ids).
    hand_events_path = args.run_dir / "hand_events.json"
    if not hand_events_path.exists():
        raise SystemExit(f"{hand_events_path} does not exist: run ../hand_events.py --hands {args.hands} first")
    hand_events_file = json.loads(hand_events_path.read_text())
    if hand_events_file["source"] != args.hands:
        raise SystemExit(f"{hand_events_path} was built from {hand_events_file['source']}, not {args.hands}; rerun hand_events.py")
    if not (args.run_dir / "face_detail.json").exists():
        print("face filter inactive: no face_detail.json, so hand_events.json has no face-contact events")
    face_spans = {}
    for e in hand_events_file["events"]:
        if e["kind"] in FACE_EVENT_KINDS:
            face_spans.setdefault(e["hand"], []).append((e["start"], e["end"]))

    events = []
    for i, hand in enumerate(hands):
        mine = {t: p for (j, t), p in probs.items() if j == i}
        events += hand_events(hand, mine, classifier.classes, NO_GESTURE[args.model], face_spans.get(hand["id"], []))
    events.sort(key=lambda e: (e["start"], e["hand"]))

    out = dict(LICENCES[args.model])
    out.update({"classes": classifier.classes, "source": args.hands, "device": "cpu",
                "timings": {"total_s": round(time.perf_counter() - start, 2)}, "events": events})
    path = args.run_dir / ("emblems.json" if args.model == "stgcnpp" else "emblems_mediapipe.json")
    path.write_text(json.dumps(out, indent=1))
    counts = {}
    for e in events:
        counts[e["label"]] = counts.get(e["label"], 0) + 1
    print(f"{path.name} from {args.hands}: {len(events)} events {counts}; {stats}; {out['timings']['total_s']} s")


if __name__ == "__main__":
    main()
