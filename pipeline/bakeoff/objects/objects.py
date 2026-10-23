"""Object stage of the bake-off: what things are on screen, tracked within shots and linked across them.

1. Vocabulary: Claude Haiku 5.5 sees each shot's three keyframes and names up to
   MAX_OBJECTS concrete, trackable objects (no people or background). It is shown the
   names it used in EARLIER shots and told to reuse them, so one bike is "bicycle"
   throughout and labels can be linked. Cached in runs/<name>/object_vocabulary.json.
2. Segment + track: each shot is decoded at OBJECT_FPS and run through SAM 3 video
   (facebook/sam3, Sam3VideoModel) with the shot's nouns as text prompts. SAM 3
   detects on every frame and tracks with memory, so objects entering mid-shot are
   picked up. A session never spans a cut.
3. Embedding: each track is embedded as the mean DINOv3 ViT-L/16 embedding (CLS) of up to
   EMBED_SAMPLES tight box crops, letterboxed to a square so thin objects are not cut. The
   crops are re-cut from the video at the published sample boxes, so --relink gives the
   same embeddings as a full run.
4. Checks (Haiku, cached): one call classifies the clip's names into families ("pearl necklace" and
   "necklace" are one), lookalike or individual, and worn or not (object_vocabulary.json "classes");
   one crop per track is checked for really being its label (object_verification.json). Tracks that
   fail are kept in "rejected_tracks" and not linked.
5. Identities: worn tracks get a wearer, the face-clustered person whose posed body holds them.
   Tracks of one family are merged by average-linkage clustering of adjusted cosines above
   LINK_THRESHOLD (pair_score), starting from groups of duplicate tracks (one object tracked under two names
   at once). Never merged: two other tracks on screen at the same moment, items worn by
   two certainly different people (face distance), and tiny tracks (< MIN_LINK_PX) not on the same wearer. Lookalike
   families (trees, posts, poles...) are never linked. Each identity's crop in runs/<name>/object_crops/
   is the view (of those embedded) closest to the identity's mean embedding, which skips
   frames where a mask drifted.

SAM 3 on Apple silicon: transformers' Sam3VideoModel takes NMS, hole filling and
sprinkle removal from a CUDA-only hub kernel (kernels-community/cv-utils) and,
without it, silently skips all three. install_cpu_cv_utils() puts in their place
the CPU implementations from Meta's reference code (facebookresearch/sam3 perflib),
so the post-processing SAM 3 was designed with does run here.

On an NVIDIA machine pass --device cuda (the pinned torch wheel from PyPI is the CUDA build on
Linux). SAM 3 then runs in bf16, as Meta's reference runs it under CUDA autocast; the CPU
post-processing stand-ins are still used so results match across machines (the CUDA hub kernel is
faster but was not available here). The CUDA path has NOT been run yet: only MPS and CPU were tested.

Usage (from objects/): uv run python objects.py ../runs/<name> [--device mps|cuda|cpu]
           [--shots 3,5,17] [--fps 6] [--refresh-vocabulary] [--cache PATH.npz]
       uv run python objects.py ../runs/<name> --relink PATH.npz   (re-embed and re-link only; no SAM 3)
"""

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

BAKEOFF = Path(__file__).resolve().parent.parent  # extract.json's video.source is relative to this
sys.path.insert(0, str(BAKEOFF))
from caption import MODELS as CAPTION_MODELS  # noqa: E402  (one place for model prices)
from extract import load_env  # noqa: E402

OBJECT_FPS = 6                 # SAM 3 cost is per frame; 6 fps keeps fast bike motion trackable
MAX_OBJECTS = 8                # nouns per shot; each is a separate SAM 3 text prompt on every frame
VOCAB_KEY = "claude-haiku-5-5"
SEGMENTER = "facebook/sam3"    # transformers Sam3VideoModel (detector + tracker)
# bf16 on MPS is ~2.5x faster than fp32; on two NES shots it gave the same objects on every frame and
# matched masks at median IoU 0.998 (p10 0.979, 1 of 128 masks off), so it is the default on MPS.
SEGMENTER_DTYPE = {"mps": torch.bfloat16, "cuda": torch.bfloat16, "cpu": torch.float32}
# DINOv3 ViT-L/16 (gated on Hugging Face; HF_TOKEN from bakeoff.env). Chosen over PE-Core-L14-336, DINOv2-L
# and DINOv3-H+ in the linking bake-off on NES - Biking (objects/link_reference_nes.json): with tight box
# crops it ranks same-object pairs above different-object pairs far better than PE-Core (AP 0.84 vs 0.49)
# and, at the threshold below, wrongly merges 4 of 96 different ramps vs PE-Core's 25, linking the same pairs.
EMBEDDER = "facebook/dinov3-vitl16-pretrain-lvd1689m"
EMBED_PX = 224                 # DINOv3's pretraining resolution; 448 was not better in the bake-off
EMBED_MEAN, EMBED_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet, as DINOv3's processor uses
GREY = 128                     # letterbox bars and masked-out background in crops
POLY_MAX_POINTS = 40           # polygon budget per sample
MIN_TRACK_SAMPLES = 2          # a track seen on a single 6 fps frame is almost always a flicker detection
EMBED_SAMPLES = 8              # crops averaged per track embedding, evenly spaced over the track
MIN_CROP_PX = 24               # crops smaller than this on both sides carry no identity information
CROP_PAD = 0.1                 # fraction of the box added on each side of a displayed crop
CROP_MAX_PX = 384              # displayed crops are kept at most this size
# Cosine for average-linkage identity merging, chosen as before: the highest value that keeps the two
# objects verified by eye whole, the one bike (20 tracks) and the white fastback in shots 14, 20 and 33
# (its three tracks merge at 0.625). It is also the best pair F1 on the reference set's thing labels.
# The bike splits above 0.66.
LINK_THRESHOLD = 0.62
# Cross-shot linking rules added after OOO (2026-10-23), where Haiku named one necklace "pearl necklace" and
# "necklace", and DINOv3 merged small necklaces worn by different people. Tracks link within a name FAMILY
# (classify_names), not only within one exact name. The margins shift a pair's cosine before average linkage.
CROSS_NAME_MARGIN = 0.05       # different names in one family need a little more evidence (provisional)
SAME_WEARER_BONUS = 0.05       # worn items on the same face-clustered person need a little less (provisional)
# Below this median box side a crop carries too little identity (rings were 18-33 px on OOO): such tracks link
# only to tracks on the same wearer, never on appearance alone. Provisional: about OOO's 10th percentile (49 px).
MIN_LINK_PX = 48
WEARER_SHARE = 0.5             # a worn track's wearer is the person whose body holds >= this share of its samples
# Two wearers are certainly different people when their ArcFace centroids (faces/identity_similarity.py) are at
# least this far apart. On OOO, cluster pairs of one panelist were 0.33-0.67 apart and different panelists
# 0.87-1.04, so 0.8 sits in the gap (provisional, one clip). Face distance is used rather than "ever on screen
# together" because over-split clusters can hold two people: Rayvan's clusters E and L share a shot through L's
# stray member, while Aala's and Alyssa's clusters were never on screen together.
DIFFERENT_PERSON_DISTANCE = 0.8
# SAM 3 removes duplicate detections within one text prompt but not across prompts, so a shot prompted with both
# "necklace" and "pearl necklace" tracks Rayvan's necklace twice at once; the on-screen-together rule then forbids
# ever linking the two names. Same-family tracks whose boxes overlap at least this much (mean IoU over their
# shared frames) are one object. On OOO such duplicates were 0.82-1.0 and distinct neighbours at most 0.33
# (Alyssa's two necklaces 0.03).
DUPLICATE_IOU = 0.6

LICENCE = ("SAM 3 weights and code: Meta SAM License (commercial use permitted; acknowledge use in "
           "publications; trade-control, no-reverse-engineering and indemnity terms). "
           "DINOv3 ViT-L/16: Meta DINOv3 License (commercial use permitted; acknowledge use in publications; "
           "redistribution only under the same licence with a copy of it; trade-control, "
           "no-reverse-engineering and indemnity terms; Meta may amend the terms). "
           "Vocabulary, name families and crop verification: Claude Haiku 5.5 output via the Anthropic API.")
LICENCE_MODE = "commercial-ok"

VOCAB_SCHEMA = {
    "type": "object",
    "properties": {"objects": {"type": "array", "items": {"type": "string"}}},
    "required": ["objects"],
    "additionalProperties": False,
}

VOCAB_INSTRUCTIONS = f"""These are three frames from one shot of a video. List up to {MAX_OBJECTS} concrete \
physical objects that are clearly visible in the frames and could be outlined and tracked with a \
segmentation mask.

Rules:
- Short noun phrases a detector would understand: 1-3 words, singular, lower case \
(e.g. "bicycle", "wooden ramp", "coffee mug", "car").
- Never people, faces, hands or other body parts.
- Never generic background or scenery: wall, floor, ground, sky, grass, trees, forest, fog, \
road, water, light, shadow.
- Clothing or worn items only if notable (e.g. "full-face helmet"); not ordinary shirts or pants.
- On-screen text and logos are not objects.
- If an object of the same kind was named in an earlier shot, reuse exactly that phrase.
- Most prominent first. Return an empty list if there is nothing that fits."""

NAMES_SCHEMA = {
    "type": "object",
    "properties": {"names": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "family": {"type": "string"},
                       "kind": {"type": "string", "enum": ["individual", "lookalike"]}, "worn": {"type": "boolean"}},
        "required": ["name", "family", "kind", "worn"], "additionalProperties": False}}},
    "required": ["names"],
    "additionalProperties": False,
}

NAMES_INSTRUCTIONS = """These noun phrases were used to name objects in the shots of one video. Different \
shots sometimes name the same physical object differently ("pearl necklace" and "necklace"), and a \
detector sometimes attaches a name loosely ("white top" on a yellow jacket). Classify every name:

- family: a short common noun shared by all names that could refer to the same physical object. Group \
coarsely by what the thing is: everything worn on the upper body (shirt, t-shirt, top, tank top, cardigan, \
jacket, sweater) is one family, "upper-body clothing"; all necklaces are "necklace"; all chairs are "chair". \
Names that can never be the same object (a ring and a necklace, a car and a bicycle) get different families.
- kind: "individual" if a particular instance could be recognised again in another shot by its appearance \
(a specific bicycle, car, poster, laptop, or a person's jacket, hat or necklace); "lookalike" if it is \
scenery or usually comes in many near-identical copies that cannot be told apart (trees, bushes, posts, \
poles, pillars, street lamps, rocks, dirt mounds, power lines, ceiling lights).
- worn: true if it is worn or carried on a person's body (clothing, jewellery, glasses, hats, held bags).

Return every input name exactly once, spelled exactly as given."""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"views": {"type": "array", "items": {
        "type": "object",
        "properties": {"view": {"type": "integer"},
                       "verdict": {"type": "string", "enum": ["object", "graphic", "absent"]}},
        "required": ["view", "verdict"], "additionalProperties": False}}},
    "required": ["views"],
    "additionalProperties": False,
}

VERIFY_BATCH = 16          # tracks per verification call
# Version of the crop check (views + prompt). Cached verdicts of another version are recomputed.
# v1 (tight middle crop, "clearly inside") rejected 36 of 169 OOO tracks, mostly real necklaces, rings and
#   clothing, and passed six on-screen dollar-sign graphics: a tight crop shows skin or a green shape, no context.
# v2 (one middle context view) fixed those but rejected a bike track whose outline had drifted onto grass at
#   its middle frame, and could only say yes or no to edited-in graphics (movie posters with year captions).
# v3: three moments per track side by side, and a three-way verdict that keeps graphics apart from absences.
VERIFY_VERSION = 3
CONTEXT_SCALE = 2.6        # each view: this many box sides around the box, with the box outlined
CONTEXT_MIN_PX = 200       # ... and at least this many pixels, so tiny objects show their surroundings
CONTEXT_PX = 256           # each moment is resized to this side; three are tiled left to right
CONTEXT_MOMENTS = (0.2, 0.5, 0.8)  # positions in the track's samples


def verify_instructions(label, n):
    return f"""Each of the {n} numbered images (view 1 to view {n}, in order) shows one tracked region of a \
video at three moments, left to right, with a yellow outline drawn around what a segmentation model labelled \
"{label}". For each view, judge what the outline mostly covers across the three moments:

- "object": a real physical {label}, or part of one, or the same broad kind of thing under a slightly \
different name (a jacket called a cardigan, trousers called jeans, a case called a box all count). Small items \
such as jewellery count when the outline sits where they are worn.
- "graphic": a digital graphic, animation, caption or picture added to the video in editing (overlaid \
images, icons, animated scenes), or something only depicted inside such a graphic. Physical things filmed by \
the camera are never "graphic".
- "absent": something else physical: a different kind of object, a body part, empty floor, wall, stage or \
lighting, or an unrecognisable blur.

Answer every view."""


def shot_frames(video, start, end, fps, width, height):
    """Decode [start, end) at fps as full-resolution RGB frames; frame k is at start + k / fps."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
         "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
    times = [round(start + k / fps, 3) for k in range(len(frames))]
    keep = [i for i, t in enumerate(times) if t < end]  # fps rounding can emit one frame past the cut
    return frames[keep], [times[i] for i in keep]


def synchronize(device):
    """Wait for queued GPU work, so per-shot timings measure the work and not just its submission."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# --- SAM 3 post-processing kernels on CPU ----------------------------------------

class CpuCvUtils:
    """Stand-in for the CUDA-only kernels-community/cv-utils kernel, from Meta's sam3 perflib CPU paths."""

    @staticmethod
    def generic_nms(ious, scores, iou_threshold, use_iou_matrix=True):
        ious_np = ious.float().cpu().numpy()
        order = scores.float().cpu().numpy().argsort()[::-1]
        kept = []
        while order.size > 0:
            i = order[0]
            kept.append(i)
            order = order[1:][ious_np[i, order[1:]] <= iou_threshold]
        return torch.tensor(kept, dtype=torch.int64, device=scores.device)

    @staticmethod
    def cc_2d(mask, get_counts=True):
        """8-connected components per (N, 1, H, W) mask; counts = size of each pixel's component (0 off-mask)."""
        masks = mask.cpu().numpy().astype(np.uint8)
        labels_out = np.zeros(masks.shape, np.int32)
        counts_out = np.zeros(masks.shape, np.int32)
        for i in range(masks.shape[0]):
            _, labels, stats, _ = cv2.connectedComponentsWithStats(masks[i, 0], connectivity=8)
            areas = stats[:, cv2.CC_STAT_AREA].astype(np.int32)
            areas[0] = 0  # label 0 is the off-mask region
            labels_out[i, 0] = labels
            counts_out[i, 0] = areas[labels]
        return torch.from_numpy(labels_out).to(mask.device), torch.from_numpy(counts_out).to(mask.device)


def install_cpu_cv_utils():
    from transformers.models.sam3_video import modeling_sam3_video
    modeling_sam3_video.cv_utils_kernel = CpuCvUtils()


# --- geometry ----------------------------------------------------------------------

def mask_polygon(mask):
    """Largest external contour, simplified until it has at most POLY_MAX_POINTS points."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(contour, True)
    poly = cv2.approxPolyDP(contour, epsilon, True)
    while len(poly) > POLY_MAX_POINTS:
        epsilon *= 1.5
        poly = cv2.approxPolyDP(contour, epsilon, True)
    return [[int(x), int(y)] for x, y in poly[:, 0]]


def mask_box(mask):
    ys, xs = np.nonzero(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    return [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1]


def padded_crop(frame, box):
    x, y, w, h = box
    px, py = int(w * CROP_PAD), int(h * CROP_PAD)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(frame.shape[1], x + w + px), min(frame.shape[0], y + h + py)
    return (y0, y1, x0, x1), frame[y0:y1, x0:x1]


def shrink(image):
    h, w = image.shape[:2]
    scale = CROP_MAX_PX / max(h, w)
    return image.copy() if scale >= 1 else cv2.resize(image, (round(w * scale), round(h * scale)),
                                                      interpolation=cv2.INTER_AREA)


def letterbox(image, size):
    """Scale the longer side to size and pad the shorter one with grey: the whole object, undistorted."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    out = np.full((size, size, 3), GREY, np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def object_views(frame, box):
    """(embedded crop, displayed crop). The embedded one is the tight box, letterboxed to a square:
    centre-cropping (the embedders' default) cut thin poles and posts down to their middle, and in the
    bake-off neither masking out the background nor adding context linked better than the plain box.
    The displayed one has a margin, which is what a person wants to see."""
    x, y, w, h = box
    tight = frame[y:y + h, x:x + w]
    return letterbox(tight, EMBED_PX), shrink(padded_crop(frame, box)[1])


def context_view(frame, box):
    """The crop check's view: CONTEXT_SCALE box sides around the box (at least CONTEXT_MIN_PX), box outlined."""
    x, y, w, h = box
    side = max(max(w, h) * CONTEXT_SCALE, CONTEXT_MIN_PX)
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(frame.shape[1], cx + side / 2)), int(min(frame.shape[0], cy + side / 2))
    view = frame[y0:y1, x0:x1].copy()
    thickness = max(2, round(max(view.shape[:2]) / 120))
    cv2.rectangle(view, (x - x0, y - y0), (x - x0 + w, y - y0 + h), (255, 230, 0), thickness)  # RGB yellow
    scale = CONTEXT_PX / max(view.shape[:2])
    view = cv2.resize(view, (round(view.shape[1] * scale), round(view.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    out = np.zeros((CONTEXT_PX, CONTEXT_PX, 3), np.uint8)
    out[:view.shape[0], :view.shape[1]] = view
    return out


def context_strip(track, frame_at):
    """The crop check's image for a track: context_view at CONTEXT_MOMENTS of its samples, left to right,
    separated by white bars."""
    samples = track["samples"]
    picks = [samples[min(len(samples) - 1, int(m * len(samples)))] for m in CONTEXT_MOMENTS]
    bar = np.full((CONTEXT_PX, 6, 3), 255, np.uint8)
    views = [context_view(frame_at[s["t"]], s["box"]) for s in picks]
    return np.hstack([x for v in views for x in (v, bar)][:-1])


def frame_sample(t, mask, score):
    return {"t": t, "box": mask_box(mask), "poly": mask_polygon(mask), "score": round(float(score), 3)}


# --- vocabulary ---------------------------------------------------------------------

def ask_haiku(content, schema, max_tokens=1000):
    """One structured Haiku call: (parsed JSON, input tokens, output tokens)."""
    import anthropic

    response = anthropic.Anthropic().messages.create(
        model=CAPTION_MODELS[VOCAB_KEY]["model"], max_tokens=max_tokens,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {response.stop_details}")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise RuntimeError(f"no text in the response: stop_reason {response.stop_reason}, "
                           f"blocks {[b.type for b in response.content]}, usage {response.usage}")
    return json.loads(text), response.usage.input_tokens, response.usage.output_tokens


def haiku_cost(tokens_in, tokens_out):
    price_in, price_out = CAPTION_MODELS[VOCAB_KEY]["price"]
    return round((tokens_in * price_in + tokens_out * price_out) / 1e6, 4)


def jpeg_block(data):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(data).decode()}}


def ask_vocabulary(images, earlier):
    earlier_text = ", ".join(sorted(earlier)) or "(none yet)"
    prompt = f"{VOCAB_INSTRUCTIONS}\n\nNames used in earlier shots: {earlier_text}"
    content = [jpeg_block(p.read_bytes()) for p in images] + [{"type": "text", "text": prompt}]
    out, n_in, n_out = ask_haiku(content, VOCAB_SCHEMA)
    return out["objects"], n_in, n_out


def build_vocabulary(run_dir, shots):
    """One Haiku call per shot, in order, each told the names used before it (never after)."""
    tokens_in = tokens_out = 0
    vocab, earlier = {}, set()
    t0 = time.time()
    for shot in shots:
        names, n_in, n_out = ask_vocabulary([run_dir / k["file"] for k in shot["keyframes"]], earlier)
        # The schema cannot cap list length; the prompt asks for most prominent first, so keep the head.
        names = list(dict.fromkeys(n.strip().lower() for n in names if n.strip()))[:MAX_OBJECTS]
        vocab[str(shot["index"])] = names
        earlier.update(names)
        tokens_in, tokens_out = tokens_in + n_in, tokens_out + n_out
        print(f"  vocabulary shot {shot['index']}: {names}")
    return {"model": VOCAB_KEY, "cost_usd": haiku_cost(tokens_in, tokens_out), "input_tokens": tokens_in,
            "output_tokens": tokens_out, "seconds": round(time.time() - t0, 1), "shots": vocab}


def classify_names(vocabulary):
    """One Haiku call over the clip's whole name list: family, kind (individual/lookalike) and worn, per name.
    Replaces the old hard-coded list of look-alike labels, which only covered NES - Biking."""
    names = sorted({n for shot in vocabulary["shots"].values() for n in shot})
    t0 = time.time()
    out, n_in, n_out = ask_haiku([{"type": "text", "text": NAMES_INSTRUCTIONS + "\n\nNames:\n" + "\n".join(names)}],
                                 NAMES_SCHEMA, max_tokens=4000)
    classes = {c["name"].strip().lower(): {k: c[k] for k in ("family", "kind", "worn")} for c in out["names"]}
    missing, extra = set(names) - set(classes), set(classes) - set(names)
    if missing or extra:
        raise RuntimeError(f"name classification does not match the vocabulary: missing {sorted(missing)}, "
                           f"unexpected {sorted(extra)}")
    for c in classes.values():
        c["family"] = c["family"].strip().lower()
    return {"model": VOCAB_KEY, "cost_usd": haiku_cost(n_in, n_out), "seconds": round(time.time() - t0, 1),
            "names": classes}


def verify_tracks(tracks, cache_path):
    """Is each track really the thing it is named after? Haiku sees each track's context strip (context_strip),
    VERIFY_BATCH tracks of one label per call, and answers object / graphic / absent. Verdicts are cached by
    (track id, label) in cache_path and saved after every call, so a re-link only pays for tracks it has not
    seen; a cache of another VERIFY_VERSION is discarded. Tracks without a context strip get no verdict.
    Returns {track id: verdict} and the cache's total cost."""
    from PIL import Image
    import io

    cache = json.loads(cache_path.read_text()) if cache_path.exists() else None
    if cache is None or cache.get("version") != VERIFY_VERSION:
        cache = {"model": VOCAB_KEY, "version": VERIFY_VERSION, "cost_usd": 0.0, "verdicts": {}}
    todo = [t for t in tracks if t.get("context") is not None
            and cache["verdicts"].get(t["id"], {}).get("label") != t["label"]]
    for label in sorted({t["label"] for t in todo}):
        group = [t for t in todo if t["label"] == label]
        for start in range(0, len(group), VERIFY_BATCH):
            batch = group[start:start + VERIFY_BATCH]
            content = []
            for k, t in enumerate(batch, 1):
                buf = io.BytesIO()
                Image.fromarray(t["context"]).save(buf, "JPEG", quality=90)
                content += [{"type": "text", "text": f"view {k}"}, jpeg_block(buf.getvalue())]
            content.append({"type": "text", "text": verify_instructions(label, len(batch))})
            out, n_in, n_out = ask_haiku(content, VERIFY_SCHEMA, max_tokens=4000)
            answers = {a["view"]: a["verdict"] for a in out["views"]}
            if sorted(answers) != list(range(1, len(batch) + 1)):
                raise RuntimeError(f"verification of '{label}' answered views {sorted(answers)}, "
                                   f"expected 1..{len(batch)}")
            for k, t in enumerate(batch, 1):
                cache["verdicts"][t["id"]] = {"label": label, "verdict": answers[k]}
            cache["cost_usd"] = round(cache["cost_usd"] + haiku_cost(n_in, n_out), 4)
            cache_path.write_text(json.dumps(cache, indent=1))
    return {tid: v["verdict"] for tid, v in cache["verdicts"].items()}, cache["cost_usd"]


# --- segmentation + tracking ----------------------------------------------------------

def load_segmenter(device):
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    install_cpu_cv_utils()
    model = Sam3VideoModel.from_pretrained(SEGMENTER, torch_dtype=SEGMENTER_DTYPE[device.type]).to(device).eval()
    return model, Sam3VideoProcessor.from_pretrained(SEGMENTER)


@torch.inference_mode()
def segment_shot(segmenter, frames, times, nouns, device):
    """Run SAM 3 over one shot; return {sam_obj_id: track} with per-sample geometry."""
    model, processor = segmenter
    session = processor.init_video_session(video=list(frames), inference_device=device, processing_device="cpu",
                                           video_storage_device="cpu", dtype=SEGMENTER_DTYPE[device.type])
    processor.add_text_prompt(session, nouns)
    tracks = {}
    for out in model.propagate_in_video_iterator(inference_session=session):
        post = processor.postprocess_outputs(session, out)
        prompt_of = {obj: text for text, objs in post["prompt_to_obj_ids"].items() for obj in objs}
        masks = post["masks"].cpu().numpy()
        for obj_id, det_score, mask in zip(post["object_ids"].tolist(), post["scores"].tolist(), masks):
            if not mask.any():
                continue
            sample = frame_sample(times[out.frame_idx], mask, out.obj_id_to_tracker_score.get(obj_id, det_score))
            track = tracks.setdefault(obj_id, {"label": prompt_of[obj_id], "score": det_score, "samples": []})
            track["samples"].append(sample)
    del session
    return tracks


# --- embedding + identities ---------------------------------------------------------------

def load_embedder(device):
    from transformers import AutoModel
    return AutoModel.from_pretrained(EMBEDDER).to(device).eval()


@torch.inference_mode()
def embed_crops(model, crops, device):
    """Crops are EMBED_PX squares (object_views); returns (normalised mean, normalised per-crop CLS)."""
    batch = torch.from_numpy(np.stack(crops)).to(device).permute(0, 3, 1, 2).float() / 255
    mean, std = (torch.tensor(v, device=device)[None, :, None, None] for v in (EMBED_MEAN, EMBED_STD))
    feats = model(pixel_values=(batch - mean) / std).pooler_output
    feats = torch.nn.functional.normalize(feats.float(), dim=-1)
    return torch.nn.functional.normalize(feats.mean(0), dim=-1).cpu().numpy(), feats.cpu().numpy()


def embed_tracks(tracks, video, video_meta, shots, fps, model, device):
    """Embed every track from crops re-cut at its sample boxes, so a --relink matches a full run.
    Adds embedding, view_embeddings and views (displayed crops) to each track."""
    by_shot = {}
    for t in tracks:
        by_shot.setdefault(t["shot"], []).append(t)
    for shot in shots:
        if shot["index"] not in by_shot:
            continue
        frames, times = shot_frames(video, shot["start"], shot["end"], fps, video_meta["width"],
                                    video_meta["height"])
        frame_at = {t: f for t, f in zip(times, frames)}
        for track in by_shot[shot["index"]]:
            usable = [s for s in track["samples"] if max(s["box"][2:]) >= MIN_CROP_PX]
            picks = [usable[int(i)] for i in np.linspace(0, len(usable) - 1, min(EMBED_SAMPLES, len(usable)))] \
                if usable else []
            views = [object_views(frame_at[s["t"]], s["box"]) for s in picks]
            track["embedding"], track["view_embeddings"] = \
                embed_crops(model, [e for e, _ in views], device) if views else (None, None)
            track["views"] = [shown for _, shown in views]
            track["context"] = context_strip(track, frame_at)
        synchronize(device)


def overlaps(a, b):
    return a["shot"] == b["shot"] and a["start"] <= b["end"] and b["start"] <= a["end"]


def contains(box, x, y):
    bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh


def assign_wearers(tracks, run_dir, classes):
    """Who wears each worn track: the face-clustered person (faces.json "person") whose posed body
    (pose_rtmw.json, linked to a face track) holds the object's box centre in >= WEARER_SHARE of its samples.
    Where bodies overlap, the smallest containing body box wins; faces.json's catch-all "Unknown" is never a
    wearer. Returns the set of person pairs that are certainly different people (DIFFERENT_PERSON_DISTANCE),
    or None when the run has no identity_similarity.json: then the different-wearer rule is off, and
    write_output records that.
    Persons are the machine clusters; the annotator's human renames and merges are not applied here."""
    faces = json.loads((run_dir / "faces.json").read_text())["tracks"]
    pose = json.loads((run_dir / "pose_rtmw.json").read_text())
    step = 1 / pose["fps"]
    bodies = {}  # shot -> [(t, box, person)]
    for body in pose["tracks"]:
        if body.get("face_track") is None or faces[body["face_track"]]["person"] == "Unknown":
            continue
        person = faces[body["face_track"]]["person"]
        bodies.setdefault(body["shot"], []).extend((s["t"], s["box"], person) for s in body["samples"])
    for t in tracks:
        t["wearer"] = None
        if not classes[t["label"]]["worn"]:
            continue
        votes = {}
        for s in t["samples"]:
            x, y = s["box"][0] + s["box"][2] / 2, s["box"][1] + s["box"][3] / 2
            near = [(b[2] * b[3], person) for bt, b, person in bodies.get(t["shot"], [])
                    if abs(bt - s["t"]) <= step / 2 + 1e-6 and contains(b, x, y)]
            if near:
                person = min(near)[1]
                votes[person] = votes.get(person, 0) + 1
        if votes:
            person, n = max(votes.items(), key=lambda kv: kv[1])
            if n >= WEARER_SHARE * len(t["samples"]):
                t["wearer"] = person
    similarity = run_dir / "identity_similarity.json"
    if not similarity.exists():
        return None
    return {frozenset((p["a"], p["b"])) for p in json.loads(similarity.read_text())["pairs"]
            if p["distance"] >= DIFFERENT_PERSON_DISTANCE}


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return inter / (aw * ah + bw * bh - inter)


def duplicates(a, b):
    """Two tracks of one object in the same frames (SAM 3 cross-prompt duplicates): mean box IoU over shared
    sample times >= DUPLICATE_IOU."""
    if not overlaps(a, b):
        return False
    boxes_b = {s["t"]: s["box"] for s in b["samples"]}
    shared = [box_iou(s["box"], boxes_b[s["t"]]) for s in a["samples"] if s["t"] in boxes_b]
    return bool(shared) and float(np.mean(shared)) >= DUPLICATE_IOU


def track_size(t):
    return float(np.median([max(s["box"][2:]) for s in t["samples"]]))


def pair_score(a, b, different):
    """Adjusted cosine for linking a and b, or None when they may never be one object.
    different: person pairs that are certainly different people (assign_wearers), or None if unknown."""
    if overlaps(a, b) or a["embedding"] is None or b["embedding"] is None:
        return None
    wa, wb = a.get("wearer"), b.get("wearer")
    if different is not None and wa and wb and wa != wb and frozenset((wa, wb)) in different:
        return None  # worn by two different people
    same_wearer = bool(wa) and wa == wb
    if min(track_size(a), track_size(b)) < MIN_LINK_PX and not same_wearer:
        return None  # too small to identify by appearance alone
    score = float(a["embedding"] @ b["embedding"])
    if a["label"] != b["label"]:
        score -= CROSS_NAME_MARGIN
    if same_wearer:
        score += SAME_WEARER_BONUS
    return score


def link_identities(tracks, threshold, classes, different):
    """Average-linkage clustering within each name family; a merge needs every cross pair linkable (pair_score
    not None) and an average adjusted score >= threshold. Clustering starts from groups of duplicate tracks
    (duplicates()), which are one object from the outset. Lookalike families never link across tracks.
    Each identity is named after its members' most common label."""
    identities, numbering = [], {}
    for family in sorted({classes[t["label"]]["family"] for t in tracks}):
        members = [t for t in tracks if classes[t["label"]]["family"] == family]
        lookalike = all(classes[t["label"]]["kind"] == "lookalike" for t in members)
        n = len(members)
        score = [[pair_score(members[i], members[j], different) if i != j else None for j in range(n)]
                 for i in range(n)]
        group = list(range(n))  # union-find over duplicate pairs
        def root(i):
            while group[i] != i:
                group[i] = group[group[i]]
                i = group[i]
            return i
        for i in range(n):
            for j in range(i + 1, n):
                if duplicates(members[i], members[j]):
                    group[root(j)] = root(i)
        clusters = {}
        for i in range(n):
            clusters.setdefault(root(i), []).append(i)
        clusters = list(clusters.values())
        while not lookalike:
            best = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    pairs = [score[a][b] for a in clusters[i] for b in clusters[j]]
                    if any(p is None for p in pairs):
                        continue
                    avg = float(np.mean(pairs))
                    if avg >= threshold and (best is None or avg > best[0]):
                        best = (avg, i, j)
            if best is None:
                break
            _, i, j = best
            clusters[i] += clusters.pop(j)
        clusters = [[members[i] for i in c] for c in clusters]
        clusters.sort(key=lambda c: min(t["start"] for t in c))
        for cluster in clusters:
            labels = [t["label"] for t in cluster]
            label = max(sorted(set(labels)), key=labels.count)
            numbering[label] = numbering.get(label, 0) + 1
            wearers = {t["wearer"] for t in cluster if t.get("wearer")}
            identities.append({"id": f"{label} {numbering[label]}", "label": label, "family": family,
                               "worn_by": sorted(wearers),
                               "tracks": [t["id"] for t in sorted(cluster, key=lambda t: t["start"])]})
    return identities


def crop_name(identity_id):
    return re.sub(r"[^a-z0-9]+", "_", identity_id.lower()).strip("_") + ".jpg"


# --- main ------------------------------------------------------------------------------

def write_cache(path, tracks, timings, segmenter_name, fps):
    """The SAM 3 output (tracks with their samples), so --relink can re-embed and re-link without SAM 3."""
    keep = ("id", "label", "shot", "start", "end", "score", "samples")
    np.savez(path, meta=json.dumps({"tracks": [{k: t[k] for k in keep} for t in tracks], "timings": timings,
                                    "segmenter": segmenter_name, "fps": fps}))


def read_cache(path):
    """Tracks, timings, segmenter and fps. Caches written before 2026-10-23 also hold PE-Core embeddings and
    crops ("arrays"); they are not read, because embeddings are now always recomputed from the video."""
    meta = json.loads(str(np.load(path)["meta"]))
    return meta["tracks"], meta["timings"], meta["segmenter"], meta["fps"]


def representative_view(identity, by_id):
    """The embedded view closest to the identity's mean embedding."""
    members = [by_id[t] for t in identity["tracks"] if by_id[t]["embedding"] is not None]
    if not members:
        return None
    centre = np.mean([t["embedding"] for t in members], axis=0)
    views = [(float(e @ centre), v) for t in members for e, v in zip(t["view_embeddings"], t["views"])]
    return max(views, key=lambda x: x[0])[1]


def name_classes(run_dir):
    """The vocabulary's name families, classified once and cached in object_vocabulary.json under "classes"."""
    vocab_path = run_dir / "object_vocabulary.json"
    vocabulary = json.loads(vocab_path.read_text())
    names = {n for shot in vocabulary["shots"].values() for n in shot}
    if set(vocabulary.get("classes", {}).get("names", {})) != names:
        vocabulary["classes"] = classify_names(vocabulary)
        vocab_path.write_text(json.dumps(vocabulary, indent=1))
    return vocabulary["classes"]


def write_output(run_dir, tracks, timings, segmenter_name, fps):
    from PIL import Image
    classes = name_classes(run_dir)
    verdict, verify_cost = verify_tracks(tracks, run_dir / "object_verification.json")
    rejected = [t for t in tracks if verdict.get(t["id"]) in ("graphic", "absent")]
    tracks = [t for t in tracks if verdict.get(t["id"]) not in ("graphic", "absent")]
    different = assign_wearers(tracks, run_dir, classes["names"])
    if different is None:
        print(f"different-wearer rule OFF: {run_dir}/identity_similarity.json does not exist")
    identities = link_identities(tracks, LINK_THRESHOLD, classes["names"], different)
    crop_dir = run_dir / "object_crops"
    crop_dir.mkdir(exist_ok=True)
    for old in crop_dir.glob("*.jpg"):  # identities are renumbered on every run
        old.unlink()
    by_id = {t["id"]: t for t in tracks}
    for identity in identities:
        view = representative_view(identity, by_id)
        if view is None:
            identity["crop"] = None
            continue
        name = crop_name(identity["id"])
        Image.fromarray(view).save(crop_dir / name, quality=90)
        identity["crop"] = f"object_crops/{name}"
    identity_of = {t: i["id"] for i in identities for t in i["tracks"]}
    out = {"segmenter": segmenter_name, "detector": None, "embedder": EMBEDDER,
           "vocabulary": "object_vocabulary.json", "verification": "object_verification.json", "fps": fps,
           "link_threshold": LINK_THRESHOLD, "cross_name_margin": CROSS_NAME_MARGIN,
           "same_wearer_bonus": SAME_WEARER_BONUS, "min_link_px": MIN_LINK_PX,
           "different_wearer_rule": (f"on: face distance >= {DIFFERENT_PERSON_DISTANCE}" if different is not None
                                     else "off: no identity_similarity.json (run faces/identity_similarity.py)"),
           "licence_mode": LICENCE_MODE, "licence": LICENCE, "timings": timings,
           "haiku_cost_usd": {"names": classes["cost_usd"], "verification": verify_cost},
           "tracks": [{"id": t["id"], "label": t["label"], "identity": identity_of[t["id"]], "shot": t["shot"],
                       "start": t["start"], "end": t["end"], "score": t["score"], "wearer": t["wearer"],
                       "verdict": verdict.get(t["id"]), "samples": t["samples"]} for t in tracks],
           "identities": identities,
           # tracks the crop check judged "graphic" (edited-in images, e.g. movie posters with year captions:
           # meaningful, but not objects in the scene) or "absent": never linked or shown as objects
           "rejected_tracks": [{"id": t["id"], "label": t["label"], "verdict": verdict[t["id"]], "shot": t["shot"],
                                "start": t["start"], "end": t["end"], "samples": t["samples"]} for t in rejected]}
    (run_dir / "objects.json").write_text(json.dumps(out))

    print(f"\n{len(tracks)} tracks ({len(rejected)} rejected by the crop check), {len(identities)} identities "
          f"({sum(len(i['tracks']) > 1 for i in identities)} linking more than one track), "
          f"{sum(1 for t in tracks if t['wearer'])} tracks with a wearer")
    for family in sorted({i["family"] for i in identities}):
        ids = [i for i in identities if i["family"] == family]
        n_t = sum(len(i["tracks"]) for i in ids)
        print(f"  {family}: {n_t} tracks -> {len(ids)} identities")
    for t in rejected:
        print(f"  rejected ({verdict[t['id']]}): {t['id']} '{t['label']}' shot {t['shot']} at {t['start']} s")
    print(f"timings: {timings}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--shots", help="comma-separated shot indices (default: all)")
    parser.add_argument("--fps", type=float, default=OBJECT_FPS)
    parser.add_argument("--refresh-vocabulary", action="store_true")
    parser.add_argument("--cache", type=Path, help="save the SAM 3 tracks here (.npz)")
    parser.add_argument("--relink", type=Path, help="skip SAM 3: re-embed and re-link the tracks of a --cache file")
    args = parser.parse_args()
    load_env()

    extract = json.loads((args.run_dir / "extract.json").read_text())
    video_meta = extract["video"]
    video = BAKEOFF / video_meta["source"]
    shots = extract["shots"]
    device = torch.device(args.device)

    if args.relink:
        tracks, timings, segmenter_name, fps = read_cache(args.relink)
        t0 = time.time()
        embed_tracks(tracks, video, video_meta, shots, fps, load_embedder(device), device)
        # per_shot_s in older caches summed SAM 3 and the old embedder's time; segment_per_shot_s replaces it.
        timings = {k: v for k, v in timings.items() if k != "per_shot_s"}
        timings["embed_s"] = round(time.time() - t0, 1)  # includes decoding the shots and loading the model
        write_output(args.run_dir, tracks, timings, segmenter_name, fps)
        return

    timings = {}

    vocab_path = args.run_dir / "object_vocabulary.json"
    if vocab_path.exists() and not args.refresh_vocabulary:
        vocabulary = json.loads(vocab_path.read_text())
    else:
        vocabulary = build_vocabulary(args.run_dir, shots)
        vocab_path.write_text(json.dumps(vocabulary, indent=1))
    timings["vocabulary_s"] = vocabulary["seconds"]  # time of the Haiku calls, also when read from the cache
    print(f"vocabulary: {vocabulary['model']}, ${vocabulary['cost_usd']}, "
          f"{sum(len(v) for v in vocabulary['shots'].values())} nouns over {len(vocabulary['shots'])} shots")

    selected = set(map(int, args.shots.split(","))) if args.shots else None
    t0 = time.time()
    segmenter = load_segmenter(device)
    load_s = time.time() - t0

    tracks = []
    segment_s = 0.0
    per_shot = []
    for shot in shots:
        if selected is not None and shot["index"] not in selected:
            continue
        nouns = vocabulary["shots"][str(shot["index"])]
        if not nouns:
            print(f"  shot {shot['index']}: no nouns")
            continue
        t0 = time.time()
        frames, times = shot_frames(video, shot["start"], shot["end"], args.fps, video_meta["width"],
                                    video_meta["height"])
        shot_tracks = segment_shot(segmenter, frames, times, nouns, device)
        synchronize(device)
        seg = time.time() - t0

        kept = 0
        for obj_id, track in sorted(shot_tracks.items()):
            if len(track["samples"]) < MIN_TRACK_SAMPLES:
                continue
            tracks.append({"id": f"obj:{len(tracks)}", "label": track["label"], "shot": shot["index"],
                           "start": track["samples"][0]["t"], "end": track["samples"][-1]["t"],
                           "score": round(float(track["score"]), 3), "samples": track["samples"]})
            kept += 1
        segment_s += seg
        per_shot.append({"shot": shot["index"], "frames": len(frames), "nouns": len(nouns),
                         "segment_s": round(seg, 2), "tracks": kept})
        print(f"  shot {shot['index']} ({shot['end'] - shot['start']:.1f} s, {len(frames)} frames, "
              f"{len(nouns)} nouns): {kept} tracks of {len(shot_tracks)}, segment {seg:.1f} s")

    del segmenter
    t0 = time.time()
    embed_tracks(tracks, video, video_meta, shots, args.fps, load_embedder(device), device)
    timings.update({"segment_s": round(segment_s, 1), "embed_s": round(time.time() - t0, 1),
                    "segment_per_shot_s": round(segment_s / max(1, len(per_shot)), 2)})
    segmenter_name = f"{SEGMENTER} (Sam3VideoModel, {str(SEGMENTER_DTYPE[device.type]).removeprefix('torch.')})"
    if args.cache:
        write_cache(args.cache, tracks, timings, segmenter_name, args.fps)
    write_output(args.run_dir, tracks, timings, segmenter_name, args.fps)
    print(f"model load {load_s:.1f} s (not in timings)")
    print("per shot: " + json.dumps(per_shot))
    if device.type == "mps":
        print(f"MPS driver allocated at end: {torch.mps.driver_allocated_memory() / 1e9:.1f} GB")
    elif device.type == "cuda":
        print(f"CUDA peak allocated: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
