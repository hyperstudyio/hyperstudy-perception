"""Stage 2 of the bake-off: per-shot captioning, one model at a time.

Every model gets identical inputs per shot: the shot's three keyframes, the
transcript lines that overlap it, and its OWN earlier output (roster of people
and the previous shot's description). Nothing from later shots is ever sent,
so a caption cannot describe the future.

Usage: uv run python caption.py runs/<name> [--with-faces] MODEL_KEY [MODEL_KEY ...]
Model keys are the keys of MODELS below. --with-faces uses faces.json (and
speaker_links.json if present): keyframes get labelled boxes and the prompt
gets the face roster, so the model reuses those labels instead of inventing
its own. Output is saved as captions/<key>+faces.json.
"""

import base64
import json
import sys
import time
from pathlib import Path

from extract import load_env

# USD per 1M tokens (input, output). None = price not known to this script;
# the report shows tokens and leaves cost blank.
MODELS = {
    "claude-opus-5-5": {"provider": "anthropic", "model": "claude-opus-5-5", "price": (4.00, 20.00)},
    "claude-haiku-5-5": {"provider": "anthropic", "model": "claude-haiku-5-5", "price": (0.10, 0.50)},
    "gpt-6.1-sol": {"provider": "openai", "model": "gpt-6.1-sol", "price": None},
    "qwen3.6-35b-a3b-local": {"provider": "local", "model": "qwen3.6-35b-a3b", "price": (0.0, 0.0),
                              "base_url": "http://127.0.0.1:8080/v1"},
}

SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "setting": {"type": "string"},
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "appearance": {"type": "string"},
                    "action": {"type": "string"},
                    "speaking": {"type": "boolean"},
                },
                "required": ["label", "appearance", "action", "speaking"],
                "additionalProperties": False,
            },
        },
        "onscreen_text": {"type": "array", "items": {"type": "string"}},
        "mood": {"type": "string"},
    },
    "required": ["description", "setting", "people", "onscreen_text", "mood"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You are writing one entry of a time-coded description track for a short film. \
An AI participant will read these entries while watching alongside people, so each entry must \
say what someone watching would see and hear during THIS shot only.

Rules:
- Describe only what is visible in the keyframes or audible in the transcript lines for this shot. \
Do not guess what happens later or why, and do not describe earlier shots again.
- people: everyone visible. Reuse a label from the roster when it is clearly the same person; \
otherwise use the next unused letter ("Person A", "Person B", ...). Never use a real name unless \
the name is spoken in the transcript or written on screen.
- onscreen_text: legible text in the frames, verbatim; [] if none.
- description: 1-3 plain present-tense sentences. setting and mood: a short phrase each.
Return only the JSON object."""


MAX_WORD_SECONDS = 2.0  # a longer "word" is a forced-alignment failure that absorbed preceding silence or music


def word_onset(word):
    if word["end"] - word["start"] > MAX_WORD_SECONDS:
        return word["end"] - 0.4
    return word["start"]


def shot_speech(shot, transcript):
    """Lines of speech whose words start inside the shot, grouped by speaker.

    Uses word times rather than segment times: WhisperX segment starts inherit
    any misaligned first word (one segment here began 23 s before its speech).
    """
    lines = []
    for segment in transcript["segments"]:
        for word in segment["words"]:
            if word["start"] is None or not shot["start"] <= word_onset(word) < shot["end"]:
                continue
            speaker = word["speaker"] or segment["speaker"] or "SPEAKER"
            if lines and lines[-1]["speaker"] == speaker:
                lines[-1]["words"].append(word["word"])
            else:
                lines.append({"t": word_onset(word), "speaker": speaker, "words": [word["word"]]})
    return "\n".join(f"[{l['t']:.1f}s {l['speaker']}] {' '.join(l['words'])}" for l in lines) or "(no speech)"


def shot_prompt(shot, transcript, roster, previous):
    speech = shot_speech(shot, transcript)
    roster_text = "\n".join(f"- {label}: {appearance}" for label, appearance in roster.items()) or "(nobody yet)"
    return (f"{INSTRUCTIONS}\n\nShot {shot['index']} runs {shot['start']:.1f}s to {shot['end']:.1f}s. "
            f"Keyframes are at {', '.join(str(k['t']) + 's' for k in shot['keyframes'])}.\n\n"
            f"Speech during this shot:\n{speech}\n\nPeople seen in earlier shots:\n{roster_text}\n\n"
            f"Previous shot: {previous or '(this is the first shot)'}")


FACES_INSTRUCTIONS = """Face tracking has already identified people. Labelled boxes are drawn on the \
keyframes. Use exactly these labels for the boxed people and the roster below; do not invent new \
labels for them. Someone visible but not boxed (back turned, too small) may be described without a \
label, or as "Unlabelled person" in people. The boxes and labels were added for you; they are not \
part of the film, so never describe or mention them."""

BOX_COLOURS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]


def annotate_keyframes(run_dir, shot, faces):
    """Draw each tracked face that is on screen within 0.2 s of a keyframe, with its label."""
    from PIL import Image, ImageDraw

    colour = {p["label"]: BOX_COLOURS[i % len(BOX_COLOURS)] for i, p in enumerate(faces["people"])}
    out_dir = run_dir / "keyframes_faces"
    out_dir.mkdir(exist_ok=True)
    paths = []
    for keyframe in shot["keyframes"]:
        path = out_dir / Path(keyframe["file"]).name
        paths.append(path)
        if path.exists():  # drawn once per run, shared by every model
            continue
        image = Image.open(run_dir / keyframe["file"]).convert("RGB")
        scale = image.width / faces["video_width"]
        draw = ImageDraw.Draw(image)
        for track in faces["tracks"]:
            if track["person"] == "Unknown":  # unattributed faces are not labelled for the model
                continue
            near = [s for s in track["series"] if abs(s["t"] - keyframe["t"]) <= 0.2]
            if not near:
                continue
            x, y, w, h = (v * scale for v in min(near, key=lambda s: abs(s["t"] - keyframe["t"]))["box"])
            c = colour[track["person"]]
            draw.rectangle([x, y, x + w, y + h], outline=c, width=3)
            tab_top = max(0, y - 18)  # label tab above the box, pushed inside the frame at the top edge
            draw.rectangle([x, tab_top, x + 8 * len(track["person"]) + 6, tab_top + 18], fill=c)
            draw.text((x + 3, tab_top + 2), track["person"], fill="white")
        image.save(path, quality=90)
    return paths


def faces_context(shot, faces, links):
    seen = [p for p in faces["people"] if p["first_seen"] < shot["end"]]  # never people who appear later
    roster = "\n".join(f"- {p['label']}" for p in seen) or "(nobody yet)"
    voices = "\n".join(f"- {speaker} is {link['person']}" if link["person"] else f"- {speaker} is not on screen"
                        for speaker, link in (links or {}).items())
    return f"{FACES_INSTRUCTIONS}\n\nFace roster so far:\n{roster}" + (f"\n\nVoices:\n{voices}" if voices else "")


def b64(path):
    return base64.standard_b64encode(path.read_bytes()).decode()


def call_anthropic(cfg, prompt, images):
    import anthropic

    client = anthropic.Anthropic()
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(p)}}
               for p in images] + [{"type": "text", "text": prompt}]
    response = client.messages.create(
        model=cfg["model"], max_tokens=4000,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {response.stop_details}")
    text = next(b.text for b in response.content if b.type == "text")
    return text, response.usage.input_tokens, response.usage.output_tokens


def call_openai(cfg, prompt, images):
    from openai import OpenAI

    client = OpenAI()
    content = [{"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64(p)}"} for p in images]
    content.append({"type": "input_text", "text": prompt})
    response = client.responses.create(
        model=cfg["model"],
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_schema", "name": "shot_caption", "schema": SCHEMA, "strict": True}},
    )
    return response.output_text, response.usage.input_tokens, response.usage.output_tokens


def call_local(cfg, prompt, images):
    from openai import OpenAI

    client = OpenAI(base_url=cfg["base_url"], api_key="local")
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(p)}"}} for p in images]
    content.append({"type": "text", "text": prompt})
    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_schema", "json_schema": {"name": "shot_caption", "schema": SCHEMA}},
    )
    return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens


CALLERS = {"anthropic": call_anthropic, "openai": call_openai, "local": call_local}


def run_model(run_dir, key, with_faces):
    cfg = MODELS[key]
    extract = json.loads((run_dir / "extract.json").read_text())
    faces = links = None
    if with_faces:
        faces = json.loads((run_dir / "faces.json").read_text())
        faces["video_width"] = extract["video"]["width"]
        links_path = run_dir / "speaker_links.json"
        links = json.loads(links_path.read_text())["links"] if links_path.exists() else None
    roster, previous, shots_out = {}, None, []
    totals = {"input_tokens": 0, "output_tokens": 0, "seconds": 0.0, "errors": 0}

    for shot in extract["shots"]:
        prompt = shot_prompt(shot, extract["transcript"], roster, previous)
        images = [run_dir / k["file"] for k in shot["keyframes"]]
        if faces:
            prompt += "\n\n" + faces_context(shot, faces, links)
            images = annotate_keyframes(run_dir, shot, faces)
        t0 = time.time()
        try:
            text, tokens_in, tokens_out = CALLERS[cfg["provider"]](cfg, prompt, images)
            caption, error = json.loads(text), None
        except Exception as exc:  # recorded per shot so one failure does not hide the rest
            caption, error, tokens_in, tokens_out = None, f"{type(exc).__name__}: {exc}", 0, 0
            totals["errors"] += 1
        elapsed = time.time() - t0

        totals["input_tokens"] += tokens_in
        totals["output_tokens"] += tokens_out
        totals["seconds"] += elapsed
        shots_out.append({"index": shot["index"], "caption": caption, "error": error, "seconds": round(elapsed, 2),
                          "input_tokens": tokens_in, "output_tokens": tokens_out})
        if caption:
            for person in caption["people"]:
                roster.setdefault(person["label"], person["appearance"])
            previous = caption["description"]
        print(f"  {key} shot {shot['index']}: {elapsed:.1f}s" + (f" ERROR {error}" if error else ""))

    price = cfg["price"]
    totals["cost_usd"] = (None if price is None else
                          round((totals["input_tokens"] * price[0] + totals["output_tokens"] * price[1]) / 1e6, 4))
    totals["seconds"] = round(totals["seconds"], 1)
    name = f"{key}+faces" if with_faces else key
    out = {"model": name, "config": {k: v for k, v in cfg.items() if k != "price"}, "price_per_mtok": price,
           "totals": totals, "roster": roster, "shots": shots_out}
    (run_dir / "captions").mkdir(exist_ok=True)
    (run_dir / "captions" / f"{name}.json").write_text(json.dumps(out, indent=2))
    print(f"{name}: {totals}")


def main():
    load_env()
    run_dir = Path(sys.argv[1])
    with_faces = "--with-faces" in sys.argv
    for key in [a for a in sys.argv[2:] if a != "--with-faces"]:
        run_model(run_dir, key, with_faces)


if __name__ == "__main__":
    main()
