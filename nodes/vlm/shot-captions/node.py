"""vlm.shot-captions: one description per shot from its keyframes and speech, with the face roster.

Ported from pipeline/bakeoff/caption.py (Claude Haiku 5.5 with the face roster: 126 of 135 boxed people named on
OOO, about $0.04 a clip). Each shot sees its three keyframes, with tracked faces boxed and labelled, the speech that
starts inside it, the people seen so far and the previous shot's description; never anything from later shots.
That chaining is why the calls run in order rather than as a Message Batch.

Writes the shots layer with captions (it refines the decode node's shots layer) and the on-screen text layer.
"""

import base64
import io
import json
import time

from perception_node import Node, NodeError

PRICE_PER_MTOK = {"claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.00, 20.00)}   # input, output (USD)

SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "setting": {"type": "string"},
        "people": {"type": "array", "items": {
            "type": "object",
            "properties": {"label": {"type": "string"}, "appearance": {"type": "string"}, "action": {"type": "string"},
                           "speaking": {"type": "boolean"}},
            "required": ["label", "appearance", "action", "speaking"], "additionalProperties": False}},
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

FACES_INSTRUCTIONS = """Face tracking has already identified people. Labelled boxes are drawn on the \
keyframes. Use exactly these labels for the boxed people and the roster below; do not invent new \
labels for them. Someone visible but not boxed (back turned, too small) may be described without a \
label, or as "Unlabelled person" in people. The boxes and labels were added for you; they are not \
part of the film, so never describe or mention them."""

BOX_COLOURS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]
MAX_WORD_SECONDS = 2.0   # a longer "word" is an alignment failure that absorbed preceding silence or music


def label(person_id: str) -> str:
    return person_id.split(":", 1)[1]


def shot_speech(shot, words) -> str:
    lines = []
    for w in words:
        onset = w["end"] - 0.4 if w["end"] - w["start"] > MAX_WORD_SECONDS else w["start"]
        if not shot["start"] <= onset < shot["end"]:
            continue
        speaker = label(w["voice"]) if w.get("voice") else "SPEAKER"
        if lines and lines[-1]["speaker"] == speaker:
            lines[-1]["words"].append(w["w"])
        else:
            lines.append({"t": onset, "speaker": speaker, "words": [w["w"]]})
    return "\n".join(f"[{l['t']:.1f}s {l['speaker']}] {' '.join(l['words'])}" for l in lines) or "(no speech)"


def labelled_keyframes(shot, keyframes_dir, faces, width) -> list[bytes]:
    """Each keyframe with every tracked, assigned face within 0.2 s of it boxed and labelled."""
    from PIL import Image, ImageDraw

    people = [p["id"] for p in faces["people"]]
    colour = {p: BOX_COLOURS[i % len(BOX_COLOURS)] for i, p in enumerate(people)}
    out = []
    for kf in shot["keyframes"]:
        image = Image.open(keyframes_dir / kf["file"].removeprefix("assets/")).convert("RGB")
        scale = image.width / width
        draw = ImageDraw.Draw(image)
        for track in faces["tracks"]:
            if track["person"] is None:   # unattributed faces are not labelled for the model
                continue
            near = [s for s in track["samples"] if abs(s["t"] - kf["t"]) <= 0.2]
            if not near:
                continue
            x, y, w, h = (v * scale for v in min(near, key=lambda s: abs(s["t"] - kf["t"]))["box"])
            name, c = label(track["person"]), colour[track["person"]]
            draw.rectangle([x, y, x + w, y + h], outline=c, width=3)
            top = max(0, y - 18)
            draw.rectangle([x, top, x + 8 * len(name) + 6, top + 18], fill=c)
            draw.text((x + 3, top + 2), name, fill="white")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        out.append(buf.getvalue())
    return out


def caption(client, model, effort, prompt, images):
    import anthropic

    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(b).decode()}}
               for b in images] + [{"type": "text", "text": prompt}]
    try:
        response = client.messages.create(model=model, max_tokens=4000, messages=[{"role": "user", "content": content}],
                                          output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}})
    except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
        raise NodeError("transient", f"{type(e).__name__}: {e}") from None   # the SDK already retried twice
    except anthropic.AuthenticationError:
        raise NodeError("input", "the Anthropic API rejected the key (ANTHROPIC_API_KEY)") from None
    if response.stop_reason == "refusal":
        raise NodeError("input", f"refusal: {response.stop_details}")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise NodeError("internal", f"no text block in the response (stop_reason {response.stop_reason})")
    return json.loads(text), response.usage.input_tokens, response.usage.output_tokens


def spent(model, tokens_in, tokens_out):
    price = PRICE_PER_MTOK[model]
    return round((tokens_in * price[0] + tokens_out * price[1]) / 1e6, 4)


with Node() as node:
    import anthropic

    c = node.config
    model, effort = c["model"], c["effort"]
    client = anthropic.Anthropic(api_key=node.secret("ANTHROPIC_API_KEY"))
    shots_doc = node.input_json("shots")
    words = node.input_json("transcript")["words"]
    faces = node.input_json("faces")
    width = node.input_json("media")["width"]
    keyframes_dir = node.input("keyframes")
    first_seen = {p["id"]: p["first_seen"] for p in faces["people"]}

    roster, previous, tokens_in, tokens_out, t0 = {}, None, 0, 0, time.time()
    shots, texts = [], []
    for shot in shots_doc["shots"]:
        seen = [label(p) for p, t in first_seen.items() if t is not None and t < shot["end"]]   # never people who appear later
        roster_text = "\n".join(f"- {k}: {v}" for k, v in roster.items()) or "(nobody yet)"
        prompt = (f"{INSTRUCTIONS}\n\nShot {shot['index']} runs {shot['start']:.1f}s to {shot['end']:.1f}s. "
                  f"Keyframes are at {', '.join(str(k['t']) + 's' for k in shot['keyframes'])}.\n\n"
                  f"Speech during this shot:\n{shot_speech(shot, words)}\n\nPeople seen in earlier shots:\n{roster_text}\n\n"
                  f"Previous shot: {previous or '(this is the first shot)'}\n\n{FACES_INSTRUCTIONS}\n\nFace roster so far:\n"
                  + ("\n".join(f"- {s}" for s in seen) or "(nobody yet)"))
        cap, n_in, n_out = caption(client, model, effort, prompt, labelled_keyframes(shot, keyframes_dir, faces, width))
        tokens_in, tokens_out = tokens_in + n_in, tokens_out + n_out
        # The running total after every call: if a later shot fails, the response still reports what was spent.
        node.usage(usd=spent(model, tokens_in, tokens_out), tokens_in=tokens_in, tokens_out=tokens_out, calls=len(shots) + 1)
        for person in cap["people"]:
            roster.setdefault(person["label"], person["appearance"])
        previous = cap["description"]
        shots.append({**shot, "caption": {"description": cap["description"], "setting": cap["setting"], "mood": cap["mood"],
                                          "people": cap["people"], "model": model}})
        for string in cap["onscreen_text"]:   # the same string in consecutive shots is one item
            same = next((t for t in reversed(texts) if t["text"] == string and t["end"] == shot["start"]), None)
            if same:
                same["end"] = shot["end"]
            else:
                texts.append({"id": f"text:{len(texts)}", "text": string, "start": shot["start"], "end": shot["end"], "box": None})

    node.model(model, licence_mode="api-terms", licence="Anthropic API terms")
    node.model("PySceneDetect AdaptiveDetector", licence_mode="commercial-ok", licence="BSD-3-Clause")   # the shots it refines
    timings = {"captions_s": round(time.time() - t0, 1)}
    node.write_json("shots", "shots.json", {**shots_doc, "producer": node.producer(timings=timings, device="Anthropic API"), "shots": shots})
    node.write_json("text", "text.json", {"layer": "text", "schema_version": "0.1.0",
                                          "producer": node.producer(timings=timings, device="Anthropic API"), "items": texts})
