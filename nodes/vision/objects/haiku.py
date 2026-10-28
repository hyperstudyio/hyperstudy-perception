"""The node's three Claude Haiku 5.5 calls, with the bake-off's prompts and schemas (objects.py), unchanged:

1. vocabulary: per shot, in order, its three keyframes and the names used in earlier shots (never later ones) -> up to
   MAX_OBJECTS concrete, trackable object names (no people or background). The bake-off ran this separately from the
   captions; PLAN.md §4 says the two passes over the same keyframes are to fold into one (a follow-up).
2. name classes: one call over the clip's whole name list -> each name's family, kind (individual / lookalike) and
   whether it is worn.
3. crop check: per label, VERIFY_BATCH tracks a call, each track as three outlined moments side by side ->
   object / graphic / absent.

What leaves the machine: keyframes and crops of video frames, which show the people on screen (frames and faces, the
manifest's privacy.external_upload), and the names themselves. Calls go one at a time, as the bake-off made them (the vocabulary's chaining needs order).
"""

from __future__ import annotations

import base64
import io
import json

from perception_node import NodeError

MODEL = "claude-haiku-5-5"
PRICE_PER_MTOK = (0.10, 0.50)   # input, output (USD), prompts up to 100K tokens
MAX_OBJECTS = 8                 # names per shot; each is a separate SAM 3 text prompt on every frame
VERIFY_BATCH = 16               # tracks per crop-check call

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


def verify_instructions(label: str, n: int) -> str:
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


def jpeg_block(data: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(data).decode()}}


def jpeg(rgb) -> bytes:
    """An RGB array as the bake-off encoded crop-check strips: PIL JPEG, quality 90."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def spent(tokens_in: int, tokens_out: int) -> float:
    return round((tokens_in * PRICE_PER_MTOK[0] + tokens_out * PRICE_PER_MTOK[1]) / 1e6, 4)


def name_classes_max_tokens(n_names: int) -> int:
    """Room for the name classification's reply: the bake-off's 4000 tokens for a clip's few dozen names, and about 40
    a name beyond that (an entry is ~25 tokens), so a long film's vocabulary is not cut off."""
    return max(4000, 2000 + 40 * n_names)


def reply_json(response):
    """The JSON a structured reply holds, or NodeError: refused, cut off at max_tokens, or with no text."""
    if response.stop_reason == "refusal":
        raise NodeError("input", f"refusal: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise NodeError("internal", "Haiku's reply reached max_tokens: its JSON is cut off")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise NodeError("internal", f"no text block in the response (stop_reason {response.stop_reason})")
    return json.loads(text)


class Haiku:
    """Structured Haiku calls, each billed through spend(tokens_in, tokens_out) as soon as it returns."""

    STREAM_FROM = 16000   # the SDK wants a long reply streamed (a non-streaming request could time out)

    def __init__(self, api_key: str, spend):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.spend = spend

    def ask(self, content: list, schema: dict, max_tokens: int = 1000):
        import anthropic
        request = dict(model=MODEL, max_tokens=max_tokens, messages=[{"role": "user", "content": content}],
                       output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}})
        try:
            if max_tokens >= self.STREAM_FROM:
                with self.client.messages.stream(**request) as stream:
                    response = stream.get_final_message()
            else:
                response = self.client.messages.create(**request)
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            raise NodeError("transient", f"{type(e).__name__}: {e}") from None   # the SDK already retried twice
        except anthropic.AuthenticationError:
            raise NodeError("input", "the Anthropic API rejected the key (ANTHROPIC_API_KEY)") from None
        self.spend(response.usage.input_tokens, response.usage.output_tokens)   # billed even if the reply is then rejected
        return reply_json(response)

    def vocabulary(self, images: list[bytes], earlier: set[str]) -> list[str]:
        prompt = f"{VOCAB_INSTRUCTIONS}\n\nNames used in earlier shots: {', '.join(sorted(earlier)) or '(none yet)'}"
        return self.ask([jpeg_block(b) for b in images] + [{"type": "text", "text": prompt}], VOCAB_SCHEMA)["objects"]

    def name_classes(self, names: list[str]) -> list[dict]:
        text = NAMES_INSTRUCTIONS + "\n\nNames:\n" + "\n".join(names)
        return self.ask([{"type": "text", "text": text}], NAMES_SCHEMA, max_tokens=name_classes_max_tokens(len(names)))["names"]

    def check_crops(self, label: str, strips: list) -> list[dict]:
        content = []
        for k, strip in enumerate(strips, 1):
            content += [{"type": "text", "text": f"view {k}"}, jpeg_block(jpeg(strip))]
        content.append({"type": "text", "text": verify_instructions(label, len(strips))})
        return self.ask(content, VERIFY_SCHEMA, max_tokens=4000)["views"]
