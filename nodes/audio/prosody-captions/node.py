"""audio.prosody-captions: how each utterance is said, in words and tags, from gpt-audio-1.5 through the OpenAI API,
written into the prosody node's layer (it refines it: export takes this one).

Ported from the bake-off's prosody/prosody.py captions (style_captions.py: prompts, tags, prices). A node of its own,
not part of the prosody node: it sends each utterance's audio, with a second of context each side, and its transcript
lines to OpenAI, and the run's consent to uploads (--allow-uploads) is per node, so the local measures and emotion run
without it. Only per-utterance snippets are sent, never the clip.

Beside the layer, layers/prosody_captions.parquet keeps every prompt and the raw replies (captions_raw.py), pointed at
with raw.captions; the prosody node's side files stay the layer's (its raw pointers are kept).
"""

import base64
import io
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import captions_raw
import style_captions as sc
from perception_node import Node, NodeError


def read_samples(path) -> bytes:
    """Decode's audio (16 kHz mono 16-bit WAV), its PCM bytes."""
    with wave.open(str(path)) as w:
        got = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        if got != (1, 2, sc.RATE):
            raise NodeError("input", f"the audio is {got[0]} channel(s), {8 * got[1]}-bit, {got[2]} Hz; "
                                     f"decode writes 1 channel, 16-bit, {sc.RATE} Hz")
        return w.readframes(w.getnframes())


def wav_b64(pcm: bytes) -> str:
    """A snippet's own 16-bit samples as a WAV file, base64 (the bake-off wrote its float copy back to 16-bit PCM)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sc.RATE)
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode()


def call(fn):
    """One API call, its errors as the node's: the SDK has already retried twice."""
    import openai
    try:
        return fn()
    except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError) as e:
        raise NodeError("transient", f"{type(e).__name__}: {e}") from None
    except openai.AuthenticationError:
        raise NodeError("input", "the OpenAI API rejected the key (OPENAI_API_KEY)") from None


with Node() as node:
    from openai import OpenAI

    client = OpenAI(api_key=node.secret("OPENAI_API_KEY"))
    transcript, prosody = node.input_json("transcript"), node.input_json("prosody")
    pcm = read_samples(node.input("audio"))
    try:
        units = sc.targets(transcript, prosody, len(pcm) // 2)
    except ValueError as e:
        raise NodeError("input", str(e)) from None

    lock, totals = threading.Lock(), {"usd": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0}

    def spend(usage: dict):
        """The running total after every billed call: if a later one fails, the response still reports it."""
        with lock:
            totals["usd"] += usage["usd"]
            totals["calls"] += 1
            totals["tokens_in"] += usage["text_in"] + usage["audio_in"]
            totals["tokens_out"] += usage["out"]
            node.usage(usd=round(totals["usd"], 4), calls=totals["calls"], tokens_in=totals["tokens_in"],
                       tokens_out=totals["tokens_out"])

    def caption(k: int) -> dict:
        u = units[k]
        i, j = u["samples"]
        text = sc.prompt(units, k)
        r = call(lambda: client.chat.completions.create(
            model=sc.GPT_MODEL, modalities=["text"],
            tools=[{"type": "function", "function": {"name": "record_style", "description": "Record how the target is said",
                                                     "parameters": sc.CAPTION_SCHEMA, "strict": True}}],
            tool_choice={"type": "function", "function": {"name": "record_style"}},
            messages=[{"role": "system", "content": sc.GPT_INSTRUCTIONS},
                      {"role": "user", "content": [{"type": "text", "text": text},
                                                   {"type": "input_audio", "input_audio": {"data": wav_b64(pcm[2 * i:2 * j]), "format": "wav"}}]}]))
        d = r.usage.prompt_tokens_details
        usage = sc.cost(sc.GPT_MODEL, r.usage.prompt_tokens, d.audio_tokens if d else 0, r.usage.completion_tokens)
        spend(usage)
        choice = r.choices[0]
        if not choice.message.tool_calls:
            raise NodeError("internal", f"{u['id']}: no function call in the reply (finish_reason {choice.finish_reason})")
        arguments = choice.message.tool_calls[0].function.arguments
        try:
            parsed = sc.parse_caption(arguments)
        except ValueError as e:
            raise NodeError("internal", f"{u['id']}: {e}") from None
        return {"parsed": parsed, "usage": usage, "prompt": text,
                "raw": {"model": r.model, "finish_reason": choice.finish_reason, "arguments": arguments,
                        "usage": r.usage.model_dump(mode="json")}}

    def tag(description: str) -> dict:
        r = call(lambda: client.chat.completions.create(
            model=sc.TAGGER,
            response_format={"type": "json_schema", "json_schema": {"name": "tags", "schema": sc.TAG_SCHEMA, "strict": True}},
            messages=[{"role": "system", "content": sc.TAGGER_INSTRUCTIONS},
                      {"role": "user", "content": f"Description:\n{description}"}]))
        d = r.usage.prompt_tokens_details
        usage = sc.cost(sc.TAGGER, r.usage.prompt_tokens, d.audio_tokens if d else 0, r.usage.completion_tokens)
        spend(usage)
        content = r.choices[0].message.content
        try:
            tags = sc.parse_tags(content)
        except ValueError as e:
            raise NodeError("internal", f"the tagger: {e}") from None
        return {"tags": tags, "usage": usage,
                "raw": {"model": r.model, "finish_reason": r.choices[0].finish_reason, "content": content,
                        "usage": r.usage.model_dump(mode="json")}}

    t0 = time.time()
    with ThreadPoolExecutor(sc.WORKERS) as pool:
        captions = list(pool.map(caption, range(len(units))))
    caption_s = round(time.time() - t0, 1)
    t0 = time.time()
    with ThreadPoolExecutor(sc.WORKERS) as pool:
        tagged = list(pool.map(lambda c: tag(c["parsed"]["description"]), captions))
    tagger_s = round(time.time() - t0, 1)

    for m in prosody["producer"]["models"]:   # the measures' and emotion's models, as the prosody node recorded them
        node.model(m["id"], licence_mode=m["licence_mode"], source=m.get("source"), revision=m.get("revision"), licence=m.get("licence"))
    node.model(sc.GPT_MODEL, licence_mode="api-terms", licence=sc.LICENCE)
    node.model(sc.TAGGER, licence_mode="api-terms", licence=sc.TAGGER_LICENCE)

    entries = [sc.caption_entry(c["parsed"], u["snippet"], c["usage"], t["tags"], t["usage"])
               for u, c, t in zip(units, captions, tagged)]
    captions_raw.write(node.path("captions", "prosody_captions.parquet"),
                       [captions_raw.row(u["id"], u["snippet"], c["prompt"], c["raw"], t["raw"])
                        for u, c, t in zip(units, captions, tagged)], sc.request())
    node.write_json("prosody", "prosody.json", sc.refine(
        prosody, entries, node.producer(timings={"caption_s": caption_s, "tagger_s": tagger_s}, device="OpenAI API"),
        {"file": captions_raw.FILE, "format": "parquet", "description": captions_raw.DESCRIPTION}))
    node.usage(utterances=len(units), workers=sc.WORKERS,
               usd_by_model={sc.GPT_MODEL: round(sum(c["usage"]["usd"] for c in captions), 4),
                             sc.TAGGER: round(sum(t["usage"]["usd"] for t in tagged), 4)})
