"""Style captions per utterance, as the bake-off's prosody/prosody.py made them with gpt-audio (its default captioner,
the one the annotator draws): the prompts, the fixed tag vocabulary, the text-only tagger, prices, and the prosody layer
with the captions in. Standard library only; the API calls are node.py's.

Each utterance is heard with CONTEXT_S of audio on each side (cut at the clip's ends), its target span named in the
prompt with its transcript and the lines before and after it. gpt-audio-1.5 (the only OpenAI family that took audio
input on the bake-off's key) writes a 1-3 sentence description, picks tags from STYLE_TAGS and lists the emphasised
words, through a strict function call (the audio models do not take json_schema). The description is then mapped onto
the same tags by a text-only model (TAGGER, gpt-5.4-mini), as tags_from_text: the bake-off's way to compare captioners
tag by tag. The app shows the captioner's own tags. The Qwen3-Omni captioner the bake-off also ran is not ported.

Sampled at the API's default temperature: captions differ run to run (the node's result is cached, as the shot
captions' are). The target's span ends at its analysed end (the prosody node's: a word is cut WORD_MAX_S after its
onset), which this module recomputes from the transcript and checks against the prosody layer.
"""

from __future__ import annotations

CONTEXT_S = 1.0
RATE = 16000
WORD_MAX_S = 1.2               # the prosody node's (prosody_units.WORD_MAX_S); checked against its layer
GPT_MODEL = "gpt-audio-1.5"
TAGGER = "gpt-5.4-mini"
KEY = "gpt"                    # the layer's caption key, the bake-off's (the app names it "GPT audio")
WORKERS = 8
# USD per 1M tokens, from developers.openai.com/api/docs/models (the bake-off checked them 2026-10-23).
PRICES = {GPT_MODEL: {"text_in": 2.50, "audio_in": 32.00, "out": 10.00},
          TAGGER: {"text_in": 0.75, "audio_in": None, "out": 4.50}}

# Fixed vocabulary, after SpeechCraft's (pitch, speed, energy, emotion) and ParaSpeechCaps' (situational and
# voice-quality tags) attribute sets, plus the conversational attitudes and laughter this footage needs.
STYLE_TAGS = {
    "pitch": ["high-pitch", "low-pitch", "wide-pitch-range", "monotone", "rising-end", "falling-end"],
    "pace": ["fast", "slow", "rushed", "drawn-out"],
    "loudness": ["loud", "soft", "whispered", "shouting"],
    "emphasis": ["emphatic", "stressed-words"],
    "voice_quality": ["breathy", "creaky", "tense", "nasal", "husky", "clear", "shaky"],
    "emotion": ["neutral", "happy", "amused", "excited", "calm", "sad", "angry", "annoyed", "surprised",
                "fearful", "anxious", "confused", "bored", "confident"],
    "attitude": ["sarcastic", "teasing", "playful", "sincere", "questioning", "assertive", "doubtful",
                 "enthusiastic"],
    "hesitation": ["hesitant", "filled-pause", "stammering", "trailing-off"],
    "laughter": ["laughing-while-speaking", "laughter-nearby", "smiling-voice"],
    "delivery": ["conversational", "narration", "dramatic", "enunciated", "mumbled"],
}
ALL_TAGS = [t for group in STYLE_TAGS.values() for t in group]
TAG_GLOSSARY = ("rising-end/falling-end: final pitch movement of the target. stressed-words: one or more words "
                "clearly stressed. laughter-nearby: laughter just before/after the target or by other people "
                "during it. smiling-voice: audible smile without laughter. narration: voice-over or storytelling "
                "delivery.")

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string", "enum": ALL_TAGS}},
        "emphasized_words": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["description", "tags", "emphasized_words"],
    "additionalProperties": False,
}
TAG_SCHEMA = {
    "type": "object",
    "properties": {"tags": {"type": "array", "items": {"type": "string", "enum": ALL_TAGS}}},
    "required": ["tags"],
    "additionalProperties": False,
}

GPT_INSTRUCTIONS = """You are an expert in speech prosody and paralinguistics. You will hear a short audio \
snippet from a video. It contains one target utterance with up to 1 s of surrounding audio before and after it.

Describe HOW the target utterance is said, judged from the voice, not from the words alone: pitch (level, \
range, the movement at the end), pace, loudness, emphasis (which words are stressed), voice quality, emotion, \
attitude (e.g. sarcasm, teasing, sincerity), hesitation, and laughter in or around the speech. Mention the \
context only where it matters (e.g. others laughing). Do not invent what you cannot hear; if the target is \
too short or masked to judge something, say so.

Return through the function: a 1-3 sentence description, the tags from the fixed list that clearly apply \
(only those you can hear; "neutral" only if no emotion is audible), and the words of the target that are \
audibly emphasized (copied from the transcript; empty if none). Tag notes: """ + TAG_GLOSSARY

TAGGER_INSTRUCTIONS = """You map a free-text description of how someone speaks onto a fixed tag list. \
Choose only tags the description states or clearly implies about the main/target speech. Do not infer \
from your own guesses about the situation. If the description covers several speakers or sounds, tag what \
it says about the speech, and use laughter-nearby for laughter by others or around the speech. Tag notes: """ \
    + TAG_GLOSSARY

LICENCE = "OpenAI API terms (outputs owned by the user; audio sent to OpenAI)"
TAGGER_LICENCE = "OpenAI API terms (text only)"


def word_end(word: dict) -> float:
    return min(word["end"], word["start"] + WORD_MAX_S)


def targets(transcript: dict, prosody: dict, n_samples: int) -> list[dict]:
    """Per utterance of the prosody layer: {id, text, start, analysed_end, snippet: (a, b) seconds, samples: (i, j)}.
    ValueError when the prosody layer is not the transcript's (ids, order, analysed ends)."""
    words = transcript["words"]
    utts = transcript["utterances"]
    if [u["id"] for u in utts] != [p["utt"] for p in prosody["utterances"]]:
        raise ValueError("the prosody layer's utterances are not the transcript's: it must come from the same transcript")
    clip_s = n_samples / RATE
    out = []
    for u, p in zip(utts, prosody["utterances"]):
        end = round(word_end(words[u["words"][1]]), 3)
        duration = (p.get("measures") or {}).get("duration")
        if duration is None or abs(round(end - u["start"], 3) - duration) > 1e-9:
            raise ValueError(f"{u['id']}: the prosody layer measured {duration} s, the transcript gives "
                             f"{round(end - u['start'], 3)} s to its analysed end")
        a, b = max(0.0, u["start"] - CONTEXT_S), min(clip_s, end + CONTEXT_S)
        out.append({"id": u["id"], "text": u["text"], "start": u["start"], "analysed_end": end, "snippet": (a, b),
                    "samples": (int(a * RATE), int(b * RATE))})
    return out


def prompt(units: list[dict], k: int) -> str:
    """The user text sent with utterance k's snippet, as the bake-off wrote it."""
    u = units[k]
    a, b = u["snippet"]
    i, j = u["samples"]
    prev_text = units[k - 1]["text"] if k > 0 else "(none)"
    next_text = units[k + 1]["text"] if k + 1 < len(units) else "(none)"
    return (f"Snippet length {(j - i) / RATE:.2f} s. The target utterance runs from {round(u['start'] - a, 2):.2f} s "
            f"to {round(u['analysed_end'] - a, 2):.2f} s of the snippet.\n"
            f"Target transcript (automatic, may contain errors): \"{u['text']}\"\n"
            f"Previous line in the conversation: \"{prev_text}\"\nNext line: \"{next_text}\"")


def cost(model: str, prompt_tokens: int, audio_tokens: int | None, completion_tokens: int) -> dict:
    """{text_in, audio_in, out, usd} of one call at PRICES."""
    p = PRICES[model]
    audio_in = audio_tokens or 0
    text_in = prompt_tokens - audio_in
    usd = (text_in * p["text_in"] + audio_in * (p["audio_in"] or 0) + completion_tokens * p["out"]) / 1e6
    return {"text_in": text_in, "audio_in": audio_in, "out": completion_tokens, "usd": usd}


def caption_attempt(ask) -> dict:
    """One call for one target's caption: `ask()` makes it and gives (its function call's arguments, or None when it
    made none; its usage, as cost() gives it; its raw reply). {parsed, arguments, raw, usage}, or {rejected: {problem,
    finish_reason, arguments}, usage} when the reply is not the schema's. Why one can fail the schema is known: the
    audio models do not constrain their function call to it (strict or not; the same reason off-list tags are dropped,
    parse_caption), so a sample can be malformed (one was cut short of a comma on a pod's run, 2026-10-31)."""
    arguments, usage, raw = ask()
    try:
        if arguments is None:
            raise ValueError(f"no function call in the reply (finish_reason {raw.get('finish_reason')})")
        return {"parsed": parse_caption(arguments), "arguments": arguments, "raw": raw, "usage": usage}
    except ValueError as e:
        return {"rejected": {"problem": str(e), "finish_reason": raw.get("finish_reason"), "arguments": arguments},
                "usage": usage}


def rejected_text(rejected: dict) -> str:
    return (f"{rejected['problem']} (finish_reason {rejected['finish_reason']}, "
            f"arguments {str(rejected['arguments'])[:300]!r})")


def may_ask_again(spent_usd: float, caption_usd: float, tag_usd: float, utterances: int) -> bool:
    """Whether a rejected caption may be asked again: once every other call of the step has been made (spent_usd is
    then all but this), only if this caption's second call and its tag stay within what the step reserved (the
    manifest's cost, an utterance: review of #53)."""
    return spent_usd + caption_usd + tag_usd <= utterances * COST_PER_UTTERANCE


def asked_again(first: dict, second: dict) -> dict:
    """A caption asked a second time: the second's, with both calls' usage and the first's rejected reply
    (asked_again). ValueError when the second is not the schema's either."""
    if "rejected" in second:
        raise ValueError(f"asked twice, and neither reply was the schema's: {rejected_text(first['rejected'])}; "
                         f"{rejected_text(second['rejected'])}")
    usage = {k: first["usage"][k] + second["usage"][k] for k in first["usage"]}
    return {**second, "usage": usage, "asked_again": [first["rejected"]]}


# what the step reserves an utterance (manifest.yaml's cost; a test checks they agree): one caption call and its tags
COST_PER_UTTERANCE = 0.008


def parse_caption(arguments: str) -> dict:
    """The function call's arguments -> {description, tags, emphasized_words}; ValueError when they are not that. Its tags
    are those of the list only: gpt-audio-1.5 does not hold a function call to the schema's enum (strict or not), so a tag
    outside it is dropped here (outside_tags says which; the side file keeps the reply as it came)."""
    import json
    out = json.loads(arguments)
    if not (isinstance(out, dict) and set(out) == {"description", "tags", "emphasized_words"}):
        raise ValueError(f"the caption is not {{description, tags, emphasized_words}}: {sorted(out) if isinstance(out, dict) else type(out).__name__}")
    if not isinstance(out["tags"], list):
        check_tags(out["tags"])   # not a list: said as such
    if not all(isinstance(t, str) for t in out["tags"]):
        raise ValueError(f"the caption's tags are not strings: {[t for t in out['tags'] if not isinstance(t, str)][:5]}")
    return {**out, "tags": [t for t in out["tags"] if t in ALL_TAGS]}


def outside_tags(arguments: str) -> list[str]:
    """The tags a caption's reply gave that are not in the list (dropped by parse_caption)."""
    import json
    tags = json.loads(arguments).get("tags")
    return [t for t in tags if t not in ALL_TAGS] if isinstance(tags, list) else []


def parse_tags(content: str) -> list[str]:
    import json
    out = json.loads(content)
    if not (isinstance(out, dict) and set(out) == {"tags"}):
        raise ValueError("the tagger's reply is not {tags}")
    return check_tags(out["tags"])


def check_tags(tags) -> list[str]:
    unknown = [t for t in tags if t not in ALL_TAGS] if isinstance(tags, list) else [tags]
    if unknown:
        raise ValueError(f"tags outside the vocabulary: {unknown[:5]}")
    return tags


def caption_entry(caption: dict, snippet: tuple[float, float], usage: dict, tags_from_text: list[str], tagger_usage: dict) -> dict:
    """One utterance's caption in the layer, the bake-off's fields (convert_bakeoff copied them) less the call's
    seconds, a timing (in the side file)."""
    return {"description": caption["description"], "tags": caption["tags"], "emphasized_words": caption["emphasized_words"],
            "snippet": [round(snippet[0], 2), round(snippet[1], 2)], "usage": usage, "tags_from_text": tags_from_text,
            "tagger_usage": tagger_usage}


def refine(prosody: dict, entries: list[dict], producer: dict, pointer: dict) -> dict:
    """The prosody layer with every utterance's captions[KEY], caption_models[KEY] and raw.captions (`pointer`, at the
    captions side file); the rest (and the other raw pointers) as the prosody node wrote it."""
    if len(entries) != len(prosody["utterances"]):
        raise ValueError(f"{len(entries)} captions for {len(prosody['utterances'])} utterances")
    utterances = [{**u, "captions": {**u.get("captions", {}), KEY: e}} for u, e in zip(prosody["utterances"], entries)]
    return {**prosody, "producer": producer, "utterances": utterances,
            "caption_models": {**prosody.get("caption_models", {}), KEY: {"model": GPT_MODEL, "licence_mode": "api-terms"}},
            "raw": {**prosody.get("raw", {}), "captions": pointer}}


def request() -> dict:
    """What every call shares, for the side file's metadata."""
    return {"model": GPT_MODEL, "tagger": TAGGER, "context_s": CONTEXT_S, "rate": RATE, "prices_usd_per_mtok": PRICES,
            "tags": STYLE_TAGS, "tag_notes": TAG_GLOSSARY, "instructions": GPT_INSTRUCTIONS,
            "tagger_instructions": TAGGER_INSTRUCTIONS, "caption_schema": CAPTION_SCHEMA, "tag_schema": TAG_SCHEMA}


def fan_out(fn, items, workers: int = WORKERS) -> list:
    """[fn(x) for x in items] on `workers` threads, in order, raising the first failure. At most `workers` calls are
    submitted at a time, the next only once one has finished, so after a failure no call starts: those running finish
    (each billed call reports its spend itself) and the failure is raised. pool.map submitted every call at once and
    cancelled the queued ones only when its in-order results reached the failure: until every earlier call had
    finished, its workers went on starting queued calls (behind one slow early call, all of them), so a failed run, and
    its node retry, could pay for every call."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    items = list(items)
    out, running, nxt = [None] * len(items), {}, 0
    with ThreadPoolExecutor(workers) as pool:   # on a failure, leaving it waits only for the calls already running
        while nxt < len(items) or running:
            while nxt < len(items) and len(running) < workers:
                running[pool.submit(fn, items[nxt])] = nxt
                nxt += 1
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                out[running.pop(f)] = f.result()
    return out
