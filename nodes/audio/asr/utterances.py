"""The transcript layer's utterances from its words, for both of the node's runners (node.py on MLX, node_nemo.py on
NeMo): a new utterance at a pause over 0.25 s, a sentence end, or a shot cut, as the annotator prototype did
(pipeline/bakeoff/label_page.py). Voices come from the diarization node, not here: every utterance's voice is null."""

PAUSE_SPLIT = 0.25            # seconds of silence between words that starts a new utterance
SENTENCE_END = (".", "?", "!")


def utterances(words: list[dict], shots: list[dict]) -> list[dict]:
    cuts = [s["start"] for s in shots[1:]]
    units, current = [], []
    for i, word in enumerate(words):
        if current:
            prev = words[current[-1]]
            crosses_cut = any(prev["end"] <= c <= word["start"] or prev["start"] < c <= word["start"] for c in cuts)
            if word["start"] - prev["end"] > PAUSE_SPLIT or prev["w"].endswith(SENTENCE_END) or crosses_cut:
                units.append(current)
                current = []
        current.append(i)
    if current:
        units.append(current)

    def shot_at(t):
        return next(s["id"] for s in reversed(shots) if s["start"] <= t)

    return [{"id": f"utt:{n}", "start": words[u[0]]["start"], "end": words[u[-1]]["end"],
             "text": " ".join(words[i]["w"] for i in u), "shot": shot_at((words[u[0]]["start"] + words[u[-1]]["end"]) / 2),
             "words": [u[0], u[-1]], "voice": None} for n, u in enumerate(units)]
