"""Checks for prosody.json that do not need anyone to listen.

1. Laughter: utterances overlapping PE-A-Frame "laughter" spans (audio_events.json) against the rest, for
   arousal, emotion2vec+ "happy", and laughter tags. Parakeet writes no laughter tokens, so the transcript
   has no laughter to check against (reported).
2. Questions: measured final pitch movement for utterances ending in "?" against those ending in "." or "!".
3. Agreement: GPT vs Qwen tags (both mapped from free text by the same tagger), GPT's own tags vs the
   tagger's reading of GPT's text, categorical vs dimensional emotion, and caption tags vs measurements.
4. Pitch sanity: octave jumps between 10 ms frames and between 20 Hz contour bins, and words whose mean F0
   sits more than an octave from the speaker's median (a halved or doubled stretch, which jump counts miss).
5. Spot check: prints utterances with everything each source said about them.
6. Two dimensional models (audEERING vs the Odyssey 2024 baseline, when prosody.json has both): correlation
   per dimension, laughter vs arousal AUC for each, each valence against emotion2vec+ P(happy), and how much
   of each valence a text-only sentiment of the transcript explains (TEXT_SENTIMENT, run locally; the words
   alone, so valence that tracks it may be reading lexical content rather than the voice).

Usage (from prosody/): uv run python evaluate_prosody.py ../runs/<name> [--spot 12,35,...] [--no-text]
"""

import argparse
import json
from pathlib import Path

import numpy as np

DIMENSIONAL = {"audeering": "dimensional", "odyssey": "dimensional_odyssey"}
# Small local text sentiment model (RoBERTa-base on ~124M tweets, TweetEval sentiment; CC-BY-4.0). Score = P(pos) - P(neg).
TEXT_SENTIMENT = "cardiffnlp/twitter-roberta-base-sentiment-latest"
TEXT_SENTIMENT_REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
LAUGH_TAGS = {"laughing-while-speaking", "laughter-nearby", "smiling-voice"}
# Word stems that show a caption is describing how speech is delivered, not the recording or the words.
STYLE_GROUPS = {
    "pitch/intonation": ["pitch", "intonation", "inflection", "rising", "falling", "upward", "downward", "monoton"],
    "pace": ["pace", "paced", "rapid", "quick", "slow", "hurried", "deliberate", "tempo"],
    "loudness/emphasis": ["loud", "soft", "volume", "emphas", "stress", "whisper", "shout"],
    "emotion/attitude": ["tone", "emotion", "excite", "enthusias", "amuse", "happy", "playful", "sarcas", "teas",
                         "nervous", "anxious", "confident", "annoy", "surpris", "calm", "upbeat", "cheerful"],
    "laughter": ["laugh", "chuckl", "giggl"],
    "hesitation": ["hesita", "pause", "filler", "stammer", "trail"],
}
STYLE_WORDS = [w for g in STYLE_GROUPS.values() for w in g]


def overlaps(a0, a1, spans):
    return any(s < a1 and e > a0 for s, e in spans)


def kappa(x, y):
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    po = np.mean(x == y)
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return None if pe == 1 else (po - pe) / (1 - pe)


def auc(pos, neg):
    """Probability that a random positive scores above a random negative (ties count half)."""
    if not pos or not neg:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def text_sentiment(texts):
    """P(positive) - P(negative) per text from TEXT_SENTIMENT, on CPU."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging
    logging.set_verbosity_error()  # the unused pooler weights are expected for this head
    tok = AutoTokenizer.from_pretrained(TEXT_SENTIMENT, revision=TEXT_SENTIMENT_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(TEXT_SENTIMENT, revision=TEXT_SENTIMENT_REVISION).eval()
    label = {v: int(k) for k, v in model.config.id2label.items()}
    with torch.no_grad():
        p = model(**tok(texts, return_tensors="pt", padding=True, truncation=True)).logits.softmax(-1).numpy()
    return p[:, label["positive"]] - p[:, label["negative"]]


def compare_dimensional(units, spans, no_text):
    """Section 6: audEERING vs Odyssey on the same utterances."""
    from scipy.stats import spearmanr
    dims = ("arousal", "valence", "dominance")
    v = {m: {k: np.array([u["emotion"][key][k] for u in units]) for k in dims} for m, key in DIMENSIONAL.items()}
    print("\n[two dimensional models] audEERING (wav2vec2-large-robust, MSP-Podcast) vs Odyssey 2024 baseline "
          "(WavLM-large, MSP-Podcast)")
    for m in v:
        print(f"  {m:9s} " + "  ".join(f"{k} mean {v[m][k].mean():.3f} sd {v[m][k].std():.3f} "
                                        f"[{v[m][k].min():.2f}, {v[m][k].max():.2f}]" for k in dims))
    print("  between models, same dimension: " + "; ".join(
        f"{k} r {np.corrcoef(v['audeering'][k], v['odyssey'][k])[0, 1]:.2f} "
        f"rho {spearmanr(v['audeering'][k], v['odyssey'][k])[0]:.2f}" for k in dims))
    for m in v:
        print(f"  within {m}: r(arousal, valence) {np.corrcoef(v[m]['arousal'], v[m]['valence'])[0, 1]:.2f}, "
              f"r(arousal, dominance) {np.corrcoef(v[m]['arousal'], v[m]['dominance'])[0, 1]:.2f}")
    if spans:
        for name, pad in (("during utterance", 0), ("within 1 s", 1)):
            m = np.array([overlaps(u["start"] - pad, u["analysed_end"] + pad, spans) for u in units])
            print(f"  laughter {name} ({m.sum()} vs {(~m).sum()}): " + "; ".join(
                f"{k} arousal AUC {fmt(auc(list(v[k]['arousal'][m]), list(v[k]['arousal'][~m])))}, "
                f"valence AUC {fmt(auc(list(v[k]['valence'][m]), list(v[k]['valence'][~m])))}" for k in v))
    ph = np.array([u["emotion"]["categorical"]["scores"]["happy"] for u in units])
    print("  r(valence, emotion2vec+ P happy): " + "; ".join(
        f"{k} {np.corrcoef(v[k]['valence'], ph)[0, 1]:.2f} (rho {spearmanr(v[k]['valence'], ph)[0]:.2f})" for k in v))
    def partial(x, y, z):
        """r(x, y) after removing a straight-line fit on z from both."""
        rx, ry = x - np.polyval(np.polyfit(z, x, 1), z), y - np.polyval(np.polyfit(z, y, 1), z)
        return np.corrcoef(rx, ry)[0, 1]

    print("  r(valence, P happy) controlling for the same model's arousal: " + "; ".join(
        f"{k} {partial(v[k]['valence'], ph, v[k]['arousal']):.2f}" for k in v))
    if no_text:
        return
    sent = text_sentiment([u["text"] for u in units])
    print(f"  text sentiment ({TEXT_SENTIMENT}, P(pos) - P(neg) of the transcript; mean {sent.mean():.2f}, "
          f"{(sent > 0.5).sum()} clearly positive, {(sent < -0.5).sum()} clearly negative):")
    for k in v:
        r = np.corrcoef(v[k]["valence"], sent)[0, 1]
        ra = np.corrcoef(v[k]["arousal"], sent)[0, 1]
        print(f"    {k:9s} valence r {r:.2f} (R^2 {r * r:.2f}, rho {spearmanr(v[k]['valence'], sent)[0]:.2f}); "
              f"arousal r {ra:.2f} (control)")
    r_happy = np.corrcoef(ph, sent)[0, 1]
    print(f"    emotion2vec+ P happy r {r_happy:.2f} (R^2 {r_happy * r_happy:.2f})")
    # does each model's valence still agree with emotion2vec+ once text sentiment is partialled out?
    print("    after removing text sentiment: " + "; ".join(
        f"r({k} valence, P happy) {partial(v[k]['valence'], ph, sent):.2f}" for k in v)
          + f"; r(audeering, odyssey valence) {partial(v['audeering']['valence'], v['odyssey']['valence'], sent):.2f}")


def fmt(x, n=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--spot", default="")
    parser.add_argument("--no-text", action="store_true", help="skip the text-sentiment check (loads a local model)")
    args = parser.parse_args()
    d = json.loads((args.run_dir / "prosody.json").read_text())
    events = json.loads((args.run_dir / "audio_events.json").read_text())
    units = d["utterances"]
    caps = [u.get("captions", {}) for u in units]
    has = {k: all(k in c for c in caps) for k in ("gpt", "qwen")}
    print(f"{args.run_dir.name}: {len(units)} utterances; captions: {[k for k, v in has.items() if v]}")

    # 1. laughter
    spans = events["models"]["peaframe"]["queries"].get("laughter", {}).get("spans", [])
    print(f"\n[laughter] PE-A-Frame large 'laughter': {len(spans)} spans, {sum(e - s for s, e in spans):.1f} s; "
          "transcript laughter tokens: none (Parakeet emits none)")
    if spans:
        during = [overlaps(u["start"], u["analysed_end"], spans) for u in units]
        near = [overlaps(u["start"] - 1, u["analysed_end"] + 1, spans) for u in units]
        aro = [u["emotion"]["dimensional"]["arousal"] for u in units]
        val = [u["emotion"]["dimensional"]["valence"] for u in units]
        happy = [u["emotion"]["categorical"]["top"] == "happy" for u in units]
        for name, mask in (("during utterance", during), ("within 1 s (caption window)", near)):
            m = np.array(mask)
            print(f"  {name}: {m.sum()} laughter utts vs {(~m).sum()} others")
            print(f"    arousal mean {np.mean(np.array(aro)[m]):.3f} vs {np.mean(np.array(aro)[~m]):.3f}, "
                  f"AUC {fmt(auc(list(np.array(aro)[m]), list(np.array(aro)[~m])))}; valence "
                  f"{np.mean(np.array(val)[m]):.3f} vs {np.mean(np.array(val)[~m]):.3f}; "
                  f"emotion2vec happy {np.mean(np.array(happy)[m]):.2f} vs {np.mean(np.array(happy)[~m]):.2f}")
            for src, field in (("gpt", "tags"), ("gpt", "tags_from_text"), ("qwen", "tags_from_text")):
                if not has[src]:
                    continue
                tagged = np.array([bool(LAUGH_TAGS & set(c[src][field])) for c in caps])
                print(f"    {src}.{field} any laughter tag: {tagged[m].mean():.2f} vs {tagged[~m].mean():.2f}"
                      f"  (hits {tagged[m].sum()}/{m.sum()}, false alarms {tagged[~m].sum()}/{(~m).sum()})")
            if has["qwen"]:
                said = np.array(["laugh" in c["qwen"]["description"].lower() for c in caps])
                print(f"    qwen text mentions 'laugh': {said[m].mean():.2f} vs {said[~m].mean():.2f}")

    # 2. questions
    print("\n[questions] measured final pitch movement (last 300 ms voiced, +-1.5 st)")
    for label, pick in (("ends '?'", lambda t: t.endswith("?")), ("ends '.'/'!'", lambda t: t.endswith((".", "!")))):
        sel = [u for u in units if pick(u["text"].strip())]
        mov = [u["prosody"]["final_movement"] for u in sel]
        n = sum(m is not None for m in mov)
        deltas = [u["prosody"]["final_delta_st"] for u in sel if u["prosody"]["final_delta_st"] is not None]
        print(f"  {label}: n={len(sel)} (measurable {n}): rise {mov.count('rise')}, level {mov.count('level')}, "
              f"fall {mov.count('fall')}; rise share {fmt(mov.count('rise') / n if n else None)}; "
              f"median delta {fmt(float(np.median(deltas)) if deltas else None)} st")
        if has["gpt"]:
            print(f"    gpt rising-end tag: {np.mean(['rising-end' in u['captions']['gpt']['tags'] for u in sel]):.2f}"
                  f"  (GPT sees the transcript, so this is not independent)")

    # 3a. does the capped Qwen caption get to how things are said?
    if has["qwen"]:
        cap = d["captions"]["qwen"].get("max_new_tokens")
        print(f"\n[qwen coverage] captions cut at {cap} tokens; style words = {len(STYLE_WORDS)} prosody/emotion stems")
        reach, first, truncated = [], [], []
        for c in caps:
            text = c["qwen"]["description"].lower()
            hits = [text.find(w) for w in STYLE_WORDS if w in text]
            reach.append(bool(hits))
            truncated.append(c["qwen"]["tokens"] >= cap)
            if hits:
                first.append(min(hits) / max(1, len(text)))  # where the first style word sits, as a share of the text
        reach, truncated = np.array(reach), np.array(truncated)
        print(f"  hit the cap: {truncated.sum()}/{len(caps)}; mention any style word: {reach.sum()}/{len(caps)}; "
              f"median position of the first one: {np.median(first) if first else float('nan'):.2f} of the text")
        opens = [c["qwen"]["description"].lower()[:160] for c in caps]
        artefact = sum(any(w in o for w in ("hiss", "hum", "microphone", "recording", "fidelity", "click", "rustl"))
                       for o in opens)
        print(f"  first ~160 chars about the recording (hiss/hum/mic/fidelity/click): {artefact}/{len(caps)}")
        for label, words in STYLE_GROUPS.items():
            n = sum(any(w in c["qwen"]["description"].lower() for w in words) for c in caps)
            print(f"    {label}: {n}/{len(caps)}")

    # 3. agreement
    if has["gpt"] and has["qwen"]:
        print("\n[agreement] GPT vs Qwen, both via the text tagger (tags with >= 5 positives in either)")
        tags = [t for g in d["captions"]["tags"].values() for t in g]
        rows = []
        for t in tags:
            g = [t in c["gpt"]["tags_from_text"] for c in caps]
            q = [t in c["qwen"]["tags_from_text"] for c in caps]
            if sum(g) >= 5 or sum(q) >= 5:
                rows.append((t, sum(g), sum(q), kappa(g, q)))
        for t, ng, nq, k in sorted(rows, key=lambda r: -(r[3] or -9)):
            print(f"  {t:24s} gpt {ng:3d}  qwen {nq:3d}  kappa {fmt(k)}")
        jac = [len(set(c["gpt"]["tags_from_text"]) & set(c["qwen"]["tags_from_text"])) /
               max(1, len(set(c["gpt"]["tags_from_text"]) | set(c["qwen"]["tags_from_text"]))) for c in caps]
        print(f"  mean Jaccard per utterance {np.mean(jac):.2f}")
    if has["gpt"]:
        jac = [len(set(c["gpt"]["tags"]) & set(c["gpt"]["tags_from_text"])) /
               max(1, len(set(c["gpt"]["tags"]) | set(c["gpt"]["tags_from_text"]))) for c in caps]
        print(f"  GPT own tags vs tagger on GPT text: mean Jaccard {np.mean(jac):.2f} (tagger consistency)")

    print("\n[emotion] dimensional by emotion2vec+ top category")
    by = {}
    for u in units:
        by.setdefault(u["emotion"]["categorical"]["top"], []).append(u["emotion"]["dimensional"])
    for cat, v in sorted(by.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:10s} n={len(v):3d}  arousal {np.mean([x['arousal'] for x in v]):.3f}  "
              f"valence {np.mean([x['valence'] for x in v]):.3f}  dominance {np.mean([x['dominance'] for x in v]):.3f}")
    ph = [u["emotion"]["categorical"]["scores"]["happy"] for u in units]
    vv = [u["emotion"]["dimensional"]["valence"] for u in units]
    aa = [u["emotion"]["dimensional"]["arousal"] for u in units]
    pn = [u["emotion"]["categorical"]["scores"]["neutral"] for u in units]
    print(f"  r(P happy, valence) {np.corrcoef(ph, vv)[0, 1]:.2f}; r(P happy, arousal) {np.corrcoef(ph, aa)[0, 1]:.2f}; "
          f"r(P neutral, arousal) {np.corrcoef(pn, aa)[0, 1]:.2f}")
    long = [u for u in units if u["prosody"]["duration"] >= 1.0]
    print(f"  utterances >= 1 s: {len(long)} of {len(units)} (shorter ones are weak evidence for any model)")

    if has["gpt"]:
        print("\n[captions vs measurements] GPT tags (direct)")
        for tag, key, label in (("rising-end", "final_movement", "rise"), ("falling-end", "final_movement", "fall")):
            sel = [u for u in units if u["prosody"][key] is not None]
            g = [tag in u["captions"]["gpt"]["tags"] for u in sel]
            m = [u["prosody"][key] == label for u in sel]
            print(f"  {tag} vs measured {label}: kappa {fmt(kappa(g, m))} (gpt {sum(g)}, measured {sum(m)}, n {len(sel)})")
        for tag in ("fast", "slow"):
            sel = [u for u in units if u["prosody"]["n_words"] >= 3]
            on = [u["prosody"]["articulation_rate_syl_s"] for u in sel if tag in u["captions"]["gpt"]["tags"]]
            off = [u["prosody"]["articulation_rate_syl_s"] for u in sel if tag not in u["captions"]["gpt"]["tags"]]
            print(f"  {tag}: articulation rate {fmt(np.mean(on) if on else None)} syl/s tagged (n {len(on)}) vs "
                  f"{fmt(np.mean(off) if off else None)} untagged (utterances of 3+ words)")

    if all(key in u["emotion"] for u in units for key in DIMENSIONAL.values()):
        compare_dimensional(units, spans, args.no_text)

    # 4. pitch sanity
    print("\n[pitch] octave checks")
    sp = d["measure"]["speakers"]
    pairs = sum(v["octave_jumps"]["pairs"] for v in sp.values())
    jumps = sum(v["octave_jumps"]["n_gt_12st"] for v in sp.values())
    pairs1 = sum(v["octave_jumps_first_pass"]["pairs"] for v in sp.values())
    jumps1 = sum(v["octave_jumps_first_pass"]["n_gt_12st"] for v in sp.values())
    print(f"  10 ms frame pairs > 12 st: {jumps}/{pairs} = {jumps / pairs:.4f} (first pass 75-600 Hz: "
          f"{jumps1}/{pairs1} = {jumps1 / pairs1:.4f})")
    c = d["measure"]["contour"]
    st, who = c["f0_st"], c["speaker"]
    bins = [(a, b) for i, (a, b) in enumerate(zip(st, st[1:])) if a is not None and b is not None and who[i] == who[i + 1]]
    big = sum(abs(b - a) > 12 for a, b in bins)
    print(f"  20 Hz contour adjacent bins > 12 st: {big}/{len(bins)} = {big / len(bins):.4f}")
    words = [w for u in units for w in u["words"] if w["f0_mean_st"] is not None]
    off = [w for w in words if abs(w["f0_mean_st"]) > 12]
    print(f"  words with mean F0 > 12 st from speaker median: {len(off)}/{len(words)}"
          + (f" {[(w['word'], w['f0_mean_st']) for w in off][:8]}" if off else ""))
    for s, v in sp.items():
        print(f"  {s}: {v['floor_hz']}-{v['ceiling_hz']} Hz, median {v['median_hz']} Hz, voiced {v['voiced_s']} s")

    # 5. spot check
    if args.spot:
        print("\n[spot check]")
        by_id = {u["id"]: u for u in units}
        for i in [int(x) for x in args.spot.split(",")]:
            u = by_id[i]
            p, e = u["prosody"], u["emotion"]
            print(f"\n#{i} {u['start']:.1f}-{u['end']:.1f}s {u['speaker']} ({u['speaker_source']}): {u['text']!r}")
            print(f"  measured: rate {p['articulation_rate_syl_s']} syl/s, range {p['f0_range_st']} st, end "
                  f"{p['final_movement']} ({p['final_delta_st']} st), prominent: "
                  f"{[w['word'] for w in u['words'] if (w['prominence'] or -9) > 0.8]}")
            print(f"  emotion: A {e['dimensional']['arousal']:.2f} V {e['dimensional']['valence']:.2f} "
                  f"D {e['dimensional']['dominance']:.2f}; e2v {e['categorical']['top']} {e['categorical']['top_score']:.2f}"
                  + (f"; odyssey A {e['dimensional_odyssey']['arousal']:.2f} V {e['dimensional_odyssey']['valence']:.2f} "
                     f"D {e['dimensional_odyssey']['dominance']:.2f}" if "dimensional_odyssey" in e else ""))
            if "gpt" in u.get("captions", {}):
                g = u["captions"]["gpt"]
                print(f"  gpt: {g['description']} | {g['tags']} | emph {g['emphasized_words']}")
            if "qwen" in u.get("captions", {}):
                q = u["captions"]["qwen"]
                print(f"  qwen: {q['description'][:600]} | {q['tags_from_text']}")


if __name__ == "__main__":
    main()
