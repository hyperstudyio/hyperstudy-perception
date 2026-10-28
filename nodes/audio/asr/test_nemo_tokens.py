"""NeMo's greedy TDT hypothesis as parakeet-mlx's tokens and sentences (nemo_tokens.py): the same times to the last
bit, the same texts and sentences, and through raw_tokens.to_words the same layer words, as the MLX runner gets from
the same tokens; and NeMo's configured confidence measure is parakeet-mlx's. Run: uv run --no-project --with pytest
--with numpy --with-editable ./sdk pytest nodes/audio/asr"""

import math

import numpy as np
import pytest

from nemo_tokens import Sentence, Token, sentences, time_ratio, tokens
from raw_tokens import to_words

DURATIONS = [0, 1, 2, 3, 4]
VOCAB = ["<unk>", "▁As", "▁rec", "ent", "▁tr", "end", "s", ".", "▁", "Yeah", "?", "Okay", "▁U", ".S", "▁fine", "!"]
RATIO = time_ratio(8, 16000, 0.01)


def nemo(*steps):
    """(vocabulary piece, frame, duration frames, confidence) -> the hypothesis' parallel lists, as NeMo gives them."""
    return ([VOCAB.index(p) for p, *_ in steps], [f for _, f, _, _ in steps], [d for _, _, d, _ in steps],
            [c for *_, c in steps])


def test_times_are_parakeet_mlx_floats_and_texts_its_decoded_pieces():
    assert RATIO == 8 / 16000 * 160   # BaseParakeet.time_ratio: subsampling / rate * hop
    got = tokens(*nemo(("▁As", 2, 2, 0.97), ("▁tr", 3, 1, 0.9), ("end", 3, 0, 0.5)), VOCAB, RATIO, DURATIONS)
    assert [t.text for t in got] == [" As", " tr", "end"]
    # parakeet-mlx: start = step * time_ratio, duration = durations[decision] * time_ratio, end = start + duration
    assert [(t.start, t.duration, t.end) for t in got] == [(2 * RATIO, 2 * RATIO, 2 * RATIO + 2 * RATIO),
                                                           (3 * RATIO, RATIO, 3 * RATIO + RATIO), (3 * RATIO, 0.0, 3 * RATIO)]
    assert tokens(*nemo(("▁As", 35, 1, 1.0)), VOCAB, RATIO, DURATIONS)[0].start == 2.8000000000000003   # rounded later
    assert [t.id for t in got] == [1, 4, 5] and [t.confidence for t in got] == [0.97, 0.9, 0.5]


def test_sentences_end_where_parakeet_mlx_ends_them():
    got = tokens(*nemo(("▁As", 0, 1, 1), ("▁U", 1, 1, 1), (".S", 2, 1, 1), ("▁tr", 3, 1, 1), ("end", 4, 1, 1),
                       ("s", 5, 1, 1), (".", 6, 1, 1), ("▁", 7, 1, 1), ("Yeah", 8, 1, 1), ("?", 9, 1, 1),
                       ("Okay", 10, 1, 1), ("!", 11, 1, 1), ("▁fine", 12, 1, 1), (".", 13, 1, 1)), VOCAB, RATIO, DURATIONS)
    texts = ["".join(t.text for t in s.tokens) for s in sentences(got)]
    # ".S" holds a "." and the next token, " tr", a space: parakeet-mlx ends a sentence after "U.S" too, as it does at
    # "." before a lone space; "?" and "!" end one whatever follows
    assert texts == [" As U.S", " trends.", " Yeah?", "Okay!", " fine."]
    assert sentences([]) == []
    assert ["".join(t.text for t in s.tokens) for s in sentences(got[:5])] == [" As U.S", " trend"]   # the rest


def test_a_dot_inside_a_word_does_not_end_a_sentence():
    got = tokens(*nemo(("▁U", 0, 1, 1), (".S", 1, 1, 1), ("s", 2, 1, 1)), VOCAB, RATIO, DURATIONS)
    assert [[t.text for t in s.tokens] for s in sentences(got)] == [[" U", ".S", "s"]]


def test_the_layer_words_are_the_ones_the_mlx_runner_makes_from_the_same_tokens():
    steps = (("▁As", 2, 2, 0.97), ("▁rec", 8, 2, 0.9999), ("ent", 10, 2, 0.98), ("▁tr", 12, 1, 0.9), ("end", 13, 1, 0.99),
             ("s", 14, 2, 0.95), (".", 16, 0, 0.6), ("▁", 20, 1, 0.7), ("Yeah", 21, 3, 0.8), ("?", 24, 1, 0.75))
    mine = sentences(tokens(*nemo(*steps), VOCAB, RATIO, DURATIONS))
    # what parakeet-mlx builds from the same decoder output: AlignedTokens at step * time_ratio, split at . and ?
    theirs = [Sentence([Token(VOCAB.index(p), p.replace("▁", " "), f * RATIO, d * RATIO, c) for p, f, d, c in part])
              for part in (steps[:7], steps[7:])]
    assert mine == theirs
    words, rows = to_words(mine)
    assert [w["w"] for w in words] == ["As", "recent", "trends.", "Yeah?"]
    assert [(w["start"], w["end"]) for w in words] == [(0.16, 0.32), (0.64, 0.96), (0.96, 1.28), (1.6, 2.0)]
    assert [r["word"] for r in rows] == [0, 1, 1, 2, 2, 2, 2, 3, 3, 3] and [r["sentence"] for r in rows] == [0] * 7 + [1] * 3


def test_a_hypothesis_that_is_not_one_token_per_entry_is_refused():
    ids, frames, durations, conf = nemo(("▁As", 0, 1, 1.0), ("ent", 1, 1, 1.0))
    with pytest.raises(ValueError, match="2 token ids, 1 frames"):
        tokens(ids, frames[:1], durations, conf, VOCAB, RATIO, DURATIONS)
    with pytest.raises(ValueError, match=r"token 1: id 16 is not in the 16-token vocabulary"):   # the blank
        tokens([1, len(VOCAB)], frames, durations, conf, VOCAB, RATIO, DURATIONS)
    with pytest.raises(ValueError, match="duration 5 is not one the model predicts"):
        tokens(ids, frames, [1, 5], conf, VOCAB, RATIO, DURATIONS)
    with pytest.raises(ValueError, match="frame 0 is before the previous token's, 1"):
        tokens(ids, [1, 0], durations, conf, VOCAB, RATIO, DURATIONS)
    assert tokens([], [], [], [], VOCAB, RATIO, DURATIONS) == []


def test_nemos_entropy_gibbs_lin_measure_is_parakeet_mlxs_confidence():
    """NeMo (asr_confidence_utils, alpha 1): 1 + sum(exp(x) * x) / ln(v) over x = log_softmax(vocabulary and blank
    logits), v = blank_id + 1. parakeet-mlx (decode_greedy): 1 - H / ln(V + 1), H = -sum(p * ln(p + 1e-10)). The same
    up to that 1e-10 (tokens it cannot be told apart from 0 in float32 make it): here within 3e-7."""
    rng = np.random.default_rng(0)
    v = 8193
    for scale in (0.5, 3, 10, 30):   # flat to peaked distributions
        logits = rng.normal(0, scale, v)
        x = logits - logits.max() - math.log(np.exp(logits - logits.max()).sum())   # log_softmax
        nemo_conf = 1 + (np.exp(x) * x).sum() / math.log(v)
        p = np.exp(x)
        mlx_conf = 1 - (-(p * np.log(p + 1e-10)).sum()) / math.log(v)
        assert abs(nemo_conf - mlx_conf) < 3e-7, scale
