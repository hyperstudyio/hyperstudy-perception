"""NeMo's greedy TDT hypothesis as parakeet-mlx's tokens and sentences, so the CUDA runner (node_nemo.py) merges words
and writes the side file with the same code as the MLX one (raw_tokens.to_words, raw_tokens.write).

NeMo 3.0.0's greedy_batch decoding (tdt_label_looping.py) returns, for the one utterance: the token ids (y_sequence),
the encoder frame each was emitted at (timestamp), its predicted duration in frames (token_duration, with
tdt_include_token_duration) and its confidence at the emitting step (token_confidence, with preserve_token_confidence
and exclude_blank). parakeet-mlx 0.5.3 (Apache-2.0) gives each token as AlignedToken(id, text, start = frame *
time_ratio, duration = frames * time_ratio, confidence), its text the vocabulary piece with SentencePiece's "▁" as a
space (tokenizer.decode), and splits them into sentences (alignment.tokens_to_sentences with its default
SentenceConfig: only the punctuation rule applies). Both are restated here, for the same times to the last bit and the
same sentences from the same tokens. Pure Python: no torch, no NeMo.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Token:   # parakeet-mlx's alignment.AlignedToken: the fields raw_tokens.to_words reads
    id: int
    text: str
    start: float
    duration: float
    confidence: float
    end: float = field(init=False)

    def __post_init__(self):
        self.end = self.start + self.duration


@dataclass
class Sentence:   # parakeet-mlx's alignment.AlignedSentence: its tokens, sorted by start (stable, as it sorts them)
    tokens: list[Token]

    def __post_init__(self):
        self.tokens = sorted(self.tokens, key=lambda t: t.start)


def time_ratio(subsampling_factor: int, sample_rate: int, window_stride: float) -> float:
    """Seconds an encoder frame, computed as parakeet-mlx computes it (BaseParakeet.time_ratio, hop_length an int), so a
    frame's time is the same float: 8 / 16000 * 160 for Parakeet TDT."""
    return subsampling_factor / sample_rate * int(window_stride * sample_rate)


def tokens(ids, frames, durations, confidences, vocabulary: list[str], ratio: float, model_durations) -> list[Token]:
    """The hypothesis' parallel lists -> Tokens. Refused (ValueError) when they are not one per token, an id is not in
    the vocabulary (the blank, len(vocabulary), is never a token), a duration is not one the model predicts, or the
    frames go back in time."""
    ids, frames, durations, confidences = list(ids), list(frames), list(durations), list(confidences)
    if not len(ids) == len(frames) == len(durations) == len(confidences):
        raise ValueError(f"{len(ids)} token ids, {len(frames)} frames, {len(durations)} durations and "
                         f"{len(confidences)} confidences: not one each per token")
    out = []
    for k, (i, f, d, c) in enumerate(zip(ids, frames, durations, confidences)):
        i, f, d = int(i), int(f), int(d)
        if not 0 <= i < len(vocabulary):
            raise ValueError(f"token {k}: id {i} is not in the {len(vocabulary)}-token vocabulary")
        if d not in model_durations:
            raise ValueError(f"token {k}: duration {d} is not one the model predicts ({list(model_durations)})")
        if out and f < frames[k - 1]:
            raise ValueError(f"token {k}: frame {f} is before the previous token's, {frames[k - 1]}")
        out.append(Token(i, vocabulary[i].replace("▁", " "), f * ratio, d * ratio, float(c)))
    return out


def sentences(tokens: list[Token]) -> list[Sentence]:
    """parakeet-mlx's tokens_to_sentences with the default SentenceConfig: a sentence ends at a token holding ! ? 。 ？
    or ！, or holding . when it is the last token or the next one holds a space; the rest is the last sentence."""
    out, current = [], []
    for idx, token in enumerate(tokens):
        current.append(token)
        if ("!" in token.text or "?" in token.text or "。" in token.text or "？" in token.text or "！" in token.text
                or ("." in token.text and (idx == len(tokens) - 1 or " " in tokens[idx + 1].text))):
            out.append(Sentence(current))
            current = []
    if current:
        out.append(Sentence(current))
    return out
