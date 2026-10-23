"""Second transcriber for the bake-off: NVIDIA Parakeet TDT 0.6B v3 via MLX.

Writes runs/<name>/transcript_parakeet.json in the same shape as the WhisperX
transcript in extract.json (segments with word times and pyannote speakers),
then compares word onsets against WhisperX for the words both systems heard.

Usage: uv run python transcribe_parakeet.py runs/<name>
"""

import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

from extract import load_env

PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


def to_words(sentence):
    """Merge Parakeet's subword tokens into words; a leading space starts a new word."""
    words = []
    for token in sentence.tokens:
        if not words or token.text.startswith(" "):
            words.append({"word": token.text.strip(), "start": token.start, "end": token.end})
        else:
            words[-1]["word"] += token.text
            words[-1]["end"] = token.end
    return [w for w in words if w["word"]]


def diarization_turns(wav):
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    frame = DiarizationPipeline(token=os.environ["HF_TOKEN"], device="cpu")(whisperx.load_audio(str(wav)))
    return [{"start": r.start, "end": r.end, "speaker": r.speaker} for r in frame.itertuples()]


def speaker_for(word, turns):
    overlaps = {}
    for turn in turns:
        overlap = min(word["end"], turn["end"]) - max(word["start"], turn["start"])
        if overlap > 0:
            overlaps[turn["speaker"]] = overlaps.get(turn["speaker"], 0) + overlap
    return max(overlaps, key=overlaps.get) if overlaps else None


def norm(word):
    return re.sub(r"[^a-z0-9']", "", word.lower())


def compare(whisperx_words, parakeet_words):
    a, b = [norm(w["word"]) for w in whisperx_words], [norm(w["word"]) for w in parakeet_words]
    pairs = []
    for block in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        for i in range(block.size):
            wx, pk = whisperx_words[block.a + i], parakeet_words[block.b + i]
            if wx["start"] is not None:
                pairs.append({"word": pk["word"], "whisperx": wx["start"], "parakeet": round(pk["start"], 3),
                              "diff": round(pk["start"] - wx["start"], 3)})
    diffs = sorted(abs(p["diff"]) for p in pairs)
    return {
        "matched_words": len(pairs), "whisperx_words": len(a), "parakeet_words": len(b),
        "median_abs_onset_diff_s": diffs[len(diffs) // 2] if diffs else None,
        "over_0.5s": [p for p in pairs if abs(p["diff"]) > 0.5],
        "only_whisperx": [whisperx_words[i]["word"] for op, i1, i2, _, _ in
                          difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
                          if op in ("delete", "replace") for i in range(i1, i2)],
        "only_parakeet": [parakeet_words[j]["word"] for op, _, _, j1, j2 in
                          difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
                          if op in ("insert", "replace") for j in range(j1, j2)],
    }


def main():
    load_env()
    from parakeet_mlx import from_pretrained

    run_dir = Path(sys.argv[1])
    wav = run_dir / "audio.wav"

    t0 = time.time()
    result = from_pretrained(PARAKEET_MODEL).transcribe(str(wav))
    seconds = round(time.time() - t0, 1)
    turns = diarization_turns(wav)

    segments = []
    for sentence in result.sentences:
        words = to_words(sentence)
        for w in words:
            w["start"], w["end"] = round(w["start"], 3), round(w["end"], 3)
            w["speaker"] = speaker_for(w, turns)
        speakers = [w["speaker"] for w in words if w["speaker"]]
        segments.append({"start": round(sentence.start, 3), "end": round(sentence.end, 3),
                         "speaker": max(set(speakers), key=speakers.count) if speakers else None,
                         "text": sentence.text.strip(), "words": words})

    transcript = {"model": PARAKEET_MODEL, "diarized": True, "seconds": seconds, "segments": segments}
    (run_dir / "transcript_parakeet.json").write_text(json.dumps(transcript, indent=2))

    whisperx = json.loads((run_dir / "extract.json").read_text())["transcript"]
    comparison = compare([w for s in whisperx["segments"] for w in s["words"]],
                         [w for s in segments for w in s["words"]])
    (run_dir / "transcript_comparison.json").write_text(json.dumps(comparison, indent=2))

    print(f"Parakeet: {seconds}s")
    for s in segments:
        print(f"  [{s['start']:.1f}-{s['end']:.1f} {s['speaker']}] {s['text']}")
    print(json.dumps({k: v for k, v in comparison.items()}, indent=2))


if __name__ == "__main__":
    main()
