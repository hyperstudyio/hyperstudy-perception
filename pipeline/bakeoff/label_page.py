"""Ground-truth labelling page: who is speaking, utterance by utterance.

Builds runs/<name>/label_speakers.html, a local page that plays the clip from
disk (nothing is uploaded) with the py-feat face tracks drawn over the video.
For each utterance (Parakeet words split at sentence ends, pauses and cuts) the
labeller picks the speaking face by its number, or off-screen / visible but
not boxed / several speakers / not speech. Model predictions are not shown, so
they cannot bias the labels. Labels autosave in the browser and export as JSON
(speaker_labels.json), which evaluate_asd.py scores ASD and diarization against.

Usage: uv run python label_page.py runs/<name>
"""

import json
import sys
from pathlib import Path

PAUSE_SPLIT = 0.25  # seconds of silence between words that starts a new utterance
SENTENCE_END = (".", "?", "!")


def utterances(transcript, shots):
    cuts = [s["start"] for s in shots[1:]]
    words = [w for segment in transcript["segments"] for w in segment["words"]]
    units, current = [], []
    for word in words:
        if current:
            previous = current[-1]
            crosses_cut = any(previous["end"] <= c <= word["start"] or previous["start"] < c <= word["start"] for c in cuts)
            if (word["start"] - previous["end"] > PAUSE_SPLIT or previous["word"].endswith(SENTENCE_END)
                    or crosses_cut):
                units.append(current)
                current = []
        current.append(word)
    if current:
        units.append(current)

    def shot_at(t):
        return next(s["index"] for s in reversed(shots) if s["start"] <= t)

    return [{"id": i, "start": u[0]["start"], "end": u[-1]["end"], "text": " ".join(w["word"] for w in u),
             "shot": shot_at((u[0]["start"] + u[-1]["end"]) / 2)} for i, u in enumerate(units)]


def main():
    run_dir = Path(sys.argv[1])
    extract = json.loads((run_dir / "extract.json").read_text())
    transcript = json.loads((run_dir / "transcript_parakeet.json").read_text())
    faces = json.loads((run_dir / "faces.json").read_text())
    video = Path(extract["video"]["source"])
    video_src = Path("..") / ".." / video if not video.is_absolute() else video

    crops = {p["label"]: p["crop"] for p in faces["people"]}
    tracks = [{"id": i, "person": t["person"], "shot": t["shot"], "start": t["start"], "end": t["end"],
               "t": [s["t"] for s in t["series"]], "box": [s["box"] for s in t["series"]]}
              for i, t in enumerate(faces["tracks"])]
    data = {"clip": video.name, "sha256": extract["video"]["sha256"], "width": extract["video"]["width"],
            "height": extract["video"]["height"], "units": utterances(transcript, extract["shots"]),
            "tracks": tracks, "crops": crops, "transcriber": transcript["model"]}

    template = (Path(__file__).parent / "label_page_template.html").read_text()
    page = (template.replace("__VIDEO_SRC__", str(video_src))
            .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    out = run_dir / "label_speakers.html"
    out.write_text(page)
    print(f"{out}: {len(data['units'])} utterances, {len(tracks)} face tracks")


if __name__ == "__main__":
    main()
