# Track bundle format, v0.1 (draft)

The contract between the pipeline (this repo), the annotator app and HyperStudy. A bundle is a directory:

```
manifest.json            what the bundle is: video identity, pipeline, layers with checksums and licence modes
layers/<name>.json       one file per layer (shots, transcript, speakers, faces, bodies, hands, gestures, facing,
                         gaze, objects, sounds, text, prosody); every layer is optional
layers/faces_detail.json heavy per-sample arrays live in side files the layer points to
assets/                  optional small images (keyframes, person and object crops) referenced by relative path
```

`bundle-0.1.schema.json` validates the manifest (the root schema) and each layer file
(`#/$defs/layer/<name>`). `pipeline/bundle/validate.py` checks a bundle directory against both, plus
cross-references the schema cannot express (every id a layer points at exists).

## Conventions

- **Video identity.** A bundle never contains the video. `manifest.video.sha256` identifies it; readers ask
  for the file and check the hash. Stimuli are often copyrighted, and films are large.
- **Time** is seconds from the start of the video (float, millisecond precision). Spans are `start`/`end`,
  samples are `t`.
- **Space** is source-video pixels. Boxes are `[x, y, w, h]`. Keypoints are `[x, y, score]`, where each
  layer states its score scale and the threshold its producer used (`score_threshold`).
- **IDs** are strings `"<kind>:<local>"`, unique within a bundle: `shot:3`, `utt:17`, `voice:SPEAKER_01`,
  `face:126` (a face track), `person:Person A` (a machine identity cluster), `body:41`, `hand:125`, `obj:47`
  (an object track), `objid:white box 1` (an object identity), `sound:9`, `text:2`, `gesture:4`,
  `facing:7`, `gaze:31`. References between layers use these ids. Ids are stable for a given pipeline run,
  and they change when a stage re-runs; corrections re-attach by time and geometry, not by id alone.
- **Persons are machine clusters.** The pipeline over-splits on purpose (a wrong merge misleads agents, a split
  is fixed in review); names and merges are corrections, never written into a bundle.
- **Producer.** Every layer file carries `producer`: stage name and version, the models used (id, source,
  revision, licence, licence mode), the configuration that affects outputs, run time, device and timings.
- **Licence mode** per model and per layer: `academic-only`, `commercial-ok`, `needs-review`, or `api-terms`
  (output of a commercial API, governed by its terms). A layer's mode is the most restrictive of its models'.
  A commercial deployment drops layers that are not `commercial-ok` or `api-terms`.
- **Extensibility.** Core fields are required and typed; objects allow extra, producer-specific fields
  (`additionalProperties` is open), so stages can add detail without breaking readers. Readers ignore fields they
  do not know.
- **Versioning.** `schema_version` is semver. Minor versions only add optional fields or layers; a major version
  may change required fields. Readers accept bundles whose major version they know.

## Corrections and added data (not part of a bundle)

The annotator stores corrections as an append-only op log over a bundle (`perception-corrections`), plus the
user's own annotation types, notes and imported layers. They reference bundle ids and the video sha256, and are
exported as their own files. The prototype's op log (v1) uses bare track numbers and person labels; mapping it to
v0.1 ids is mechanical (`track 126` -> `face:126`, `Person O` -> `person:Person O`).

## Open for v0.2

- Arrow/Parquet side files for heavy per-frame arrays (a 2-hour film's face detail would be ~1 GB of JSON).
- A `social` layer once the social-signal rules are settled (today: gaze events and facing).
- Per-utterance speaker evidence from more than one detector (today: LR-ASD candidates + TalkNet agreement).
