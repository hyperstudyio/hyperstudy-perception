# Track bundle format, v0.1 (draft)

The contract between the pipeline (this repo), the annotator app and HyperStudy. A bundle is a directory:

```
manifest.json                 what the bundle is: video identity, pipeline, layers with checksums and licence modes
layers/<name>.json            one file per layer (shots, transcript, speakers, faces, bodies, hands, gestures, facing,
                              gaze, objects, sounds, text, prosody); every layer is optional
layers/faces_detail.json      side files: the model output a layer does not draw, beside the layer that points to it
layers/faces_mesh.json        (JSON when small, Parquet for large arrays)
layers/faces_embeddings.json
layers/gaze_heatmaps.parquet
layers/sounds_tags.parquet
layers/sounds_queries.parquet
assets/                       optional small images (keyframes, person and object crops) referenced by relative path
```

`bundle-0.1.schema.json` validates the manifest (the root schema), each layer file (`#/$defs/layer/<name>`) and each
JSON side file (`#/$defs/side/<name>` for `layers/<name>.json`), and documents each Parquet side file's columns
(`#/$defs/parquet/<name>` for `layers/<name>.parquet`). The manifest lists a layer's side files under its `side_files`,
with their checksums.
The faces layer's: `faces_detail` (its `detail.file`: 68
landmarks, action units, emotions, valence and arousal, head pose and gaze per sample), `faces_mesh` (its
`mesh.file`: the 478-point 2D face mesh per sample, all x then all y in whole source pixels; about 11 MB for a
two-minute clip, so readers fetch it only when they draw the mesh) and `faces_embeddings` (its `embeddings.file`:
the 512-d ArcFace identity embedding per sample, the vectors the layer's people and `person_pairs` were computed
from, kept for reuse such as finding the same person in another video or re-clustering; as py-feat returns them,
not L2-normalised (`normalised: false`), to 4 decimal places, which keeps cosine distances within 2e-5 of full
precision; about 10 MB for a two-minute clip).
The gaze layer's: `gaze_heatmaps` (its `raw.heatmaps`: Gaze-LLE's 64x64 heatmap for every head, of which the layer
keeps the peak, with the unrounded in-frame score; one row per sample; each map as uint8 relative to its own maximum,
`floor(255 * heat / max)`, with that maximum as float32, so `(cell + 0.5) / 255 * max` is within `max / 510` of the
model's value and the stored map's argmax is the layer's peak; about 0.9 KB a head, 2.4 MB for a two-minute clip).
The sounds layer's: `sounds_tags` (its `raw.tags`: the AudioSet taggers' scores for all 527 classes, PANNs Cnn14's
every 0.32 s segment, which its 100 frames/s output repeats, and EfficientAT mn40's every 1 s step; a row per model
output with the scores as a fixed-size list in the order of the file metadata's `labels`) and `sounds_queries` (its
`raw.queries`: PE-A-Frame's probability for each query every 40 ms frame, in the order of the metadata's `queries`).
Both are the models' float32 scores, unrounded, so thresholding them again gives the layer's events exactly; about
26 MB and 6 MB an hour of soundtrack (0.9 MB and 0.2 MB for a two-minute clip).
`pipeline/bundle/validate.py` checks a bundle directory against them, plus cross-references the schema cannot
express (every id a layer points at exists; faces_detail, faces_mesh and faces_embeddings each hold a sample for each
of the faces layer's samples, at the same times; gaze_heatmaps holds a row for each of the gaze layer's samples).

The embeddings are identity data: a pipeline step that reads them on another machine needs the graph policy's
`allow_upload` and the run's consent to name `embeddings`. Like the faces layer, they come from InsightFace's ArcFace
weights, which are licensed for non-commercial research only (`academic-only`). The gaze heatmaps hold no pixels, but
each is computed from the whole frame, and together they are a coarse picture of where the people and things in each
frame are: a step that reads them elsewhere needs `allow_upload` to name `frames` (not `faces`: they hold no face
image, landmarks or box). The sounds tags and queries hold no samples, but they are a dense measure of the whole
soundtrack (as the waveform is), speech classes included: a step that reads them elsewhere needs `allow_upload` to
name `audio`.

## Side files: keeping a model's full output

Every pipeline step keeps its model's full per-sample output, not only what the app's timeline draws (decided
2026-10-27). The layer stays the app's view; the rest goes in side files beside it: JSON when small, Parquet for large
arrays (per-frame scores, maps, embeddings of long films). A layer points at its side files with `raw`, one entry per
role:

```json
"raw": {"heatmaps": {"file": "layers/gaze_heatmaps.parquet", "format": "parquet",
                     "description": "Gaze-LLE's 64x64 heatmap per head (...), a row per sample"}}
```

`file` is `layers/<name>.<format>`, `format` is `json` or `parquet`, `description` is one line on what it holds. The
faces layer's `detail`, `mesh` and `embeddings` predate `raw` and keep working; a layer may use both.

To add one (a step's raw output: diarization turns, per-frame class scores, word confidences...):

1. **Register it** in `orchestrator/src/perception/artifacts.py` `SIDE_FILES`: `"<name>": Side("<layer>", (<kinds>),
   "json" | "parquet")`. Its type is `bundle.<name>@0.1` and its file `layers/<name>.<format>`. The kinds are what it
   holds in `policy.allow_upload`'s terms (`video`, `audio`, `frames`, `faces`, `transcript`, `embeddings`), counted
   for its content rather than its source: say why in the comment there.
2. **Describe it** in `bundle-0.1.schema.json`: a JSON one as a schema at `#/$defs/side/<name>`, which the orchestrator
   validates every output against; a Parquet one as its columns at `#/$defs/parquet/<name>` (name, Arrow type as
   pyarrow prints it, description; `per_sample: true` when it holds a row per sample of the layer's tracks, which
   validate.py then checks). A test fails while a registered side file is not described, or a described one is not
   registered.
3. **Write it** from the node: declare an output of type `bundle.<name>@0.1` in its manifest, write the file
   (`node.write_json`, or `node.path` and your Parquet writer), and put the `raw` pointer in the layer. Bump the node's
   version and move its pin in `graphs/`.
4. **Parquet: own the columns and the bytes.** The orchestrator does not read Parquet (it has no pyarrow, to stay light
   for the app): it checks the file's PAR1 magic at both ends and its footer length, and hashes it like any file. The
   node, with pyarrow pinned exactly in its environment, writes a fixed schema with no nullable columns and fixed
   writer options (compression and level, row group size, dictionary and statistics columns, format and data page
   versions), so the same rows are the same bytes (a stored artifact must be, or the cache breaks downstream); no
   times in the file. Test the columns against `#/$defs/parquet/<name>` and the bytes by writing twice
   (`nodes/vision/gaze/heatmaps.py` and its tests are the pattern). Write rows in fixed-size groups as they come
   rather than holding a film's worth in memory.
5. **Export and validate.** `perception export` copies every side file a layer's node wrote, byte for byte, lists it
   under the layer's `side_files`, and stops if a pointer names a file that node did not write. validate.py checks
   pointers, checksums, JSON side schemas and Parquet magic; with pyarrow (its dev group, which `uv run` installs) it
   also reads each Parquet file's columns and rows, and without pyarrow it says it skipped that check.

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

- Moving the faces layer's heavy JSON side files (detail, mesh, embeddings) to Parquet: a 2-hour film's face detail
  would be ~1 GB of JSON. Parquet side files exist in 0.1 (above); moving existing ones changes what readers load.
- A `social` layer once the social-signal rules are settled (today: gaze events and facing).
- Per-utterance speaker evidence from more than one detector (today: LR-ASD candidates + TalkNet agreement).
