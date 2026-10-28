# Plan: perception pipeline, track bundle, and annotator app

Status: draft, 2026-10-23. Repos: **hyperstudy-perception** (this pipeline) and **hyperstudy-annotator** (the app).
Both start **private under `ljchang`** (next to the main app and Flowmail) and move to `hyperstudyio`, home of the
open-source helpers, when open-sourced. Licence is an open decision (end).

## 1. Shape

Two products joined by one file format.

```
video ──► PIPELINE (this repo: Python package + CLI) ──► TRACK BUNDLE (versioned files) ──► ANNOTATOR APP (separate repo)
            stages chosen per run; each stage's models           the contract                Tauri + Svelte 5: view, edit,
            and uv environment fetched on first use;                                         add data; SQLite
            runs on the user's Mac or NVIDIA box, or on cloud GPUs     │                     optional "Extract…": drives the
                ▲                                                      │                     pipeline CLI locally or in the cloud
                └──────────────── app shells out to the CLI ───────────┼─────────────────────────────┘
                                                                       └──► HYPERSTUDY: stores bundles, serves them to
                                                                            agents, embeds the annotator component
```

- The pipeline owns models, environments, GPUs and model licences, and is complete on its own as a CLI. It is
  installable on its own (`uv tool install`), so it also runs headless on a lab server or in HyperStudy's backend
  (`ptrack extract video.mp4 --stages faces,speakers,prosody`, `ptrack stages`, `ptrack models pull`).
- The app is the annotator: it views bundles, edits them (corrections) and adds data (annotations, imported layers).
  Import-only use needs nothing else. Extraction is optional: the app installs the pipeline CLI with a bundled `uv`
  (pyfeat-live's pattern), shows the stage catalogue (what each stage extracts, its models, download size, licence
  mode, hardware), and runs the stages the user picks on their own hardware or on cloud GPUs. Models are fetched only for
  chosen stages, and licences are shown before download, so the app itself still ships no weights.
- The bundle format is the only coupling. Both sides pin a schema version and validate against it.
- Corrections never modify a bundle. They are an append-only op log over it (the annotator prototype's design),
  exported as their own file and re-attachable after the pipeline re-runs.

## 2. Repositories

| Repo | Contents | Consumers |
|---|---|---|
| **hyperstudy-perception** (this one) | Stage code, one environment per stage group, the graph runner and CLI, the cloud backend, evaluation scripts and labels, **the bundle JSON Schema** (the producer owns it) | The lab, HyperStudy's backend |
| **hyperstudy-annotator** (new) | Tauri 2 + Svelte 5 desktop app; the annotator as a Svelte 5 package with a data adapter; a pinned copy of the bundle schema with fixture tests | Researchers; HyperStudy imports the annotator package |

A third repo for the schema alone is not worth it yet. If HyperStudy, the app and the pipeline start drifting, move
`schema/` out and publish it as both a Python and an npm package.

## 3. The track bundle (the contract)

A directory (or a `.ptrack` zip of it):

```
manifest.json        schema version; video sha256, duration, fps, size; pipeline version; per stage: model id and
                     revision, licence mode (academic-only / commercial-ok / needs-review), config hash, timings
shots.json           cuts, keyframe refs, shot captions
transcript.json      words with times; utterances; diarized voices
speakers.json        utterance -> speaker (face track / person / off-screen voice) with scores and evidence
faces.json           face tracks, identity clusters, per-sample geometry (or a pointer to faces_detail.arrow)
bodies.json, hands.json, gestures.json, facing.json, gaze.json, social.json
objects.json         object tracks (outlines), identities, wearer, verdicts
sounds.json          sound events with sources and agreement
prosody.json         pitch contour, per-utterance measures, arousal/valence, delivery captions and tags
crops/, keyframes/   small images for review (optional)
```

- **The video is referenced by sha256, never embedded** (size, and stimuli are often copyrighted). The app asks for
  the file on first open and checks the hash.
- Large per-frame arrays (face landmarks and mesh, per-frame scores) move to Arrow/Parquet side files once films
  are long; the JSON keeps pointers. The 2-minute panel clip's per-frame face data was already ~14 MB of JSON.
  Since 2026-10-27 every step keeps its model's full per-sample output in side files beside its layer (the layer's
  `raw` pointers), JSON when small and Parquet for large arrays; the gaze heatmaps were the first Parquet one
  (schema/README.md, "Side files").
- Every record carries its licence mode, so a commercial deployment can refuse academic-only layers.
- Corrections file: `perception-corrections/1` (already used by the prototype): video sha256, bundle version, ops.

First task: write the JSON Schema from today's outputs, then make every stage emit exactly it. The prototype's
per-stage files are close but inconsistent (ids, time fields, licence fields).

## 4. Pipeline (this repo)

**Graph.** Shared work is done once:

```
extract (shots, keyframes, audio, one frame cache)
 ├─ AUDIO: Parakeet + pyannote ─┬─ prosody (Parselmouth, Odyssey, captions)
 │         audio events ────────┘
 ├─ FACES: py-feat + ArcFace ── identity similarity
 │    ├─ speakers: LR-ASD (+ audio) ── voice links (gated, word share >= 0.3)
 │    └─ PEOPLE: one person-box pass ── RTMW body → RTMPose-hand → gestures ; MediaPipe facing
 ├─ VLM: one Haiku call per shot: caption + object names (+ face roster)
 │    └─ OBJECTS: SAM 3 → DINOv3 linking (+ bodies, faces)
 └─ GAZE: Gaze-LLE (faces + objects) ── SOCIAL rules
```

Duplicated today and to be removed: per-stage video decoding, three person-detector passes, two Haiku passes over
the same keyframes, a second model call to tag prosody captions.

**Stage contract.** `stage run --bundle DIR --video PATH` reads its inputs from the bundle and writes its files plus
a manifest entry. Re-runs are keyed by hash(video, stage version, config, upstream output hashes, and the code that
runs with its lock, the accelerator and the platform, never the machine), so only stale stages run, and a result made on
one machine is reused on any other of its platform.

**Environments.** Nine today (torch 2.5–2.14, transformers 4.52–5.20); most of the spread is rejected models (GRW,
HaGRID emblems, TalkNet, Sapiens2, RTMW3D, audEERING, emotion2vec+). Keep one uv project per environment (uv makes
many environments cheap: shared package cache, locked, built in seconds once cached), and merge where the locks
agree. Probable groups, to be confirmed by resolving them together: **core** (CPU: extract, linking, social rules,
Parselmouth, bundle writing), **audio** (Parakeet, pyannote, taggers, PE-A-Frame, Odyssey), **faces** (py-feat,
ArcFace, LR-ASD), **vision** (rtmlib/RTMW, RTMPose-hand, MediaPipe, Gaze-LLE, SAM 3, DINOv3), and **api** (Haiku and
audio-caption calls; no torch). Torch-heavy groups that refuse to share a lock stay separate.

**Stage catalogue.** Each stage declares what it extracts, its inputs, its environment, its models (id, revision,
download size, licence mode, gated or not), its hardware (CPU, Apple GPU, NVIDIA; minimum memory) and its outputs.
`ptrack stages` prints it; the app shows it as the "what to extract" picker. A run resolves the chosen stages plus
their upstream dependencies, fetches only those environments and weights, and asks for Hugging Face tokens only for
gated models (SAM 3, DINOv3, pyannote).

**Model sources: Hugging Face first.** Every model in the catalogue is a pinned source: `hf://repo@revision[/file]`
where one exists, else `url + sha256`. One download path, one token for gated models, one cache shared by all
environments. Status of the chosen stack (2026-10-23):

| On Hugging Face (official) | Not on Hugging Face | Mirror to a lab HF org? |
|---|---|---|
| Parakeet (`nvidia/parakeet-tdt-0.6b-v3`; MLX conversion on Macs), pyannote community-1 (gated), PE-A-Frame, Odyssey (`3loi/…`) + WavLM, py-feat v2 + ArcFace (`py-feat/…`), SAM 3 (`facebook/sam3`, gated), DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`, gated); also Qwen3-Omni, PaGE | PANNs Cnn14 (Zenodo) · EfficientAT mn40 (GitHub) · RTMW, RTMPose-hand, RTMDet-hand, YOLOX-m (OpenMMLab server) · MediaPipe pose/hands (Google storage) | Yes: CC-BY-4.0, MIT, Apache-2.0 and Apache-2.0 respectively, with credit, original licence and checksum |
| | LR-ASD (weights inside its GitHub repo) · Gaze-LLE (GitHub releases, + a DINOv2 file from Meta's server) | Not until their weight licences are confirmed (training data VoxCeleb2/LRS3, GazeFollow/VideoAttentionTarget); keep url + sha256 |

The py-feat organisation on Hugging Face is the natural home for the mirrors. Mirroring does not change any
licence mode: weights trained on non-commercial data stay academic-tier.

**Where it runs.** The same stage code in native uv environments everywhere: on the user's machine (a Mac uses the
Apple GPU), on cloud machines and on lab servers. No containers (2026-10-24, `docs/plans/GRAPH_MVP_PLAN.md`). Cloud GPUs
are reached through one SkyPilot backend (decided 2026-10-23, after a survey of orchestration frameworks): the same
locked environments run on whichever provider is cheapest and available (RunPod, Lambda, Vast, AWS, GCP, Azure, Kubernetes, Slurm), using the user's own
accounts, with inputs passed as signed URLs and outputs returned as artifacts. Local runs schedule one
heavy GPU stage at a time (overlapping SAM 3 and Sapiens2 pushed a 128 GB Mac into 45 GB of swap). Nothing has been
timed on NVIDIA yet.

**Cloud host: chosen on measured cost, not assumed.** No provider is preferred; the user will use whichever is best
and cheapest (2026-10-23). Estimates per 2-hour film from the hosting report (unmeasured GPU time): AWS Batch Spot L4
$1.00–3.60 (films stay next to S3; Dartmouth billing), Hugging Face Jobs L4 $1.40–4.30, Modal L4 $1.50–4.80; the
spec cites RunPod serverless from $0.69/h for 24 GB GPUs (not yet checked). Plan: time one 10-minute clip on one
NVIDIA GPU through SkyPilot, then price the measured stage timings across providers and pick. Modal is not reachable
through SkyPilot; it would need its own adapter, worth writing only if a measurement favours it.

**Chosen models (academic tier).** From the bake-off's measured results (kept with the lab's research notes, not in this repository).

## 5. Annotator app (new repo)

- **Stack:** Tauri 2, Svelte 5, Vite, SQLite through sqlx with migrations: the HyperStudy Bridge stack. Copy the
  release setup from **Flowmail** (`ljchang/flowmail`), which already signs, notarizes and auto-updates a private
  Tauri app from this account (`.github/workflows/release-macos.yml`, `scripts/notarize.sh`, an update token for
  private releases); Bridge's `scripts/build-and-sign-mac.sh` is the other reference. Trim the entitlements: no serial devices, JIT or
  debugger; keep network client and user-selected files.
- **Reuse from pyfeat-live** (same stack, cosanlab, MIT): `OverlayCanvas`, `MeshCanvas`, `ScrubBar`,
  `TimeseriesPlot`, `ValenceArousalPlot`, and the identity review components (`IdentityClusterPanel`,
  `IdentityMergeSuggestions`, `IdentityAssignDialog`), `AnnotationPopover`. If both apps keep them, extract a shared
  Svelte component package later; copy first.
- **SQLite holds:** the library of imported bundles and their videos (paths, sha256), the corrections op log
  (append-only, one row per op), annotation types and notes, review state and per-viewer layout. Bundle files are
  stored as imported, read-only, in the app's data directory. This replaces the prototype's browser storage, and
  with it the multi-tab overwrite problem.
- **View, edit, add.** Import a bundle → locate the video → review and correct (identities, speakers, objects,
  inclusion for agents) → add data: the researcher's own annotation types and notes, and **imported layers**
  (another tool's timed events as CSV or JSON, or a newer pipeline stage's output), stored as additional layers
  beside the bundle, never inside it → export corrections, annotations and layers. Later: open a bundle from a
  HyperStudy URL.
- **Extract (optional).** "Extract…" on a video opens the stage catalogue; the app installs or updates the pipeline
  CLI through its bundled `uv` (as pyfeat-live installs its runtime), shows download sizes and licences, runs the
  chosen stages locally or on cloud GPUs, streams progress from the CLI's JSON log, and imports the bundle when done. The
  app never imports pipeline code; it only runs the CLI, so the two repos version independently.
- **The annotator as a package:** port `pipeline/bakeoff/annotator/template.html` to Svelte 5 components behind a
  data adapter (load bundle, media URL, load and append corrections). The app supplies a SQLite adapter; HyperStudy
  supplies its own.

## 6. HyperStudy

HyperStudy integrates the bundles and the annotator through interfaces of its own.

## 7. Phases

| # | Deliverable | Depends on |
|---|---|---|
| 0 | This repo: bake-off code, notes, labels, plan | — |
| 1 | Bundle JSON Schema v0.1 from today's outputs; converter from bake-off runs to a bundle; fixtures (OOO, NES) | 0 |
| 2 | Annotator app skeleton: Tauri + Svelte 5 + SQLite, import a bundle, locate video, play with overlays, signing CI | 1 |
| 3 | Annotator ported into the app as a package; corrections in SQLite; export | 2 |
| 4 | Pipeline refactor to the graph, stage contract and stage catalogue; `ptrack` CLI; shared decode and person boxes; environments merged where locks agree | 1 |
| 5 | First NVIDIA timing through SkyPilot in uv environments (done on an L4, 2026-10-24); a 10-minute clip; price the timings across providers; pick; full film | 4 |
| 5b | App "Extract…": install the CLI with bundled uv, stage picker, local and cloud runs, auto-import | 3, 4 |
| 6 | HyperStudy: trigger, storage, annotator embed, agent read API | 3, 5 |
| ∥ | Evaluation set grown through the app (identities, speakers, gaze, prosody ratings) | 3 |

Phases 2–3 and 4–5 can run in parallel once the schema exists.

## 8. Open decisions

- Visibility (private to start was proposed). CLI command name (`ptrack` is a placeholder).
  Decided 2026-10-23: private under ljchang, moved to hyperstudyio when open-sourced; repos hyperstudy-perception and
  hyperstudy-annotator.
  Signing secrets are per repository: add Flowmail's set (Apple certificate, identity, Apple ID, team ID, Tauri
  updater key) to hyperstudy-annotator when signed builds start.
- Licence for the code (MIT, like pyfeat-live, with a `LICENSE-MODELS.md` for the pipeline's weights).
- Bundle as a directory or a single `.ptrack` zip for transfer (both can be supported).
