# Perception graph: Phase 0–1 implementation plan

Implements `PERCEPTION_GRAPH_SPEC.md` (the spec of record) up to its Phase 1 local MVP. This file records the
choices the spec leaves open and where the spec meets what this repo already has (`PLAN.md`, the bundle schema,
the bake-off stages). Search the code for `TODO: [PERCEPTION-GRAPH]` for unfinished steps.

## Decisions (defaults; each can be revisited)

| Question | Choice for the MVP | Why |
|---|---|---|
| Canonical artifact types | **The bundle's layer schemas are the artifact types** (`bundle.faces@0.1` validates against `schema/bundle-0.1.schema.json#/$defs/layer/faces`), plus a few internal types (`media.v1`, `audio.wav.v1`, and `assets.v1`: keyframes and crops, merged into the bundle's `assets/`). A bundle is a published *view* over committed artifacts (manifest + layer files), not a second format. | Acceptance criterion 9: the annotator and HyperStudy already read bundles. One schema, one validator. |
| Artifact encoding | JSON for v0.1 layers; Parquet/Arrow for heavy time series when the bundle moves to v0.2 (already planned there). | Changing encoding is a schema-version change on both sides; keep it to one move. |
| Time | Seconds (float, ms precision) in layers, as the bundle does today. `media.v1` records the container timebase, `r_frame_rate` vs `avg_frame_rate` and whether the stream is variable-rate, so the integer-µs mapping can be added without re-decoding. | Integer µs everywhere is a bundle major version; flagged as an open decision rather than smuggled in. |
| Scheduler | Custom asyncio coordinator + SQLite, as the spec recommends. The Prefect spike is dropped (2026-10-23): a survey of 17 frameworks found none that meets per-node environments, the Apple GPU, a private cross-run cache, cost caps and Windows together (`docs/research/orchestration_frameworks.md`). | Every node runs in its own environment (py-feat, Parakeet-MLX, SAM 3 cannot share a lock), so a framework would mostly launch subprocesses; Prefect's task runners are per flow, not per task. |
| Native runner | `uv run --project envs/<env> python -m <module>`: one uv project per environment group (`PLAN.md` §4). The runner hashes the env's `uv.lock` into the cache key. | uv environments are cheap and locked; this is what the bake-off already does. |
| OCI runner | `docker run` with the image digest; linux/amd64 + CUDA. Built in Phase 1, exercised for real in Phase 2 (no NVIDIA box here). | Docker on macOS cannot reach the Apple GPU (spec agrees). |
| Node protocol | Files, not sockets: the coordinator writes `request.json` into a fresh attempt directory; the node writes outputs under `out/` and `response.json`. The coordinator validates, hashes and commits. A node never writes into the store. | Atomic commit and safe crash recovery (criterion 3) fall out of it: an orphaned attempt can only write into its own directory. The same contract works over SSH/HTTPS later (stage inputs, run, collect). |
| Node SDK | `sdk/perception_node`: stdlib only, added as a path dependency of every env. Reads the request, resolves input paths, writes the response, maps exceptions to error codes (`transient`, `input`, `schema`, `privacy`, `licence`, `internal`). | Nodes stay plain scripts; no orchestrator code in model environments. |
| GPU budget on a Mac | One `gpu: exclusive` node at a time; `gpu: shared` nodes up to the free unified memory. | Measured: SAM 3 overlapping Sapiens2 pushed a 128 GB Mac into 45 GB of swap. |
| Weights | Nodes declare their weights (`hf://repo@revision`, size, gated, licence). `perception doctor`/`plan` report what is missing; `perception models pull` fetches. A run never downloads silently: a missing weight fails the plan. | Spec: never silently download large weights. |
| Secrets | Names only in YAML and requests (`secrets: [HF_TOKEN]`); values from the environment or `~/.config/hyperstudy/*.env`, passed in the child's environment, never written to disk or logs. | Spec security section. |
| Privacy | `privacy.external_upload` per node and per data kind (video, audio, face crops, transcripts, embeddings). A graph's `policy.privacy: local-only` makes any node that needs upload a plan error. | Copyrighted stimuli: only Anthropic/OpenAI were approved for the bake-off clips. |
| CLI name | `perception` (the spec's sketch); package `hyperstudy-perception`. Replaces the `ptrack` placeholder. | Final name is still an open decision. |

## Phase 0 — contracts

1. Graph IR: YAML → validated model (pydantic): inputs, outputs, policy, nodes, `$inputs.x` / `$nodes.n.out`
   references, topological order, cycle and type checks against node manifests.
2. Node manifests: `nodes/<namespace>/<name>/manifest.yaml` (id, version, inputs, outputs with artifact types,
   runners, resources, execution, privacy, licence, weights, secrets). A registry loads them all.
3. Artifact types: a registry mapping type ids to validators (JSON Schema refs into the bundle schema, or a
   file-shape check for binaries).
4. `perception graph validate`, `perception plan --explain`.
5. ~~Prefect spike~~: dropped after the framework survey (`docs/research/orchestration_frameworks.md`).

## Phase 1 — local MVP

1. Content-addressed store: `store/objects/sha256/ab/cd…` (files) and tree objects for directories; artifact
   records (type, hash, size, producer task) in SQLite. Atomic publish: write to a temp path, fsync, rename.
2. Run state: runs, tasks (pending, ready, leased, running, succeeded, cached, failed, canceled), attempts,
   leases with expiry and heartbeat, events log. Resume reclaims expired leases.
3. Cache: key = hash(manifest id@version, runner identity (env lock hash or image digest), canonical config,
   input artifact hashes, artifact schema versions, partition); a hit whose objects all exist marks the task
   `cached`.
4. Coordinator: asyncio loop; ready = all inputs committed; placement by runner eligibility and resources;
   retries with backoff for `transient` only; `max_cost_usd` reservation (zero for local nodes).
5. Backends: native subprocess (uv) and Docker, behind `probe_capabilities / estimate / stage_inputs / submit /
   poll / cancel / collect / cleanup`.
6. `perception doctor` (OS, CPU, RAM, Apple GPU / CUDA / VRAM, Docker, uv, ffmpeg), `run`, `status`, `logs`,
   `resume`, `rerun --from`, `artifacts`, `export --format bundle`.
7. Nodes: `core.decode` (probe, sha256, timebase, shots, keyframes, 16 kHz mono audio), `vision.face-track`
   (py-feat + ArcFace clustering, from `pipeline/bakeoff/faces`), `audio.asr` (Parakeet-MLX on Apple silicon,
   NeMo Parakeet on CUDA, + pyannote diarization), all emitting bundle layers.
8. Tests: graph validation, cache keys, a crash mid-run (kill the coordinator, resume, no partial artifacts), a
   transient failure retried, a schema failure terminal, independent nodes in parallel, a changed config re-running
   only descendants. Fake nodes for scheduler tests; the three real nodes on the NES clip as an integration run.

## Status (2026-10-23)

Done and tested:
- **Contracts.** The graph IR with typed references, config, cycle and privacy checks (`graph validate`). Node manifests. Artifact types (bundle layers, `media.v1`, `audio.wav.v1`, `assets.v1`).
- **Store and state.** A content-addressed store that keeps file names. SQLite run state with numbered migrations.
- **Coordinator.**
  - Leases, and adopting the attempts a dead coordinator left.
  - Retries for transient errors only.
  - CPU, memory and GPU admission.
  - The budget cap.
  - Atomic publication.
  - `rerun --from`.
- **Native and self backends**, plus `doctor`, `plan` (placement, cache, missing weights and secrets, spend), `run`, `status`, `logs`, `resume`, `rerun`, `cancel`, `artifacts` and `export --format bundle`.
- **Tests:** 17 pytest (`orchestrator/tests`), covering acceptance criteria 2–5 and 7–8 with real processes, including killing the coordinator mid-run.
- **Nodes on the NES clip.** All three reproduce the bake-off:
  - `core.decode`: 36 shots, the same boundaries; 17 s.
  - `audio.asr`: Parakeet MLX; 81 of 81 words identical; 5 s.
  - `vision.face-track`: py-feat on MPS; 20 tracks with identical boxes and the same 8 people as the current bake-off code; 62 s.
- **Export.** The exported bundle passes `validate.py`. The bake-off's NES bundle has 14 people because it came from the Oct 21 clustering, before the two-pass anchor clustering.

Not done:
- **OCI backend.** Written but not yet run against a real image. It needs a CUDA box or a CPU test image.
- **CUDA runners** for ASR and faces.
- **`perception models pull`.**
- **Cloud backend.** SkyPilot across providers, chosen on measured cost (no provider preferred); the first NVIDIA timing needs a GPU provider account.
- **Windows.** Untested.
