# Perception graph: Phase 0–1 implementation plan

Implements `PERCEPTION_GRAPH_SPEC.md` (the spec of record) up to its Phase 1 local MVP. This file records the
choices the spec leaves open and where the spec meets what this repo already has (`PLAN.md`, the bundle schema,
the bake-off stages). Search the code for `TODO: [PERCEPTION-GRAPH]` for unfinished steps.

## Decisions (defaults; each can be revisited)

| Question | Choice for the MVP | Why |
|---|---|---|
| Canonical artifact types | **The bundle's layer schemas are the artifact types** (`bundle.faces@0.1` validates against `schema/bundle-0.1.schema.json#/$defs/layer/faces`), plus a few internal types (`media.v1`, `audio.wav.v1`, and `assets.v1`: keyframes and crops, merged into the bundle's `assets/`). A bundle is a published *view* over committed artifacts (manifest + layer files), not a second format. | Acceptance criterion 9: the annotator and HyperStudy already read bundles. One schema, one validator. |
| Artifact encoding | JSON for v0.1 layers; Parquet/Arrow for heavy time series when the bundle moves to v0.2 (already planned there). Revised 2026-10-27: side files may be Parquet in v0.1 (the gaze heatmaps first), because every step now keeps its model's full output beside its layer. Layers stay JSON; moving an existing JSON side file to Parquet is still a v0.2 move. | Changing an existing file's encoding is a schema-version change on both sides; a new Parquet side file is additive (readers that do not know it ignore it), and the orchestrator checks Parquet at the format level, so it needs no pyarrow. |
| Time | Seconds (float, ms precision) in layers, as the bundle does today. `media.v1` records the container timebase, `r_frame_rate` vs `avg_frame_rate` and whether the stream is variable-rate, so the integer-µs mapping can be added without re-decoding. | Integer µs everywhere is a bundle major version; flagged as an open decision rather than smuggled in. |
| Scheduler | Custom asyncio coordinator + SQLite, as the spec recommends. The Prefect spike is dropped (2026-10-23): a survey of 17 frameworks found none that meets per-node environments, the Apple GPU, a private cross-run cache, cost caps and Windows together. | Every node runs in its own environment (py-feat, Parakeet-MLX, SAM 3 cannot share a lock), so a framework would mostly launch subprocesses; Prefect's task runners are per flow, not per task. |
| Native runner | `uv run --project envs/<env> python -m <module>`: one uv project per environment group (`PLAN.md` §4). The runner hashes the env's `uv.lock` into the cache key. | uv environments are cheap and locked; this is what the bake-off already does. |
| No containers | uv is the only runner, everywhere: this machine, cloud machines, lab servers (2026-10-24; the spec's OCI runner was built, never run against a real image, and removed). The annotator app bundles uv. | Every node installs with uv (pip and Hugging Face, plus pinned research code for some bake-off stages); the L4 cloud runs used plain uv environments and matched the Mac. Docker on macOS cannot reach the Apple GPU, images would be per platform anyway, and weights that are non-commercial or gated cannot go into a published image. Revisit (a CUDA image built from the same uv locks, weights fetched at run time) only if a node cannot be installed with uv, or cloud start-up time matters. Revisited 2026-10-25 for RunPod: an image of our own with the faces environment in it (6.8 GB) took 145 s to start on hosts that had not seen it, against about 30 s for uv to fetch the environment in the datacenter. So RunPod pods start from RunPod's cached PyTorch image with tools added (cloud/runpod/image: ffmpeg, rsync, uv, runpodctl; no environments, code or weights), and uv stays the only runner. |
| Node protocol | Files, not sockets: the coordinator writes `request.json` into a fresh attempt directory; the node writes outputs under `out/` and `response.json`. The coordinator validates, hashes and commits. A node never writes into the store. | Atomic commit and safe crash recovery (criterion 3) fall out of it: an orphaned attempt can only write into its own directory. The same contract works over SSH/HTTPS later (stage inputs, run, collect). |
| Node SDK | `sdk/perception_node`: stdlib only, added as a path dependency of every env. Reads the request, resolves input paths, writes the response, maps exceptions to error codes (`transient`, `input`, `schema`, `privacy`, `licence`, `internal`). | Nodes stay plain scripts; no orchestrator code in model environments. |
| GPU budget on a Mac | One `gpu: exclusive` node at a time; `gpu: shared` nodes up to the free unified memory. | Measured: SAM 3 overlapping Sapiens2 pushed a 128 GB Mac into 45 GB of swap. |
| Weights | Nodes declare their weights (`hf://repo@revision`, size, gated, licence). `perception doctor`/`plan` report what is missing; `perception models pull` fetches. A run never downloads silently: a missing weight fails the plan. | Spec: never silently download large weights. |
| Secrets | Names only in YAML and requests (`secrets: [HF_TOKEN]`); values from, in order, the environment, the system keychain (`perception keys set`; names only in `keys.json`) and `~/.config/hyperstudy/*.env` files only the user can read (others are refused, not read). Passed only in the environment of a node that declares them; never written to run files or logs. The only places a value is written: the keychain, or `keys.env` (0600, in a 0700 folder) where there is no keychain (2026-10-25, keys.py). | Spec security section; the keys design doc. |
| Privacy | `privacy.external_upload` per node and per data kind (video, audio, face crops, transcripts, embeddings). A graph's `policy.privacy: local-only` makes any node that needs upload a plan error. A whole graph run on a cloud machine (`cloud/skypilot/`) is outside this check: the clip is sent only when named at launch. | Copyrighted stimuli: Anthropic and OpenAI were approved for the bake-off clips, and OOO clip 2 for our own temporary AWS machines (2026-10-23). |
| CLI name | `perception` (the spec's sketch); package `hyperstudy-perception`. Replaces the `ptrack` placeholder. | Final name is still an open decision. |

## Phase 0 — contracts

1. Graph IR: YAML → validated model (pydantic): inputs, outputs, policy, nodes, `$inputs.x` / `$nodes.n.out`
   references, topological order, cycle and type checks against node manifests.
2. Node manifests: `nodes/<namespace>/<name>/manifest.yaml` (id, version, inputs, outputs with artifact types,
   runners, resources, execution, privacy, licence, weights, secrets). A registry loads them all.
3. Artifact types: a registry mapping type ids to validators (JSON Schema refs into the bundle schema, or a
   file-shape check for binaries).
4. `perception graph validate`, `perception plan --explain`.
5. ~~Prefect spike~~: dropped after the framework survey.

## Phase 1 — local MVP

1. Content-addressed store: `store/objects/sha256/ab/cd…` (files) and tree objects for directories; artifact
   records (type, hash, size, producer task) in SQLite. Atomic publish: write to a temp path, fsync, rename.
2. Run state: runs, tasks (pending, ready, leased, running, succeeded, cached, failed, canceled), attempts,
   leases with expiry and heartbeat, events log. Resume reclaims expired leases.
3. Cache: key = hash(manifest id@version, runner identity (env lock hash), canonical config,
   input artifact hashes, artifact schema versions, partition); a hit whose objects all exist marks the task
   `cached`.
4. Coordinator: asyncio loop; ready = all inputs committed; placement by runner eligibility and resources;
   retries with backoff for `transient` only; `max_cost_usd` reservation (zero for local nodes).
5. Backends: native subprocess (uv), behind `probe_capabilities / estimate / stage_inputs / submit /
   poll / cancel / collect / cleanup`.
6. `perception doctor` (OS, CPU, RAM, Apple GPU / CUDA / VRAM, uv, ffmpeg), `run`, `status`, `logs`,
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
- ~~**OCI backend.**~~ Removed 2026-10-24: uv is the only runner (see the decisions table).
- **CUDA runner** for ASR (faces has one: native on Linux, first run 2026-10-24).
- **`perception models pull`.**
- ~~**Cloud backend in the coordinator.**~~ Done 2026-10-24: `perception cloud on` offers nodes with a CUDA runner the
  place `cloud:cuda`, a g6.2xlarge (L4, 8 vCPUs) started for the node through SkyPilot and deleted after it. It is the
  SSH backend for one attempt: the driver starts the machine (sky.yaml: no laptop credentials, only perception-node),
  makes an OS shutdown terminate it, probes it, runs guard.sh before anything is sent, and deletes it whatever happens.
  The price for the whole deadline (20 minutes plus the video's length) is reserved; the time the machine existed is
  charged (at most the deadline once the machine's own shutdown is set; all of it before). If SkyPilot cannot
  delete the machine, the driver terminates its instance through EC2; a machine with no shutdown set that neither could
  delete fails the attempt without a retry, naming it. Whole graphs still run through `cloud/skypilot/launch.sh`.
- ~~**RunPod.**~~ Done 2026-10-25: `perception cloud on runpod` (an account, credit and `RUNPOD_API_KEY`) offers the place
  `runpod:cuda`: a Secure Cloud pod in the US, Canada or Europe, the cheapest free GPU of sixteen with at least 20 GB, Ampere, Ada
  or Blackwell ($0.25–1.19/h; widened 2026-10-25 after US capacity for the first seven ran out for minutes at a time),
  through RunPod's REST API directly (SkyPilot would copy `~/.runpod/config.toml` to every machine, AWS ones too). The
  key stays in the driver's process; the pod gets a key made for the attempt, its own disk only (no volume: the inputs
  go with it), and a start command that sets its removal at the deadline before anything else runs (`runpodctl` with
  its pod-scoped key, which can remove that pod and no other, nor list the account's). cloud/runpod/guard.sh checks that
  removal is running, the pod's own key is refused by the API and no account config is there. Deleted at the end and
  checked gone (404); a create with no clear answer (a 5xx, a timeout, a cancel) is looked for by its name; a launch
  also deletes perception pods past the deadline in their names. Reserved at the dearest GPU's price for the deadline, charged at the pod's own price. First
  real run, the face node on a 120 s synthetic video: an RTX A5000 at $0.27/h, 2 min 6 s in all (launch 22 s, setup 22,
  prepare 60, run 13), about $0.01, its output byte-identical to the AWS L4's. Uploads from this laptop: 4 MB/s on one
  stream, 6.6 MB/s on four.
- ~~**Nodes on a lab server.**~~ Done 2026-10-24: `perception hosts add NAME USER@HOST` records a machine reached over SSH
  (probed once: platform, cores, memory, GPU, uv, ffmpeg), and its places (`ssh-NAME:cuda`) join each node's options.
  A local driver process per attempt (`remote_driver.py`) sends the node's git-tracked code and its inputs (by content:
  once per host), starts it detached in its own session, streams its log, and brings out/ and response.json back, so
  the coordinator's cancel, adoption and publication are unchanged. Sending inputs is an upload: the graph's
  allow_upload and the run's --allow-uploads apply. Checked with a stand-in ssh (tests/test_remote.py) and the real
  decode node through it: outputs byte-identical to a local run.
- **Windows.** Untested.

### Phase 2 progress (2026-10-23)

- **First API node, `vlm.shot-captions`.** Claude Haiku 5.5 with the face roster, as chosen in the bake-off. It refines decode's shots layer, and export takes the downstream version. It also writes the on-screen text layer.
  - On NES: 36 calls, $0.0215, 161 s.
  - The calls run in order, because each shot's prompt carries the roster and the previous caption. A Message Batch (50% off) would mean dropping that chaining, which is a quality decision still open.
- **Cost estimates.** A manifest declares `cost: {usd, per: shot|minute|run, input}`. The estimate is reserved against `max_cost_usd` before the node starts, and the run is charged what the node reports (`usage.usd`).
- **Upload permissions per data kind.** `policy.allow_upload` names each kind that may leave the machine: video, audio, frames, faces, transcript or embeddings. A node that uploads anything else is a graph error.
- **Deterministic stored layers.** The SDK no longer writes the run time or timings into layer content; export adds them to the bundle's copy. Rerunning decode on NES now leaves ASR, faces and captions as cache hits.
- **`rerun --no-cache` forces only the named node.** Downstream nodes rerun only if its outputs changed.
- **The node SDK is part of each native node's cache identity.**
- **Tests:** 33 (`orchestrator/tests`).

### First cloud runs (2026-10-24)

SkyPilot on AWS us-east-1 (`cloud/skypilot/`), always through `launch.sh`:
- **Credentials.** No laptop credentials are copied to the machine, and its only role is `perception-node`, which can end `perception-*` machines and do nothing else (`sky.yaml`, `iam-perception-node.json`). `guard.sh` checks both on the machine and fails the job otherwise.
  - The first runs below used `remote_identity: NO_UPLOAD`. On AWS that only skips copying credential files: SkyPilot still attached its default `skypilot-v1` role, which has EC2 and S3 full access. The guard caught it on 2026-10-24. CloudTrail shows the role was used only for SkyPilot's own autodown (describe and terminate calls).
- **Uploads.** The workdir is a staged copy of the files git tracks, nothing untracked (`.gitignore` does not cover every image, crop or bundle folder a working copy can hold). A clip is sent only when named with `--env CLIP_SRC=...`.
- **Wall-clock limit.** `MAX_MINUTES` (default 60) deletes the machine whatever the job is doing.
- **Deletion.** Every machine is deleted after its job: `sky down` when `launch.sh` exits, idle autodown, and a shutdown deadline that terminates the instance (`launch.sh` sets on-demand machines to terminate, not stop, on shutdown).

- **CPU smoke test** (`smoke.yaml`, spot `m6i.large`): decode of a synthetic 2-minute video, 11.8 s. The machine deleted itself 40 s after the job.
- **First NVIDIA run** (`gpu-faces.yaml`, `graphs/faces.yaml`, OOO clip 2): on-demand `g6.xlarge` (L4, 4 vCPUs, driver 580, torch 2.14.1+cu130). There was no L4 or A10G spot capacity in any us-east-1 zone, and the account's G spot quota is 0 in other regions. About $0.12.
  - The face node ran on its new CUDA runner. Against the Mac's run: 185 of 186 tracks match (median box IoU 0.994), the same 52 shots, 97.9% agreement on which tracks are the same person, and 18 people in both. The Mac left 31 tracks unassigned and the L4 21: tiny numeric differences move borderline tracks across the clustering threshold.
  - Timing: decode 43 s (Mac 15 s); faces 141 s (detection 106 s), against 95 s on the Mac (detection 76 s). The batch size was the default 32, and GPU utilisation was not recorded. Next: record utilisation, sweep the batch size, and try a machine with more vCPUs, to find what limits it.
- **What limits the face node on an L4** (`gpu-tune.yaml`, g6.2xlarge, 8 vCPUs, OOO clip 2, 2026-10-24, about $0.40).
  - **Frame decoding in the detector's process was the bottleneck.** With `num_workers: 4`, detection took 57 s
    instead of 99 s, and the whole face node 87 s instead of 128 s, faster than the Mac (95 s).
  - **The thread count was a smaller factor.** Ray sets `OMP_NUM_THREADS=1` for SkyPilot jobs. At 8 threads, detection
    took 99 s instead of 110 s.
  - **Batch size barely matters:** 8, 16 and 32 are within 3 s of each other, and 64 runs out of GPU memory, because
    py-feat feeds whole 1080p frames.
  - **Output.** Workers do not change it: the boxes are byte-identical. Batch size changes boxes very slightly (GPU
    kernels depend on batch shape), with the same 189 tracks and 18 people. Batch size is in the cache key.
  - **Still idle.** The GPU is idle about 40% of the run: model loading, and thumbnails and crops cut by ffmpeg on the
    CPU. `num_workers` now defaults to 4 where workers fork (#4). Next: GPU frame decoding (torchcodec on CUDA).
- **Thumbnails.** Each person's crop is chosen among shortlisted faces on exposure, colour cast, blur and blown-out pixels (`thumbnails.py`), instead of size and frontality alone (which picked a red-lit face on clip 2).

### Phase 3: the bake-off's other models as nodes (from 2026-10-27)

Only four of the bake-off's capabilities had become nodes: decode, asr, face-track and shot-captions. The rest are
ported one node at a time, each with the model the bake-off chose (PLAN.md §4) and writing the layer the annotator
already draws (`pipeline/bundle/convert_bakeoff.py`'s `*_layer` functions). The order follows what each one reads:

1. Wave 1: bodies, diarization, sounds, waveform, the face detail output (in face-track, not a new node), and gaze
   from faces.
2. Wave 2: speakers (LR-ASD, recomputed on Parakeet's diarized transcript), hands, facing and objects.
3. Wave 3: gestures, prosody, gaze with objects, and asr on CUDA (NeMo, a native runner).

- **`vision.body-pose` (bodies).** RTMW-l 384x288, plus YOLOX-m, the person detector RTMW needs because it is
  top-down. Both run through rtmlib on ONNX Runtime, in the new `envs/vision`: CoreML (MLProgram) on a Mac, CUDA via
  onnxruntime-gpu on Linux, or CPU.
  - On OOO it matches the bake-off's `pose_rtmw.json` exactly: 244 tracks, 185 linked to faces, all 3136 samples
    matched, keypoint difference 0. It took 114 s for the 119 s clip on CoreML (the bake-off took 103 s), plus 16 s
    to load.
  - The CUDA runner is not yet run on a GPU.
- **Weights from a URL.** A model's `source` may be an https URL with its `sha256`. The node fetches the file into
  one shared cache keyed by that sha256 (`perception_node.weights`), and checks it before use. `plan` lists such a
  file as a download until it is in the cache.
- **Follow-up: weights we host or bake into the image (to revisit).** Weights are keyed by sha256, so a copy we
  host, or one already in the cloud image's `PERCEPTION_WEIGHTS`, is the same file to every node. `fetch()` could try
  our mirror before the source. A cloud run could then start with no download, so a pay-per-use GPU isn't billed
  while weights arrive. The mirrorable models are already listed in PLAN.md §4.
