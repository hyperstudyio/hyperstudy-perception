# Run anywhere: one pipeline on any machine, or split across machines

Drafted 2026-10-28 from three research passes (the graph's duplicated work, the orchestrator's placement and data
paths, and RunPod / Hugging Face / Tailscale / Ray-Dask-Prefect-SkyPilot docs). What the user asked for:

1. The full pipeline runs **locally, on a lab server, or on RunPod**.
2. The graph is **efficient**: no step does work another step already did.
3. One pipeline's steps can be **spread over the Mac, lab servers, cloud GPUs and APIs**, with each step's inputs
   sent where it runs and everything collected on the Mac.

Claims marked **unverified** come from code reading or docs, not a run. Every verification below runs on
the lab server or RunPod, never the laptop (it kernel-panicked under the objects experiments on 2026-10-28).

## Where things stand (2026-10-28, before phases 1–3; line numbers as of then)

| | Today | Where |
|---|---|---|
| Steps that can run off the Mac | 12 of 16. Never remote: captions, prosody_captions (API keys, uploads), objects (Anthropic key, HF token, uploads), diarize (HF token) | `backends/remote.py:98` `remote_refusal` |
| Where a run is coordinated | Only on the machine that starts it (the app's Mac). A remote step whose driver loses touch for 600 s is ended there (`lost_after_s`), so a closed lid ends long remote steps | `backends/remote.py:77`, `remote_driver.py:10,50` |
| Data between machines | The Mac is the hub. Inputs go to a host once each, by sha256 (`objects/<sha256>`); outputs come back and the host's copy is deleted, so the next step on that host is sent them again | `remote_driver.py:230` `send_inputs`, `:393` `fetch` |
| RunPod | One pod per step: environment, weights and inputs again for each. 40 GB disk, no volume | `remote_driver.py:753`, `backends/cloud.py:159` |
| Gated weights off the Mac | Never: the steps that need them are refused before weights matter | as above |
| Cache identity | Was `ssh-<host>:hash(…)` remotely and platform-free natively; since #40 (2026-10-28) one rule for every backend, `Backend.identity` (`backends/base.py`) | |
| The lab server | two NVIDIA cards; the probe records only the largest card, so one exclusive GPU step at a time. `code/` is 20 GB for 8 code versions (a `.venv` per node code hash). Disk 90% full | `hosts.py:57`, probe 2026-10-28 |

## Decisions

| Question | Proposed | Why |
|---|---|---|
| Where the coordinator runs | **Either the Mac or the machine doing the work.** "Run on the lab server" / "Run on RunPod" starts the coordinator there, detached; the Mac starts it, watches, and fetches the bundle. Mixed runs keep the coordinator on the Mac. | A detached coordinator survives the laptop sleeping. The coordinator already runs on Linux (`cloud/skypilot/gpu-faces.yaml` runs `perception run` on AWS). |
| API keys (Anthropic, OpenAI) | **Decided 2026-10-28: the machine that runs the whole pipeline gets the keys its steps need**, so runs are fully unattended. They go over SSH stdin into the remote coordinator's memory (never a file, argv or log there) and from it, as now (`child_env`), only to the steps that declare them. One switch per host in Settings ("Run whole pipelines here, with my keys"); the plan says which keys go where. In a run coordinated on the Mac, API steps stay on the Mac (a pod would idle at $0.25–1.19/h waiting on serial calls). | Unattended runs need the keys where the run is. The rejected alternatives: the Mac as a worker for API steps (the run stalls until the Mac wakes) and a reverse-tunnel proxy (needs the Mac awake). |
| What a key on a host exposes | On a lab server, root and processes of the same user can read a running process's environment (`/proc/<pid>/environ`) for the run's length; nothing is left after it. On a pod, that pod and the machine's operator, who has root on it: on RunPod's Community Cloud an unvetted third party, and a pod that coordinates holds every key the run needs. So keys go only to Secure Cloud pods by default; a Community pod needs its own yes, and the switch says who can read them. Settings recommends scoped keys: an Anthropic workspace with a monthly spend limit and a short-expiry key, an OpenAI project with a hard limit. | Neither vendor can issue keys per run through an API, so scoping is the limit on what a leaked key can cost. |
| Hugging Face token | **Only for downloads.** Gated steps run offline from the cache (`HF_HUB_OFFLINE=1`); the token goes only to `prepare`. A lab server downloads with the running user's own token, sent with the other keys, into that user's private cache on the host; a host's own login is used only when it is the running user's own account, since gated access is the user's, not the lab's. | Access is granted to individual users, never to organisations (HF docs), so each user needs their own approvals and token (the Settings screen). |
| Gated weights | **Decided 2026-10-28: downloaded in each pod session** (in the datacenter, with `HF_XET_HIGH_PERFORMANCE=1`); **kept on lab servers**, in a cache per Hugging Face account (`HF_HOME` under the host's root, named by the account `whoami` returns), so they download there once per person and weights fetched under one person's approval never serve another's run, even when people share the host's Unix account. Never in the public image; no RunPod volume. | Meta's licences (SAM 3, DINOv3). |
| Non-gated weights in the image | Measure first. With one download per pod session (below) it matters less; a 6.8 GB image took 145 s to start against ~30 s for the cached base (2026-10-25). | |
| RunPod granularity | **One pod session for a run, or for a chain of steps**, not one pod per step. Environments and weights once; intermediate outputs stay on the pod; a session drains (every output fetched or pushed elsewhere) before it is deleted. | |
| Cache identity | **One rule for every backend:** hash(code, environment lock, SDK, accelerator, platform), with no host name; where a step ran goes to `usage`. The one-time rerun of cached results is accepted (2026-10-28). | Lets a result made on the lab be reused by a pod and a remote coordinator's store be imported. |
| Data between machines | **Outputs stay where they are made; bytes move only to a consumer.** Per edge: same host → keep (publish into the host's `objects/<sha256>`); two remote hosts → direct, started by the side that can reach the other (the lab can reach a pod's public SSH; a pod cannot reach the lab); otherwise relay through the Mac. In a run coordinated on the Mac every output still comes back to the Mac before it is committed, and no step starts from an uncommitted input; what this saves is the *upload* to the next machine, which is kept on the host or sent host to host instead of from the Mac's (usually slow) uplink. In a run coordinated elsewhere, that coordinator's store holds the run and the Mac imports it at fetch. A step is `succeeded` only once its outputs are checked (`_check_outputs`) and published in the coordinator's store, and a later step starts only from committed inputs (the spec's rule): on the same machine it then uses the kept copy, so nothing is sent again, but it waits for the commit. A session lost before its outputs reach the store leaves those steps not yet succeeded, so they run again; no committed task ever names an object the store lacks. | Ray and Dask keep outputs where they are made too; unlike them, our machines cannot all reach each other, so relaying through the Mac remains as the fallback. Content addressing (sha256, checked on arrival) means the route never changes a result. |
| Privacy across routes | Consent stays per (step, data kind, destination), as `sends()` / `uploads()` do now; the route does not change who receives the bytes. **"Run on X" is consent for X too:** with the coordinator on X every step is local to X, so `sends()` would see nothing; `remote run` therefore checks the graph's `policy.allow_upload` (and `--allow-uploads`) for the input `video` and every kind the run will make, with X as the destination, before the clip leaves the Mac, and a `local-only` graph cannot be run on X. Outputs kept on a host stay in its 0700 root, like inputs, but not indefinitely: an output is kept only while a later step of the run on that host reads it, and anything kept is removed, as a backstop, after a retention period (default 7 days) or when the host's disk is short, oldest first, never while a step holds it: `send_inputs` takes the hold before it looks for `.complete` (a marker `holds/<sha256>/<run>-<node>-<attempt>` beside `objects/`, never inside an object's folder, which `publish` renames into place and then makes read-only; written in the same command), so no cleanup can fall between the check and `in/`. The attempt's folder is created before any input is sent (today `start()` makes it afterwards), so a hold always names an existing folder; it is released when that folder is removed, and a hold whose folder is gone is stale and ignored. Cleanup skips held objects. | GRAPH_MVP_PLAN.md accepted the cloud/skypilot exception only because the clip is named at launch; a one-click "Run on" needs the policy check. The lab server is shared and 90% full. |
| Pods on the tailnet | Deferred. Userspace Tailscale with an ephemeral tagged key and a grant limited to port 22 would let a pod reach the lab, but it adds a credential and is unverified on RunPod. Lab-started transfers cover lab ↔ pod. | |
| Objects vs captions | **Decided by measurement** (the user prefers one call if the information is redundant, and the best results): objects is split, and its GPU step reads a `vocabulary` input that either a names-only step or the captions call (one response, two JSON parts) can produce. Both are run on held-out clips on the lab and the better one becomes the graph's default (below). | Where the names come from is then one edge of the graph, so both can be built and compared without duplicating the GPU work. |

## The graph: removing duplicated and idle work

Ranked by measured effect on OOO (2-minute clip):

1. **Split `vision.objects` into three steps.** Objects is `gpu: exclusive` (`manifest.yaml:34`), so no other GPU step
   runs beside it (`scheduler.py:441`), yet it spends ~295 s of its 1431 s waiting on Haiku: vocabulary 124 s (PR #19), crop
   check 171 s (PR #36). Its naming also waits for bodies, though it needs only keyframes.
   - **The names** (CPU, API): the vocabulary and name classes, written as a `vocabulary` artifact (per shot: up to 8
     nouns; per name: family, kind, worn). From one of the two sources compared in 1b.
   - `objects.track` (GPU, no API key, no uploads): reads `vocabulary`; SAM 3, the garment pass, DINOv3; writes
     tracks, embeddings and the crops for the check. It can run on any GPU machine once the HF token is download-only.
   - `objects.link` (CPU, API): the crop check, then wearers and identities. The crop check cannot join any other
     call: it needs SAM 3's tracks.
   What each declares (their union must equal 1.1.0's `external_upload: [frames, faces]` and secrets, checked by a test):
   names `external_upload: [frames, faces]` (raw keyframes show faces, as captions declares) and `ANTHROPIC_API_KEY`;
   track `[]`, with `HF_TOKEN` for its `prepare` only; link `[frames, faces]` (a worn item's crop centres on its
   wearer's face) and `ANTHROPIC_API_KEY`. Cost: names and link each declare their own (`per: minute` of media, as
   1.1.0 does), track none; the same test checks that the two add up to at least 1.1.0's. Under B the captions
   step's estimate is measured again, since each call returns more.
   The objects layer and the app are unchanged: the names and track steps declare `part_of: vision.objects`, and the
   app shows the three as one feature, Objects, in Settings and Extract (docs/cli-json.md). Parity: output diff against
   the unsplit 1.1.0 on OOO, on the lab.

   **1b. Where the names come from: measured, then chosen.**

   | | A: `objects.names`, names only | B: the captions call names the objects too |
   |---|---|---|
   | Calls | one per shot, Haiku, raw keyframes | none of its own: captions' one call per shot returns `{caption, objects}`, and the node writes the shots layer and `vocabulary` |
   | Starts | after decode (keyframes only) | after faces and diarize (captions draws face boxes and reads the speech) |
   | Effect on objects | names ready before bodies finishes, so `objects.track` waits for bodies only | captions runs shot by shot (150–282 s measured for 36–52 shots), so `objects.track` may wait ~0.5–3 min longer (estimate from those timings) |
   | Cost | ~44 extra calls per 2-minute clip (~$0.006, estimate) | saved |
   | Cache | names rerun only when keyframes or the names prompt change | names rerun whenever captions does: a transcript change, a captions prompt change, or switching captions to Opus (~40× Haiku's price, from the repo's prices) reruns SAM 3 (1,000–3,200 s) |
   | Could be better because | prompt tuned for SAM 3 nouns (PR #36); clean frames | sees the speech and the roster, so names what people talk about and handle; the caption and the object names agree |
   | Could be worse because | no context beyond the frames | boxed frames and two jobs in one prompt (descriptive captions vs generic nouns) |

   Compare on 3–4 clips not used in tuning (on the lab): objects' PR #36 measures (people with upper/lower garments,
   wearer right/wrong, splits, merges, crop-check rejections), the captions against today's (the user reads both, blind,
   in the annotator), the run's critical path and cost. If B's objects and captions are at least as good, B becomes the
   default and A stays available; otherwise A.
2. **Run the crop check's calls in parallel** (prosody_captions already runs 8 workers). 97–253 s today, measured;
   gaze and gestures wait on it.
3. **Drop prosody_captions' second call per utterance** (`tags_from_text`, gpt-5.4-mini): 10.5 s of its 28.6 s on
   OOO; the app reads it only when `tags` is missing (`rows.ts:63`), and `tags` is required. A config flag; by default the
   second call is not made; the side file's `tagger_content` column becomes nullable, a schema change with its
   own version bump.
4. **Smaller:** rotation into `media.v1` so gestures no longer takes the video; decode could write the 48 kHz audio
   sounds resamples (check that the samples match).

Not worth doing (measured): a shared frame cache. Nine steps decode the video themselves, but each pass is 1–4 s of a
2-minute clip (<1% of a run). A JPEG cache would be 4–6 GB an hour and would break the bit-exact bake-off matches; the
compressed video is already the smallest thing to send a host. Little genuine model duplication: one person detector
(bodies), one face detector (faces), SAM 3's detector reused for garments.

## The three ways to run

```
 HERE or MIXED (coordinator on the Mac)          ON X (X = a lab server or one RunPod pod)
 Mac: coordinator, API steps (keys)              Mac --ssh: start (+ keys on stdin) / watch / fetch-->  X: coordinator
  |--ssh--> lab: step -> objects/<sha>                                                  |  (detached; keys in memory)
  |          ^ lab -> lab: kept on the host                                             |- every step, API ones too
  |          `-- lab-started rsync --> pod                                              |- environments + weights once
  |--ssh--> pod session: A -> B -> C, outputs stay on the pod                           |  (lab: kept for later runs)
  `<== every output is fetched to the Mac and committed (before its consumers start)    `- bundle + cache -> Mac's store
```

## Phases

Each phase is PR-sized; verification runs on the lab server or RunPod.

| # | Change | Repo | Unlocks | Depends on | Verify |
|---|---|---|---|---|---|
| 0 | ~~Merge PR #36 (objects 1.1.0)~~ merged 2026-10-28; still to do: judge it on 3–4 clips it was not tuned on | pipeline | the objects split builds on it | — | held-out clips on the lab (after 3, or a one-off run there) |
| 1 | Done 2026-10-28 (#39); 1b (retention) done 2026-10-29: holds under `objects.lock`, the cleanup on each attempt (at most every 10 min), hourly by a sweeper on the host the attempts start (until nothing is kept; none after a restart until the next attempt) and by `hosts clean NAME --expired`, `--keep-days` and `--min-free-gb` per host (default 20 GB, or a tenth of the disk if less). Not done: removing, as the run ends, the outputs no later step there reads; until then the 7 days are the limit. **Keep outputs on the host**: `fetch` files `out/` in the host's `objects/<sha256>` before deleting the attempt, as a copy, under the hash the Mac computes from what it fetched with the store's own functions, only after the host has checked every file's sha256 and the file count against it, and never with a link or special file in it (PR #39). An object filed for an output the coordinator then rejects is unreachable: later steps are sent only committed hashes; then (1b) the retention rule above: kept only for later steps there, removed after 7 days or when disk is short | pipeline | same-host chains send nothing twice | — | decode + faces on the lab: no "copying" of decode's outputs for faces; bundle equals a Mac-placed run |
| 2 | Done 2026-10-28 (#40). **One cache identity** for all backends (platform in native, no host name) | pipeline | lab results reused on pods; stores importable | — | run on the lab, plan with RunPod: shown cached |
| 3 | **Environments by lock hash (and the SDK's content: the SDK is an editable path dependency outside the lock, so a shared environment must never import an older SDK than its cache key records) (done in #41: the SDK is left out of the shared environment and imported from the code's own folder); every GPU in the probe; `CUDA_VISIBLE_DEVICES` slots, each taken through a host-side lease (a per-GPU `flock`, `~/.cache/hyperstudy-perception-gpus/<machine>-<index>`: a private folder of its own beside the host's root, not in it, so coordinators with different roots, or one on the server itself, meet there, and a `~/.cache/hyperstudy-perception` holding weights without a root's marker does not refuse it; done in #42) taken by the wrapper that starts the detached step and held for the step's whole life, so it outlives a lost SSH session (a coordinator only asks for a slot and reads who holds it), so a lab-coordinated run and a Mac-coordinated one cannot both put an exclusive step on one card** | pipeline | lab disk stops growing per edit; two GPU steps at once on the lab | — | `du` of `code/` flat across node edits; faces and asr together on the two cards |
| 4 | Done 2026-10-29 (#43); the parity run on OOO on the lab still to do (after 7). **Split objects** (names A, track, link); parallel crop check: graph ids `objects_names`, `objects_track` and `objects` (`vision.objects-names` 1.0.0, `vision.objects-track` 1.0.0, `vision.objects` 2.0.0), passing `vocabulary.v1` and `object_tracks.v1`; byte-identical to 1.1.0 with stand-in models and answers (`nodes/vision/objects/test_split.py`) | pipeline | objects' GPU work placeable anywhere; ~2–5 min off a clip | 0 | output diff against 1.1.0 on OOO, on the lab |
| 4b | **Names from the captions call** (B: one response, two JSON parts), then the A/B comparison and the default chosen | pipeline | one call per shot, if as good | 4, 7 | the comparison above, on the lab |
| 5 | Done 2026-10-29 (manifests' `download_secrets`): diarize and objects_track get no token and run with `HF_HUB_OFFLINE=1` and no stored token (`HF_HUB_DISABLE_IMPLICIT_TOKEN`); here, a step whose weights are missing (or whose prepare never finished: `perception-prepared/` marks) has its environment synced without the token, its prepare run with it, and the step started by `execve` without it, its GPU lease taken only then (`prepare_first.py`); on a host with `hosts keys HOST on`, the token goes over ssh's stdin to the prepare alone, into that account's cache (`<root>/hf/<account>`, `HF_HUB_CACHE` too), and a host's own login is never used; a remote coordinator is sent it with the run's keys and the account's name, and gives that cache to the gated steps alone; pods still refuse gated steps (10). Checked on liquidswords2 (OOO clip 1, diarize per-step and under `remote run`): a monitor of every process there saw the token only in the prepare's processes, and the same pid without it after `execve`; in no command line (116 processes) and no file under the root; pyannote loads offline. Still to do: a pod with the token (10); the host's own login when it is the user's account. **HF token download-only**: gated steps run offline from the cache; refusal narrowed to API keys; host's own login, or the user's token to `prepare` | pipeline | diarize and objects.track on the lab and on pods | 4 | diarize on the lab with the running user's own token, sent to `prepare`; on a pod with the token; token absent from the step's `/proc/<pid>/environ` |
| 6 | Done 2026-10-28 (#38, annotator #50). **Hugging Face access in Settings** (steps, links, per-model status: designed 2026-10-28) + `keys check HF_TOKEN` reporting each gated model's access | both | users can set up gated models themselves | — | Chrome; a token without access shows "Not yet" per model |
| 7 | Done 2026-10-29 (`perception remote run / status / attach / fetch`, `hosts keys HOST on\|off`; GRAPH_MVP_PLAN.md, "Whole runs on a lab server"): on liquidswords2, decode, asr, diarize, faces and captions ran with this machine's side killed 10 s into captions, and the host finished the run alone 4 min 16 s later; the fetched decode, asr and faces artifacts are byte-identical to a run coordinated here with them placed on the lab (14 of 14), their cache keys equal, and a re-plan here finds them cached; keys in no file, argv or log there, and only in the environments of the steps that declare them. A run's inputs are held on the host (1b's holds) until it ends. Still to do: the 30-minute lid-shut full-graph run, and the A/B of 4b. **Coordinator on a lab server, with the keys**: `perception remote run / status / attach / fetch` (the run's `--max-cost-usd`, `--set`, `--only`, `--place` and `--allow-uploads` go with it to the lab coordinator; a remote coordinator refuses `runpod:*` and `cloud:*` places, so the account-wide RunPod key and AWS credentials never leave the Mac, as GRAPH_MVP_PLAN.md decided, and a remote run's spend is only what `max_cost_usd` bounds), store import, which needs the remote coordinator to run from a listed copy of the pipeline (the tracked-file listing a release ships as `RELEASE.json`, sent with the code) so its cache keys equal the Mac's and imported results are hits (the Mac picks the run id, and `remote run` attaches instead of starting a second coordinator for an id already running there; imported attempts get new local numbers and never overwrite local cache entries); keys over stdin into the coordinator only; the host's pipeline pinned by content hash | pipeline | requirement 1 on the lab, unattended; runs survive a closed lid | 2 | full run on the lab, API steps included, with the laptop shut 30 min; bundle equals a Mac-coordinated run; security review: keys in no file, argv or log there |
| 8 | ~~The Mac as a worker for steps that hold keys~~: dropped 2026-10-28 (keys go with the run) | | | | |
| 9 | (merged into 7) | | | | |
| 10 | **A RunPod session as coordinator**, keys and HF token included: disk ~100 GB (estimate), gated weights downloaded in the session, deadline and spend from the sum of the steps, a linger window for the fetch, self-removal. One `max_cost_usd` for the whole run: the Mac reserves the pod (its deadline including the linger window) and passes the remote coordinator `max_cost_usd` minus that reservation as its cap for API steps | pipeline | requirement 1 on RunPod | 3, 5, 7 | short clip: one pod in the console; killing the Mac's driver → removal at the deadline; deleted after fetch or linger |
| 11 | **Pod sessions for chains in mixed runs**, each with its own removal deadline and spend reservation covering its steps plus the wait to drain (a session that cannot drain, e.g. the Mac asleep with no other place to push to, is removed at its deadline and its steps rerun); lab-started direct transfers (a one-key ssh-agent forwarded to the lab for that pod: while it is forwarded, root on the shared lab server can use that one key to reach that one pod, which holds the run's inputs and, if it coordinates, its keys; so a pod that holds keys is never reached this way, and the agent is dropped when the transfer ends); relay otherwise; outputs fetched to the Mac and committed before their consumers start, as now; placement scored by bytes to move | pipeline | requirement 3 without the Mac's uplink as a bottleneck | 1, 2, 3 | decode on the lab, faces on a pod, gaze on the lab: lab → pod direct in the log; killing the pod reruns only faces |
| A1 | Sleep assertion while a run has remote steps (does not stop a lid-close) | app | fewer lost remote steps now | — | `pmset -g assertions` during a run |
| A2 | Settings > Compute per host: "Run whole pipelines here, with my keys" (names the keys it sends), and the scoped-key advice | app | 5, 7 from the app | 5, 7 | Chrome |
| A3 | Extract: a run-level "Run on" (This Mac / a lab server / RunPod) that sets every place; remote runs listed, attached and fetched on wake | app | 7, 10 from the app | 7, 10 | Chrome, then a real run on the lab |
| A4 | Extract: a mixed run's routes and what each machine receives | app | 11 from the app | 11 | Chrome |

Phases 1–3 and 6 are independent and can go first. A3 is where requirement 1 reaches users; 11 completes requirement 3.
4b's comparison needs 7 (a full run on the lab), so the default for the names is chosen after the lab coordinator works.

## Phase 10: a RunPod pod as the run's coordinator (design, 2026-10-31)

Phase 7's whole run (`remote run`), with a pod made for the run as the host. The pod's lifecycle (create it, wait
for ssh, probe it, guard.sh, delete it) is the per-step RunPod backend's, moved out of `remote_driver.Driver` into
`pod_session.py` so both use one implementation.

- **Surface.** `perception remote run runpod GRAPH ...` (`runpod` is a reserved host name: `hosts add` refuses it),
  and `remote status | attach | fetch | cancel runpod RUN`, which reach the pod through the run's record
  (`remote/<run>.json` gains `pod`: its id, name, deadline, address, ssh port and price). The pod's ssh key and
  known-hosts file are the run's own (`remote/<run>.pod/`, 0700): pods reuse addresses, so nothing goes in the user's
  `~/.ssh`.
- **Consent.** `perception cloud on runpod` (the account key, as for per-step pods), and for the run's keys a yes of
  their own: `perception cloud keys runpod on` (off by default; the app's A2). Without it a run whose steps need keys
  is refused, naming them, as on a lab server. The keys go on ssh's stdin to the coordinator there, as in phase 7; the
  RunPod account key never leaves this machine (guard.sh checks the pod holds none).
- **Deadline.** Set at creation (in the pod's name and its start command): setup (the orchestrator's and nodes'
  environments, the gated weights: 30 min until measured) + 12 × the video's length (an hour when it is not known) +
  the linger window (`--linger-min`, 10 to 1440, 30 by default: shorter, the pod could go before this machine has seen the run end and begun its fetch, which then keeps it 30 minutes), rounded up to 5 minutes, at least an hour (pod_run.deadline_s).
  Not the plan's estimates: they come from a place's own runs, and a pod made for a run has none.
- **One spending cap.** The pod is reserved at the dearest listed GPU's price for its whole deadline. The run's
  `max_cost_usd` minus that reservation is the cap the coordinator there is given for its API steps; a run whose
  remainder is below the plan's API estimate is refused before a pod is made. The pod is charged for the time it
  existed at the price RunPod gave it, recorded with the run.
- **Linger and removal.** When the run ends there, the coordinator writes the time the pod is to go
  (`/perception-remove-at`: the end plus the linger window); the removal its start command set (removal_at_boot) acts
  at whichever is sooner, that or the deadline. `fetch` deletes the pod through the API once the run is imported,
  whether it succeeded or not; `cancel` cancels the run there, imports it, and deletes the pod through the API
  whatever the pod answered (one that cannot be reached is deleted too, ending its run). A start whose answer was lost
  leaves the pod to the run that may be going there (`attach` follows it, `cancel` ends it). A pod never fetched goes at the end of its linger window, and a stray past
  its deadline is deleted by the next RunPod launch (runpod_sweep), as now.
- **Weights.** Downloaded in the session with the user's token (decided 2026-10-28), into the pod's disk (100 GB
  until measured). Then, measured on real runs: a RunPod network volume, an R2 bucket and an image with the weights,
  against downloading each time (time to ready, cost, bandwidth). The licences allow a private copy: the SAM License
  (2025-11-19) and the DINOv3 License (2025-08-19) each grant the right to "reproduce, distribute, copy" their materials,
  weights included, with the agreement given with any copy passed on (read 2026-10-31; not legal advice). A copy in
  the user's own bucket or volume is theirs to keep; the weights never go in the public image (ghcr.io), which would
  hand gated models to anyone who pulls it, past Meta's own gate. A private image with them, under the organisation on
  GitHub, is wanted as a fourth way (decided 2026-10-31): RunPod pulls it with registry credentials kept in the RunPod
  account (a GitHub token that reads packages, never on a pod); its weights layer (~5 GB) misses RunPod's cache of the
  base image, which the comparison measures. R2 (also wanted, 2026-10-31) charges nothing to download (Cloudflare's
  pricing page, read 2026-10-31: $0.015/GB-month stored, the first 10 GB free; egress free), so ~5 GB of weights there
  costs nothing or cents a month however many pods fetch them; a pod reads them with a read-only token scoped to that
  bucket, sent as the run's keys are.
- **Verify.** Tests with RunPod's API stood in (test_runpod's FakeRunPod) and the pod a stand-in host; then a short
  clip on a real pod: one pod in the console, killing this machine's side leaves the run going and the pod removed at
  its linger window; deleted after fetch. A run whose coordinator on the pod is gone (killed, the pod restarted) never
  ends there: following it (`remote run`, `attach`) or fetching it cancels it there, imports what it did and deletes
  the pod, as `remote cancel runpod` does, instead of leaving the pod idle until its deadline (reviews of #53, #55).
  A detached run nobody follows or fetches is still bounded only by its deadline (its reservation).

### Phase 10 on real pods (2026-10-31)

The full graph on OOO clip 3 (92 s), `remote run runpod`, from a throwaway store:

- **Ready in about 2 minutes:** pod requested, made, guard.sh, setup, the pipeline's copy, the orchestrator's
  environment and the video sent, the first steps started. Pods got: RTX 4000 Ada and RTX PRO 4000 Blackwell
  (EU-RO-1), RTX 3090 (EU-CZ-1), at $0.28-0.57/h.
- **The run in 19.5-20 minutes** on an RTX PRO 4000 Blackwell; objects_track 591 s with SAM 3 and DINOv3 downloaded in
  the session (the token to its prepare alone); diarize's gated weights the same way. The pod cost $0.21-0.22 and the
  API steps about $0.32: $0.53 of the run's $1.50. The deadline (80 min: setup 30, 12x the video, linger 30) was a
  quarter used: PER_VIDEO_S and SETUP_S can come down once longer clips are measured.
- **What the first pods found, each fixed and then passing on a pod:** the image keeps uv off PATH (the pod runs
  node-setup.sh); uv picked a Python without av's wheels (this machine's Python, in the environment's name); facing's
  MediaPipe needs libEGL and libGLESv2 (in the image); sounds ran out of memory on 24 GB cards (expandable segments,
  and its measured 23 GB: whole-run pods get GPUs with what the graph's CUDA steps need); one malformed reply from
  gpt-audio failed prosody_captions (asked again once).
- **Lifecycle:** a failed run is fetched and its pod deleted; no pod left on the account after six; three that failed
  before the run started cost $0.00-0.04 each.

## Decided 2026-10-28

1. **Unattended runs:** the machine running the whole pipeline gets the keys (over stdin, held in memory).
2. **Gated weights:** downloaded in each pod session; kept on lab servers.
3. **Cache reset:** accepted.
4. **Objects and captions:** consolidate if the results are as good: the A/B comparison in 1b decides.

## Unverified

Lab → pod SSH reachability and agent forwarding on the lab server; whether RunPod's create-pod `env` accepts
`{{ RUNPOD_SECRET_… }}`; whether several pods can attach one network volume; Tailscale inside RunPod containers;
offline loading for transformers with `HF_HUB_OFFLINE=1` (pyannote's: it works, 2026-10-29, phase 5); the disk and transfer estimates. (Whether a
remote coordinator's bundle matches the Mac's: it does, 2026-10-29, phase 7.)
