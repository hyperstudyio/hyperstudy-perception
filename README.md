# hyperstudy-perception

Turn a video into a time-indexed **perception track** (who is on screen and speaking, what they say and how,
faces, gaze, bodies, hands, gestures, objects, sounds, shots) and review or correct it in an annotator.

Built first for HyperStudy, whose AI agents perceive the stimuli participants watch, but usable on its own.
This repository is the **pipeline**: video in, a versioned track bundle out. Viewing and correcting bundles is a
separate desktop app (Tauri + Svelte 5) that only imports files and never runs models; HyperStudy imports the
same bundle format and the app's annotator component. See [PLAN.md](PLAN.md).

Status: research prototype. `pipeline/bakeoff/` is the model bake-off as it stood on 2026-10-23, moved here from
the HyperStudy repo before the refactor described in [PLAN.md](PLAN.md).

| Path | Contents |
|---|---|
| `pipeline/bakeoff/` | Bake-off stages (one uv environment each), the HTML annotator prototype, evaluation scripts |
| `eval/labels/` | Hand labels and reference sets (track ids and times only; no video) |
| `docs/research/`, `docs/reports/` | Model surveys, measured results (`docs/research/Agent video perception audio and objects/measured_bakeoff_results.md`), hosting report |
| `docs/plans/` | Annotator plan |

Model weights are downloaded at run time under their own licences; several (py-feat, ArcFace, RTMW, Gaze-LLE)
are non-commercial. No video or model weights are stored in this repository.
