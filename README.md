# hyperstudy-perception

Turn a video into a time-indexed **perception track** (who is on screen and speaking, what they say and how,
faces, gaze, bodies, hands, gestures, objects, sounds, shots) and review or correct it in an annotator.

Built first for HyperStudy, whose AI agents perceive the stimuli participants watch, but usable on its own.
This repository is the **pipeline**: video in, a versioned track bundle out. Viewing and correcting bundles is a
separate desktop app (Tauri + Svelte 5) that only imports files and never runs models; HyperStudy imports the
same bundle format and the app's annotator component. See [PLAN.md](PLAN.md).

Status: research prototype. `pipeline/bakeoff/` is the model bake-off as it stood on 2026-10-23, moved here from
the HyperStudy repo before the refactor described in [PLAN.md](PLAN.md).

Install it with `uv tool install hyperstudy-perception` (or pip); [orchestrator/README.md](orchestrator/README.md) shows
how to run it, and [RELEASING.md](RELEASING.md) how a release is cut.

| Path | Contents |
|---|---|
| `orchestrator/` | The `perception` CLI: graphs, planning, the run coordinator and store, the places nodes run |
| `nodes/`, `envs/`, `sdk/` | The nodes, the locked uv environments they run in, and the SDK they write outputs with |
| `graphs/` | The graphs `perception run` takes by name (`perception graphs` lists them) |
| `schema/` | Track bundle format: conventions and the JSON Schema (v0.1 draft) |
| `cloud/` | Cloud GPU images and scripts (RunPod, SkyPilot) |
| `pipeline/bundle/` | `convert_bakeoff.py` (bake-off run → bundle) and `validate.py` (schema, checksums, cross-references) |
| `pipeline/bakeoff/` | Bake-off stages (one uv environment each), the HTML annotator prototype, evaluation scripts |
| `docs/` | The `--json` protocol the app reads, the graph spec and the MVP plan |

The code is MIT licensed ([LICENSE](LICENSE)). Model weights are downloaded at run time under their own licences;
several (py-feat, ArcFace, RTMW, Gaze-LLE) are non-commercial. No study video or model weights are stored in this
repository (`nodes/testdata/` holds tiny synthetic clips for tests).
