# `perception --json`

What the annotator app reads when it drives the CLI as a sidecar (`docs/BACKEND_PLAN.md` step 2 in hyperstudy-annotator).
Put `--json` before the command: `perception --json run graph.yaml --input video=clip.mp4`. `plan` and `run` also take
`--place NODE=PLACE` (where a node runs; `plan` lists its options) and `--only NODE[,NODE]` (just those nodes and the
nodes they need).

**The protocol.**
- Every line on stdout is one JSON object with `"v"` (the protocol version, now 1) and `"type"`. Nothing else is
  printed there.
- Adding a field is compatible. Changing what a field means, or removing it, bumps `v`.
- The exit code is 0 on success and 1 otherwise. A failure always ends with an object saying why: `error`, a `plan`
  with `ok: false`, or `run.finished` with `status: "failed"`.
- Times are UTC ISO 8601. Money is USD.

| type | from | fields |
|---|---|---|
| `doctor` | `doctor` | `platform`, `cpu_cores`, `memory_gb`, `accelerators`, `vram_gb`, `tools` (name → path or null), `store`, `nodes` [{`ref`, `stage`, `description`, `where` (or null), `why_not`}] |
| `plan` | `plan`; `run` when it cannot start | `ok`, `estimated_usd`, `nodes` [{`id`, `uses`, `where`, `why_not`, `cached`, `downloads` [{`id`, `source`, `size_gb`, `gated`, `licence_mode`}], `missing_secrets`, `cost_usd`, `cost_note`, `uploads` (data kinds it sends off the machine: `video`, `audio`, `frames`, `faces`, `transcript`, `embeddings`), `place` (where it would run, `backend:accelerator`), `options` (every place it can run here, best first; `--place NODE=PLACE` chooses), `estimate_s` (seconds, or null), `estimate_basis` (what the estimate rests on, or why there is none), `needs` (the nodes whose outputs it takes)}] |
| `run.created` | `run`, `resume`, `rerun` | `run`, `store` |
| `event` | the same, as they happen (`resume` and `rerun` stream only their own, not the run's history) | `run`, `node` (or null for the run), `kind` (`task.started`, `task.cached`, `task.retry`, `task.succeeded`, `task.failed`, `run.succeeded`, `run.failed`, ...), `message`, `at` |
| `run.finished` | the same, last | `run`, `status` (`succeeded`, `failed`, `canceled`) |
| `status` | `status RUN` (a task's `cost_usd`: what is reserved while it runs, what it was charged once it succeeded; a reservation charged by a cancel is in the run's `spent_usd` and the task shows 0) | `run`, `pipeline`, `status`, `created_at`, `spent_usd`, `max_cost_usd`, `tasks` [{`node`, `state`, `attempt`, `runner`, `started_at`, `finished_at`, `seconds`, `cost_usd`, `error_code`, `error`}] |
| `runs` | `status` | `runs` [{`run`, `pipeline`, `status`, `created_at`}], newest first |
| `exported` | `export` | `run`, `path` (the bundle folder) |
| `nodes` | `nodes` | `nodes` [{`ref`, `id`, `version`, `stage`, `description`, `inputs` (name → type), `outputs`, `config` (name → JSON Schema), `places` (where it can run here, best first), `why_not`, `cost` (or null), `uploads`, `secrets` (names), `models` [{`id`, `source`, `size_gb`, `gated`, `licence_mode`}], `gpu`}] |
| `graph` | `graph validate` | `valid`, `nodes` (ids in run order), `problems` |
| `models` | `models list` | `models` [{`node`, `id`, `source`, `size_gb`, `gated`, `licence_mode`}] |
| `artifacts` | `artifacts RUN` | `run`, `artifacts` [{`node`, `name`, `type`, `hash`, `path`}] |
| `event` | `logs RUN` | as above, the run's history (`--follow` keeps streaming; the node's own log text is not included) |
| `canceled` | `cancel` | `run` |
| `error` | any command | `problems` [strings] (an invalid graph, an unknown run, or an unexpected exception) |

A node's `error_code` is one of `transient`, `input`, `schema`, `privacy`, `licence`, `resources` (out of memory: the
message names the setting to lower) and `internal`. Only `transient` is retried. Two more are set by the coordinator:
`budget` (the run's cost cap stopped the node before it started) and `upstream` (canceled because a node it depends on
failed).
