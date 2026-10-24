# `perception --json`

What the annotator app reads when it drives the CLI as a sidecar (`docs/BACKEND_PLAN.md` step 2 in hyperstudy-annotator).
Put `--json` before the command: `perception --json run graph.yaml --input video=clip.mp4`. `plan` and `run` also take
`--place NODE=PLACE` (where a node runs; `plan` lists its options), `--only NODE[,NODE]` (just those nodes and the
nodes they need) and `--set NODE.KEY=VALUE` (a node's setting; `plan` lists the fixed choices). `run` sends nothing off
the machine without `--allow-uploads` (and then only what the graph's `policy.allow_upload` names), as it downloads no
weights without `--allow-downloads`. The consent is kept with the run: `resume` and `rerun` keep to it (they take
`--allow-uploads` to give it), and a node that would upload in a run without it fails before it starts (`policy`).
`--place` and `--set` may name a node `--only` leaves out. On Apple silicon a node that runs on the GPU has no CPU
place (CPU and GPU share memory), so `--place NODE=native:cpu` is refused for it there. A node placed on another machine (`--place NODE=ssh-gpu1:cuda`, a host from `perception hosts add`) sends its inputs there: it is an upload like an API's, offered only when the graph's `policy.allow_upload` names each kind in its inputs and run only with `--allow-uploads`. Nodes that call an API or need secrets always run on this machine. The app runs `graphs/full.yaml` and picks features with `--only`.

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
| `plan` | `plan`; `run` when it cannot start | `ok`, `estimated_usd`, `places` (every place in any node's `options` -> {`title` (for people: `This machine`, `gpu1 (SSH)`), `remote` (another machine: it receives the node's inputs), `accelerator`}), `nodes` [{`id`, `uses`, `where`, `why_not`, `cached`, `downloads` [{`id`, `source`, `size_gb`, `gated`, `licence_mode`}], `missing_secrets`, `cost_usd`, `cost_note`, `uploads` (data kinds that leave the machine where it would run: `video`, `audio`, `frames`, `faces`, `transcript`, `embeddings`; to its API, or its inputs to another machine), `upload_to` (where they go: the API's service, or the other machine's title, e.g. `gpu1 (SSH)`; null when nothing leaves), `place` (where it would run, `backend:accelerator`: `native:mps`, `ssh-gpu1:cuda`), `options` (every place it can run, best first: this machine's, then other machines' that the graph's policy lets its inputs go to; another machine is never the default, only chosen with `--place NODE=PLACE`), `estimate_s` (seconds, or null), `estimate_basis` (what the estimate rests on, or why there is none), `needs` (the nodes whose outputs it takes), `estimates` (each of `options` -> {`estimate_s`, `estimate_basis`, `cost_usd` (what running there costs, upper estimate), `uploads`, `upload_to` (as above, if it ran there)}, to compare places before choosing one), `title` (what it gives, in a few words), `service` (the hosted API that does its work, e.g. `Anthropic API`, or null when it runs where `place` says), `description`, `outputs` (name → artifact type), `models` [{`id`, `source`, `licence_mode`, `size_gb`}], `settings` [{`key`, `description`, `value`, `choices`, `prices` (choice → USD per `per`, when the cost depends on it, else null), `per`}]}]; `ok` is false while a node that would run uploads and `run` was not given `--allow-uploads` (a `plan` alone never is) |
| `run.created` | `run`, `resume`, `rerun` | `run`, `store` |
| `event` | the same, as they happen (`resume` and `rerun` stream only their own, not the run's history) | `run`, `node` (or null for the run), `kind` (`task.started`, `task.cached`, `task.retry`, `task.succeeded`, `task.failed`, `run.succeeded`, `run.failed`, ...), `message`, `at` |
| `run.finished` | the same, last | `run`, `status` (`succeeded`, `failed`, `canceled`) |
| `status` | `status RUN` (a task's `cost_usd`: what is reserved while it runs, what it was charged once it succeeded; a reservation charged by a cancel is in the run's `spent_usd` and the task shows 0) | `run`, `pipeline`, `status`, `created_at`, `spent_usd`, `max_cost_usd`, `tasks` [{`node`, `state`, `attempt`, `runner`, `started_at`, `finished_at`, `seconds`, `cost_usd`, `error_code`, `error`}] |
| `runs` | `status` | `runs` [{`run`, `pipeline`, `status`, `created_at`}], newest first |
| `exported` | `export` | `run`, `path` (the bundle folder) |
| `nodes` | `nodes` | `nodes` [{`ref`, `id`, `version`, `stage`, `description`, `inputs` (name → type), `outputs`, `config` (name → JSON Schema), `places` (where it can run here, best first), `why_not`, `cost` (or null), `uploads`, `secrets` (names), `models` [{`id`, `source`, `size_gb`, `gated`, `licence_mode`}], `gpu`}] |
| `hosts` | `hosts add`, `list`, `check`, `remove`, `clean` | `hosts` [{`name`, `dest` (the ssh destination), `root` (its folder on the host), `caps` {`platform`, `cpu_cores`, `memory_gb`, `accelerators`, `vram_gb`, `tools` (uv, ffmpeg, ffprobe → path on the host, or null)}, `probed_at`}]: what each host had when it was added or last checked. Its places are `ssh-NAME:ACCELERATOR`. A host that cannot be reached, or a bad name, destination or root, is an `error` |
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
