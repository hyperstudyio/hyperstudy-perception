"""Dry run: what `perception run` would do, without doing it (spec acceptance criterion 6).

For each node: where it would run (or why it cannot), whether its result is already cached, which model weights are
not on this machine yet, which secrets are missing, and the estimated spend. Privacy conflicts are graph errors
(ir.py), so a graph that reaches planning has none.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .estimates import estimate, media_seconds, probe_seconds
from .ir import Graph
from .manifests import ModelRef
from .scheduler import Coordinator, manifest_cost
from .store import Artifact


@dataclass
class NodePlan:
    id: str
    uses: str
    where: str | None
    why_not: str = ""
    cached: bool = False
    downloads: list[ModelRef] = field(default_factory=list)
    missing_secrets: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    cost_note: str = ""
    uploads: list[str] = field(default_factory=list)   # data kinds the node sends off the machine (its manifest's privacy)
    place: str | None = None                           # where it would run (backend:accelerator)
    options: list[str] = field(default_factory=list)   # every place it can run here, best first
    estimate_s: float | None = None                    # seconds, from its measured runs here (estimates.py)
    estimate_basis: str = ""
    needs: list[str] = field(default_factory=list)     # the nodes whose outputs it takes (a stage picker's edges)
    estimates: dict[str, dict] = field(default_factory=dict)   # every option -> {estimate_s, estimate_basis}: to compare places
    title: str = ""
    service: str | None = None                                  # the hosted API that does its work, if it runs elsewhere
    description: str = ""
    outputs: dict[str, str] = field(default_factory=dict)      # name -> artifact type
    models: list[ModelRef] = field(default_factory=list)       # every model it uses (downloads: the ones not here yet)
    settings: list[dict] = field(default_factory=list)         # its fixed-choice settings: {key, description, value, choices, prices, per}


@dataclass
class Plan:
    nodes: list[NodePlan]
    allow_downloads: bool = False
    allow_uploads: bool = False   # the run's consent; the graph's policy.allow_upload is the most it may ever send

    @property
    def ok(self) -> bool:
        return all(n.where and not n.missing_secrets and (self.allow_downloads or not n.downloads)
                   and (self.allow_uploads or n.cached or not n.uploads) for n in self.nodes)

    def render(self, explain: bool = False) -> str:
        lines = []
        for n in self.nodes:
            status = "cached" if n.cached else (n.where or "CANNOT RUN")
            lines.append(f"{n.id:16s} {n.uses:28s} {status}")
            if n.why_not:
                lines.append(f"{'':16s}   {n.why_not}")
            for m in n.downloads:
                size = f"{m.size_gb:.1f} GB" if m.size_gb else "size unknown"
                lines.append(f"{'':16s}   needs download: {m.id} ({size}{', gated' if m.gated else ''}, {m.licence_mode}) from {m.source}")
            for s in n.missing_secrets:
                lines.append(f"{'':16s}   missing secret: {s}")
            if n.cost_usd or n.cost_note:
                lines.append(f"{'':16s}   estimated ${n.cost_usd:.2f}" + (f" ({n.cost_note})" if n.cost_note else ""))
        total = sum(n.cost_usd for n in self.nodes if not n.cached)
        lines.append(f"estimated spend: ${total:.2f}")
        downloads = [m for n in self.nodes for m in n.downloads]
        if downloads and not self.allow_downloads:
            lines.append("weights are missing: fetch them first (perception models pull) or run with --allow-downloads")
        uploading = [n for n in self.nodes if n.uploads and not n.cached]
        if uploading and not self.allow_uploads:
            kinds = sorted({k for n in uploading for k in n.uploads})
            lines.append(f"{', '.join(n.id for n in uploading)} send {', '.join(kinds)} off this machine: run with --allow-uploads, or leave them out (--only)")
        return "\n".join(lines)


def hf_cached(source: str) -> bool:
    """Is hf://org/name[@revision] in the local Hugging Face cache?"""
    repo = source.removeprefix("hf://").split("@")[0]
    parts = repo.split("/")
    if len(parts) < 2:
        return False
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    return (hub / f"models--{parts[0]}--{parts[1]}").is_dir()


def settings(m, config: dict) -> list[dict]:
    """A node's settings that are a fixed list of choices (a model, an effort), with the price of each when the node's
    cost is keyed by that setting: what the app offers as menus. Free-form settings stay at their tuned defaults."""
    out = []
    for key, schema in m.config.items():
        if "enum" not in schema:
            continue
        priced = m.cost and m.cost.by == key and isinstance(m.cost.usd, dict)
        out.append({"key": key, "description": schema.get("description", ""), "value": config.get(key), "choices": list(schema["enum"]),
                    "prices": dict(m.cost.usd) if priced else None, "per": m.cost.per if priced else None})
    return out


def plan(coord: Coordinator, graph: Graph, allow_downloads: bool = False, allow_uploads: bool = False) -> Plan:
    known: dict[str, dict[str, Artifact]] = {}   # node -> outputs, when its result is cached
    graph_inputs = {name: coord._graph_input(graph, name) for name in graph.spec.inputs}
    out = []
    video_s = None   # the video's length: from a cached decode below, else ffprobe on the graph input
    for name, spec in graph.spec.inputs.items():
        if spec.type == "video.v1" and not video_s:
            video_s = probe_seconds(graph.input_path(name))
    for node in graph.nodes.values():
        m = node.manifest
        placement, why = coord.place(m, node.spec.backend, node.spec.place)
        np = NodePlan(node.id, m.ref, f"{placement.backend.kind} ({placement.runner.accelerator})" if placement else None, why)
        np.options = [p.label for p in coord.placements(m, node.spec.backend)[0]]
        np.needs = sorted(node.upstream)
        np.place = placement.label if placement else None
        np.downloads = [x for x in m.models if x.source and x.source.startswith("hf://") and not hf_cached(x.source)]
        np.missing_secrets = [s for s in m.secrets if s not in coord.secrets]
        np.uploads = list(m.privacy.external_upload)
        np.title, np.service, np.description, np.outputs, np.models = m.title, m.service, m.description, dict(m.outputs), list(m.models)
        np.settings = settings(m, node.config)
        if placement:
            inputs = {}
            for name, ref in node.inputs.items():
                if ref.node is None:
                    inputs[name] = graph_inputs[ref.name]
                elif ref.node in known:
                    inputs[name] = known[ref.node][ref.name]
            if len(inputs) == len(node.inputs):
                key = coord.cache_key(node, placement, inputs)
                hit = coord.store.db.execute("SELECT outputs FROM cache WHERE key=?", (key,)).fetchone()
                if hit:
                    outs = {k: Artifact.from_json(v) for k, v in json.loads(hit["outputs"]).items()}
                    if all(coord.store.has(a) for a in outs.values()):
                        np.cached = True
                        np.downloads = []
                        known[node.id] = outs
                        video_s = media_seconds(json.loads(hit["outputs"])) or video_s
            np.cost_usd = placement.backend.estimate(m, placement.runner, {}).cost_usd
            if m.cost:   # units are known once the input exists (from the cache, or a graph input)
                np.cost_usd += manifest_cost(m, node.config, inputs) if m.cost.per == "run" or m.cost.input in inputs else 0.0
                np.cost_note = "" if m.cost.per == "run" or m.cost.input in inputs else f"${m.cost.unit_usd(node.config)} per {m.cost.per}, units known after {node.inputs[m.cost.input].node}"
        for option in np.options:   # cached or not: moving a node elsewhere runs it again
            seconds, basis = estimate(coord.store, m.ref, option, video_s)
            np.estimates[option] = {"estimate_s": seconds, "estimate_basis": basis}
        if placement and not np.cached:
            chosen = np.estimates[placement.label]
            np.estimate_s, np.estimate_basis = chosen["estimate_s"], chosen["estimate_basis"]
        out.append(np)
    return Plan(out, allow_downloads, allow_uploads)
