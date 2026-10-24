"""Graph files: parse, check against the node manifests, and compile into an ordered graph.

    schema_version: 1
    pipeline: perception-mvp
    inputs:
      video: {type: video.v1, uri: 'file:///data/clip.mp4'}
    policy: {privacy: local-only, max_cost_usd: 0, retries: 2}
    nodes:
      - id: decode
        uses: core.decode@1.0.0
        inputs: {video: $inputs.video}
      - id: faces
        uses: vision.face-track@1.0.0
        inputs: {video: $inputs.video, shots: $nodes.decode.shots}
        config: {fps: 8}

References are `$inputs.<name>` or `$nodes.<node id>.<output name>`. Compilation checks that every reference
exists, that its artifact type is the one the consuming node's manifest declares, that every declared input is
supplied, that config keys are known and valid, and that the graph has no cycles. The privacy policy is checked here
too, so a graph that would upload what it may not never starts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import artifacts
from .manifests import Manifest, Registry

REF = re.compile(r"^\$(inputs)\.([A-Za-z0-9_-]+)$|^\$nodes\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)$")


class GraphError(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputSpec(Strict):
    type: str
    uri: str


UploadKind = Literal["video", "audio", "frames", "faces", "transcript", "embeddings"]
UPLOAD_KINDS = UploadKind.__args__


class Policy(Strict):
    placement: Literal["local", "hybrid", "remote"] = "local"
    # local-only: nothing leaves the machine. Otherwise a node may send out only the kinds of data allow_upload names
    # (spec: video, audio, frames, faces, transcripts and embeddings are permitted separately).
    privacy: Literal["local-only", "local-preferred", "allow-external"] = "local-only"
    allow_upload: list[UploadKind] = Field(default_factory=list)
    max_cost_usd: float = 0.0
    retries: int = Field(default=2, ge=0, le=10)


class NodeSpec(Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    uses: str = Field(pattern=r"^[a-z0-9_.-]+@\d+\.\d+\.\d+$")
    inputs: dict[str, str] = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)
    backend: Literal["auto", "native", "self"] = "auto"
    # where it runs, among the places it can run (`perception plan` lists them): native:mps, native:cpu, ssh-gpu1:cuda...
    place: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]*:[a-z]+$")


class GraphSpec(Strict):
    schema_version: Literal[1]
    pipeline: str
    inputs: dict[str, InputSpec]
    outputs: dict[str, str] = Field(default_factory=dict)
    policy: Policy = Field(default_factory=Policy)
    nodes: list[NodeSpec]


@dataclass
class Ref:
    node: str | None      # None: a graph input
    name: str

    def __str__(self):
        return f"$inputs.{self.name}" if self.node is None else f"$nodes.{self.node}.{self.name}"


@dataclass
class CompiledNode:
    spec: NodeSpec
    manifest: Manifest
    inputs: dict[str, Ref]
    config: dict                              # manifest defaults + the graph's values
    upstream: set[str] = field(default_factory=set)

    @property
    def id(self) -> str:
        return self.spec.id


@dataclass
class Graph:
    spec: GraphSpec
    nodes: dict[str, CompiledNode]            # in topological order
    source: Path | None = None

    def input_path(self, name: str) -> Path:
        uri = self.spec.inputs[name].uri
        parsed = urlparse(uri)
        if parsed.scheme not in ("file", ""):
            raise GraphError([f"input {name}: only file:// inputs are supported locally ({uri})"])
        path = Path(unquote(parsed.path)) if parsed.scheme else Path(uri)
        if not path.is_absolute() and self.source:
            path = self.source.parent / path
        return path

    def descendants(self, node_id: str) -> set[str]:
        out, frontier = set(), {node_id}
        while frontier:
            frontier = {n.id for n in self.nodes.values() if n.upstream & frontier} - out
            out |= frontier
        return out


def parse_ref(text: str) -> Ref | None:
    m = REF.match(text)
    if not m:
        return None
    return Ref(None, m.group(2)) if m.group(1) else Ref(m.group(3), m.group(4))


def load(path: Path, registry: Registry) -> Graph:
    doc = yaml.safe_load(path.read_text())
    graph = compile_graph(doc, registry)
    graph.source = path.resolve()
    return graph


def compile_graph(doc: dict, registry: Registry) -> Graph:
    try:
        spec = GraphSpec.model_validate(doc)
    except ValidationError as e:
        raise GraphError([f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]) from None

    problems: list[str] = []
    for name, inp in spec.inputs.items():
        if inp.type not in artifacts.TYPES:
            problems.append(f"input {name}: unknown artifact type {inp.type}")
    ids = [n.id for n in spec.nodes]
    for dup in {i for i in ids if ids.count(i) > 1}:
        problems.append(f"node id {dup} is used twice")

    manifests: dict[str, Manifest] = {}
    for n in spec.nodes:
        try:
            manifests[n.id] = registry.get(n.uses)
        except KeyError as e:
            problems.append(f"node {n.id}: {e.args[0]}")

    compiled: dict[str, CompiledNode] = {}
    for n in spec.nodes:
        m = manifests.get(n.id)
        if m is None:
            continue
        refs: dict[str, Ref] = {}
        for name in n.inputs.keys() - m.inputs.keys():
            problems.append(f"node {n.id}: {m.ref} has no input {name!r} (inputs: {', '.join(m.inputs) or 'none'})")
        for name, want in m.inputs.items():
            text = n.inputs.get(name)
            if text is None:
                problems.append(f"node {n.id}: input {name!r} ({want}) is not connected")
                continue
            ref = parse_ref(text)
            if ref is None:
                problems.append(f"node {n.id}: input {name}: {text!r} is not $inputs.<name> or $nodes.<id>.<output>")
                continue
            if ref.node is None:
                got = spec.inputs[ref.name].type if ref.name in spec.inputs else None
                if got is None:
                    problems.append(f"node {n.id}: input {name}: graph has no input {ref.name!r}")
            else:
                src = manifests.get(ref.node)
                if ref.node not in ids:
                    problems.append(f"node {n.id}: input {name}: no node {ref.node!r}")
                    continue
                if src is None:
                    continue
                got = src.outputs.get(ref.name)
                if got is None:
                    problems.append(f"node {n.id}: input {name}: {src.ref} has no output {ref.name!r} (outputs: {', '.join(src.outputs)})")
            if got is not None and got != want:
                problems.append(f"node {n.id}: input {name} wants {want} but {text} is {got}")
            refs[name] = ref

        for key in n.config.keys() - m.config.keys():
            problems.append(f"node {n.id}: {m.ref} has no config {key!r} (config: {', '.join(m.config) or 'none'})")
        config = {**m.defaults(), **{k: v for k, v in n.config.items() if k in m.config}}
        for key, value in config.items():
            for err in Draft202012Validator(m.config[key]).iter_errors(value):
                problems.append(f"node {n.id}: config {key}: {err.message}")

        if spec.policy.privacy == "local-only" and m.privacy.external_upload:
            problems.append(f"node {n.id}: {m.ref} uploads {', '.join(m.privacy.external_upload)}, but the graph's privacy policy is local-only")
        elif not_allowed := [k for k in m.privacy.external_upload if k not in spec.policy.allow_upload]:
            problems.append(f"node {n.id}: {m.ref} uploads {', '.join(not_allowed)}, which policy.allow_upload does not permit")

        compiled[n.id] = CompiledNode(n, m, refs, config, {r.node for r in refs.values() if r.node})

    order = _topological(compiled, problems)
    if problems:
        raise GraphError(problems)
    return Graph(spec, {i: compiled[i] for i in order})


def _topological(nodes: dict[str, CompiledNode], problems: list[str]) -> list[str]:
    order, done, visiting = [], set(), []

    def visit(i: str):
        if i in done:
            return
        if i in visiting:
            cycle = visiting[visiting.index(i):] + [i]
            problems.append(f"cycle: {' -> '.join(cycle)}")
            return
        visiting.append(i)
        for up in sorted(nodes[i].upstream):
            if up in nodes:
                visit(up)
        visiting.pop()
        done.add(i)
        order.append(i)

    for i in nodes:
        visit(i)
    return order
