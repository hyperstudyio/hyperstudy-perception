"""Node manifests: what a node needs, produces and runs on. One `manifest.yaml` per node directory under `nodes/`.

A manifest is the only thing the coordinator knows about a node. The graph refers to it as `uses: <id>@<version>`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DATA_KINDS = ("video", "audio", "frames", "faces", "transcript", "embeddings")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Runner(Strict):
    """One way to run the node. `native`: a script in a uv environment on this machine. `oci`: a container image
    pinned by digest. `self`: the coordinator's own interpreter (tests and trivial nodes only)."""
    kind: Literal["native", "oci", "self"]
    entry: str | None = None                 # native/self: script path, relative to the node directory
    env: str | None = None                   # native: uv project, relative to the repository root
    image: str | None = None                 # oci: registry/name@sha256:...
    platforms: list[str] = Field(default_factory=lambda: ["darwin/arm64", "linux/amd64", "linux/arm64", "windows/amd64"])
    accelerator: Literal["cpu", "mps", "cuda"] = "cpu"
    minimum_vram_gb: float = 0

    @field_validator("image")
    @classmethod
    def pinned(cls, v):
        if v is not None and "@sha256:" not in v:
            raise ValueError("container images must be pinned by digest (name@sha256:...)")
        return v


class Resources(Strict):
    cpu_cores: float = 1
    memory_gb: float = 1
    # exclusive: one such node at a time on a device (large models; measured: two at once swapped a 128 GB Mac)
    gpu: Literal["none", "shared", "exclusive"] = "none"


class Execution(Strict):
    partitionable: bool = False
    deterministic: bool = True


class Privacy(Strict):
    external_upload: list[Literal["video", "audio", "frames", "faces", "transcript", "embeddings"]] = Field(default_factory=list)


class Cost(Strict):
    """A billable node's estimate, reserved against the run's budget before it starts; what it actually spent comes
    back in its response (usage.usd). `per` counts units in one of the node's inputs. When the price depends on a
    config value (the model), `usd` is a table keyed by that value and `by` names the config key."""
    usd: float | dict[str, float]                # per unit, an upper estimate
    by: str | None = None                        # the config key `usd` is keyed by
    per: Literal["shot", "minute", "run"]
    input: str | None = None                     # the input holding the units (shots layer; media for minutes)

    def unit_usd(self, config: dict) -> float:
        return self.usd[config[self.by]] if isinstance(self.usd, dict) else self.usd


class ModelRef(Strict):
    """A model the node uses: where its weights come from and under what terms."""
    id: str
    source: str | None = None                # hf://repo@revision[/file], or a URL
    revision: str | None = None
    licence: str | None = None
    licence_mode: Literal["academic-only", "commercial-ok", "needs-review", "api-terms"]
    size_gb: float | None = None
    gated: bool = False


class Manifest(Strict):
    id: str
    version: str
    description: str = ""
    stage: str | None = None                 # the bundle producer's stage name (default: the node id)
    inputs: dict[str, str] = Field(default_factory=dict)      # name -> artifact type
    outputs: dict[str, str]                                    # name -> artifact type
    config: dict[str, dict] = Field(default_factory=dict)     # name -> JSON Schema (with default)
    runners: list[Runner]
    resources: Resources = Field(default_factory=Resources)
    execution: Execution = Field(default_factory=Execution)
    privacy: Privacy = Field(default_factory=Privacy)
    models: list[ModelRef] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)
    licence: dict = Field(default_factory=dict)
    cost: Cost | None = None

    directory: Path = Field(default=Path("."), exclude=True)

    @model_validator(mode="after")
    def _cost_counts_an_input(self):
        if self.cost and self.cost.per != "run" and self.cost.input not in self.inputs:
            raise ValueError(f"cost: per {self.cost.per} counts units in an input, but cost.input "
                             f"{self.cost.input!r} is not one of this node's inputs ({', '.join(self.inputs) or 'none'})")
        return self

    @model_validator(mode="after")
    def _cost_covers_its_config(self):
        """A price table must price every value the config allows, or a new model would be reserved at nothing."""
        if self.cost is None or not (isinstance(self.cost.usd, dict) or self.cost.by):
            return self
        if not (isinstance(self.cost.usd, dict) and self.cost.by):
            raise ValueError("cost: a price table (usd: {value: price}) and `by` go together")
        if self.cost.by not in self.config or "enum" not in self.config[self.cost.by]:
            raise ValueError(f"cost.by: {self.cost.by!r} is not a config key with an enum")
        missing = [v for v in self.config[self.cost.by]["enum"] if v not in self.cost.usd]
        if missing:
            raise ValueError(f"cost: no price for {self.cost.by} {', '.join(map(repr, missing))}")
        return self

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    def defaults(self) -> dict:
        return {k: s["default"] for k, s in self.config.items() if "default" in s}

    def code_hash(self) -> str:
        """Hash of every file in the node's directory: changing node code invalidates its cache."""
        h = hashlib.sha256()
        for p in sorted(self.directory.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(self.directory)).encode())
                h.update(p.read_bytes())
        return h.hexdigest()


class Registry:
    def __init__(self, manifests: list[Manifest]):
        self.by_ref: dict[str, Manifest] = {}
        for m in manifests:
            if m.ref in self.by_ref:
                raise ValueError(f"two manifests define {m.ref}: {self.by_ref[m.ref].directory} and {m.directory}")
            self.by_ref[m.ref] = m

    @classmethod
    def load(cls, *roots: Path) -> "Registry":
        manifests = []
        for root in roots:
            for path in sorted(root.rglob("manifest.yaml")):
                doc = yaml.safe_load(path.read_text())
                m = Manifest.model_validate(doc)
                m.directory = path.parent.resolve()
                manifests.append(m)
        return cls(manifests)

    def get(self, ref: str) -> Manifest:
        if ref not in self.by_ref:
            same_id = [r for r in self.by_ref if r.split("@")[0] == ref.split("@")[0]]
            hint = f" (available: {', '.join(same_id)})" if same_id else ""
            raise KeyError(f"no node {ref}{hint}")
        return self.by_ref[ref]

    def __iter__(self):
        return iter(self.by_ref.values())
