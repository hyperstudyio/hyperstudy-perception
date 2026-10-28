"""The node side of the perception graph's file protocol. Standard library only: it is installed into every model
environment, which share nothing else.

The coordinator makes a fresh attempt directory, writes request.json into it and runs the node with
`--request PATH`. The node reads its inputs (paths), writes outputs under the request's out_dir, and finishes by
writing response.json next to the request. The coordinator then validates, hashes and publishes the outputs; a node
never writes into the artifact store, so a node that crashes or is orphaned cannot corrupt it.

    from perception_node import Node, NodeError

    with Node() as node:
        video = node.input("video")              # a Path
        shots = detect(video, node.config["threshold"])
        node.write_json("shots", "shots.json", shots)
        node.model("PySceneDetect AdaptiveDetector", licence_mode="commercial-ok", licence="BSD-3-Clause")

Errors: raise NodeError(code, message). Codes: transient (worth retrying: a network error, a busy device),
input (the inputs cannot be processed), schema, privacy, licence (terminal). Any other exception is reported as
`internal` with its traceback, and is terminal: retrying a bug does not fix it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

PROTOCOL = 1
# resources: the node ran out of memory (GPU or RAM). Terminal like the others: the same input on the same device would
# run out again, so the fix is a smaller memory setting in the graph's config, or another device.
ERROR_CODES = ("transient", "input", "schema", "privacy", "licence", "resources", "internal")
MEMORY_SETTINGS = ("batch_size",)   # config keys that trade memory for speed, named in the message


def out_of_memory(exc: BaseException) -> bool:
    """PyTorch's OutOfMemoryError (CUDA), MPS's RuntimeError, and Python's MemoryError."""
    return (isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError"
            or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()))


class NodeError(Exception):
    def __init__(self, code: str, message: str):
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code {code!r}")
        super().__init__(message)
        self.code = code


class Node:
    def __init__(self, argv: list[str] | None = None):
        parser = argparse.ArgumentParser()
        parser.add_argument("--request", type=Path, required=True)
        args = parser.parse_args(argv)
        self.request_path: Path = args.request
        self.request: dict = json.loads(self.request_path.read_text())
        if self.request.get("protocol") != PROTOCOL:
            raise SystemExit(f"request protocol {self.request.get('protocol')} is not {PROTOCOL}")
        self.out_dir = Path(self.request["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config: dict = self.request.get("config", {})
        self.accelerator: str = self.request.get("accelerator", "cpu")   # cpu | mps | cuda: where placement put this run
        self._outputs: dict[str, dict] = {}
        self._models: list[dict] = []
        self._usage: dict = {}
        self._t0 = time.time()

    # --- inputs ---------------------------------------------------------------------------------------------

    def input(self, name: str) -> Path:
        try:
            return Path(self.request["inputs"][name]["path"])
        except KeyError:
            raise NodeError("input", f"no input {name!r} in the request") from None

    def input_json(self, name: str):
        return json.loads(self.input(name).read_text())

    def has_input(self, name: str) -> bool:
        """Whether the request carries this input: one of the manifest's optional_inputs is absent when the graph left it
        unconnected (or `--only` left its producer out)."""
        return name in self.request.get("inputs", {})

    def secret(self, name: str) -> str:
        """A secret the request names, from the environment the coordinator gave this process."""
        if name not in self.request.get("secrets", []):
            raise NodeError("input", f"secret {name} was not requested by this node's manifest")
        value = os.environ.get(name)
        if not value:
            raise NodeError("input", f"secret {name} is not set")
        return value

    # --- outputs ---------------------------------------------------------------------------------------------

    def path(self, name: str, relative: str) -> Path:
        """Where to write output `name` (a file, or a directory for tree outputs). Declares the output."""
        p = self.out_dir / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        self._outputs[name] = {"path": relative}
        return p

    def write_json(self, name: str, relative: str, doc) -> Path:
        p = self.path(name, relative)
        p.write_text(json.dumps(doc, separators=(",", ":")))
        return p

    def producer(self, timings: dict | None = None, device: str | None = None) -> dict:
        """The bundle layer `producer` block: stage and models from the manifest, and the device.

        Deliberately without the run time or timings: a stored layer must be the same bytes whenever the same inputs
        produce the same data, or every downstream node would miss the cache after an upstream rerun. Timings go to
        the response's usage; `perception export` writes run_at and timings into the bundle's copy of the layer."""
        if timings:
            self._usage.setdefault("timings", {}).update(timings)
        base = self.request.get("producer", {})
        return {**base, "models": self._models or base.get("models", []), "config": self.config,
                "device": device, "run_at": None, "timings": {}}

    def model(self, id: str, licence_mode: str, source: str | None = None, revision: str | None = None, licence: str | None = None):
        """Record a model this run used (the coordinator checks it against the manifest's licence mode)."""
        self._models.append({"id": id, "source": source, "revision": revision, "licence": licence, "licence_mode": licence_mode})

    def usage(self, **values):
        """Billable or measured usage: seconds, tokens_in, tokens_out, usd, device..."""
        self._usage.update(values)

    # --- lifecycle -------------------------------------------------------------------------------------------

    def __enter__(self) -> "Node":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        error = None
        if exc is not None:
            if isinstance(exc, NodeError):
                error = {"code": exc.code, "message": str(exc)}
            elif isinstance(exc, KeyboardInterrupt):
                error = {"code": "transient", "message": "interrupted"}
            elif out_of_memory(exc):
                knobs = [f"{k} (now {self.config[k]})" for k in MEMORY_SETTINGS if k in self.config]
                hint = f"lower {' or '.join(knobs)} in the graph's config for this node" if knobs else "run it on a device with more memory"
                error = {"code": "resources", "message": f"out of memory on {self.accelerator}: {exc}. To fix: {hint}.",
                         "traceback": traceback.format_exc()}
            else:
                error = {"code": "internal", "message": f"{exc_type.__name__}: {exc}", "traceback": traceback.format_exc()}
        response = {
            "protocol": PROTOCOL, "ok": error is None, "outputs": self._outputs if error is None else {},
            "models": self._models, "usage": {"seconds": round(time.time() - self._t0, 3), **self._usage},
            "host": {"platform": sys.platform, "machine": platform.machine(), "python": platform.python_version()},
            "error": error,
        }
        tmp = self.request_path.with_name("response.json.tmp")
        tmp.write_text(json.dumps(response, indent=1))
        os.replace(tmp, self.request_path.with_name("response.json"))   # the coordinator never sees half a response
        if error:
            print(f"node failed ({error['code']}): {error['message']}", file=sys.stderr)
            sys.exit(1)
        return False
