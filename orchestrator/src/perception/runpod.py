"""RunPod's REST API (https://rest.runpod.io/v1), as perception uses it: create a pod, read it, delete it, list pods.

The account key comes from the user's own keys (RUNPOD_API_KEY in ~/.config/hyperstudy/*.env, or the environment) and is
used only on this machine: never written to a file, never sent to a pod. RunPod's own CLI and SkyPilot keep it in
~/.runpod/config.toml, which SkyPilot uploads to every machine it starts (AWS ones too), so that file is not used.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v1"
# Cloudflare in front of the API refuses Python's default User-Agent (error 1010)
USER_AGENT = "hyperstudy-perception"
# a pod perception started: perception-<run>-<node>-<hash>-<attempt>-d<the deadline, as a Unix time>
POD_NAME = re.compile(r"^perception-[a-z0-9-]+-d(\d{9,11})$")


class RunPodError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"RunPod answered {status}: {message}" if status else f"RunPod could not be reached: {message}")
        self.status = status   # 0: no answer


def api_base() -> str:
    return os.environ.get("PERCEPTION_RUNPOD_API", API)   # the tests' stand-in server


class RunPod:
    def __init__(self, key: str, base: str | None = None, timeout: float = 60):
        self.key, self.base, self.timeout = key, (base or api_base()).rstrip("/"), timeout

    def call(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(self.base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                                              "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                text = r.read().decode()
        except urllib.error.HTTPError as e:
            raise RunPodError(e.code, e.read().decode(errors="replace").strip()[:300]) from None
        except (urllib.error.URLError, OSError) as e:   # no answer (a timeout too): the request may or may not have been done
            raise RunPodError(0, str(getattr(e, "reason", e))) from None
        try:
            return json.loads(text) if text.strip() else None
        except json.JSONDecodeError:
            raise RunPodError(0, f"an answer that is not JSON: {text[:200]!r}") from None

    def create_pod(self, body: dict) -> dict:
        return self.call("POST", "/pods", body)

    def pod(self, pod_id: str) -> dict | None:
        """The pod, or None once it is gone (404)."""
        try:
            return self.call("GET", f"/pods/{pod_id}")
        except RunPodError as e:
            if e.status == 404:
                return None
            raise

    def delete_pod(self, pod_id: str):
        try:
            self.call("DELETE", f"/pods/{pod_id}")
        except RunPodError as e:
            if e.status != 404:   # already gone
                raise

    def pods(self) -> list[dict]:
        return self.call("GET", "/pods") or []
