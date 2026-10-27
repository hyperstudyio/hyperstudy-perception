"""Whether a provider accepts a key (`perception keys check NAME`): one request that reads something small and costs
nothing (Anthropic's model list, your Hugging Face account, your RunPod pods). The key is sent only to its own
provider, never through a redirect (runpod.OPENER)."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request

from .runpod import OPENER, RunPod, RunPodError

ANTHROPIC = "https://api.anthropic.com/v1/models?limit=1"
HUGGING_FACE = "https://huggingface.co/api/whoami-v2"
CHECKED = ("ANTHROPIC_API_KEY", "HF_TOKEN", "RUNPOD_API_KEY")


def get(url: str, headers: dict[str, str]) -> tuple[int, bytes]:
    """The status and body of a GET (an HTTP error's too); a provider that cannot be reached raises OSError."""
    try:
        with OPENER.open(urllib.request.Request(url, headers=headers), timeout=20) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def account(body: bytes) -> str | None:
    """The account name in Hugging Face's answer, if it says one (the key was accepted either way)."""
    try:
        d = json.loads(body or b"{}")
    except ValueError:
        return None
    return d.get("name") if isinstance(d, dict) else None


def check(name: str, value: str | None) -> tuple[bool, str]:
    """(accepted, what to tell the user)."""
    if name not in CHECKED:
        raise ValueError(f"no check for {name}: perception can check {', '.join(CHECKED)}")
    if not value:
        return False, f"{name} is not set"
    try:
        if name == "RUNPOD_API_KEY":
            try:
                RunPod(value).pods()
            except RunPodError as e:
                return False, ("RunPod refused the key" if e.status in (401, 403) else f"RunPod could not be checked: {e}")
            return True, "RunPod accepted the key"
        if name == "ANTHROPIC_API_KEY":
            status, _ = get(ANTHROPIC, {"x-api-key": value, "anthropic-version": "2023-06-01"})
            provider = "Anthropic"
        else:
            status, body = get(HUGGING_FACE, {"Authorization": f"Bearer {value}"})
            provider = "Hugging Face"
            if status == 200:
                return True, f"Hugging Face accepted the key{f' (as {who})' if (who := account(body)) else ''}"
    except (OSError, http.client.HTTPException) as e:   # no whole answer (a reply cut short is not an OSError)
        return False, f"could not reach the provider to check: {e}"
    if status == 200:
        return True, f"{provider} accepted the key"
    if status in (401, 403):
        return False, f"{provider} refused the key"
    return False, f"{provider} could not be checked (HTTP {status})"
