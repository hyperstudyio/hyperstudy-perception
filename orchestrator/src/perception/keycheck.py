"""Whether a provider accepts a key (`perception keys check NAME`): one request that reads something small and costs
nothing (Anthropic's model list, your Hugging Face account, your RunPod pods). The key is sent only to its own
provider, never through a redirect (runpod.OPENER).

A Hugging Face token is also checked against each gated model the nodes declare: whether its account has been granted
access at the revision the node pins (a HEAD of a small file there), and how Hugging Face grants it (`manual`: the
owner reviews each request; `auto`: on accepting the terms), from the model's public page, asked without the token.
Access is granted to people, never to organisations, so each user asks for it once per model."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .runpod import OPENER, RunPod, RunPodError

ANTHROPIC = "https://api.anthropic.com/v1/models?limit=1"
HUGGING_FACE = "https://huggingface.co/api/whoami-v2"
HF_FILE = "https://huggingface.co/{repo}/resolve/{revision}/.gitattributes"   # every repo has one, and it is small
HF_MODEL = "https://huggingface.co/api/models/{repo}"
CHECKED = ("ANTHROPIC_API_KEY", "HF_TOKEN", "RUNPOD_API_KEY")
NOT_REACHED = (OSError, http.client.HTTPException)   # no whole answer (a reply cut short is not an OSError)


def get(url: str, headers: dict[str, str]) -> tuple[int, bytes]:
    """The status and body of a GET (an HTTP error's too); a provider that cannot be reached raises OSError."""
    try:
        with OPENER.open(urllib.request.Request(url, headers=headers), timeout=20) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def head(url: str, headers: dict[str, str]) -> tuple[int, dict[str, str]]:
    """The status and headers (names in lower case) of a HEAD, an HTTP error's too; a redirect is answered, not followed."""
    try:
        with OPENER.open(urllib.request.Request(url, headers=headers, method="HEAD"), timeout=20) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}


def account(body: bytes) -> str | None:
    """The account name in Hugging Face's answer, if it says one (the key was accepted either way)."""
    try:
        d = json.loads(body or b"{}")
    except ValueError:
        return None
    return d.get("name") if isinstance(d, dict) else None


def gated(registry) -> list[dict]:
    """Every gated model on Hugging Face the nodes declare, once per repo and revision: `id`, `repo`, `revision` (the
    pinned one, else main) and `nodes` (the refs of the nodes that use it)."""
    found: dict[tuple[str, str], dict] = {}
    for m in registry:
        for x in m.models:
            if not (x.gated and x.source and x.source.startswith("hf://")):
                continue
            repo, _, pinned = x.source.removeprefix("hf://").partition("@")
            repo = "/".join(repo.split("/")[:2])
            revision = x.revision or pinned.partition("/")[0] or "main"
            entry = found.setdefault((repo, revision), {"id": x.id, "repo": repo, "revision": revision, "nodes": []})
            if m.ref not in entry["nodes"]:
                entry["nodes"].append(m.ref)
    return list(found.values())


def gating(repo: str) -> str | None:
    """How Hugging Face grants access to a repo (`manual`, `auto`), from its public page; None if it does not say."""
    try:
        status, body = get(HF_MODEL.format(repo=repo), {})
        d = json.loads(body or b"{}") if status == 200 else {}
    except (*NOT_REACHED, ValueError):
        return None
    mode = d.get("gated") if isinstance(d, dict) else None
    return mode if mode in ("manual", "auto") else None


def access(token: str, repo: str, revision: str) -> tuple[str, str | None]:
    """(`granted`, `not-granted` or `unknown`; why it is unknown). Only the file itself, answered, is `granted`."""
    try:
        status, headers = head(HF_FILE.format(repo=repo, revision=revision), {"Authorization": f"Bearer {token}"})
    except NOT_REACHED as e:
        return "unknown", f"could not reach Hugging Face to check: {e}"
    if status == 200:
        return "granted", None
    if status == 403 and headers.get("x-error-code") == "GatedRepo":   # pending and never asked are answered alike
        return "not-granted", None
    if status == 401:
        return "unknown", "Hugging Face refused the key"
    if status == 404:
        return "unknown", f"Hugging Face has no revision {revision} of {repo}"
    message = headers.get("x-error-message")
    return "unknown", f"Hugging Face could not be checked (HTTP {status}{f': {message}' if message else ''})"


def models(token: str | None, wanted: list[dict], why_not: str | None) -> list[dict]:
    """Each of `wanted` with `gating`, `access` and `reason` (why it is unknown, else null), asked for every model at
    once. With no token Hugging Face accepted (`why_not`), only the public gating mode is asked."""
    def one(m: dict) -> dict:
        state, reason = access(token, m["repo"], m["revision"]) if token and not why_not else ("unknown", why_not)
        return {**m, "gating": gating(m["repo"]), "access": state, "reason": reason}
    if not wanted:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(wanted))) as pool:
        return list(pool.map(one, wanted))


def hugging_face(value: str | None, registry=()) -> dict:
    """`ok` and `detail` as for every key, `account` (its name, once accepted) and `models` (the registry's gated(),
    each with its access)."""
    who = None
    if not value:
        ok, detail = False, "HF_TOKEN is not set"
    else:
        try:
            status, body = get(HUGGING_FACE, {"Authorization": f"Bearer {value}"})
            ok = status == 200
            who = account(body) if ok else None
            detail = (f"Hugging Face accepted the key{f' (as {who})' if who else ''}" if ok
                      else "Hugging Face refused the key" if status in (401, 403)
                      else f"Hugging Face could not be checked (HTTP {status})")
        except NOT_REACHED as e:
            ok, detail = False, f"could not reach the provider to check: {e}"
    return {"ok": ok, "detail": detail, "account": who, "models": models(value, gated(registry), None if ok else detail)}


def check(name: str, value: str | None) -> tuple[bool, str]:
    """(accepted, what to tell the user)."""
    if name not in CHECKED:
        raise ValueError(f"no check for {name}: perception can check {', '.join(CHECKED)}")
    if name == "HF_TOKEN":
        r = hugging_face(value)
        return r["ok"], r["detail"]
    if not value:
        return False, f"{name} is not set"
    try:
        if name == "RUNPOD_API_KEY":
            try:
                RunPod(value).pods()
            except RunPodError as e:
                return False, ("RunPod refused the key" if e.status in (401, 403) else f"RunPod could not be checked: {e}")
            return True, "RunPod accepted the key"
        status, _ = get(ANTHROPIC, {"x-api-key": value, "anthropic-version": "2023-06-01"})
    except NOT_REACHED as e:
        return False, f"could not reach the provider to check: {e}"
    if status == 200:
        return True, "Anthropic accepted the key"
    if status in (401, 403):
        return False, "Anthropic refused the key"
    return False, f"Anthropic could not be checked (HTTP {status})"
