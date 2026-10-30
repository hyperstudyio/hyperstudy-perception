"""A native node whose weights are downloaded by its prepare (download_secrets) and are not in the cache yet: its prepare
first, with those secrets, then the node, without them (backends/native.py, NativeBackend.submit):

    python -I prepare_first.py --secrets HF_TOKEN[,...] --offline OFFLINE_JSON [--sync SYNC_JSON]
        --prepare PREPARE_JSON --marks MARKS_JSON -- NODE_COMMAND... REQUEST

Run by its path, with the standard library alone: it starts in the node's environment (child_env), which has none of
the coordinator's import path (a coordinator on a host imports this package from its copy, through PYTHONPATH), so it
imports nothing of the package; -I keeps its own folder, this package's, off its import path. OFFLINE_JSON is the
environment the node runs offline with (backends/base.OFFLINE).

SYNC_JSON and PREPARE_JSON are commands as JSON lists: the node's environment synced first, without the secrets (a
build step of a locked package never sees them), then its prepare, in that environment as it is. The secrets come in
this process's environment. The node's command (with its GPU lease, when it has a card: so the card is held from when
the node starts, not while its weights download) is started with execve, without them and offline
(OFFLINE_JSON): a process's environment is what execve gave it, so the node never holds a download secret
(/proc/<pid>/environ there shows none), only this process does while its prepare runs. The node's command ends with its
request's path, as every node's does (`--request PATH`, or a host coordinator's sh script's last argument): a
restarted coordinator recognises this process as the node's by it (native._alive), and a failed prepare's response goes
beside it.

MARKS_JSON lists the files that say its weights are complete in the cache it reads (plan.prepared_mark), made once its
prepare succeeds: a download cut short leaves a snapshot that looks there, which the node, offline, could not finish.

A prepare that fails is the attempt's own and final, as a host's is (remote_driver.PrepareFailed): a response with
error code internal, beside the request, and its output at the end of the attempt's log.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def fail(request: Path, message: str) -> int:
    tmp = request.with_name("response.json.tmp")
    tmp.write_text(json.dumps({"ok": False, "usage": {}, "error": {"code": "internal", "message": message}}))
    os.replace(tmp, request.with_name("response.json"))
    return 1


def main(argv: list[str]) -> int:
    split = argv.index("--")
    head, node = argv[:split], argv[split + 1:]
    opts = dict(zip(head[0::2], head[1::2]))
    secrets = [s for s in opts["--secrets"].split(",") if s]
    offline = json.loads(opts["--offline"])
    request = Path(node[-1])
    # online, for the download, with the token it was sent as HF_TOKEN, and implicit tokens on only then: transformers
    # fetches a model's config without the token it is given when they are off. Its implicit token is then that one: no
    # OIDC identity (which huggingface_hub takes first) and no stored token (its path where there is none). As
    # base.PREPARE_UNSET, OIDC_LOGINS and NO_STORED_TOKEN, which this script cannot import
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "HF_OIDC_RESOURCE", "HF_OIDC_ID_TOKEN")}
    implicit = bool((env.get("HF_TOKEN") or "").strip())
    prepare_env = {**{k: v for k, v in env.items() if not (implicit and k == "HF_HUB_DISABLE_IMPLICIT_TOKEN")},
                   "HF_TOKEN_PATH": "/dev/null/no-stored-token", **({} if implicit else {"HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"})}
    if "--sync" in opts:
        print("syncing the node's environment", flush=True)
        done = subprocess.run(json.loads(opts["--sync"]), env={k: v for k, v in env.items() if k not in secrets},
                              stdin=subprocess.DEVNULL)
        if done.returncode:
            return fail(request, f"syncing the node's environment failed (exit {done.returncode}): see the end of the log")
    print("preparing the node (its weights, into the cache it reads)", flush=True)
    done = subprocess.run(json.loads(opts["--prepare"]), env=prepare_env, stdin=subprocess.DEVNULL)   # its output: the log
    if done.returncode:
        return fail(request, f"preparing the node failed (exit {done.returncode}): see the end of the log")
    for mark in json.loads(opts.get("--marks", "[]")):
        Path(mark).parent.mkdir(parents=True, exist_ok=True)
        Path(mark).touch()
    env = {k: v for k, v in os.environ.items() if k not in secrets}
    os.execvpe(node[0], node, {**env, **offline})


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
