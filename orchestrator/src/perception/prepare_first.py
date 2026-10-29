"""A native node whose weights are downloaded by its prepare (download_secrets) and are not in the cache yet: its prepare
first, with those secrets, then the node, without them (backends/native.py, NativeBackend.submit):

    python -m perception.prepare_first --secrets HF_TOKEN[,...] --prepare PREPARE_JSON -- NODE_COMMAND... --request REQUEST

PREPARE_JSON is the prepare command as a JSON list. The secrets come in this process's environment. The node is started
with execve, without them and offline (backends/base.OFFLINE): a process's environment is what execve gave it, so the
node never holds a download secret (/proc/<pid>/environ there shows none), only this process does while its prepare
runs. The node's command ends with its request, as every node's does: a restarted coordinator recognises this process
as the node's by it (native._alive).

A prepare that fails is the attempt's own and final, as a host's is (remote_driver.PrepareFailed): a response with
error code internal, beside the request, and its output at the end of the attempt's log.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .backends.base import OFFLINE


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
    prepare = json.loads(opts["--prepare"])
    request = Path(node[node.index("--request") + 1])
    print("preparing the node (its weights, into the cache it reads)", flush=True)
    env = {k: v for k, v in os.environ.items() if k not in OFFLINE}
    done = subprocess.run(prepare, env=env, stdin=subprocess.DEVNULL)   # its output goes to the attempt's log
    if done.returncode:
        return fail(request, f"preparing the node failed (exit {done.returncode}): see the end of the log")
    env = {k: v for k, v in os.environ.items() if k not in secrets}
    os.execvpe(node[0], node, {**env, **OFFLINE})


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
