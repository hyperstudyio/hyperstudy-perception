#!/usr/bin/env bash
# The pipeline as an app ships it: the files a run needs, as committed at HEAD, into an empty folder, with RELEASE.json
# saying which commit they are and listing them (the copy has no .git; orchestrator/src/perception/release.py).
#   scripts/export-release.sh DEST
set -euo pipefail
cd "$(dirname "$0")/.."
dest=${1:?usage: scripts/export-release.sh DEST}
# what runs read: the CLI and its lock, the nodes, their environments' locks, the node SDK, the graphs, the layer schemas
# and the cloud scripts nodes run with on other machines. Not the tests, plans, evaluations or bake-off code.
paths=(cloud envs graphs nodes orchestrator/pyproject.toml orchestrator/uv.lock orchestrator/src schema sdk)

[ -z "$(git status --porcelain -- "${paths[@]}")" ] || { echo "commit first: the copy is HEAD, and says which commit it is" >&2; exit 1; }
[ ! -e "$dest" ] || [ -z "$(ls -A "$dest")" ] || { echo "$dest is not empty" >&2; exit 1; }
mkdir -p "$dest"
git archive --format=tar HEAD -- "${paths[@]}" | tar -x -C "$dest"
git ls-files -z -- "${paths[@]}" | python3 -c '
import json, sys
files = sorted(f for f in sys.stdin.read().split("\0") if f)
json.dump({"commit": sys.argv[1], "files": files}, open(sys.argv[2], "w"), indent=0)
' "$(git rev-parse HEAD)" "$dest/RELEASE.json"
echo "exported $(git rev-parse --short HEAD) to $dest"
