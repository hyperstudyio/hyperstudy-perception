#!/usr/bin/env bash
# Build the package PyPI publishes (orchestrator/dist: a wheel and an sdist). The package carries the pipeline's files,
# as committed at HEAD, in perception/_pipeline: the nodes, their environments' locks, the node SDK they install, the
# graphs, the layer schemas and the cloud scripts nodes run with on other machines (not the tests, plans or bake-off
# code), with RELEASE.json naming the commit and listing them (the package has no .git: release.py).
#   scripts/build-dist.sh
set -euo pipefail
cd "$(dirname "$0")/.."
paths=(cloud envs graphs nodes schema sdk)
[ -z "$(git status --porcelain -- "${paths[@]}" orchestrator)" ] || { echo "commit first: the package is HEAD, and says which commit it is" >&2; exit 1; }
data=orchestrator/src/perception/_pipeline
rm -rf "$data" orchestrator/dist
mkdir -p "$data"
git archive --format=tar HEAD -- "${paths[@]}" | tar -x -C "$data"
git ls-files -z -- "${paths[@]}" | python3 -c '
import json, sys
files = sorted(f for f in sys.stdin.read().split("\0") if f)
json.dump({"commit": sys.argv[1], "files": files}, open(sys.argv[2], "w"), indent=0)
' "$(git rev-parse HEAD)" "$data/RELEASE.json"
cp LICENSE orchestrator/LICENSE   # a real file: the sdist would keep a link to ../LICENSE, which points nowhere there
(cd orchestrator && uv build --quiet)
rm -rf "$data" orchestrator/LICENSE   # a checkout runs from its own files (release.py)
ls orchestrator/dist
