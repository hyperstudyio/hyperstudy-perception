#!/usr/bin/env bash
# Build the package PyPI publishes (orchestrator/dist: a wheel and an sdist), from HEAD, in a staging folder:
# - it carries the pipeline's files in perception/_pipeline: the nodes, their environments' locks, the node SDK they
#   install, the graphs, the layer schemas and the cloud scripts nodes run with on other machines (not the tests, plans
#   or bake-off code), with RELEASE.json naming the commit and listing them (the package has no .git: release.py);
# - it requires exactly the versions orchestrator/uv.lock holds, its dependencies' dependencies too: an install gets
#   the set CI tested, not whatever is newest that day (the steps' own environments are installed from their locks);
# - it carries the repository's LICENSE.
#   scripts/build-dist.sh
set -euo pipefail
cd "$(dirname "$0")/.."
paths=(cloud envs graphs nodes schema sdk)
[ -z "$(git status --porcelain -- "${paths[@]}" orchestrator LICENSE)" ] || { echo "commit first: the package is HEAD, and says which commit it is" >&2; exit 1; }
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
git archive --format=tar HEAD -- orchestrator/pyproject.toml orchestrator/README.md orchestrator/src | tar -x -C "$stage" --strip-components=1
cp LICENSE "$stage/LICENSE"
data="$stage/src/perception/_pipeline"
mkdir -p "$data"
git archive --format=tar HEAD -- "${paths[@]}" | tar -x -C "$data"
git ls-files -z -- "${paths[@]}" | python3 -c '
import json, sys
files = sorted(f for f in sys.stdin.read().split("\0") if f)
json.dump({"commit": sys.argv[1], "files": files}, open(sys.argv[2], "w"), indent=0)
' "$(git rev-parse HEAD)" "$data/RELEASE.json"
uv export --project orchestrator --frozen --no-dev --no-hashes --no-emit-project --no-header --no-annotate > "$stage/pins.txt"
python3 - "$stage/pyproject.toml" "$stage/pins.txt" <<'PY'
import json, re, sys
project, pins = sys.argv[1], [l.strip() for l in open(sys.argv[2]) if l.strip()]
text = open(project).read()
deps = re.search(r"^dependencies = \[\n.*?^\]\n", text, re.S | re.M)
assert deps, "no dependencies list in pyproject.toml"
open(project, "w").write(text[:deps.start()] + "dependencies = [\n" + "".join(f"  {json.dumps(p)},\n" for p in pins) + "]\n" + text[deps.end():])
PY
rm -rf orchestrator/dist
(cd "$stage" && uv build --quiet --out-dir "$OLDPWD/orchestrator/dist")
ls orchestrator/dist
