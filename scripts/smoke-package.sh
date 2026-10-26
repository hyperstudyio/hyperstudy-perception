#!/usr/bin/env bash
# A built package, used as the annotator app uses it: installed with uv as a tool into folders of its own, on a Python
# uv fetches, then run with nothing on PATH but uv (an app started from the Finder has /usr/bin:/bin:/usr/sbin:/sbin),
# decoding a small clip made here. Fails if any of it does.
#   scripts/smoke-package.sh orchestrator/dist/hyperstudy_perception-X.Y.Z-py3-none-any.whl
set -euo pipefail
wheel=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
uvdir=$(dirname "$(command -v uv)")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
in_app() {
  env -i HOME="$HOME" PATH="$uvdir:/usr/bin:/bin:/usr/sbin:/sbin" UV_TOOL_DIR="$work/tools" UV_TOOL_BIN_DIR="$work/bin" \
    UV_PYTHON_INSTALL_DIR="$work/python" UV_PYTHON_PREFERENCE=only-managed UV_CACHE_DIR="$work/cache" UV_NO_CONFIG=1 \
    PERCEPTION_HOME="$work/home" "$@"
}
# 3 s of 160x120 video at 25 fps with a 440 Hz tone, through PyAV (no ffmpeg here either)
uv run --quiet --no-project --with av==16.1.0 python - "$work/clip.mp4" <<'PY'
import array, math, sys
import av
with av.open(sys.argv[1], "w") as out:
    v = out.add_stream("libx264", rate=25)
    v.width, v.height, v.pix_fmt = 160, 120, "yuv420p"
    a = out.add_stream("aac", rate=48000, layout="mono")
    for i in range(75):
        frame = av.VideoFrame(160, 120, "yuv420p")
        for plane in frame.planes:
            plane.update(bytes([i * 3 % 256]) * plane.buffer_size)
        out.mux(v.encode(frame))
    out.mux(v.encode())
    for s in range(0, 48000 * 3, 1024):
        frame = av.AudioFrame(format="flt", layout="mono", samples=1024)
        frame.planes[0].update(array.array("f", (0.3 * math.sin(2 * math.pi * 440 * (s + k) / 48000) for k in range(1024))).tobytes())
        frame.sample_rate, frame.pts = 48000, s
        out.mux(a.encode(frame))
    out.mux(a.encode())
PY
# every requirement exact (scripts/build-dist.sh pins the lock's versions): an install gets what CI tested
python3 - "$wheel" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as w:
    meta = w.read(next(n for n in w.namelist() if n.endswith(".dist-info/METADATA"))).decode()
reqs = [l.split(":", 1)[1].strip() for l in meta.splitlines() if l.startswith("Requires-Dist:")]
loose = [r for r in reqs if "==" not in r.split(";")[0]]
assert reqs and not loose, f"requirements not pinned exactly: {loose}"
print(f"requirements: {len(reqs)}, all exact")
PY
in_app uv tool install --quiet --python 3.12 "$wheel"
in_app "$work/bin/perception" --json doctor | python3 -c '
import json, sys
d = json.load(sys.stdin)
assert d["pipeline"]["version"] and d["pipeline"]["commit"], d["pipeline"]
assert any(n["ref"] == "core.decode@1.0.0" and n["where"] for n in d["nodes"]), d["nodes"]
print("doctor:", d["pipeline"]["version"], d["pipeline"]["commit"][:7])'
in_app "$work/bin/perception" run decode.yaml --input "video=$work/clip.mp4" | tail -2
ls "$work/home/envs" | grep -q '^core-' || { echo "decode's environment is not in the store" >&2; exit 1; }
if find "$work/tools" -name .venv -path '*_pipeline*' | grep -q .; then echo "an environment was made inside the package" >&2; exit 1; fi
echo "the package installs and decodes with nothing but uv"
