#!/usr/bin/env bash
# First step on every RunPod pod, before anything is sent (remote_driver.py runs it over ssh):
#   bash -s MINUTES < cloud/runpod/guard.sh
# 1. The pod removes itself after MINUTES (runpodctl, with the pod's own key): the backstop if the laptop never comes back
#    to delete it. That key is scoped to the pod: it can remove the pod, and cannot list or reach the account's others.
# 2. No key to the account on the pod: the pod's own key is refused by the API, and there is no RunPod or AWS config.
set -euo pipefail
minutes=$1
environ=${PERCEPTION_POD_ENVIRON:-/proc/1/environ}   # the container's own variables (ssh sessions do not inherit them)
var() { tr '\0' '\n' < "$environ" | sed -n "s/^$1=//p"; }
pod=$(var RUNPOD_POD_ID)
[ -n "$pod" ] || { echo "guard: no RUNPOD_POD_ID in the pod's environment: not a RunPod pod" >&2; exit 1; }
command -v runpodctl > /dev/null || { echo "guard: runpodctl is missing: the pod could not remove itself at its deadline" >&2; exit 1; }
for f in ~/.runpod/config.toml ~/.runpod/config.yaml ~/.aws/credentials ~/.aws/config; do
  [ ! -e "$f" ] || { echo "guard: $f exists on the pod: a key to an account must never be there" >&2; exit 1; }
done
# the pod's key must not reach the account: RunPod answers 401 or 403 to it for the account's pods
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -A hyperstudy-perception \
       -H "Authorization: Bearer $(var RUNPOD_API_KEY)" "${PERCEPTION_RUNPOD_API:-https://rest.runpod.io/v1}/pods" || true)
case "$code" in
  401|403) ;;
  *) echo "guard: the pod's own key got $code listing the account's pods (401 or 403 expected): not used" >&2; exit 1 ;;
esac
# its removal at the deadline, detached from this session (it outlives ssh), reading the key only when it runs
cat > ~/.perception-deadline.sh <<EOF
sleep $(( minutes * 60 ))
RUNPOD_API_KEY=\$(tr '\\0' '\\n' < "$environ" | sed -n 's/^RUNPOD_API_KEY=//p') runpodctl remove pod "$pod"
EOF
setsid nohup sh ~/.perception-deadline.sh > ~/.perception-deadline.log 2>&1 < /dev/null &
kill -0 $! 2> /dev/null || { echo "guard: the removal at the deadline did not start" >&2; exit 1; }
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | head -1 || true)
echo "guard: ${gpu:-no GPU seen} in $(var RUNPOD_DC_ID); no key to the account; the pod removes itself in $minutes min"
