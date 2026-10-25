#!/usr/bin/env bash
# First step on every RunPod pod, before anything is sent (remote_driver.py runs it over ssh):
#   bash -s MIN_DRIVER < cloud/runpod/guard.sh       (the oldest NVIDIA driver the nodes' torch runs on, e.g. 580)
# 1. The pod's removal at its deadline is running: its start command set it (runpodctl, with the pod's own key), the
#    backstop if the laptop never comes back to delete it. That key is scoped to the pod: it removes the pod, and can
#    neither list the account's pods nor remove another (both checked on RunPod, 2026-10-25).
# 2. No key to the account on the pod: the pod's own key is refused by the API, and there is no RunPod or AWS config.
set -euo pipefail
min_driver=$1
environ=${PERCEPTION_POD_ENVIRON:-/proc/1/environ}   # the container's own variables (ssh sessions do not inherit them)
var() { tr '\0' '\n' < "$environ" | sed -n "s/^$1=//p"; }
pod=$(var RUNPOD_POD_ID)
[ -n "$pod" ] || { echo "guard: no RUNPOD_POD_ID in the pod's environment: not a RunPod pod" >&2; exit 1; }
command -v runpodctl > /dev/null || { echo "guard: runpodctl is missing: the pod could not remove itself at its deadline" >&2; exit 1; }
for f in ~/.runpod/config.toml ~/.runpod/config.yaml ~/.aws/credentials ~/.aws/config; do
  [ ! -e "$f" ] || { echo "guard: $f exists on the pod: a key to an account must never be there" >&2; exit 1; }
done
# its own key: what removes it at the deadline
[ -n "$(var RUNPOD_API_KEY)" ] || { echo "guard: no RUNPOD_API_KEY in the pod's environment: it could not remove itself at its deadline" >&2; exit 1; }
# the pod's key must not reach the account: RunPod answers 401 or 403 to it for the account's pods. Exit 75: the check could
# not be made (no answer, RunPod's own trouble), which says nothing about the key (the driver tries another pod)
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -A hyperstudy-perception \
       -H "Authorization: Bearer $(var RUNPOD_API_KEY)" "${PERCEPTION_RUNPOD_API:-https://rest.runpod.io/v1}/pods" || true)
case "$code" in
  401|403) ;;
  2??) echo "guard: the pod's own key got $code listing the account's pods (401 or 403 expected): not used" >&2; exit 1 ;;
  *) echo "guard: could not check the pod's own key (RunPod answered ${code:-nothing}); another pod is tried" >&2; exit 75 ;;
esac
# its removal at the deadline, which its start command set before anything else ran (remote_driver.removal_at_boot)
pgrep -f perception-deadline > /dev/null || { echo "guard: the pod's removal at its deadline is not running" >&2; exit 1; }
# a driver new enough for the nodes' CUDA (asked of RunPod too): exit 75, another pod may have one
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null | head -1 || true)
[ "${driver%%.*}" -ge "$min_driver" ] 2> /dev/null || {
  echo "guard: the pod's NVIDIA driver is ${driver:-not known}, older than $min_driver (the nodes' CUDA); another pod is tried" >&2; exit 75; }
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | head -1 || true)
echo "guard: ${gpu:-no GPU seen} (driver $driver) in $(var RUNPOD_DC_ID); no key to the account; its removal at its deadline is running"
