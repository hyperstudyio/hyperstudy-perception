#!/usr/bin/env bash
# Run one task on AWS and delete the machine afterwards.
#   [OUT=DIR] cloud/skypilot/launch.sh TASK.yaml CLUSTER [sky launch options, e.g. --env CLIP_SRC=/path/to/clip.mp4]
# With OUT set, the machine's ~/out is copied into DIR before the machine is deleted (also when the job failed).
# - SkyPilot reads sky.yaml as the project config: no credentials copied, and only the perception-node role.
# - The workdir is a staged copy of the files git tracks, nothing else: commit a new file before a launch that needs
#   it. (Untracked files were staged too until 2026-10-24; .gitignore does not cover every image, crop or bundle
#   folder that can sit in a working copy.)
# - Once the instance exists, its OS shutdown is set to terminate it, so guard.sh's deadline deletes on-demand
#   machines as well as spot ones.
# - sky down runs on exit, whether the job succeeded, failed or was interrupted.
# - A wall-clock deadline (MAX_MINUTES, default 60) ends the launch, and so deletes the machine, whatever the job is
#   doing: a hang of any kind (a stray process holding the job open, a stuck download) costs at most that long.
set -euo pipefail
task=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); cluster=$2; shift 2
case "$cluster" in
  perception-*) ;;
  *) echo "launch.sh: cluster names start with perception- (the machine's role can only end those machines)" >&2; exit 2 ;;
esac
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
region=us-east-1
export SKYPILOT_PROJECT_CONFIG=$repo/cloud/skypilot/sky.yaml

stage=$(mktemp -d)
git -C "$repo" ls-files -z | rsync -a --from0 --files-from=- "$repo/" "$stage/"
finish() {
  kill "${watcher:-}" "${deadline:-}" 2>/dev/null || :
  if [ -n "${OUT:-}" ]; then mkdir -p "$OUT"; rsync -a "$cluster:out/" "$OUT/" || echo "launch.sh: could not fetch ~/out" >&2; fi
  sky down -y "$cluster"
  rm -rf "$stage"
}
trap finish EXIT

# SkyPilot provisions the instance a few seconds to minutes into `sky launch`; set its shutdown behaviour as soon as it
# exists. Gives up loudly after 15 minutes (then only `sky down`, autodown or MAX_MINUTES deletes it). Not under
# set -e: one failed AWS call must not end the watcher silently.
(
  set +e
  for _ in $(seq 90); do
    ids=$(aws ec2 describe-instances --region "$region" --output text --query 'Reservations[].Instances[].InstanceId' \
          --filters "Name=tag:skypilot-cluster-name,Values=$cluster*" Name=instance-state-name,Values=pending,running)
    if [ -n "$ids" ]; then
      for id in $ids; do
        if aws ec2 modify-instance-attribute --region "$region" --instance-id "$id" --instance-initiated-shutdown-behavior terminate; then
          echo "launch.sh: $id terminates on OS shutdown"
        else
          echo "launch.sh: WARNING could not make $id terminate on shutdown; its deadline will only stop it" >&2
        fi
      done
      exit 0
    fi
    sleep 10
  done
  echo "launch.sh: WARNING no instance for $cluster after 15 min; its shutdown deadline will only stop it" >&2
) &
watcher=$!

sky launch -c "$cluster" -i 15 --down -y "$task" --workdir "$stage" "$@" &
launch=$!
minutes=${MAX_MINUTES:-60}
( sleep $((minutes * 60)); echo "launch.sh: past MAX_MINUTES=$minutes; deleting the machine" >&2; kill "$launch" ) &
deadline=$!
wait "$launch"
