#!/usr/bin/env bash
# First step of every task's setup: a shutdown deadline, and a check that the machine holds no AWS credentials
# and no role beyond perception-node.
#   bash cloud/skypilot/guard.sh MINUTES
# The deadline is the backstop if the laptop never comes back to delete the machine. launch.sh makes an OS shutdown
# terminate the instance (EC2's default for on-demand machines is to stop it, which keeps the disk billing).
set -euo pipefail
sudo shutdown -h "+$1"
if [ -e ~/.aws/credentials ] || [ -e ~/.aws/config ]; then
  echo "guard: ~/.aws exists on the machine; sky.yaml was not applied" >&2; exit 1
fi
token=$(curl -sf -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
# The only role allowed is perception-node, which can end perception-* machines and nothing else (sky.yaml).
profile=$(curl -sf -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/iam/info \
          | python3 -c 'import json,sys; print(json.load(sys.stdin)["InstanceProfileArn"])' || echo none)
case "$profile" in
  */perception-node) ;;
  *) echo "guard: the instance profile is $profile, not perception-node; sky.yaml was not applied" >&2; exit 1 ;;
esac
echo "guard: no AWS credentials; role perception-node; shutdown in $1 min"
