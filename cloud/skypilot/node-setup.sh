#!/usr/bin/env bash
# What a cloud node needs on its machine (AWS or RunPod), installed by the driver (remote_driver.py) over ssh after guard.sh
# and before anything is sent, and waited for: SkyPilot runs a task's setup detached (as a job, `sky launch` returns
# before it ends), so a node started after the launch could find ffmpeg half installed.
#   bash -s < node-setup.sh        prints the uv it installed on its last line
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo=sudo; [ "$(id -u)" = 0 ] && sudo=   # a RunPod pod is root, with no sudo
# a fresh Ubuntu machine runs its own updates at boot: wait for apt's lock rather than fail on it. rsync: the driver sends
# with it (RunPod's image has none)
$sudo apt-get -o DPkg::Lock::Timeout=600 update -qq
$sudo apt-get -o DPkg::Lock::Timeout=600 install -y -qq ffmpeg rsync > /dev/null
# uv, the version the annotator app bundles, in a folder of its own (SkyPilot installs a uv of its own on the machine)
dir="$HOME/.perception-uv"
if [ ! -x "$dir/uv" ]; then
  curl -LsSf https://astral.sh/uv/0.13.5/install.sh | env UV_INSTALL_DIR="$dir" UV_NO_MODIFY_PATH=1 sh > /dev/null
fi
"$dir/uv" --version | grep -q '^uv 0\.13\.5' || { echo "node-setup: $dir/uv is not uv 0.13.5" >&2; exit 1; }
command -v ffmpeg > /dev/null && command -v ffprobe > /dev/null && command -v rsync > /dev/null
echo "$dir/uv"
