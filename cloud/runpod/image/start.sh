#!/bin/bash
# A pod's start (the image's CMD; remote_driver.removal_at_boot runs it after setting the pod's removal): ssh host keys
# of its own, the key the driver made for it (PUBLIC_KEY, from RunPod), and sshd. Nothing else runs until the driver
# connects.
set -eu
ssh-keygen -A
mkdir -p /root/.ssh
chmod 700 /root/.ssh
printf '%s\n' "${PUBLIC_KEY:?the pod was given no PUBLIC_KEY}" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
exec /usr/sbin/sshd -D -e
