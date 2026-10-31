# The RunPod image for perception's nodes (RunPodBackend.image): software only, no repository code, weights or data
# (it is public). Built by .github/workflows/runpod-image.yml from the context tag.py stages (these two files).
#
# On RunPod's own PyTorch image, which its hosts keep cached: our image's start then pulls only the small layers below
# (measured: a 6.8 GB image of our own took 145 s to start, the cached one 12-32 s). The nodes' environments are not in
# it: uv fetches them from PyPI in the datacenter faster than a host pulls them from GHCR. Its Python and torch are
# not used (uv brings each node's own).
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04@sha256:2594f25beeef751aeebcc207ba4c45a382ba174655f01e15cbbfad0e2bdb5ef8
ENV DEBIAN_FRONTEND=noninteractive

# what nodes and the driver use (setup then installs nothing): ffmpeg, rsync, and the EGL and GLES libraries MediaPipe
# loads even on the CPU (vision.facing failed for want of libEGL.so.1 on a pod, 2026-10-31). sshd takes keys only.
# Each pod makes its own host keys at start (start.sh): the ones in the base image are public, and are never used
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg rsync libegl1 libgles2 \
 && rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_* \
 && mkdir -p /run/sshd /etc/ssh/sshd_config.d \
 && printf 'PermitRootLogin prohibit-password\nPasswordAuthentication no\nKbdInteractiveAuthentication no\n' \
      > /etc/ssh/sshd_config.d/perception.conf \
 && { grep -q '^Include /etc/ssh/sshd_config.d' /etc/ssh/sshd_config || sed -i '1i Include /etc/ssh/sshd_config.d/*.conf' /etc/ssh/sshd_config; }

# runpodctl 3, for the pod's removal at its deadline with its own key (remote_driver.removal_at_boot)
ARG RUNPODCTL=3.0.1
ARG RUNPODCTL_SHA256=b66d11eca8cacbf18a2de6943c6e18a79ffbdc8a3c30aa13a49dc6cfc15d4239
RUN curl -fsSL -o /usr/local/bin/runpodctl https://github.com/runpod/runpodctl/releases/download/v${RUNPODCTL}/runpodctl-linux-amd64 \
 && echo "${RUNPODCTL_SHA256}  /usr/local/bin/runpodctl" | sha256sum -c - \
 && chmod 755 /usr/local/bin/runpodctl

# uv, the version the annotator app bundles, where node-setup.sh puts it
ARG UV=0.13.5
ARG UV_SHA256=f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f
RUN curl -fsSL -o /tmp/uv.tgz https://github.com/astral-sh/uv/releases/download/${UV}/uv-x86_64-unknown-linux-gnu.tar.gz \
 && echo "${UV_SHA256}  /tmp/uv.tgz" | sha256sum -c - \
 && mkdir -p /root/.perception-uv \
 && tar -xzf /tmp/uv.tgz -C /root/.perception-uv --strip-components=1 \
 && rm /tmp/uv.tgz \
 && /root/.perception-uv/uv --version

COPY start.sh /perception-start.sh
CMD ["/perception-start.sh"]
