# The RunPod image for perception's CUDA nodes (RunPodBackend.image): software only. Built by
# .github/workflows/runpod-image.yml from the context tag.py stages: these files, sdk/pyproject.toml and each CUDA
# environment's pyproject.toml and uv.lock (envs.txt). No repository code, no weights, no data: it is public.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

# sshd (key only; each pod makes its own host keys at start: a public image must hold none), and what nodes and the
# driver use: ffmpeg, rsync
RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-server ffmpeg rsync curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_* \
 && mkdir -p /run/sshd \
 && printf 'PermitRootLogin prohibit-password\nPasswordAuthentication no\nKbdInteractiveAuthentication no\n' \
      > /etc/ssh/sshd_config.d/perception.conf

# runpodctl, for the pod's removal at its deadline with its own key (remote_driver.removal_at_boot)
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

# each CUDA node's environment, synced from its lock into a folder of its own: the driver points uv at it
# (UV_PROJECT_ENVIRONMENT), so a pod's sync only adds the node's own package. Python goes in uv's own folder.
COPY envs.txt /src/envs.txt
COPY sdk /src/sdk
COPY envs /src/envs
RUN cd /src \
 && while read -r env; do \
      UV_PROJECT_ENVIRONMENT=/opt/perception/venvs/$env /root/.perception-uv/uv sync --project "$env" --frozen --no-dev \
        --no-install-package perception-node --link-mode copy || exit 1; \
    done < envs.txt \
 && /root/.perception-uv/uv cache clean \
 && rm -rf /src

COPY start.sh /perception-start.sh
CMD ["/perception-start.sh"]
