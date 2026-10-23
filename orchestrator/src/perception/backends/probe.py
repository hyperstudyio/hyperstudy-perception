"""What this machine has: platform, cores, memory, accelerators, tools. Used by `perception doctor` and placement."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from .base import Capabilities, this_platform


def _memory_gb() -> float:
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(out.stdout.strip()) / 2**30 if out.returncode == 0 else 0.0
    if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    return 0.0


def _cuda_vram_gb() -> float:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return 0.0
    out = subprocess.run([smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if out.returncode:
        return 0.0
    return max((float(x) / 1024 for x in out.stdout.split() if x.strip()), default=0.0)


def _docker() -> str | None:
    docker = shutil.which("docker")
    if not docker:
        return None
    ok = subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True)
    return docker if ok.returncode == 0 else None   # the CLI without a running daemon cannot run anything


def probe() -> Capabilities:
    plat = this_platform()
    memory = _memory_gb()
    accelerators = {"cpu"}
    vram = _cuda_vram_gb()
    if vram:
        accelerators.add("cuda")
    if plat == "darwin/arm64":
        accelerators.add("mps")
        vram = vram or memory   # unified memory
    return Capabilities(platform=plat, cpu_cores=os.cpu_count() or 1, memory_gb=round(memory, 1), accelerators=accelerators,
                        vram_gb=round(vram, 1), tools={"uv": shutil.which("uv"), "docker": _docker(),
                                                       "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")})
