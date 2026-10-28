"""What this machine has: platform, cores, memory, accelerators (each NVIDIA card), tools. Used by `perception doctor`
and placement."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from .base import Capabilities, Gpu, this_platform


def _memory_gb() -> float:
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(out.stdout.strip()) / 2**30 if out.returncode == 0 else 0.0
    if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    return 0.0


def _cuda_gpus() -> list[Gpu]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    out = subprocess.run([smi, f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return parse_gpus(out.stdout) if out.returncode == 0 else []


# what nvidia-smi is asked for each card (the name last: the one field that could hold a comma)
GPU_QUERY = "index,memory.total,name"


def parse_gpus(text: str) -> list[Gpu]:
    """nvidia-smi's lines for GPU_QUERY, as cards; a line it could not fill (a card in a bad state: "[N/A]") is left out."""
    gpus = []
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",", 2)]
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            gpus.append(Gpu(int(parts[0]), parts[2], round(int(parts[1]) / 1024, 1)))
    return sorted(gpus, key=lambda g: g.index)


def probe() -> Capabilities:
    plat = this_platform()
    memory = _memory_gb()
    accelerators = {"cpu"}
    gpus = _cuda_gpus()
    vram = max((g.memory_gb for g in gpus), default=0.0)   # what a runner's minimum is checked against: the largest card
    if gpus:
        accelerators.add("cuda")
    if plat == "darwin/arm64":
        accelerators.add("mps")
        vram = vram or memory   # unified memory
    return Capabilities(platform=plat, cpu_cores=os.cpu_count() or 1, memory_gb=round(memory, 1), accelerators=accelerators,
                        vram_gb=round(vram, 1), tools={"uv": shutil.which("uv"), "ffmpeg": shutil.which("ffmpeg"),
                                                       "ffprobe": shutil.which("ffprobe")}, gpus=gpus)
