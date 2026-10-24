"""How many processes decode frames beside the detector (py-feat's DataLoader workers).

On an 8-vCPU L4 (OOO clip 2, 2026-10-24), 4 workers took detection from 99 s to 57 s; batch size barely mattered, and
the boxes were byte-identical with and without workers. Workers must be forked: spawned ones (macOS, Windows)
re-import node.py and would run the node again, so there the default is none.
"""

AUTO_WORKERS = 4


def decode_workers(requested: int | None, start_method: str, cpus: int) -> int:
    """`requested` from the config; None means automatic."""
    if requested is None:
        return min(AUTO_WORKERS, max(0, cpus - 1)) if start_method == "fork" else 0
    if requested and start_method != "fork":
        raise ValueError(f"num_workers needs forked workers; this platform starts them with {start_method!r}, "
                         "which would run the node again in each")
    return requested
