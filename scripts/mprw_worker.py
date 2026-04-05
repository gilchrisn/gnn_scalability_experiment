"""
mprw_worker.py — MPRW materialization subprocess.

Runs as a child process so /usr/bin/time -v can measure peak RSS,
consistent with how Exact/KMV are measured via the C++ binary.

Memory measurement
------------------
/usr/bin/time -v captures the entire subprocess peak RSS, which includes
~300 MB of Python/PyTorch runtime overhead unrelated to the algorithm.
To isolate only the materialization memory, we also self-report a net RSS
delta: rss_after_materialize - rss_before_materialize (measured via psutil
after all setup is complete).  exp3_inference.py prefers this net figure
and falls back to /usr/bin/time -v only if psutil is unavailable.

Stdout lines (all parsed by exp3_inference.py)
-----------------------------------------------
    time: X.XXXXXX          — wall-clock seconds for materialize() only
    net_ram_mb: X.X         — net RSS increase during materialize() in MB

Usage (internal — called by exp3_inference.py)
-----------------------------------------------
    python scripts/mprw_worker.py <meta_json> <output_pt> <k> <seed>
"""
from __future__ import annotations

import json
import os
import sys
import time

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import HeteroData

from src.kernels.mprw import MPRWKernel


def _rss_mb() -> float:
    """Current process RSS in MB. Prefers psutil; falls back to /proc."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return 0.0


def main() -> None:
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <meta_json> <output_pt> <k> <seed>",
              file=sys.stderr)
        sys.exit(1)

    meta_path   = sys.argv[1]
    output_path = sys.argv[2]
    k           = int(sys.argv[3])
    seed        = int(sys.argv[4])

    with open(meta_path) as f:
        meta = json.load(f)

    target_type = meta["target_type"]

    # Reconstruct minimal HeteroData from pre-serialized edge files.
    # No features, no labels — only edge_index and num_nodes.
    g = HeteroData()
    triples = []

    for step in meta["steps"]:
        src_t     = step["src_type"]
        edge_name = step["edge_name"]
        dst_t     = step["dst_type"]

        ei = torch.load(step["edge_file"], weights_only=True)
        g[src_t].num_nodes = step["n_src"]
        g[dst_t].num_nodes = step["n_dst"]
        g[src_t, edge_name, dst_t].edge_index = ei
        triples.append((src_t, edge_name, dst_t))

    kernel = MPRWKernel(k=k, seed=seed, device=torch.device("cpu"))

    # Measure RSS baseline AFTER all setup (torch loaded, edge files read).
    # This excludes Python/PyTorch runtime overhead (~300 MB) that has nothing
    # to do with the materialization algorithm — the same overhead is absent
    # from the C++ Exact/KMV subprocesses, so including it would be unfair.
    rss_before_mb = _rss_mb()

    data, elapsed = kernel.materialize(g, triples, target_type)

    rss_after_mb  = _rss_mb()
    net_ram_mb    = max(rss_after_mb - rss_before_mb, 0.0)

    # Write output edge_index
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(data.edge_index, output_path)

    # Emit parsed fields for exp3_inference.py
    print(f"time: {elapsed:.6f}")
    print(f"net_ram_mb: {net_ram_mb:.2f}")


if __name__ == "__main__":
    main()
