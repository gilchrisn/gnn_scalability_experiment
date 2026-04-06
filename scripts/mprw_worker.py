"""
mprw_worker.py — MPRW materialization subprocess.

Memory measurement
------------------
tracemalloc is started immediately before materialize() and stopped after.
tracemalloc.clear_traces() resets the peak counter so the reported value is
the true peak Python-level allocation during materialize() only, not
accumulated from earlier setup (JSON loading, edge file reads, etc.).
This captures all tensor constructions through Python's PyMem_Raw* layer
regardless of whether new OS pages are faulted in — necessary because
PyTorch's CPU caching allocator reuses pre-mapped pages, which makes
RSS-delta approaches report 0 on small graphs.

Stdout lines (all parsed by exp3_inference.py)
-----------------------------------------------
    time: X.XXXXXX          — wall-clock seconds for materialize() only
    peak_ram_mb: X.X        — peak tracemalloc allocation during materialize() (MB)

Usage (internal — called by exp3_inference.py)
-----------------------------------------------
    python scripts/mprw_worker.py <meta_json> <output_pt> <k> <seed>
"""
from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import HeteroData

from src.kernels.mprw import MPRWKernel


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

    # Measure peak allocation during materialize() only.
    # clear_traces() resets the peak counter so setup allocations (edge files,
    # HeteroData construction) are excluded — consistent with inference_worker.
    tracemalloc.start()
    tracemalloc.clear_traces()
    data, elapsed = kernel.materialize(g, triples, target_type)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_bytes / 1e6

    # Write output edge_index
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(data.edge_index, output_path)

    # Emit parsed fields for exp3_inference.py
    print(f"time: {elapsed:.6f}")
    print(f"peak_ram_mb: {peak_ram_mb:.2f}")


if __name__ == "__main__":
    main()
