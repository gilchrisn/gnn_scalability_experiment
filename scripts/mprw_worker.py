"""
mprw_worker.py — MPRW materialization subprocess.

Runs as a child process so /usr/bin/time -v can measure peak RSS,
consistent with how Exact/KMV are measured via the C++ binary.

Inputs
------
Reads a JSON metadata file written by exp3_inference.py before the MPRW
sweep.  The metadata points to pre-serialized edge-index .pt files — one
per metapath step.  These are written from the already-loaded HeteroData,
so the subprocess never has to load the full dataset.

Output
------
Writes edge_index as a .pt file (torch.long tensor [2, E]) to the path
given as the second argument.

Stdout
------
Prints "time: X.XXXXXX" (wall-clock seconds for MPRWKernel.materialize)
so exp3_inference.py can parse it the same way it parses C++ timing output.

Usage (internal — called by exp3_inference.py)
------
    python scripts/mprw_worker.py <meta_json> <output_pt> <k> <seed>
"""
from __future__ import annotations

import json
import sys
import os
import time

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

    n_target    = meta["n_target"]
    target_type = meta["target_type"]

    # Reconstruct a minimal HeteroData from the pre-serialized edge files.
    # Only the edge_index tensors and num_nodes are needed — no features,
    # no labels, no other attributes.
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

    # Only the materialize() call is in the measured window.
    data, elapsed = kernel.materialize(g, triples, target_type)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(data.edge_index, output_path)

    # Print timing in the same format as the C++ binary so exp3 can parse it.
    print(f"time: {elapsed:.6f}")


if __name__ == "__main__":
    main()
