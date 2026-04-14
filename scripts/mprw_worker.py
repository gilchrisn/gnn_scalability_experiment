"""
mprw_worker.py — MPRW materialization subprocess.

Memory measurement
------------------
Peak RSS is measured by the parent process via /usr/bin/time -v (Linux/WSL),
identical to how Exact and KMV C++ subprocesses are measured in engine.py.
The worker itself only needs to emit algorithm time to stdout.

Stdout lines (parsed by exp3_inference.py)
------------------------------------------
    time: X.XXXXXX   — perf_counter seconds for materialize() only
                        (excludes JSON load and edge tensor reads;
                        includes CSR build + walks, analogous to C++
                        internal timer which excludes HeterGraph file load)

Usage (internal — called by exp3_inference.py via _run_mprw_subprocess)
------------------------------------------------------------------------
    python scripts/mprw_worker.py <meta_json> <output_pt> <k> <seed>
"""
from __future__ import annotations

import json
import os
import sys

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# On Windows, PyTorch extension DLLs (mprw_cpp.pyd) can't find their
# dependencies unless the torch lib dir is explicitly added to the DLL
# search path.  Must happen before the mprw import below.
if sys.platform == "win32":
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)

from torch_geometric.data import HeteroData

from src.kernels.mprw import MPRWKernel
from src.kernels.mprw import _HAS_CPP as _MPRW_CPP


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
    # (Graph load excluded from timer, same as C++ HeterGraph construction.)
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

    print(f"backend: {'cpp' if _MPRW_CPP else 'python'}", file=sys.stderr)

    kernel = MPRWKernel(k=k, seed=seed, device=torch.device("cpu"))
    data, elapsed = kernel.materialize(g, triples, target_type)
    # Peak RSS is captured by the parent via /usr/bin/time -v — no need to
    # emit it here.  Only algorithm time goes to stdout.

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(data.edge_index, output_path)

    # Emit algorithm time — same stdout protocol as C++ binary's "time:" line.
    print(f"time: {elapsed:.6f}")


if __name__ == "__main__":
    main()
