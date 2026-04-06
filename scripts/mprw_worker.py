"""
mprw_worker.py — MPRW materialization subprocess.

Memory measurement
------------------
We measure the true OS-level high-water mark RSS *during* materialize()
using a background polling thread (10 ms interval).  The reported value is
peak_RSS_during_materialize - baseline_RSS_before_materialize.

This correctly captures transient intermediate tensors (e.g. the full
N_target × k walker-state tensor) that may be garbage-collected before a
simple rss_after - rss_before snapshot is taken.  The design mirrors what
/usr/bin/time -v does for the C++ Exact/KMV subprocesses, making the
comparison methodologically consistent.

Stdout lines (all parsed by exp3_inference.py)
-----------------------------------------------
    time: X.XXXXXX          — wall-clock seconds for materialize() only
    peak_ram_mb: X.X        — true RSS high-water mark above baseline (MB)

Usage (internal — called by exp3_inference.py)
-----------------------------------------------
    python scripts/mprw_worker.py <meta_json> <output_pt> <k> <seed>
"""
from __future__ import annotations

import json
import os
import sys
import threading
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


class _PeakRSSTracker:
    """Context manager that polls process RSS every 10 ms in a daemon thread,
    capturing the true high-water mark during a specific code section.

    Correctly reports intermediate allocations that are freed before the
    section ends (e.g. transient walker-state tensors in MPRW, scatter
    buffers in GNN message passing).

    Usage:
        with _PeakRSSTracker() as tracker:
            do_work()
        net_peak = tracker.net_peak_mb   # peak - baseline, always >= 0
    """
    _POLL_INTERVAL = 0.01  # 10 ms

    def __init__(self):
        self._baseline: float = 0.0
        self._peak: float = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._baseline = _rss_mb()
        self._peak = self._baseline
        self._stop.clear()
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()

    def _run(self):
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            while not self._stop.wait(self._POLL_INTERVAL):
                try:
                    rss = proc.memory_info().rss / 1e6
                    if rss > self._peak:
                        self._peak = rss
                except Exception:
                    pass
        except ImportError:
            while not self._stop.wait(self._POLL_INTERVAL):
                rss = _rss_mb()
                if rss > self._peak:
                    self._peak = rss

    @property
    def net_peak_mb(self) -> float:
        """True RSS high-water mark minus baseline, in MB. Always >= 0."""
        return max(0.0, self._peak - self._baseline)


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

    # Measure true peak RSS during materialize() only.
    # Baseline is taken after all setup (Python + torch loaded, edges read),
    # so Python runtime overhead is excluded — consistent with C++ measurements.
    with _PeakRSSTracker() as tracker:
        data, elapsed = kernel.materialize(g, triples, target_type)
    peak_ram_mb = tracker.net_peak_mb

    # Write output edge_index
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(data.edge_index, output_path)

    # Emit parsed fields for exp3_inference.py
    print(f"time: {elapsed:.6f}")
    print(f"peak_ram_mb: {peak_ram_mb:.2f}")


if __name__ == "__main__":
    main()
