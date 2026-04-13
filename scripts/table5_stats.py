"""
scripts/table5_stats.py — HGNN Evaluation Graph Statistics (Springer Table 5)

For each dataset, finds the densest metapath (by raw HG edge count |E*|),
then runs the C++ `materialize` and `sketch` commands to collect:

  |V|      — target node count
  |E*|     — matching graph edges (unique raw HG edges along metapath)
  |Eexact| — exact relational graph edges (output of `materialize`)
  |Ekmv|   — KMV-reconstructed edges at k=32 (output of `sketch k=32`)
  rho_exact — exact graph density = |Eexact| / (|V| * (|V| - 1))

The "densest" metapath is the one with the largest |E*| (proxy: unique raw HG
edges along the metapath, computed directly from the PyG graph without any C++
calls).  Pass --metapath to override per dataset.

Usage
-----
    # All 6 paper datasets, default k=32, 10-min C++ timeout
    python scripts/table5_stats.py

    # Specific datasets only
    python scripts/table5_stats.py --datasets HGB_DBLP HGB_ACM OGB_MAG

    # Override metapath for one dataset
    python scripts/table5_stats.py --datasets HGB_DBLP \\
        --metapath "author_to_paper,paper_to_author"

    # Skip C++ calls (PyG stats only — no Eexact/Ekmv)
    python scripts/table5_stats.py --skip-cpp

    # Custom k and timeout
    python scripts/table5_stats.py --k 32 --timeout 1800

Output
------
  results/table5/table5_stats.csv  (one row per dataset)
  results/table5/table5_stats.txt  (human-readable table)
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
import types as _types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Graceful torch_sparse stub (for import-only environments)
if "torch_sparse" not in sys.modules:
    try:
        import torch_sparse  # noqa: F401
    except Exception:
        sys.modules["torch_sparse"] = _types.ModuleType("torch_sparse")
        sys.modules["torch_sparse"].spspmm = None

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from scripts.bench_utils import compile_rule_for_cpp

# ---------------------------------------------------------------------------
# Datasets for Table 5 (in paper order)
# ---------------------------------------------------------------------------

TABLE5_DATASETS: List[str] = [
    "HGB_IMDB",
    "HNE_PubMed",
    "HGB_DBLP",
    "HGB_ACM",
    "OGB_MAG",
    "OAG_CS",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _e_star_from_pyg(metapath_str: str, g_hetero) -> int:
    """Compute |E*|: total unique raw HG edges along the metapath.

    Each unique edge type in the metapath is counted exactly once, regardless
    of how many times it appears in the path string.  This gives the size of
    the matching graph G* (the HIN subgraph induced by the metapath edge types).
    """
    from src.utils import SchemaMatcher

    hops = [h.strip() for h in metapath_str.split(",")]
    seen: set = set()
    total = 0
    for rel_str in hops:
        try:
            et = SchemaMatcher.match(rel_str, g_hetero)
            if et not in seen:
                seen.add(et)
                total += g_hetero[et].num_edges
        except Exception as exc:
            print(f"  [warn] Could not resolve edge type '{rel_str}': {exc}")
    return total


def _count_adj_edges(filepath: str) -> int:
    """Count directed edges in a C++ adjacency-list file.

    Format (one node per line):  <node_id> <nbr1> <nbr2> ...
    Edge count = sum of (len(parts) - 1) over all non-empty lines.
    """
    total = 0
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) > 1:
                total += len(parts) - 1
    return total


def _run_materialize(
    binary: str,
    data_dir: str,
    rule_file: str,
    output_file: str,
    timeout: int,
) -> Optional[int]:
    """Run `materialize` and return the edge count (or None on failure)."""
    cmd = [binary, "materialize", data_dir, rule_file, output_file]
    print(f"  > {' '.join(cmd)}")
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  [timeout] materialize timed out after {timeout}s")
        return None
    except subprocess.CalledProcessError as exc:
        print(f"  [error] materialize exited {exc.returncode}")
        if exc.stderr:
            print(f"  STDERR: {exc.stderr.strip()[:300]}")
        return None

    n = _count_adj_edges(output_file)
    print(f"  |Eexact| = {n:,}")
    return n


def _run_sketch(
    binary: str,
    data_dir: str,
    rule_file: str,
    output_base: str,
    k: int,
    timeout: int,
) -> Optional[int]:
    """Run `sketch k=<k>` and return the edge count (or None on failure).

    The binary writes to `<output_base>_0` (for L=1).
    """
    cmd = [binary, "sketch", data_dir, rule_file, output_base, str(k), "1", str(k)]
    print(f"  > {' '.join(cmd)}")
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  [timeout] sketch timed out after {timeout}s")
        return None
    except subprocess.CalledProcessError as exc:
        print(f"  [error] sketch exited {exc.returncode}")
        if exc.stderr:
            print(f"  STDERR: {exc.stderr.strip()[:300]}")
        return None

    actual_output = output_base + "_0"
    n = _count_adj_edges(actual_output)
    print(f"  |Ekmv| (k={k}) = {n:,}")
    return n


def _density(e_exact: Optional[int], n_nodes: int) -> str:
    """Directed graph density: |Eexact| / (|V| * (|V| - 1)).

    Self-loops are added by downstream code (n_nodes of them), so subtract
    them before computing density.  Returns '' if inputs are invalid.
    """
    if e_exact is None or n_nodes <= 1:
        return ""
    real_edges = max(0, e_exact - n_nodes)  # strip self-loops
    denom = n_nodes * (n_nodes - 1)
    return f"{real_edges / denom:.3e}"


# ---------------------------------------------------------------------------
# Stage + collect stats for one dataset
# ---------------------------------------------------------------------------


def collect_stats(
    dataset_key: str,
    metapath_override: Optional[str],
    k: int,
    timeout: int,
    skip_cpp: bool,
) -> Dict:
    """Load graph, pick densest metapath, run C++, return stats dict."""

    cfg = config.get_dataset_config(dataset_key)
    folder_name = config.get_folder_name(dataset_key)
    data_dir = os.path.join(config.DATA_DIR, folder_name)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_key}  (folder: {folder_name})")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Load graph
    # ------------------------------------------------------------------
    print("[1/4] Loading graph …")
    t0 = time.time()
    g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    print(f"      loaded in {time.time()-t0:.1f}s")

    target_type = cfg.target_node
    n_v = g[target_type].num_nodes
    print(f"      |V| ({target_type}) = {n_v:,}")

    metapaths = cfg.suggested_paths
    if not metapaths:
        print("  [skip] no metapaths configured")
        return {"dataset": dataset_key, "error": "no metapaths"}

    # ------------------------------------------------------------------
    # Pick densest metapath (by |E*| proxy from PyG)
    # ------------------------------------------------------------------
    print("[2/4] Selecting densest metapath …")
    if metapath_override:
        best_mp = metapath_override
        best_e_star = _e_star_from_pyg(best_mp, g)
        print(f"      (override) {best_mp}  |E*|={best_e_star:,}")
    else:
        best_mp: str = metapaths[0]
        best_e_star: int = 0
        for mp in metapaths:
            e_star = _e_star_from_pyg(mp, g)
            print(f"      {mp}  |E*|={e_star:,}")
            if e_star > best_e_star:
                best_e_star = e_star
                best_mp = mp
        print(f"  --> densest: {best_mp}  |E*|={best_e_star:,}")

    # ------------------------------------------------------------------
    # Stage C++ files (skip if already present)
    # ------------------------------------------------------------------
    print("[3/4] Staging C++ files …")
    meta_dat = os.path.join(data_dir, "meta.dat")
    if os.path.exists(meta_dat):
        print("      (already staged — skipping write)")
    else:
        adapter = PyGToCppAdapter(data_dir)
        adapter.convert(g)
        print("      staging complete")

    compile_rule_for_cpp(best_mp, g, data_dir, folder_name)

    # ------------------------------------------------------------------
    # Run C++ commands
    # ------------------------------------------------------------------
    e_exact: Optional[int] = None
    e_kmv: Optional[int] = None

    if not skip_cpp:
        print("[4/4] Running C++ commands …")
        binary = config.CPP_EXECUTABLE
        rule_file = os.path.join(data_dir, f"cod-rules_{folder_name}.limit")

        # materialize → |Eexact|
        mat_out = os.path.join(data_dir, "table5_exact.adj")
        e_exact = _run_materialize(binary, data_dir, rule_file, mat_out, timeout)

        # sketch → |Ekmv|
        sketch_base = os.path.join(data_dir, "table5_sketch")
        e_kmv = _run_sketch(binary, data_dir, rule_file, sketch_base, k, timeout)
    else:
        print("[4/4] Skipping C++ (--skip-cpp)")

    rho = _density(e_exact, n_v)
    e_kmv_bound = n_v * k  # theoretical max

    return {
        "Dataset": dataset_key,
        "MetaPath": best_mp,
        "|V|": n_v,
        "|E*|": best_e_star,
        "|Eexact|": "" if e_exact is None else e_exact,
        "|Ekmv|": "" if e_kmv is None else e_kmv,
        "|V|*k_bound": e_kmv_bound,
        "rho_exact": rho,
    }


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------


def _print_table(rows: List[Dict]) -> None:
    cols = ["Dataset", "MetaPath", "|V|", "|E*|", "|Eexact|", "|Ekmv|", "rho_exact"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
              for c in cols}
    sep = "  ".join("-" * widths[c] for c in cols)
    hdr = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + hdr)
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Table 5 graph statistics.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=TABLE5_DATASETS,
        metavar="DATASET",
        help="Dataset keys to process (default: all 6 paper datasets)",
    )
    parser.add_argument(
        "--metapath",
        default=None,
        help="Override metapath (applied to every dataset — useful for single-dataset runs)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="KMV sketch width k (default: 32)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-command C++ timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--skip-cpp",
        action="store_true",
        help="Skip C++ calls; only report |V| and |E*| from PyG",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(project_root, "results", "table5"),
        help="Output directory for CSV and text table",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows: List[Dict] = []
    for ds_key in args.datasets:
        try:
            row = collect_stats(
                dataset_key=ds_key,
                metapath_override=args.metapath,
                k=args.k,
                timeout=args.timeout,
                skip_cpp=args.skip_cpp,
            )
        except Exception as exc:
            print(f"[ERROR] {ds_key}: {exc}")
            row = {"Dataset": ds_key, "error": str(exc)}
        rows.append(row)

    # Print human-readable table
    valid_rows = [r for r in rows if "error" not in r]
    if valid_rows:
        _print_table(valid_rows)

    # Write CSV
    csv_path = os.path.join(args.out_dir, "table5_stats.csv")
    all_keys = list({k for r in rows for k in r.keys()})
    ordered_keys = [
        "Dataset", "MetaPath", "|V|", "|E*|", "|Eexact|", "|Ekmv|",
        "|V|*k_bound", "rho_exact", "error",
    ]
    fieldnames = [k for k in ordered_keys if k in all_keys] + \
                 [k for k in all_keys if k not in ordered_keys]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved → {csv_path}")

    # Write text table
    txt_path = os.path.join(args.out_dir, "table5_stats.txt")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_table(valid_rows)
    with open(txt_path, "w") as fh:
        fh.write(buf.getvalue())
    print(f"TXT saved → {txt_path}")


if __name__ == "__main__":
    main()
