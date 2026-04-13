"""
scripts/table5_stats.py — HGNN Evaluation Graph Statistics (Springer Table 5)

Runs over ALL configured metapaths per dataset and collects:

  |V|       — target node count
  |E*|      — matching graph edges (sparse matmul over metapath hops)
  |Eexact|  — exact relational graph edges (C++ `materialize` output)
  |Ekmv|    — KMV-reconstructed edges at k=32 (C++ `sketch` output)
  rho_exact — exact graph density = |Eexact| / (|V| * (|V| - 1))

One CSV row per (dataset, metapath).

Usage
-----
    # All metapaths for all 6 paper datasets
    python scripts/table5_stats.py

    # Specific datasets only
    python scripts/table5_stats.py --datasets HGB_DBLP HGB_ACM

    # Single metapath override (skips the rest)
    python scripts/table5_stats.py --datasets HGB_DBLP \\
        --metapath "author_to_paper,paper_to_author"

    # PyG stats only — no C++ calls
    python scripts/table5_stats.py --skip-cpp

    # Custom k and per-command timeout
    python scripts/table5_stats.py --k 32 --timeout 1800

Output
------
  results/table5/table5_stats.csv  (one row per dataset × metapath)
  results/table5/table5_stats.txt  (human-readable table)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import subprocess
import sys
import time
import types as _types
from typing import Dict, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

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

TABLE5_DATASETS: List[str] = [
    "HGB_IMDB",
    "HNE_PubMed",
    "HGB_DBLP",
    "HGB_ACM",
    "OGB_MAG",
    "OAG_CS",
]

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _e_star_from_pyg(metapath_str: str, g_hetero, max_nodes: int = 50_000) -> Optional[int]:
    """Matching graph edges via sparse matmul: A_0 @ A_1 @ ... @ A_{n-1}.

    Binarises after each hop to count unique (src, dst) pairs.
    Returns None when any node dimension exceeds `max_nodes` (OOM guard).
    """
    import numpy as np
    import scipy.sparse as sp
    from src.utils import SchemaMatcher

    hops = [h.strip() for h in metapath_str.split(",")]
    mats = []
    for rel_str in hops:
        try:
            et = SchemaMatcher.match(rel_str, g_hetero)
        except Exception as exc:
            print(f"    [warn] cannot resolve '{rel_str}': {exc}")
            return None

        src_type, _, dst_type = et
        n_src = g_hetero[src_type].num_nodes
        n_dst = g_hetero[dst_type].num_nodes
        if n_src > max_nodes or n_dst > max_nodes:
            print(f"    [skip E*] {src_type}({n_src:,}) or {dst_type}({n_dst:,}) > {max_nodes:,}")
            return None

        ei = g_hetero[et].edge_index.numpy()
        data = np.ones(ei.shape[1], dtype=np.float32)
        mats.append(sp.csr_matrix((data, (ei[0], ei[1])), shape=(n_src, n_dst)))

    result = mats[0]
    for mat in mats[1:]:
        result = result.dot(mat)
        result = result.astype(bool).astype(np.float32)  # binarise

    return int(result.nnz)


def _count_adj_edges(filepath: str) -> int:
    """Count directed edges in a C++ adjacency-list file (node_id nbr1 nbr2 ...)."""
    if not os.path.exists(filepath):
        return 0
    total = 0
    with open(filepath) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) > 1:
                total += len(parts) - 1
    return total


def _run_materialize(binary: str, data_dir: str, rule_file: str,
                     output_file: str, timeout: int) -> Optional[int]:
    cmd = [binary, "materialize", data_dir, rule_file, output_file]
    print(f"    > materialize …")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"    [timeout] materialize timed out after {timeout}s")
        return None
    except subprocess.CalledProcessError as exc:
        print(f"    [error] materialize exited {exc.returncode}")
        if exc.stderr:
            print(f"    STDERR: {exc.stderr.strip()[:300]}")
        return None
    n = _count_adj_edges(output_file)
    print(f"    |Eexact| = {n:,}")
    return n


def _run_sketch(binary: str, data_dir: str, rule_file: str,
                output_base: str, k: int, timeout: int) -> Optional[int]:
    cmd = [binary, "sketch", data_dir, rule_file, output_base, str(k), "1", str(k)]
    print(f"    > sketch k={k} …")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"    [timeout] sketch timed out after {timeout}s")
        return None
    except subprocess.CalledProcessError as exc:
        print(f"    [error] sketch exited {exc.returncode}")
        if exc.stderr:
            print(f"    STDERR: {exc.stderr.strip()[:300]}")
        return None
    n = _count_adj_edges(output_base + "_0")
    print(f"    |Ekmv| (k={k}) = {n:,}")
    return n


def _density(e_exact: Optional[int], n_nodes: int) -> str:
    if e_exact is None or n_nodes <= 1:
        return ""
    real_edges = max(0, e_exact - n_nodes)   # strip self-loops added by engine
    return f"{real_edges / (n_nodes * (n_nodes - 1)):.3e}"


# ---------------------------------------------------------------------------
# Per-dataset collector
# ---------------------------------------------------------------------------


def collect_dataset(
    dataset_key: str,
    metapath_filter: Optional[str],
    k: int,
    timeout: int,
    skip_cpp: bool,
) -> List[Dict]:
    """Return one row dict per metapath for this dataset."""

    cfg = config.get_dataset_config(dataset_key)
    folder_name = config.get_folder_name(dataset_key)
    data_dir = os.path.join(config.DATA_DIR, folder_name)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_key}  (folder: {folder_name})")
    print(f"{'='*60}")

    # Load graph (once per dataset)
    print("[1] Loading graph …")
    t0 = time.time()
    g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    print(f"    loaded in {time.time()-t0:.1f}s")

    n_v = g[cfg.target_node].num_nodes
    print(f"    |V| ({cfg.target_node}) = {n_v:,}")

    # Decide which metapaths to run
    metapaths = cfg.suggested_paths
    if not metapaths:
        print("    [skip] no metapaths configured")
        return [{"Dataset": dataset_key, "error": "no metapaths configured"}]

    if metapath_filter:
        if metapath_filter not in metapaths:
            print(f"    [warn] --metapath not in config list; running it anyway")
        metapaths = [metapath_filter]

    # Stage C++ files once (shared across all metapaths)
    print("[2] Staging C++ files …")
    meta_dat = os.path.join(data_dir, "meta.dat")
    if os.path.exists(meta_dat):
        print("    (already staged — skipping)")
    else:
        PyGToCppAdapter(data_dir).convert(g)
        print("    staging complete")

    binary = config.CPP_EXECUTABLE
    rows: List[Dict] = []

    for i, mp in enumerate(metapaths, 1):
        print(f"\n  [{i}/{len(metapaths)}] {mp}")

        # |E*| via sparse matmul
        e_star = _e_star_from_pyg(mp, g)
        e_star_str = f"{e_star:,}" if e_star is not None else "n/a"
        print(f"    |E*| = {e_star_str}")

        e_exact: Optional[int] = None
        e_kmv: Optional[int] = None

        if not skip_cpp:
            # Compile rule for this metapath
            compile_rule_for_cpp(mp, g, data_dir, folder_name)
            rule_file = os.path.join(data_dir, f"cod-rules_{folder_name}.limit")

            # Use metapath index to avoid output file collisions
            safe_idx = str(i)
            mat_out = os.path.join(data_dir, f"table5_exact_{safe_idx}.adj")
            sketch_base = os.path.join(data_dir, f"table5_sketch_{safe_idx}")

            e_exact = _run_materialize(binary, data_dir, rule_file, mat_out, timeout)
            e_kmv = _run_sketch(binary, data_dir, rule_file, sketch_base, k, timeout)

        rows.append({
            "Dataset":   dataset_key,
            "MetaPath":  mp,
            "|V|":       n_v,
            "|E*|":      "" if e_star is None else e_star,
            "|Eexact|":  "" if e_exact is None else e_exact,
            "|Ekmv|":    "" if e_kmv is None else e_kmv,
            "|V|*k":     n_v * k,
            "rho_exact": _density(e_exact, n_v),
        })

    return rows


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


def _print_table(rows: List[Dict]) -> None:
    cols = ["Dataset", "MetaPath", "|V|", "|E*|", "|Eexact|", "|Ekmv|", "rho_exact"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
              for c in cols}
    sep = "  ".join("-" * widths[c] for c in cols)
    print("\n" + "  ".join(c.ljust(widths[c]) for c in cols))
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
        "--datasets", nargs="+", default=TABLE5_DATASETS, metavar="DATASET",
        help="Dataset keys to process (default: all 6 paper datasets)",
    )
    parser.add_argument(
        "--metapath", default=None,
        help="Run only this metapath for every specified dataset (skips all others)",
    )
    parser.add_argument(
        "--k", type=int, default=32,
        help="KMV sketch width (default: 32)",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Per-command C++ timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--skip-cpp", action="store_true",
        help="Skip C++ materialize/sketch; report only |V| and |E*| from PyG",
    )
    parser.add_argument(
        "--out-dir", default=os.path.join(project_root, "results", "table5"),
        help="Output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_rows: List[Dict] = []
    for ds_key in args.datasets:
        try:
            rows = collect_dataset(
                dataset_key=ds_key,
                metapath_filter=args.metapath,
                k=args.k,
                timeout=args.timeout,
                skip_cpp=args.skip_cpp,
            )
        except Exception as exc:
            import traceback
            print(f"[ERROR] {ds_key}: {exc}")
            traceback.print_exc()
            rows = [{"Dataset": ds_key, "error": str(exc)}]
        all_rows.extend(rows)

    valid_rows = [r for r in all_rows if "error" not in r]

    if valid_rows:
        _print_table(valid_rows)

    # CSV
    ordered_keys = ["Dataset", "MetaPath", "|V|", "|E*|", "|Eexact|", "|Ekmv|",
                    "|V|*k", "rho_exact", "error"]
    all_keys = list({k for r in all_rows for k in r})
    fieldnames = [k for k in ordered_keys if k in all_keys] + \
                 [k for k in all_keys if k not in ordered_keys]

    csv_path = os.path.join(args.out_dir, "table5_stats.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"CSV → {csv_path}")

    # TXT
    txt_path = os.path.join(args.out_dir, "table5_stats.txt")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_table(valid_rows)
    with open(txt_path, "w") as fh:
        fh.write(buf.getvalue())
    print(f"TXT → {txt_path}")


if __name__ == "__main__":
    main()
