"""
bench_mat_memory.py — Standalone materialization memory benchmark.

Compares Exact vs KMV (k=2,4,8,16,32) vs MPRW (memory-capped to each KMV level).
No GNN inference — pure materialization timing and RAM.

Runs from WSL with system Python3 (no torch required).
All three methods use the same staged .dat files and the same GNU time -v measurement.

Usage (from project root under WSL):
    python3 scripts/bench_mat_memory.py [--dataset HGB_ACM] [--metapath paper_to_term,term_to_paper] [--seed 0] [--mprw-w 200]

Output:
    results/<dataset>/bench_mat_memory.csv
    Also prints a summary table to stdout.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ = Path(__file__).resolve().parent.parent

DATASET_FOLDER = {
    "HGB_ACM":   "HGBn-ACM",
    "HGB_DBLP":  "HGBn-DBLP",
    "HGB_IMDB":  "HGBn-IMDB",
    "HGB_Freebase": "HGBn-Freebase",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_edges(filepath: str) -> int:
    """Count total directed edges in an adjacency list file."""
    n = 0
    if not os.path.exists(filepath):
        return 0
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if len(parts) > 1:
                n += len(parts) - 1
    return n


def _run(cmd: list, timeout: int = 600) -> Tuple[float, float]:
    """
    Run cmd wrapped with /usr/bin/time -v.
    Returns (algo_time_s, peak_rss_mb).

    algo_time_s: parsed from 'time: X' on stdout; falls back to wall time.
    peak_rss_mb: parsed from GNU time stderr.
    """
    full_cmd = ["/usr/bin/time", "-v"] + cmd
    t0 = time.perf_counter()
    res = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter() - t0

    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {res.returncode})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr: {res.stderr[-600:]}"
        )

    # Peak RSS from GNU time stderr
    peak_mb = 0.0
    m = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", res.stderr)
    if m:
        peak_mb = int(m.group(1)) / 1024.0

    # Algorithm time from binary stdout
    algo_t = wall
    for line in res.stdout.split("\n"):
        if line.strip().lower().startswith("time:"):
            try:
                algo_t = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
            break

    return algo_t, peak_mb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset",  default="HGB_ACM")
    parser.add_argument("--metapath", default="paper_to_term,term_to_paper")
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--mprw-w",   type=int, default=200,
                        help="Walk budget for MPRW runs (large so memory cap binds first)")
    parser.add_argument("--timeout",  type=int, default=600)
    args = parser.parse_args()

    folder   = DATASET_FOLDER.get(args.dataset, args.dataset)
    data_dir = str(PROJ / folder)
    rule     = str(PROJ / folder / f"cod-rules_{folder}.limit")
    scratch  = PROJ / "results" / args.dataset / "bench_mat_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    out_csv  = PROJ / "results" / args.dataset / "bench_mat_memory.csv"

    graph_prep = str(PROJ / "bin" / "graph_prep")
    mprw_exec  = str(PROJ / "bin" / "mprw_exec")

    if not os.path.exists(rule):
        sys.exit(f"Rule file not found: {rule}\nRun exp3_inference.py once first to stage files.")
    if not os.path.exists(graph_prep):
        sys.exit(f"graph_prep not found: {graph_prep}")
    if not os.path.exists(mprw_exec):
        sys.exit(f"mprw_exec not found: {mprw_exec}")

    rows = []

    # -----------------------------------------------------------------------
    # Exact
    # -----------------------------------------------------------------------
    print("\n[Exact] running...")
    exact_out = str(scratch / "exact.adj")
    try:
        t, mb = _run([graph_prep, "materialize", data_dir, rule, exact_out],
                     args.timeout)
        edges = _count_edges(exact_out)
        print(f"  time={t:.3f}s  ram={mb:.0f}MB  edges={edges:,}")
        rows.append({
            "Method": "Exact", "k": "", "w": "",
            "Mat_Time_s": round(t, 4), "Peak_RAM_MB": round(mb, 1),
            "Edge_Count": edges, "Status": "OK",
        })
    except Exception as e:
        print(f"  FAILED: {e}")
        rows.append({"Method": "Exact", "k": "", "w": "",
                     "Mat_Time_s": "", "Peak_RAM_MB": "", "Edge_Count": "", "Status": str(e)[:80]})
        mb = None

    # -----------------------------------------------------------------------
    # KMV for each k
    # -----------------------------------------------------------------------
    kmv_ram_by_k: dict = {}

    for k in args.k_values:
        print(f"\n[KMV k={k}] running...")
        kmv_out = str(scratch / f"kmv_k{k}.adj")
        try:
            t, mb_k = _run([graph_prep, "sketch", data_dir, rule, kmv_out,
                            str(k), "1", str(args.seed)], args.timeout)
            # graph_prep sketch appends _0 before the extension: out.adj → out_0.adj
            kmv_actual = kmv_out.replace(".adj", "_0.adj")
            if not os.path.exists(kmv_actual):
                kmv_actual = kmv_out   # fallback
            edges = _count_edges(kmv_actual)
            kmv_ram_by_k[k] = mb_k
            print(f"  time={t:.3f}s  ram={mb_k:.0f}MB  edges={edges:,}")
            rows.append({
                "Method": "KMV", "k": k, "w": "",
                "Mat_Time_s": round(t, 4), "Peak_RAM_MB": round(mb_k, 1),
                "Edge_Count": edges, "Status": "OK",
            })
        except Exception as e:
            print(f"  FAILED: {e}")
            rows.append({"Method": "KMV", "k": k, "w": "",
                         "Mat_Time_s": "", "Peak_RAM_MB": "", "Edge_Count": "",
                         "Status": str(e)[:80]})
            kmv_ram_by_k[k] = None

    # -----------------------------------------------------------------------
    # MPRW capped at each KMV memory level
    # -----------------------------------------------------------------------
    for k in args.k_values:
        mem_cap = kmv_ram_by_k.get(k)
        print(f"\n[MPRW k={k} cap={mem_cap:.0f}MB w={args.mprw_w}] running...")
        if mem_cap is None:
            print(f"  SKIPPED — no KMV RAM reference for k={k}")
            rows.append({"Method": "MPRW", "k": k, "w": args.mprw_w,
                         "Mat_Time_s": "", "Peak_RAM_MB": "", "Edge_Count": "",
                         "Status": "KMV_MISSING"})
            continue

        mprw_out = str(scratch / f"mprw_k{k}.adj")
        try:
            t, mb_m = _run([mprw_exec, "materialize", data_dir, rule, mprw_out,
                            str(args.mprw_w), str(args.seed), str(int(mem_cap))],
                           args.timeout)
            edges = _count_edges(mprw_out)
            print(f"  time={t:.3f}s  ram={mb_m:.0f}MB  edges={edges:,}  cap={mem_cap:.0f}MB")
            rows.append({
                "Method": "MPRW", "k": k, "w": args.mprw_w,
                "Mat_Time_s": round(t, 4), "Peak_RAM_MB": round(mb_m, 1),
                "Edge_Count": edges, "Status": "OK",
            })
        except Exception as e:
            print(f"  FAILED: {e}")
            rows.append({"Method": "MPRW", "k": k, "w": args.mprw_w,
                         "Mat_Time_s": "", "Peak_RAM_MB": "", "Edge_Count": "",
                         "Status": str(e)[:80]})

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    fields = ["Method", "k", "w", "Mat_Time_s", "Peak_RAM_MB", "Edge_Count", "Status"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written to: {out_csv}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "="*72)
    print(f"{'Method':<12} {'k':>4} {'w':>6} {'Time(s)':>9} {'RAM(MB)':>9} {'Edges':>12} {'Status'}")
    print("-"*72)
    for r in rows:
        t_str  = f"{r['Mat_Time_s']:>9.3f}" if r["Mat_Time_s"] != "" else f"{'—':>9}"
        mb_str = f"{r['Peak_RAM_MB']:>9.1f}" if r["Peak_RAM_MB"] != "" else f"{'—':>9}"
        e_str  = f"{int(r['Edge_Count']):>12,}" if r["Edge_Count"] != "" else f"{'—':>12}"
        print(f"{r['Method']:<12} {str(r['k']):>4} {str(r['w']):>6} {t_str} {mb_str} {e_str}  {r['Status']}")
    print("="*72)


if __name__ == "__main__":
    main()
