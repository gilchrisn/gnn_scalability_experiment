"""bench_scale_validation.py — overnight scalability test on OGB_MAG.

Materialization-only benchmark (no SAGE training/inference). For each
(dataset, metapath, method, parameter, seed):
    runs the C++ binary with /usr/bin/time -v
    captures: edges, mat_time_s, peak_rss_mb
    appends one row to results/scale_validation.csv

This is the scalability claim: KMV/KGRW/MPRW give bounded materialization
cost where Exact OOMs / takes hours / produces billions of edges. We
report curves of edge count and cost as the parameter (k or w) sweeps.

Design notes:
  - Restages cleanly per dataset (PyG → .dat + qnodes ALL).
  - Tries Exact first per metapath; on failure (timeout/OOM) records and continues.
  - Each (method, param, seed) wrapped in try/except — one failure does
    NOT kill the rest of the run.
  - Resume-safe: skips rows already in CSV.

Usage
-----
    python scripts/bench_scale_validation.py \\
        --datasets OGB_MAG \\
        --metapaths "rev_writes,writes" "rev_writes,writes,rev_writes,writes" \\
        --k-values 4 8 16 32 64 128 256 512 1024 \\
        --w-values 2 4 8 16 32 64 128 256 512 1024 \\
        --seeds 3 \\
        --exact-timeout 1800 \\
        --method-timeout 600
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
import types as _t
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_ts = _t.ModuleType("torch_sparse"); _ts.spspmm = None
sys.modules.setdefault("torch_sparse", _ts)
warnings.filterwarnings("ignore")

from src.config import config
from src.data import DatasetFactory
from src.bridge.converter import PyGToCppAdapter
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes


CSV_FIELDS = ["Dataset", "MetapathLen", "Metapath", "Method",
              "k", "w", "Seed",
              "Edge_Count", "Mat_Time_s", "Peak_RSS_MB", "Status"]


def _wsl(cmd_str: str) -> list[str]:
    if sys.platform == "win32":
        return ["wsl", "--exec", "bash", "-c", cmd_str]
    return ["bash", "-c", cmd_str]


def _run_timed(cmd_inner: str, timeout_s: int,
                max_retries: int = 2) -> tuple[bool, float, float, str]:
    """Run `/usr/bin/time -v <cmd_inner>` with cwd=ROOT, retry on transient errors.

    cmd_inner is a shell string with paths RELATIVE to project root.
    No `cd` is used; subprocess.cwd handles directory.

    Retries up to max_retries on transient FS errors (Stale handle, IO error,
    intermittent missing files), useful on networked filesystems.
    """
    cmd_str = f"/usr/bin/time -v {cmd_inner}"
    cmd_argv = _wsl(cmd_str)
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            r = subprocess.run(cmd_argv, capture_output=True, text=True,
                               timeout=timeout_s, cwd=str(ROOT))
        except subprocess.TimeoutExpired:
            return (False, 0.0, 0.0, f"TIMEOUT ({timeout_s}s)")
        except Exception as e:
            last_err = f"EXC: {type(e).__name__} {str(e)[:80]}"
            time.sleep(2); continue

        if r.returncode == 0:
            break

        last_err = (r.stderr or r.stdout or "")[-200:]
        # Retry on transient filesystem errors only
        if attempt < max_retries and (
            "Input/output error" in last_err or
            "Stale file handle"  in last_err
        ):
            time.sleep(2 + attempt * 3)
            continue
        return (False, 0.0, 0.0, f"EXIT={r.returncode} {last_err}")

    if r.returncode != 0:
        return (False, 0.0, 0.0, f"EXIT={r.returncode} {last_err}")

    # Pull binary's own time
    bin_time = 0.0
    for line in r.stdout.splitlines():
        if line.startswith("time:"):
            try: bin_time = float(line.split(":", 1)[1].strip())
            except ValueError: pass

    # Pull peak RSS from GNU time -v
    peak_rss_mb = 0.0
    for line in r.stderr.splitlines():
        if "Maximum resident set size" in line:
            try:
                kb = int(line.rsplit(":", 1)[1].strip())
                peak_rss_mb = kb / 1024.0
            except (ValueError, IndexError):
                pass

    return (True, bin_time, peak_rss_mb, "OK")


def _count_edges(adj_path_wsl: str) -> int:
    """Edges in the materialised adj — uniform across Exact/KMV/MPRW.

    Returns |{ unordered {u, v} : v ∈ adj[u], u ≠ v }|. See
    `bench_utils.count_unique_undirected_edges` for full rationale on why
    the per-method writer conventions force this canonicalisation.
    """
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_utils import AWK_UNIQUE_UNDIR_EDGES
    try:
        r = subprocess.run(_wsl(f"{AWK_UNIQUE_UNDIR_EDGES} {adj_path_wsl}"),
                           capture_output=True, text=True, timeout=120)
        s = r.stdout.strip()
        return int(s) if s.isdigit() else 0
    except Exception:
        return 0


def _restage(dataset: str, folder: str, metapath: str) -> tuple[int, int]:
    """Restage .dat + qnodes + rule. Returns (n_target, total_nodes)."""
    cfg = config.get_dataset_config(dataset)
    g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    n_target = g[cfg.target_node].num_nodes
    total = sum(g[nt].num_nodes for nt in g.node_types)
    print(f"  [restage] {dataset}: target={cfg.target_node} n={n_target:,} total={total:,}")
    PyGToCppAdapter(f"staging/{folder}").convert(g)
    compile_rule_for_cpp(metapath, g, f"staging/{folder}", folder)
    generate_qnodes(f"staging/{folder}", folder, cfg.target_node, g,
                    sample_size=n_target)
    return n_target, total


# ─── Per-row driver ────────────────────────────────────────────────────────

def _run_exact(folder: str, exact_adj_wsl: str,
                timeout_s: int) -> tuple[int, float, float, str]:
    cmd = (f"bin/graph_prep materialize staging/{folder} "
           f"staging/{folder}/cod-rules_{folder}.limit {exact_adj_wsl}")
    ok, t, rss, err = _run_timed(cmd, timeout_s)
    edges = _count_edges(exact_adj_wsl) if ok else 0
    return edges, t, rss, ("OK" if ok else err)


def _run_kmv(folder: str, out_base_wsl: str, k: int, seed: int,
              timeout_s: int) -> tuple[int, float, float, str]:
    cmd = (f"bin/graph_prep sketch staging/{folder} "
           f"staging/{folder}/cod-rules_{folder}.limit {out_base_wsl} {k} 1 {seed}")
    ok, t, rss, err = _run_timed(cmd, timeout_s)
    edges = _count_edges(out_base_wsl + "_0") if ok else 0
    return edges, t, rss, ("OK" if ok else err)


def _run_mprw(folder: str, out_adj_wsl: str, w: int, seed: int,
               timeout_s: int) -> tuple[int, float, float, str]:
    cmd = (f"bin/mprw_exec materialize staging/{folder} "
           f"staging/{folder}/cod-rules_{folder}.limit {out_adj_wsl} {w} {seed}")
    ok, t, rss, err = _run_timed(cmd, timeout_s)
    edges = _count_edges(out_adj_wsl) if ok else 0
    return edges, t, rss, ("OK" if ok else err)


def _run_kgrw(folder: str, out_adj_wsl: str, k: int, w: int, seed: int,
               timeout_s: int) -> tuple[int, float, float, str]:
    cmd = (f"bin/mprw_exec kgrw staging/{folder} "
           f"staging/{folder}/cod-rules_{folder}.limit {out_adj_wsl} {k} {w} {seed}")
    ok, t, rss, err = _run_timed(cmd, timeout_s)
    edges = _count_edges(out_adj_wsl) if ok else 0
    return edges, t, rss, ("OK" if ok else err)


# ─── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["OGB_MAG"])
    p.add_argument("--metapaths", nargs="+", required=True,
                   help="Comma-separated relation chains (e.g. 'rev_writes,writes')")
    p.add_argument("--k-values", type=int, nargs="+",
                   default=[4, 8, 16, 32, 64, 128, 256, 512, 1024])
    p.add_argument("--w-values", type=int, nargs="+",
                   default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--exact-timeout", type=int, default=3600,
                   help="seconds (default 3600 = 1h)")
    p.add_argument("--method-timeout", type=int, default=900,
                   help="seconds for one KMV/MPRW/KGRW run (default 900 = 15min)")
    p.add_argument("--skip-exact",  action="store_true")
    p.add_argument("--skip-kmv",    action="store_true")
    p.add_argument("--skip-mprw",   action="store_true")
    p.add_argument("--skip-kgrw",   action="store_true", default=True,
                   help="default true; pass --no-skip-kgrw to enable KGRW grid")
    p.add_argument("--no-skip-kgrw", dest="skip_kgrw", action="store_false")
    p.add_argument("--csv-out", default="results/scale_validation.csv")
    args = p.parse_args()

    seed_list = list(range(args.seed_base, args.seed_base + args.seeds))
    out_csv = Path(args.csv_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Resume-safe: load existing rows
    done: set = set()
    if out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["Dataset"], r["Metapath"], r["Method"],
                          r.get("k", ""), r.get("w", ""), r["Seed"]))

    fout = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS)
    if out_csv.stat().st_size == 0:
        writer.writeheader()

    def _emit(row: dict) -> None:
        writer.writerow(row)
        fout.flush()
        done.add((row["Dataset"], row["Metapath"], row["Method"],
                  row.get("k", ""), row.get("w", ""), row["Seed"]))

    print(f"[scale-bench] CSV: {out_csv}  existing rows: {len(done)}")

    for ds in args.datasets:
        folder = config.get_folder_name(ds)
        for mp in args.metapaths:
            mp_len = len(mp.split(","))
            print(f"\n{'='*70}\n{ds}  |  {mp_len}-hop  |  {mp}\n{'='*70}")

            try:
                _restage(ds, folder, mp)
            except Exception as e:
                print(f"  [restage] FAIL: {e}")
                continue

            # --- Exact ---
            if not args.skip_exact:
                key = (ds, mp, "Exact", "", "", str(args.seed_base))
                if key not in done:
                    print(f"  [Exact] running (timeout {args.exact_timeout}s)...")
                    exact_adj = f"/tmp/exact_{ds}_{mp_len}h.adj"
                    e, t, rss, status = _run_exact(folder, exact_adj, args.exact_timeout)
                    print(f"    edges={e:,}  time={t:.1f}s  rss={rss:.0f}MB  status={status}")
                    _emit({"Dataset": ds, "MetapathLen": mp_len, "Metapath": mp,
                           "Method": "Exact", "k": "", "w": "", "Seed": args.seed_base,
                           "Edge_Count": e, "Mat_Time_s": round(t, 3),
                           "Peak_RSS_MB": round(rss, 1), "Status": status[:60]})

            # --- KMV ---
            if not args.skip_kmv:
                for k in args.k_values:
                    for s in seed_list:
                        key = (ds, mp, "KMV", str(k), "", str(s))
                        if key in done: continue
                        out_base = f"/tmp/kmv_{ds}_{mp_len}h_k{k}_s{s}"
                        e, t, rss, status = _run_kmv(folder, out_base, k, s, args.method_timeout)
                        print(f"  [KMV k={k:5d} s={s}] edges={e:,}  time={t:.2f}s  rss={rss:.0f}MB  {status}")
                        _emit({"Dataset": ds, "MetapathLen": mp_len, "Metapath": mp,
                               "Method": "KMV", "k": k, "w": "", "Seed": s,
                               "Edge_Count": e, "Mat_Time_s": round(t, 3),
                               "Peak_RSS_MB": round(rss, 1), "Status": status[:60]})

            # --- MPRW ---
            if not args.skip_mprw:
                for w in args.w_values:
                    for s in seed_list:
                        key = (ds, mp, "MPRW", "", str(w), str(s))
                        if key in done: continue
                        out_adj = f"/tmp/mprw_{ds}_{mp_len}h_w{w}_s{s}.adj"
                        e, t, rss, status = _run_mprw(folder, out_adj, w, s, args.method_timeout)
                        print(f"  [MPRW w={w:5d} s={s}] edges={e:,}  time={t:.2f}s  rss={rss:.0f}MB  {status}")
                        _emit({"Dataset": ds, "MetapathLen": mp_len, "Metapath": mp,
                               "Method": "MPRW", "k": "", "w": w, "Seed": s,
                               "Edge_Count": e, "Mat_Time_s": round(t, 3),
                               "Peak_RSS_MB": round(rss, 1), "Status": status[:60]})

            # --- KGRW (off by default; enable with --no-skip-kgrw) ---
            if not args.skip_kgrw:
                for k in args.k_values:
                    for w in args.w_values:
                        for s in seed_list:
                            key = (ds, mp, "KGRW", str(k), str(w), str(s))
                            if key in done: continue
                            out_adj = f"/tmp/kgrw_{ds}_{mp_len}h_k{k}_w{w}_s{s}.adj"
                            e, t, rss, status = _run_kgrw(folder, out_adj, k, w, s,
                                                          args.method_timeout)
                            print(f"  [KGRW k={k:4d} w={w:5d} s={s}] edges={e:,}  time={t:.2f}s  rss={rss:.0f}MB  {status}")
                            _emit({"Dataset": ds, "MetapathLen": mp_len, "Metapath": mp,
                                   "Method": "KGRW", "k": k, "w": w, "Seed": s,
                                   "Edge_Count": e, "Mat_Time_s": round(t, 3),
                                   "Peak_RSS_MB": round(rss, 1), "Status": status[:60]})

    fout.close()
    print(f"\n[scale-bench] Done. CSV: {out_csv}")


if __name__ == "__main__":
    main()
