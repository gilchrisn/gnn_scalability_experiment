#!/usr/bin/env python3
"""
MPRW bias analysis — characterize the systematic bias induced by random-walk
sampling relative to KMV's uniform-hash sampling.

Hypotheses under test:
    H1. Label-homophily alignment.
        P(label(u)==label(v) | (u,v) retained) for Exact / KMV / MPRW.
        If MPRW > Exact > KMV, MPRW's hub bias oversamples homophilic edges.
    H2. Degree-product skew (inverse-propensity view).
        mean(deg_exact(u) * deg_exact(v)) across retained edges.
        If MPRW >> KMV, MPRW concentrates on hubs.
    H5. Spectral radius of retained adjacency.
        λ_max(Ã_MPRW) vs λ_max(Ã_KMV) vs λ_max(A_exact).
        If MPRW > KMV, retained graph has higher expansion.
    H6. Class distribution of retained edge endpoints.
        Cosine similarity between class_dist(E_retained) and class_dist(V).
        MPRW < KMV here would mean MPRW skews class coverage.

Output: results/kmv_properties/mprw_bias_<dataset>.csv  + summary JSON

Runs (WSL/Linux):
    python scripts/exp12_mprw_bias.py --datasets HGB_ACM HGB_DBLP HGB_IMDB
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.config import config
from src.data.factory import DatasetFactory

DATASETS = {
    "HGB_ACM":   {"source": "HGB", "name": "ACM",    "target": "paper",
                  "metapath": "paper_to_term,term_to_paper"},
    "HGB_DBLP":  {"source": "HGB", "name": "DBLP",   "target": "author",
                  "metapath": "author_to_paper,paper_to_author"},
    "HGB_IMDB":  {"source": "HGB", "name": "IMDB",   "target": "movie",
                  "metapath": "movie_to_keyword,keyword_to_movie"},
    "HNE_PubMed": {"source": "HNE", "name": "PubMed", "target": "disease",
                   "metapath": "disease_to_chemical,chemical_to_disease"},
}


# ---------------- CPP invocation ----------------

def _run_binary(cmd: List[str], cwd: str, timeout: int = 600) -> Tuple[float, float]:
    """Run a C++ binary wrapped with /usr/bin/time -v; return (algo_time, peak_rss_mb)."""
    full_cmd = ["/usr/bin/time", "-v"] + cmd
    t0 = time.perf_counter()
    res = subprocess.run(full_cmd, capture_output=True, text=True,
                         timeout=timeout, cwd=cwd)
    wall = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(
            f"cmd failed (exit {res.returncode}): {' '.join(cmd)}\n"
            f"stderr tail: {res.stderr[-800:]}"
        )
    peak_mb = 0.0
    m = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", res.stderr)
    if m:
        peak_mb = int(m.group(1)) / 1024.0
    algo_t = wall
    for line in res.stdout.splitlines():
        if line.strip().lower().startswith("time:"):
            try:
                algo_t = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
            break
    return algo_t, peak_mb


# ---------------- adj file ↔ COO ----------------

def load_adj(path: Path, node_offset: int, n_target: int) -> Tuple[np.ndarray, np.ndarray]:
    """Parse C++ adjacency list into (src, dst) local-index arrays. Filters non-target ids."""
    srcs: List[int] = []
    dsts: List[int] = []
    with path.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u_global = int(parts[0])
            u_local = u_global - node_offset
            if not (0 <= u_local < n_target):
                continue
            for tok in parts[1:]:
                v_global = int(tok)
                v_local = v_global - node_offset
                if 0 <= v_local < n_target and v_local != u_local:
                    srcs.append(u_local)
                    dsts.append(v_local)
    return np.asarray(srcs, dtype=np.int64), np.asarray(dsts, dtype=np.int64)


def node_offset_for_target(g_hetero, target: str) -> int:
    """Global-id offset the C++ binary applies: sum of node counts of ntypes alphabetically before `target`."""
    off = 0
    for nt in sorted(g_hetero.node_types):
        if nt == target:
            break
        off += g_hetero[nt].num_nodes
    return off


# ---------------- Bias metrics ----------------

def homophily(src: np.ndarray, dst: np.ndarray, labels: np.ndarray) -> Dict:
    """Fraction of edges (u,v) with label(u)==label(v). Only over edges where both endpoints have labels."""
    if src.size == 0:
        return {"homophily": float("nan"), "n_labeled_edges": 0}
    lu = labels[src]
    lv = labels[dst]
    mask = (lu >= 0) & (lv >= 0)
    if mask.sum() == 0:
        return {"homophily": float("nan"), "n_labeled_edges": 0}
    matches = (lu[mask] == lv[mask]).sum()
    return {"homophily": float(matches / mask.sum()),
            "n_labeled_edges": int(mask.sum())}


def degree_skew(src: np.ndarray, dst: np.ndarray, deg_exact: np.ndarray) -> Dict:
    """Mean and quantiles of deg_exact(u)*deg_exact(v) over retained edges."""
    if src.size == 0:
        return {"mean_deg_prod": float("nan"),
                "median_deg_prod": float("nan"),
                "mean_deg_u": float("nan")}
    dp = deg_exact[src].astype(np.float64) * deg_exact[dst].astype(np.float64)
    return {
        "mean_deg_prod":   float(dp.mean()),
        "median_deg_prod": float(np.median(dp)),
        "p90_deg_prod":    float(np.quantile(dp, 0.9)),
        "mean_deg_u":      float(deg_exact[src].mean()),
        "mean_deg_v":      float(deg_exact[dst].mean()),
    }


def spectral_radius(src: np.ndarray, dst: np.ndarray, n_target: int) -> float:
    """Largest-magnitude eigenvalue of the symmetric retained adjacency."""
    if src.size == 0:
        return float("nan")
    A = sp.coo_matrix((np.ones(src.size), (src, dst)), shape=(n_target, n_target))
    A = A.maximum(A.T).tocsr()  # symmetrize
    A.data = np.minimum(A.data, 1.0)
    try:
        from scipy.sparse.linalg import eigsh
        vals = eigsh(A.astype(np.float64), k=1, which="LM",
                     return_eigenvectors=False, maxiter=500, tol=1e-3)
        return float(vals[0])
    except Exception:
        return float("nan")


def class_distribution_skew(src: np.ndarray,
                            dst: np.ndarray,
                            labels: np.ndarray,
                            n_classes: int) -> Dict:
    """Cosine similarity between class distribution of retained endpoints vs population distribution."""
    if src.size == 0:
        return {"class_cos_sim": float("nan")}
    # endpoints union
    endpoints = np.concatenate([src, dst])
    le = labels[endpoints]
    le = le[le >= 0]
    if le.size == 0:
        return {"class_cos_sim": float("nan")}
    retained = np.bincount(le, minlength=n_classes).astype(np.float64)
    retained /= retained.sum()
    pop_mask = labels >= 0
    pop = np.bincount(labels[pop_mask], minlength=n_classes).astype(np.float64)
    pop /= pop.sum() if pop.sum() > 0 else 1.0
    cos = float((retained @ pop) / (np.linalg.norm(retained) * np.linalg.norm(pop) + 1e-12))
    # L1 distance too
    l1 = float(np.abs(retained - pop).sum())
    return {"class_cos_sim": cos,
            "class_L1": l1,
            "n_endpoints_with_label": int(le.size)}


# ---------------- Main per-dataset pipeline ----------------

def run_dataset(ds_key: str,
                k_values: List[int],
                w_values: List[int],
                seed: int,
                out_dir: Path,
                rerun: bool) -> List[Dict]:
    spec = DATASETS[ds_key]
    folder = config.get_folder_name(ds_key)
    staging = Path(config.STAGING_DIR) / folder
    rule_path = staging / f"cod-rules_{folder}.limit"
    dataset_dir = str(staging)

    graph_prep = str(PROJ / "bin" / "graph_prep")
    mprw_exec  = str(PROJ / "bin" / "mprw_exec")
    if not os.path.exists(graph_prep) or not os.path.exists(mprw_exec):
        raise FileNotFoundError("Missing C++ binaries; run `make` first")

    # ---- Load graph via PyG (for labels, degrees) ----
    g, _info = DatasetFactory.get_data(spec["source"], spec["name"], spec["target"])
    target = spec["target"]
    n_target = g[target].num_nodes
    offset = node_offset_for_target(g, target)
    labels = g[target].y.cpu().numpy() if hasattr(g[target], "y") else np.full(n_target, -1)
    if labels.dtype not in (np.int64, np.int32, np.int16):
        labels = labels.astype(np.int64)
    n_classes = int(labels.max() + 1) if (labels >= 0).any() else 0
    print(f"[{ds_key}] n_target={n_target}, n_classes={n_classes}, label_coverage="
          f"{(labels >= 0).mean():.2%}")

    adj_dir = staging / "mprw_bias"
    adj_dir.mkdir(parents=True, exist_ok=True)

    # ---- Materialize EXACT (once, cached) ----
    exact_adj = adj_dir / "mat_exact.adj"
    if rerun or not exact_adj.exists() or exact_adj.stat().st_size == 0:
        print(f"[{ds_key}] Exact materialize...")
        cmd = [graph_prep, "materialize", dataset_dir, str(rule_path), str(exact_adj)]
        t_exact, _ = _run_binary(cmd, cwd=dataset_dir, timeout=3600)
        print(f"[{ds_key}]   exact t={t_exact:.2f}s")
    src_e, dst_e = load_adj(exact_adj, offset, n_target)
    # Exact reference
    exact_edges = src_e.size
    deg_exact = np.bincount(src_e, minlength=n_target) if src_e.size else np.zeros(n_target, dtype=np.int64)
    print(f"[{ds_key}] exact edges={exact_edges}, deg_mean={deg_exact.mean():.1f}")

    rows: List[Dict] = []

    def _analyze(method: str, param_name: str, param_val, adj_path: Path):
        src, dst = load_adj(adj_path, offset, n_target)
        row = {
            "dataset": ds_key,
            "method": method,
            param_name: param_val,
            "edges_retained": int(src.size),
            "density_vs_exact": float(src.size / max(1, exact_edges)),
        }
        row.update(homophily(src, dst, labels))
        row.update(degree_skew(src, dst, deg_exact))
        row["spectral_radius"] = spectral_radius(src, dst, n_target)
        row.update(class_distribution_skew(src, dst, labels, max(1, n_classes)))
        rows.append(row)
        print(f"[{ds_key}] {method:5s} {param_name}={param_val}: "
              f"|E|={src.size} ({row['density_vs_exact']*100:.1f}% of exact), "
              f"homo={row.get('homophily', float('nan')):.3f}, "
              f"mean_deg_prod={row.get('mean_deg_prod', float('nan')):.1f}, "
              f"λ_max={row['spectral_radius']:.2f}")

    # Baseline: exact
    _analyze("Exact", "param", 0, exact_adj)

    # ---- KMV sweep (graph_prep appends "_<seed>" before extension) ----
    for k in k_values:
        adj_path_in = adj_dir / f"mat_kmv_k{k}.adj"
        adj_path    = adj_dir / f"mat_kmv_k{k}_{seed}.adj"
        if rerun or not adj_path.exists() or adj_path.stat().st_size == 0:
            print(f"[{ds_key}] KMV sketch k={k}...")
            cmd = [graph_prep, "sketch", dataset_dir, str(rule_path), str(adj_path_in),
                   str(k), "1", str(seed)]
            _run_binary(cmd, cwd=dataset_dir, timeout=1800)
        _analyze("KMV", "k", k, adj_path)

    # ---- MPRW sweep ----
    for w in w_values:
        adj_path = adj_dir / f"mat_mprw_w{w}_s{seed}.adj"
        if rerun or not adj_path.exists() or adj_path.stat().st_size == 0:
            print(f"[{ds_key}] MPRW w={w}...")
            cmd = [mprw_exec, "materialize", dataset_dir, str(rule_path), str(adj_path),
                   str(w), str(seed), "0"]
            _run_binary(cmd, cwd=dataset_dir, timeout=1800)
        _analyze("MPRW", "w", w, adj_path)

    # Persist
    out_csv = out_dir / f"mprw_bias_{ds_key}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Deterministic column order
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    keys_order = ["dataset", "method", "k", "w", "param",
                  "edges_retained", "density_vs_exact",
                  "homophily", "n_labeled_edges",
                  "mean_deg_prod", "median_deg_prod", "p90_deg_prod",
                  "mean_deg_u", "mean_deg_v",
                  "spectral_radius",
                  "class_cos_sim", "class_L1", "n_endpoints_with_label"]
    keys_order = [k for k in keys_order if k in all_keys]
    for k in all_keys:
        if k not in keys_order:
            keys_order.append(k)
    with out_csv.open("w", newline="") as f:
        w_csv = csv.DictWriter(f, fieldnames=keys_order, extrasaction="ignore")
        w_csv.writeheader()
        w_csv.writerows(rows)
    print(f"[{ds_key}] wrote {out_csv}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["HGB_ACM", "HGB_DBLP", "HGB_IMDB"])
    p.add_argument("--k-values", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--w-values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str,
                   default=str(PROJ / "results" / "kmv_properties"))
    p.add_argument("--rerun", action="store_true", help="force re-materialization")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    all_rows: Dict[str, List[Dict]] = {}
    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"[skip] unknown dataset {ds}")
            continue
        try:
            all_rows[ds] = run_dataset(ds, args.k_values, args.w_values,
                                       args.seed, out_dir, args.rerun)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_rows[ds] = [{"error": str(e)}]

    # Global summary: find matched-density pairs KMV(k) ≈ MPRW(w)
    summary = {}
    for ds, rs in all_rows.items():
        if not rs or "error" in rs[0]:
            summary[ds] = rs[0] if rs else {}
            continue
        exact = [r for r in rs if r["method"] == "Exact"][0]
        kmv = [r for r in rs if r["method"] == "KMV"]
        mprw = [r for r in rs if r["method"] == "MPRW"]
        # For each KMV k, find MPRW w with closest density
        matched = []
        for kr in kmv:
            best = min(mprw,
                       key=lambda r: abs(r["density_vs_exact"] - kr["density_vs_exact"]))
            pair = {
                "k": kr.get("k"),
                "kmv_density": kr["density_vs_exact"],
                "mprw_density": best["density_vs_exact"],
                "mprw_w": best.get("w"),
                "kmv_homophily": kr["homophily"],
                "mprw_homophily": best["homophily"],
                "exact_homophily": exact["homophily"],
                "kmv_mean_deg_prod": kr["mean_deg_prod"],
                "mprw_mean_deg_prod": best["mean_deg_prod"],
                "exact_mean_deg_prod": exact["mean_deg_prod"],
                "kmv_lambda_max": kr["spectral_radius"],
                "mprw_lambda_max": best["spectral_radius"],
                "exact_lambda_max": exact["spectral_radius"],
                "kmv_class_cos": kr["class_cos_sim"],
                "mprw_class_cos": best["class_cos_sim"],
                "exact_class_cos": exact["class_cos_sim"],
            }
            matched.append(pair)
        summary[ds] = {
            "exact": exact,
            "matched_pairs": matched,
        }

    (out_dir / "mprw_bias_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nFull summary → {out_dir / 'mprw_bias_summary.json'}")


if __name__ == "__main__":
    main()
