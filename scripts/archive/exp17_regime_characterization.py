#!/usr/bin/env python3
"""
Regime characterization: link theoretical predictor (endpoint degree spread ρ
or tail-fraction τ) to empirical KGRW-vs-MPRW CKA/F1 gap.

Inputs:
    results/<DS>/kgrw_bench.csv                 (from bench_kgrw.py)
    results/<DS>/exact_<metapath_slug>.adj     (from graph_prep materialize)

For each dataset:
    1. Compute ρ = p_max / p_min of L-step random walk stationary distribution
       on N_L(v), averaged over sources. Approximated via endpoint-degree
       histogram of the exact materialized adjacency.
    2. Compute τ = fraction of sources u with r_L(u) ≤ k (tail-fraction at k=4).
    3. From kgrw_bench.csv, aggregate KGRW-vs-MPRW CKA/F1 gap at each
       (L, k, w_prime) cell vs matched-density MPRW.
    4. Produce: per-dataset regime summary CSV + one unified plot.

Outputs:
    results/kmv_properties/regime_summary.csv
    figures/kmv_properties/regime_predictor_vs_empirical.pdf
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
RES  = PROJ / "results" / "kmv_properties"
FIG  = PROJ / "figures" / "kmv_properties"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# Datasets and their primary KGRW-bench meta-paths
# (the ones actually exercised by bench_kgrw.py — L=2)
DATASETS = {
    "HGB_ACM":  {"adj": "exact_papap.adj",  "short": "PAPAP"},
    "HGB_DBLP": {"adj": "exact_apapa.adj",  "short": "APAPA"},
    "HGB_IMDB": {"adj": "exact_mdmdm.adj",  "short": "MDMDM"},
}


# ---------- Stage 1: compute endpoint degree spread ρ ----------

def load_exact_adj(path: Path) -> Dict[int, List[int]]:
    """Read adjacency list: {src → [dst, ...]}. Skip empty lines."""
    adj: Dict[int, List[int]] = {}
    with path.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            src = int(parts[0])
            dsts = [int(t) for t in parts[1:]]
            adj[src] = dsts
    return adj


def compute_rho(adj: Dict[int, List[int]], n_sources_sample: int = 200,
                rng: np.random.Generator | None = None) -> Tuple[float, float]:
    """
    ρ = mean over sources of (p_max/p_min) where p_u = freq(endpoint u in N(src)).
    For meta-path counting adjacency (multi-edges reduced to set), we approximate
    the stationary distribution by endpoint frequency in the adjacency list.

    Returns (mean_rho, median_rho).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    srcs = list(adj.keys())
    if len(srcs) > n_sources_sample:
        srcs = rng.choice(srcs, size=n_sources_sample, replace=False).tolist()
    ratios = []
    for s in srcs:
        nbrs = adj[s]
        if len(nbrs) < 2:
            continue
        # Endpoint frequencies (for a simple adj list, each endpoint appears once,
        # so p is uniform. For a MULTI-edge KGRW source, p varies. Since we dedup'd
        # in the exact materializer, we fall back to endpoint degree as surrogate:
        # each endpoint's "weight" in a random walk is proportional to its degree).
        nbr_degs = np.array([len(adj.get(n, [])) for n in nbrs], dtype=np.float64)
        if nbr_degs.min() <= 0:
            continue
        p = nbr_degs / nbr_degs.sum()
        p_max, p_min = float(p.max()), float(p.min())
        if p_min > 0:
            ratios.append(p_max / p_min)
    if not ratios:
        return float("nan"), float("nan")
    return float(np.mean(ratios)), float(np.median(ratios))


# ---------- Stage 2: tail-fraction from exact adj ----------

def compute_tail_fraction(adj: Dict[int, List[int]], k: int) -> float:
    """
    τ = fraction of sources u with |N(u)| ≤ k. Upper bound on the fraction
    of sources for which KMV's k-sketch is exact (no sampling loss).
    """
    if not adj:
        return float("nan")
    counts = np.array([len(v) for v in adj.values()])
    return float((counts <= k).mean())


# ---------- Stage 3: aggregate kgrw_bench CKA/F1 gap ----------

def load_kgrw_bench(ds: str) -> pd.DataFrame | None:
    p = PROJ / "results" / ds / "kgrw_bench.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, low_memory=False)
    # schema: Dataset,L,Method,k,w_prime,Seed,Edge_Count,Mat_Time_s,Macro_F1,CKA,Pred_Agreement
    return df


def aggregate_gap(df: pd.DataFrame, L: int = 2) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    At fixed L, aggregate KGRW and MPRW rows across seeds (mean).
    Match KGRW(k,w') to MPRW(w) by closest Edge_Count.

    Returns:
        summary dict — key stats including MIN-BUDGET gap (not median)
                       and LARGEST GAP cell (the winning regime).
        detail DataFrame — one row per KGRW(k, w') with matched MPRW.
    """
    if df.empty:
        return {}, pd.DataFrame()
    df = df[df["L"] == L].copy()
    if df.empty:
        return {}, pd.DataFrame()
    for col in ["Edge_Count", "Macro_F1", "CKA", "Pred_Agreement", "Mat_Time_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mprw = (df[df["Method"] == "MPRW"]
            .groupby("w_prime", dropna=False).mean(numeric_only=True).reset_index())
    kgrw = (df[df["Method"] == "KGRW"]
            .groupby(["k", "w_prime"]).mean(numeric_only=True).reset_index())
    if mprw.empty or kgrw.empty:
        return {}, pd.DataFrame()
    gaps = []
    for _, kr in kgrw.iterrows():
        diffs = (mprw["Edge_Count"] - kr["Edge_Count"]).abs()
        mr = mprw.iloc[diffs.idxmin()]
        gaps.append({
            "k": int(kr["k"]),
            "w_prime": int(kr["w_prime"]),
            "kgrw_edges": kr["Edge_Count"],
            "mprw_edges": mr["Edge_Count"],
            "kgrw_cka":   kr["CKA"],
            "mprw_cka":   mr["CKA"],
            "cka_gap":    kr["CKA"] - mr["CKA"],
            "kgrw_f1":    kr["Macro_F1"],
            "mprw_f1":    mr["Macro_F1"],
            "f1_gap":     kr["Macro_F1"] - mr["Macro_F1"],
            "pa_gap":     kr["Pred_Agreement"] - mr["Pred_Agreement"],
            "kgrw_time":  kr["Mat_Time_s"],
            "mprw_time":  mr["Mat_Time_s"],
        })
    gdf = pd.DataFrame(gaps)
    gdf = gdf.sort_values("kgrw_edges").reset_index(drop=True)

    # Min-budget cell: the KGRW configuration with fewest edges
    min_row = gdf.iloc[0]
    # Best-gap cell: the KGRW configuration where KGRW beats MPRW by most
    best_row = gdf.loc[gdf["cka_gap"].idxmax()]

    # Also compute bottom-quartile (true low-budget regime)
    q25 = gdf["kgrw_edges"].quantile(0.25)
    low_q = gdf[gdf["kgrw_edges"] <= q25]

    return {
        "min_budget_k":          int(min_row["k"]),
        "min_budget_w":          int(min_row["w_prime"]),
        "min_budget_edges":      float(min_row["kgrw_edges"]),
        "min_budget_cka_gap":    float(min_row["cka_gap"]),
        "min_budget_f1_gap":     float(min_row["f1_gap"]),
        "best_gap_k":            int(best_row["k"]),
        "best_gap_w":            int(best_row["w_prime"]),
        "best_gap_edges":        float(best_row["kgrw_edges"]),
        "best_gap_cka":          float(best_row["cka_gap"]),
        "q25_mean_cka_gap":      float(low_q["cka_gap"].mean()),
        "q25_mean_f1_gap":       float(low_q["f1_gap"].mean()),
        "all_mean_cka_gap":      float(gdf["cka_gap"].mean()),
        "n_pairs":               int(len(gdf)),
    }, gdf


def main():
    rows = []
    gap_detail = {}
    for ds, meta in DATASETS.items():
        adj_path = PROJ / "results" / ds / meta["adj"]
        if not adj_path.exists():
            print(f"[{ds}] {meta['adj']} missing — skipping")
            continue
        print(f"[{ds}] loading {adj_path.name}...")
        adj = load_exact_adj(adj_path)
        rho_mean, rho_med = compute_rho(adj)
        tau4 = compute_tail_fraction(adj, k=4)
        tau8 = compute_tail_fraction(adj, k=8)
        tau16 = compute_tail_fraction(adj, k=16)
        tau32 = compute_tail_fraction(adj, k=32)
        deg_arr = np.array([len(v) for v in adj.values()])
        print(f"  ρ_mean={rho_mean:.2f}  ρ_median={rho_med:.2f}  "
              f"τ(k=4)={tau4:.2f}  τ(k=32)={tau32:.2f}  "
              f"mean_deg={deg_arr.mean():.1f}  max_deg={deg_arr.max()}")

        bench = load_kgrw_bench(ds)
        if bench is None:
            print(f"  no kgrw_bench.csv")
            continue
        gaps, detail = aggregate_gap(bench, L=2)
        gap_detail[ds] = detail
        print(f"  min-budget ({gaps.get('min_budget_edges', 0):.0f} edges): "
              f"CKA gap = {gaps.get('min_budget_cka_gap', float('nan')):+.4f}  "
              f"F1 gap = {gaps.get('min_budget_f1_gap', float('nan')):+.4f}")
        print(f"  best cell (k={gaps.get('best_gap_k',0)}, w'={gaps.get('best_gap_w',0)}, "
              f"{gaps.get('best_gap_edges', 0):.0f} edges): CKA gap = {gaps.get('best_gap_cka', float('nan')):+.4f}")
        print(f"  Q25 (lowest-quartile edges) mean CKA gap = {gaps.get('q25_mean_cka_gap', float('nan')):+.4f}")

        rows.append({
            "dataset": ds,
            "metapath": meta["short"],
            "n_sources": len(adj),
            "mean_degree": float(deg_arr.mean()),
            "max_degree":  int(deg_arr.max()),
            "rho_mean":    rho_mean,
            "rho_median":  rho_med,
            "tau_k4":      tau4,
            "tau_k8":      tau8,
            "tau_k16":     tau16,
            "tau_k32":     tau32,
            **gaps,
        })

    out_csv = RES / "regime_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n→ {out_csv}")

    # ---- Plot: theoretical predictor vs empirical gap ----
    if not rows:
        return
    df = pd.DataFrame(rows).dropna(subset=["tau_k4", "min_budget_cka_gap"])
    if df.empty:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"HGB_ACM": "#1f77b4", "HGB_DBLP": "#2ca02c", "HGB_IMDB": "#d62728"}

    # Panel 1: τ vs min-budget CKA gap (the actual winning regime)
    ax = axes[0]
    for _, r in df.iterrows():
        c = colors.get(r["dataset"], "gray")
        ax.scatter(r["tau_k4"], r["min_budget_cka_gap"], s=200, color=c,
                   label=f"{r['dataset']} ({r['metapath']})", zorder=3)
        ax.annotate(f"ρ̄={r['rho_mean']:.0f}\nedges={r['min_budget_edges']:.0f}",
                    (r["tau_k4"], r["min_budget_cka_gap"]),
                    textcoords="offset points", xytext=(10, 10), fontsize=8)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel(r"tail-fraction $\tau(k=4) = P(|N_L(v)| \leq k)$")
    ax.set_ylabel("min-budget KGRW − MPRW CKA gap (L=2)")
    ax.set_title("Regime predictor vs KGRW advantage (smallest k, w'=1)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 2: ρ vs gap
    ax = axes[1]
    for _, r in df.iterrows():
        c = colors.get(r["dataset"], "gray")
        ax.scatter(r["rho_mean"], r["min_budget_cka_gap"], s=200, color=c,
                   label=r['dataset'], zorder=3)
        ax.annotate(f"τ={r['tau_k4']:.2f}",
                    (r["rho_mean"], r["min_budget_cka_gap"]),
                    textcoords="offset points", xytext=(10, 10), fontsize=9)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"endpoint degree spread $\bar\rho = E[p_{\max}/p_{\min}]$")
    ax.set_ylabel("mean KGRW − MPRW CKA gap (low budget, L=2)")
    ax.set_title(r"Lemma 2 predictor: high $\rho$ → MPRW overhead → KMV helps")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    plt.suptitle("Theory → empirics: where KMV-guided walks beat MPRW", fontsize=11)
    plt.tight_layout()
    out_pdf = FIG / "regime_predictor_vs_empirical.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"→ {out_pdf}")

    # Sum-up print
    print("\n=== REGIME SUMMARY (sorted by tail-fraction) ===")
    df_sorted = df.sort_values("tau_k4")
    print(df_sorted[["dataset", "metapath", "tau_k4", "rho_mean",
                     "min_budget_edges", "min_budget_cka_gap",
                     "best_gap_cka", "best_gap_edges"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Per-dataset detail tables
    for ds, detail in gap_detail.items():
        if detail.empty:
            continue
        print(f"\n--- {ds}: KGRW cells sorted by edge count ---")
        print(detail[["k", "w_prime", "kgrw_edges", "mprw_edges",
                      "kgrw_cka", "mprw_cka", "cka_gap",
                      "kgrw_f1", "mprw_f1", "f1_gap"]]
              .head(12).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- Pareto frontier plots (time vs F1, time vs CKA) ----
    plot_pareto(list(DATASETS.keys()))


def plot_pareto(ds_keys):
    """Per-dataset Pareto plot: (time, F1) and (time, CKA) for MPRW vs KGRW."""
    fig, axes = plt.subplots(2, len(ds_keys), figsize=(5 * len(ds_keys), 8),
                             sharey="row")
    if len(ds_keys) == 1:
        axes = axes[:, np.newaxis]
    colors = {"MPRW": "#d62728", "KGRW": "#2ca02c"}
    markers = {"MPRW": "s", "KGRW": "o"}

    for col_i, ds in enumerate(ds_keys):
        p = PROJ / "results" / ds / "kgrw_bench.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        df = df[df["L"] == 2].copy()
        for c in ["Edge_Count", "Mat_Time_s", "Macro_F1", "CKA"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        for row_i, (metric, metric_label) in enumerate(
            [("Macro_F1", "Macro F1"), ("CKA", "Output CKA vs Exact")]
        ):
            ax = axes[row_i, col_i]
            for method in ["MPRW", "KGRW"]:
                sub = df[df["Method"] == method]
                if sub.empty:
                    continue
                if method == "MPRW":
                    agg = sub.groupby("w_prime")[["Mat_Time_s", metric]].mean().reset_index()
                else:
                    agg = sub.groupby(["k", "w_prime"])[["Mat_Time_s", metric]].mean().reset_index()
                agg = agg.sort_values("Mat_Time_s")
                ax.plot(agg["Mat_Time_s"], agg[metric],
                        marker=markers[method], color=colors[method],
                        label=method, alpha=0.85, markersize=6)
            ax.set_xscale("log")
            ax.set_xlabel("materialization time (s)")
            if col_i == 0:
                ax.set_ylabel(metric_label)
            if row_i == 0:
                ax.set_title(ds)
            ax.grid(True, which="both", alpha=0.3)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=9, loc="best")

    plt.suptitle("Pareto frontier: time vs quality (L=2, averaged across seeds)",
                 fontsize=11)
    plt.tight_layout()
    out = FIG / "pareto_time_vs_quality.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
