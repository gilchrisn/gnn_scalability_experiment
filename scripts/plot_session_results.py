"""Render session-wide plots from the per-run JSONs.

Outputs (under figures/sketch_session/):
    quality_per_dataset.pdf, .png    — bar chart: F1 across methods per dataset
    cost_breakdown_q2.pdf, .png      — stacked bar: KMV vs baseline total cost
    amortization_curve.pdf, .png     — total wall-clock vs Q (number of queries)
    similarity_speedup.pdf, .png     — sketch-vs-exact speedup per meta-path
    quality_vs_time.pdf, .png        — Pareto scatter (F1 vs train time)

All plots use the same data found by scripts/compile_master_results.py.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures" / "sketch_session"
FIG.mkdir(parents=True, exist_ok=True)

DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed"]
DS_LABEL = {"HGB_DBLP": "DBLP", "HGB_ACM": "ACM",
            "HGB_IMDB": "IMDB", "HNE_PubMed": "PubMed"}

# Method order + colour palette.
METHODS = [
    ("sketch_feature_mlp", "Sketch-feature\n(LoNe-MLP)", "#2ca02c"),
    ("sketch_sparsifier", "Sketch-sparsifier\n(SAGE)", "#1f77b4"),
    ("simple_hgn", "Simple-HGN\n(PyG port)", "#d62728"),
]

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_method_seeds(ds: str, method: str) -> List[dict]:
    """Walk per-method JSON files for a dataset + method and return blobs."""
    if method == "sketch_feature_mlp":
        pattern = "sketch_feature_pilot_k32_mlp_seed*.json"
    elif method == "sketch_sparsifier":
        pattern = "sketch_sparsifier_pilot_k32_seed*.json"
    elif method == "simple_hgn":
        pattern = "simple_hgn_baseline_seed*.json"
    else:
        return []
    out = []
    for p in sorted((RES / ds).glob(pattern)):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out


def agg_test_f1(blobs: List[dict]) -> tuple:
    vals = [b["test_f1"] for b in blobs if "test_f1" in b]
    if not vals:
        return (None, None, 0)
    return (statistics.fmean(vals),
            statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            len(vals))


def agg_train_time(blobs: List[dict]) -> tuple:
    vals = [b.get("train_time_s") for b in blobs
            if b.get("train_time_s") is not None]
    if not vals:
        return (None, None, 0)
    return (statistics.fmean(vals),
            statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            len(vals))


def load_amortization(ds: str) -> List[dict]:
    return [json.loads(p.read_text())
            for p in sorted((RES / ds).glob("multi_query_amortization_k32_seed*.json"))]


def load_similarity(ds: str) -> List[dict]:
    out = []
    for p in sorted((RES / ds).glob("sketch_similarity_pilot_k32_seed*.json")):
        out.append(json.loads(p.read_text()))
    return out


# ---------------------------------------------------------------------------
# Plot 1: Quality per dataset (grouped bar chart)
# ---------------------------------------------------------------------------


def plot_quality_per_dataset():
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DATASETS))
    width = 0.27
    for i, (m_key, m_label, color) in enumerate(METHODS):
        means, stds = [], []
        for ds in DATASETS:
            blobs = load_method_seeds(ds, m_key)
            mean, std, n = agg_test_f1(blobs)
            means.append(mean if mean is not None else 0.0)
            stds.append(std if std is not None else 0.0)
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, label=m_label,
               color=color, capsize=3, alpha=0.88)
        for xi, (m, s) in enumerate(zip(means, stds)):
            if m > 0:
                ax.text(x[xi] + offset, m + s + 0.012,
                        f"{m:.3f}", ha="center", va="bottom",
                        fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("Test macro-F1 (3 seeds)")
    ax.set_ylim(0, 1.05)
    ax.set_title("NC quality across consumer modes vs Simple-HGN baseline\n"
                 "(same splits, same machine, k=32)")
    ax.legend(loc="lower left", ncol=3, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "quality_per_dataset.pdf")
    plt.savefig(FIG / "quality_per_dataset.png")
    plt.close(fig)
    print(f"[fig]  {FIG / 'quality_per_dataset.png'}")


# ---------------------------------------------------------------------------
# Plot 2: Cost breakdown — stacked bar of total wall-clock at Q=2 per dataset
# ---------------------------------------------------------------------------


def plot_cost_breakdown():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rows = []
    for ds in DATASETS:
        amorts = load_amortization(ds)
        if not amorts:
            continue
        # Use seed 42 (only seed measured for amortization).
        a = amorts[0]
        rows.append({
            "ds": ds,
            "kmv_pre": a["kmv"]["precompute_s"],
            "kmv_nc":  a["kmv"]["nc_consume_s"],
            "kmv_sim": a["kmv"]["sim_consume_s"],
            "shgn_nc": a["baseline"]["shgn_nc_train_s"],
            "exact_sim": a["baseline"]["exact_sim_s"],
            "speedup": a["totals_q2_nc_sim"]["speedup_x"],
        })
    if not rows:
        print("[skip] no amortization data")
        return

    labels = []
    for r in rows:
        labels.append(f"{DS_LABEL[r['ds']]}\nKMV")
        labels.append(f"{DS_LABEL[r['ds']]}\nSHGN+exact")

    pre = []
    nc = []
    sim = []
    for r in rows:
        # KMV bar
        pre.append(r["kmv_pre"])
        nc.append(r["kmv_nc"])
        sim.append(r["kmv_sim"])
        # SHGN bar
        pre.append(0.0)
        nc.append(r["shgn_nc"])
        sim.append(r["exact_sim"])

    x = np.arange(len(labels))
    ax.bar(x, pre, label="Sketch precompute (one-time)", color="#9467bd")
    ax.bar(x, nc, bottom=pre, label="NC training / consume",
           color="#2ca02c")
    bottom2 = [p + n for p, n in zip(pre, nc)]
    ax.bar(x, sim, bottom=bottom2, label="Similarity (sketch / exact)",
           color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall-clock (s)")
    ax.set_title("Cost breakdown to serve {NC, Similarity} on the same graph\n"
                 "(Q=2 queries, single seed)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Annotate speedups above each pair.
    for i, r in enumerate(rows):
        kmv_total = r["kmv_pre"] + r["kmv_nc"] + r["kmv_sim"]
        shgn_total = r["shgn_nc"] + r["exact_sim"]
        max_y = max(kmv_total, shgn_total)
        ax.annotate(f"{r['speedup']:.2f}× faster",
                    xy=(2 * i + 0.5, max_y + 1.5),
                    ha="center", fontsize=9, color="#444")
    plt.tight_layout()
    plt.savefig(FIG / "cost_breakdown_q2.pdf")
    plt.savefig(FIG / "cost_breakdown_q2.png")
    plt.close(fig)
    print(f"[fig]  {FIG / 'cost_breakdown_q2.png'}")


# ---------------------------------------------------------------------------
# Plot 3: Amortization curve — total wall-clock as a function of Q
# ---------------------------------------------------------------------------


def plot_amortization_curve():
    """Project total cost across hypothetical Q values, given measured per-task numbers.

    The line is a projection (we measured Q=1 NC and Q=1 Sim once each;
    extrapolation assumes additional queries cost the same as their measured
    average). Caveats are spelled out in the report. The plot is illustrative,
    not a benchmark.
    """
    amorts = load_amortization("HGB_DBLP")
    if not amorts:
        print("[skip] no amortization for DBLP")
        return
    a = amorts[0]
    pre = a["kmv"]["precompute_s"]
    nc = a["kmv"]["nc_consume_s"]
    sim = a["kmv"]["sim_consume_s"]
    avg_kmv_consume = (nc + sim) / 2
    shgn_nc = a["baseline"]["shgn_nc_train_s"]
    exact_sim = a["baseline"]["exact_sim_s"]
    avg_baseline_per_q = (shgn_nc + exact_sim) / 2

    Qs = np.arange(0, 9)
    kmv = pre + Qs * avg_kmv_consume
    base = Qs * avg_baseline_per_q

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Qs, base, "-o", color="#d62728", label="Per-task baseline\n(SHGN-NC + exact-Jaccard avg)")
    ax.plot(Qs, kmv, "-s", color="#2ca02c", label="KMV (sketch + per-task consume)")
    ax.set_xlabel("Number of distinct queries served (Q)")
    ax.set_ylabel("Total wall-clock (s)")
    ax.set_title("Projected multi-query cost on HGB-DBLP\n"
                 "(KMV pays precompute once; per-task baselines pay full train each)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(linestyle="--", alpha=0.3)

    # Mark the crossover: smallest Q where kmv <= base.
    cross = next((q for q in Qs[1:] if kmv[q] <= base[q]), None)
    if cross is not None:
        ax.axvline(cross, color="#444", linestyle=":", linewidth=0.8)
        ax.annotate(f"crossover Q*={cross}",
                    xy=(cross, kmv[cross]),
                    xytext=(cross + 0.4, kmv[cross] + 4),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=0.7),
                    fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "amortization_curve.pdf")
    plt.savefig(FIG / "amortization_curve.png")
    plt.close(fig)
    print(f"[fig]  {FIG / 'amortization_curve.png'}")


# ---------------------------------------------------------------------------
# Plot 4: Similarity speedup per meta-path
# ---------------------------------------------------------------------------


def plot_similarity_speedup():
    fig, ax = plt.subplots(figsize=(8.5, 4))
    rows = []
    for ds in DATASETS:
        sims = load_similarity(ds)
        for s in sims:
            for mp, m in s["per_meta_path"].items():
                rows.append({
                    "ds": ds,
                    "label": f"{DS_LABEL[ds]}\n{mp[:18] + '…' if len(mp) > 18 else mp}",
                    "speedup": m["speedup_x"],
                    "mae": m["mae"],
                    "pearson": m["pearson"],
                })
    if not rows:
        print("[skip] no similarity data")
        return
    x = np.arange(len(rows))
    speedups = [r["speedup"] for r in rows]
    colors = ["#2ca02c" if r["pearson"] is not None and not (r["pearson"] != r["pearson"])
              else "#999" for r in rows]
    ax.bar(x, speedups, color=colors, alpha=0.9)
    for xi, r in enumerate(rows):
        pearson = r["pearson"]
        pearson_s = (f"r={pearson:.3f}" if pearson is not None and not (pearson != pearson)
                     else "(saturated)")
        ax.text(xi, r["speedup"] + 1, f"MAE {r['mae']:.3f}\n{pearson_s}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=8)
    ax.set_ylabel("Speedup of slot-Jaccard vs exact (×)")
    ax.set_title("Similarity from sketch vs exact meta-path Jaccard\n"
                 "(200 target nodes, 19 900 pairs)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "similarity_speedup.pdf")
    plt.savefig(FIG / "similarity_speedup.png")
    plt.close(fig)
    print(f"[fig]  {FIG / 'similarity_speedup.png'}")


# ---------------------------------------------------------------------------
# Plot 5: Quality vs train-time Pareto scatter
# ---------------------------------------------------------------------------


def plot_quality_vs_time():
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        for m_key, m_label, color in METHODS:
            blobs = load_method_seeds(ds, m_key)
            for b in blobs:
                tt = b.get("train_time_s")
                f1 = b.get("test_f1")
                if tt is None or f1 is None:
                    continue
                ax.scatter(tt, f1, color=color, s=70, alpha=0.85,
                           edgecolor="black", linewidth=0.4,
                           label=m_label.replace("\n", " "))
        ax.set_title(DS_LABEL[ds])
        ax.set_xlabel("Train wall-clock (s)")
        ax.set_ylabel("Test macro-F1")
        ax.grid(linestyle="--", alpha=0.3)
        # Dedup legend.
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        if ds == DATASETS[0]:
            ax.legend(seen.values(), seen.keys(),
                      loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Quality–cost trade-off, per (dataset, method, seed)",
                 y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / "quality_vs_time.pdf", bbox_inches="tight")
    plt.savefig(FIG / "quality_vs_time.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig]  {FIG / 'quality_vs_time.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    plot_quality_per_dataset()
    plot_cost_breakdown()
    plot_amortization_curve()
    plot_similarity_speedup()
    plot_quality_vs_time()
    print(f"\nAll figures -> {FIG}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
