#!/usr/bin/env python3
"""
Plots for the KMV invariance + MPRW-bias experiments (exp11, exp12).

Inputs:
    results/kmv_properties/i{2,3,4,5,6}_<DS>.csv
    results/kmv_properties/summary_<DS>.json
    results/kmv_properties/mprw_bias_<DS>.csv

Outputs:
    figures/kmv_properties/*.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
RES  = PROJ / "results" / "kmv_properties"
FIG  = PROJ / "figures" / "kmv_properties"
FIG.mkdir(parents=True, exist_ok=True)

DS_ORDER = ["HGB_ACM", "HGB_DBLP", "HGB_IMDB"]
COLOR = {"Exact": "#1f77b4", "KMV": "#2ca02c", "MPRW": "#d62728"}


# ---------------- Invariance plots ----------------

def plot_I4_slope_correlation(datasets: List[str]) -> None:
    """Slope of log max(K_L) vs slope of log |N_L(v)| — expansion-decay invariant."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        path = RES / f"i4_{ds}.csv"
        if not path.exists():
            ax.set_title(f"{ds} (no data)")
            continue
        df = pd.read_csv(path).dropna()
        if df.empty:
            ax.set_title(f"{ds} (empty)")
            continue
        x = df["slope_log_true_card"]
        y = df["slope_log_max"]
        ax.scatter(x, y, alpha=0.4, s=10, color="#2ca02c")
        r = np.corrcoef(x, y)[0, 1] if len(x) > 2 else float("nan")
        # y = -x reference line
        lo = float(min(x.min(), -y.max()))
        hi = float(max(x.max(), -y.min()))
        ax.plot([lo, hi], [-lo, -hi], "k--", alpha=0.5, label="y = -x (theory)")
        ax.set_xlabel(r"slope of $\log|N_L(v)|$ vs $L$")
        ax.set_ylabel(r"slope of $\log\max K_L[v]$ vs $L$")
        ax.set_title(f"{ds}  (r = {r:.3f}, n={len(df)})")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.legend(fontsize=8, loc="best")
    plt.suptitle("I4 — Expansion-decay invariant:  "
                 r"$\log\max K_L[v] \approx \log k - \log|N_L(v)|$", fontsize=12)
    plt.tight_layout()
    out = FIG / "I4_expansion_decay.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_I5_jaccard(datasets: List[str]) -> None:
    """Estimated vs true Jaccard, per-depth panels."""
    fig, axes = plt.subplots(len(datasets), 4, figsize=(16, 4 * len(datasets)),
                             sharex="col", sharey="col")
    if len(datasets) == 1:
        axes = axes[np.newaxis, :]
    for i, ds in enumerate(datasets):
        path = RES / f"i5_{ds}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for j, L in enumerate([1, 2, 3, 4]):
            ax = axes[i, j]
            sub = df[df["L"] == L]
            if sub.empty:
                ax.set_title(f"L={L} (empty)")
                continue
            x = sub["true_jac"].values
            y = sub["est_jac"].values
            ax.scatter(x, y, alpha=0.3, s=6, color="#2ca02c")
            lo, hi = 0.0, max(0.001, float(max(x.max(), y.max())))
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
            r = np.corrcoef(x, y)[0, 1] if len(x) > 2 else float("nan")
            bias = (y - x).mean()
            ax.set_title(f"{ds} L={L}\nr={r:.3f}, bias={bias:+.3g}", fontsize=9)
            if j == 0:
                ax.set_ylabel(r"estimated $\hat{J}=|K\cap K'|/|K\cup K'|$")
            if i == len(datasets) - 1:
                ax.set_xlabel(r"true Jaccard $J(N_L(u), N_L(v))$")
    plt.suptitle("I5 — Multi-hop Jaccard preservation "
                 r"($\hat{J}$ vs true; $\hat{J}$ systematically underestimates at intermediate $L$)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = FIG / "I5_jaccard_preservation.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_I2_tightness(datasets: List[str]) -> None:
    """Tightness rate vs |N_L(v)| size."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    summary = []
    for ds in datasets:
        path = RES / f"i2_{ds}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for L, sub in df.groupby("L"):
            tight = (sub["is_tight"].astype(float).mean())
            summary.append({"dataset": ds, "L": int(L),
                            "frac_tight": tight,
                            "mean_neigh_size": sub["true_neighborhood_size"].mean(),
                            "n": len(sub)})
    s = pd.DataFrame(summary)
    for ds in datasets:
        sub = s[s.dataset == ds]
        ax.plot(sub["L"], sub["frac_tight"], marker="o", label=ds)
    ax.set_xlabel("depth L (meta-path iterations)")
    ax.set_ylabel("fraction tight (sketch == exact KMV)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color="gray", ls="--", lw=0.5)
    ax.legend()
    ax.set_title("I2 — Propagation tightness empirically 100% (no winners lost)")
    plt.tight_layout()
    out = FIG / "I2_tightness.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ---------------- MPRW bias plots ----------------

def plot_H1_homophily(datasets: List[str]) -> None:
    """Homophily vs density — KMV vs MPRW curves, with Exact baseline."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        path = RES / f"mprw_bias_{ds}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ex = df[df.method == "Exact"]
        kmv = df[df.method == "KMV"].sort_values("density_vs_exact")
        mprw = df[df.method == "MPRW"].sort_values("density_vs_exact")
        if not ex.empty:
            ax.axhline(ex.iloc[0]["homophily"], color=COLOR["Exact"],
                       ls="--", lw=1.5, label=f"Exact (homo={ex.iloc[0]['homophily']:.3f})")
        ax.plot(kmv["density_vs_exact"], kmv["homophily"],
                marker="o", color=COLOR["KMV"], label="KMV (k-sweep)")
        ax.plot(mprw["density_vs_exact"], mprw["homophily"],
                marker="s", color=COLOR["MPRW"], label="MPRW (w-sweep)")
        ax.set_xscale("log")
        ax.set_xlabel("density (|E_retained| / |E_exact|)")
        ax.set_ylabel("label homophily of retained edges")
        ax.set_title(ds)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
    plt.suptitle("H1 — MPRW concentrates on homophilic edges more than KMV at matched density",
                 fontsize=11)
    plt.tight_layout()
    out = FIG / "H1_homophily_vs_density.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_H5_spectral(datasets: List[str]) -> None:
    """Spectral radius vs density."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        path = RES / f"mprw_bias_{ds}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ex = df[df.method == "Exact"]
        kmv = df[df.method == "KMV"].sort_values("density_vs_exact")
        mprw = df[df.method == "MPRW"].sort_values("density_vs_exact")
        if not ex.empty:
            ax.axhline(ex.iloc[0]["spectral_radius"], color=COLOR["Exact"],
                       ls="--", lw=1.5, label=f"Exact (λ={ex.iloc[0]['spectral_radius']:.1f})")
        ax.plot(kmv["density_vs_exact"], kmv["spectral_radius"],
                marker="o", color=COLOR["KMV"], label="KMV")
        ax.plot(mprw["density_vs_exact"], mprw["spectral_radius"],
                marker="s", color=COLOR["MPRW"], label="MPRW")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("density (|E_retained| / |E_exact|)")
        ax.set_ylabel(r"spectral radius $\lambda_{\max}(\tilde{A})$")
        ax.set_title(ds)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", alpha=0.3)
    plt.suptitle("H5 — KMV retains higher spectral radius than MPRW at matched density",
                 fontsize=11)
    plt.tight_layout()
    out = FIG / "H5_spectral_radius.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def plot_homophily_gap_table(datasets: List[str]) -> None:
    """At matched density, print homophily gap KMV vs MPRW."""
    rows = []
    for ds in datasets:
        path = RES / f"mprw_bias_{ds}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ex_homo = df[df.method == "Exact"].iloc[0]["homophily"]
        kmv = df[df.method == "KMV"].copy()
        mprw = df[df.method == "MPRW"].copy()
        for _, kr in kmv.iterrows():
            # find nearest MPRW by density
            mprw["gap"] = (mprw["density_vs_exact"] - kr["density_vs_exact"]).abs()
            mr = mprw.sort_values("gap").iloc[0]
            rows.append({
                "dataset": ds,
                "k": int(kr["k"]),
                "w_matched": int(mr["w"]),
                "kmv_density": kr["density_vs_exact"],
                "mprw_density": mr["density_vs_exact"],
                "exact_homo": ex_homo,
                "kmv_homo": kr["homophily"],
                "mprw_homo": mr["homophily"],
                "bias (mprw-kmv)": mr["homophily"] - kr["homophily"],
                "bias (mprw-exact)": mr["homophily"] - ex_homo,
                "bias (kmv-exact)": kr["homophily"] - ex_homo,
            })
    t = pd.DataFrame(rows)
    out = FIG / "homophily_bias_table.csv"
    t.to_csv(out, index=False)
    # Also print
    print(t.to_string(index=False))
    print(f"\n  → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DS_ORDER)
    args = p.parse_args()

    print("Plotting I2, I4, I5 (KMV invariances)...")
    plot_I2_tightness(args.datasets)
    plot_I4_slope_correlation(args.datasets)
    plot_I5_jaccard(args.datasets)

    print("\nPlotting H1, H5 (MPRW bias)...")
    plot_H1_homophily(args.datasets)
    plot_H5_spectral(args.datasets)

    print("\nHomophily-gap summary table:")
    plot_homophily_gap_table(args.datasets)


if __name__ == "__main__":
    main()
