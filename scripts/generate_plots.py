"""Generate publication-quality plots from experiment result CSVs."""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Style setup ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
PLOTS = RESULTS / "plots"
os.makedirs(PLOTS, exist_ok=True)

# Consistent color/marker scheme
METHOD_STYLE = {
    "GloD":  {"color": "#2ca02c", "marker": "o", "ls": "-"},
    "PerD":  {"color": "#1f77b4", "marker": "s", "ls": "-"},
    "PerD+": {"color": "#d62728", "marker": "^", "ls": "-"},
    "GloH":  {"color": "#2ca02c", "marker": "o", "ls": "--"},
    "PerH":  {"color": "#1f77b4", "marker": "s", "ls": "--"},
    "PerH+": {"color": "#d62728", "marker": "^", "ls": "--"},
}

DATASET_STYLE = {
    "HGB_ACM":  {"color": "#1f77b4", "marker": "o"},
    "HGB_DBLP": {"color": "#ff7f0e", "marker": "s"},
    "HGB_IMDB": {"color": "#2ca02c", "marker": "^"},
    "OGB_MAG":  {"color": "#d62728", "marker": "D"},
    "OAG_CS":   {"color": "#9467bd", "marker": "v"},
}

DATASET_LABELS = {
    "HGB_ACM": "ACM", "HGB_DBLP": "DBLP", "HGB_IMDB": "IMDB",
    "OGB_MAG": "OGB-MAG", "OAG_CS": "OAG-CS",
}

DEGREE_METHODS = ["GloD", "PerD", "PerD+"]
HINDEX_METHODS = ["GloH", "PerH", "PerH+"]

HGB_DATASETS = ["HGB_ACM", "HGB_DBLP", "HGB_IMDB"]


def safe_read(path):
    """Read CSV, coercing numeric columns, returning None if missing/empty."""
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"  [SKIP] {path}: {e}")
        return None


def to_numeric_safe(series):
    return pd.to_numeric(series, errors="coerce")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: F1 vs Lambda
# ═══════════════════════════════════════════════════════════════════════════
def plot_figure5():
    print("Generating Figure 5: F1 vs Lambda ...")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey="row")

    for col, ds in enumerate(HGB_DATASETS):
        df = safe_read(RESULTS / ds / "figure5.csv")
        if df is None:
            continue
        df["lambda"] = to_numeric_safe(df["lambda"])
        df["f1_or_acc"] = to_numeric_safe(df["f1_or_acc"])
        df = df.dropna(subset=["lambda", "f1_or_acc"])

        for row, methods in enumerate([DEGREE_METHODS, HINDEX_METHODS]):
            ax = axes[row, col]
            for m in methods:
                sub = df[df["method"] == m]
                if sub.empty:
                    continue
                agg = sub.groupby("lambda")["f1_or_acc"].mean().reset_index()
                agg = agg.sort_values("lambda")
                st = METHOD_STYLE[m]
                ax.plot(agg["lambda"], agg["f1_or_acc"],
                        color=st["color"], marker=st["marker"], ls=st["ls"],
                        markersize=5, linewidth=1.5, label=m)
            ax.set_xlabel(r"$\lambda$")
            if col == 0:
                ax.set_ylabel("F1" if row == 0 else "F1")
            if row == 0:
                ax.set_title(DATASET_LABELS[ds])
            ax.set_ylim(0.4, 1.05)

    # Row labels
    axes[0, 0].annotate("Degree", xy=(-0.35, 0.5), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", rotation=90, va="center")
    axes[1, 0].annotate("H-index", xy=(-0.35, 0.5), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", rotation=90, va="center")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=[0.03, 0, 1, 0.95])
    fig.savefig(PLOTS / "figure5_f1_vs_lambda.png")
    plt.close(fig)
    print(f"  Saved {PLOTS / 'figure5_f1_vs_lambda.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: F1 vs K
# ═══════════════════════════════════════════════════════════════════════════
def plot_figure6():
    print("Generating Figure 6: F1 vs K ...")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey="row")

    for col, ds in enumerate(HGB_DATASETS):
        df = safe_read(RESULTS / ds / "figure6.csv")
        if df is None:
            continue
        df["k"] = to_numeric_safe(df["k"])
        df["f1_or_acc"] = to_numeric_safe(df["f1_or_acc"])
        df = df.dropna(subset=["k", "f1_or_acc"])

        for row, methods in enumerate([DEGREE_METHODS, HINDEX_METHODS]):
            ax = axes[row, col]
            for m in methods:
                sub = df[df["method"] == m]
                if sub.empty:
                    continue
                agg = sub.groupby("k")["f1_or_acc"].mean().reset_index()
                agg = agg.sort_values("k")
                st = METHOD_STYLE[m]
                ax.plot(agg["k"], agg["f1_or_acc"],
                        color=st["color"], marker=st["marker"], ls=st["ls"],
                        markersize=5, linewidth=1.5, label=m)
            ax.set_xlabel("$k$")
            ax.set_xscale("log", base=2)
            ax.set_xticks([2, 4, 8, 16, 32])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            if col == 0:
                ax.set_ylabel("F1")
            if row == 0:
                ax.set_title(DATASET_LABELS[ds])
            ax.set_ylim(0.4, 1.05)

    axes[0, 0].annotate("Degree", xy=(-0.35, 0.5), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", rotation=90, va="center")
    axes[1, 0].annotate("H-index", xy=(-0.35, 0.5), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", rotation=90, va="center")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=[0.03, 0, 1, 0.95])
    fig.savefig(PLOTS / "figure6_f1_vs_k.png")
    plt.close(fig)
    print(f"  Saved {PLOTS / 'figure6_f1_vs_k.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Scatter |E*| vs time
# ═══════════════════════════════════════════════════════════════════════════
def plot_figure4():
    print("Generating Figure 4: Scatter (edges vs time) ...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    for col, ds in enumerate(HGB_DATASETS):
        df = safe_read(RESULTS / ds / "figure4.csv")
        if df is None:
            continue
        df["edges"] = to_numeric_safe(df["edges"])
        df["time_s"] = to_numeric_safe(df["time_s"])
        df = df.dropna(subset=["edges", "time_s"])

        ax = axes[col]
        all_methods = df["method"].unique()
        for m in all_methods:
            sub = df[df["method"] == m]
            st = METHOD_STYLE.get(m, {"color": "gray", "marker": "x", "ls": "-"})
            ax.scatter(sub["edges"], sub["time_s"],
                       c=st["color"], marker=st["marker"], s=28,
                       alpha=0.75, label=m, edgecolors="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$|E^*|$ (edges)")
        if col == 0:
            ax.set_ylabel("Time (s)")
        ax.set_title(DATASET_LABELS[ds])

    # Shared legend
    handles, labels = [], []
    for m in ["GloD", "PerD", "PerD+", "GloH", "PerH", "PerH+"]:
        st = METHOD_STYLE[m]
        handles.append(Line2D([0], [0], marker=st["marker"], color="w",
                              markerfacecolor=st["color"], markersize=6))
        labels.append(m)
    fig.legend(handles, labels, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(PLOTS / "figure4_scatter.png")
    plt.close(fig)
    print(f"  Saved {PLOTS / 'figure4_scatter.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# EXTENSION: Load + filter helper
# ═══════════════════════════════════════════════════════════════════════════
def load_extension_data():
    """Load extension CSVs for all datasets, filter bad rows.

    Keeps rows where exact failed (for KMV-only data points).
    Normalizes fractions so each dataset's max fraction = 1.0.
    Averages across metapaths per dataset per fraction.
    """
    frames = []
    for ds in ["HGB_ACM", "HGB_DBLP", "HGB_IMDB", "OGB_MAG", "OAG_CS"]:
        df = safe_read(RESULTS / ds / "extension.csv")
        if df is None:
            continue
        # Filter out FAILED snapshots
        if "snapshot" in df.columns:
            df = df[~df["snapshot"].astype(str).str.startswith("FAILED")]
        frames.append(df)
    if not frames:
        return None
    big = pd.concat(frames, ignore_index=True)
    # Coerce numeric columns
    for c in ["fraction", "speedup_kmv", "cka_kmv", "n_edges_exact", "n_edges_kmv",
              "f1_exact", "f1_kmv", "t_exact_mat", "t_kmv_mat", "pred_agreement",
              "dirichlet_exact", "dirichlet_kmv"]:
        if c in big.columns:
            big[c] = to_numeric_safe(big[c])
    # Normalize fractions: each dataset's max fraction → 100%
    for ds in big["dataset"].unique():
        mask = big["dataset"] == ds
        max_f = big.loc[mask, "fraction"].max()
        if max_f > 0:
            big.loc[mask, "norm_frac"] = big.loc[mask, "fraction"] / max_f
    return big


def avg_extension(ext):
    """Average across metapaths per dataset per normalized fraction."""
    ext["ds"] = ext["dataset"].map({
        "HGB_ACM": "ACM", "HGB_DBLP": "DBLP", "HGB_IMDB": "IMDB",
        "OGB_MAG": "OGB-MAG", "OAG_CS": "OAG-CS",
    })
    return ext.groupby(["ds", "norm_frac"]).agg(
        speedup=("speedup_kmv", "mean"),
        cka=("cka_kmv", "mean"),
        edges_exact=("n_edges_exact", "mean"),
        edges_kmv=("n_edges_kmv", "mean"),
        f1_exact=("f1_exact", "mean"),
        f1_kmv=("f1_kmv", "mean"),
        dir_exact=("dirichlet_exact", "mean"),
        dir_kmv=("dirichlet_kmv", "mean"),
    ).reset_index()


def _ext_plot(avg, col_y, ylabel, outname, log=False, hline=None, paired=False):
    """Helper for averaged extension plots with normalized fractions."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for ds in ["ACM", "DBLP", "IMDB", "OAG-CS"]:
        d = avg[avg["ds"] == ds].sort_values("norm_frac")
        if d.empty:
            continue
        st_key = {"ACM": "HGB_ACM", "DBLP": "HGB_DBLP", "IMDB": "HGB_IMDB",
                  "OAG-CS": "OAG_CS"}.get(ds, ds)
        st = DATASET_STYLE.get(st_key, {"color": "gray", "marker": "x"})
        if paired:
            # Plot exact (solid) + KMV (dashed)
            col_exact, col_kmv = col_y
            d_e = d.dropna(subset=[col_exact])
            d_k = d.dropna(subset=[col_kmv])
            if not d_e.empty:
                ax.plot(d_e["norm_frac"] * 100, d_e[col_exact],
                        color=st["color"], marker=st["marker"], ls="-",
                        markersize=5, linewidth=1.5, label=f"{ds} exact")
            if not d_k.empty:
                ax.plot(d_k["norm_frac"] * 100, d_k[col_kmv],
                        color=st["color"], marker=st["marker"], ls="--",
                        markersize=5, linewidth=1.5, alpha=0.6, label=f"{ds} KMV")
        else:
            d_clean = d.dropna(subset=[col_y])
            if d_clean.empty:
                continue
            ax.plot(d_clean["norm_frac"] * 100, d_clean[col_y],
                    color=st["color"], marker=st["marker"], ls="-",
                    markersize=5, linewidth=1.5, label=ds)
    if log:
        ax.set_yscale("log")
    if hline is not None:
        ax.axhline(y=hline, color="gray", ls=":", linewidth=1)
    ax.set_xlabel("Fraction of graph (%)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=7.5, ncol=2 if paired else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS / outname, dpi=300)
    plt.close(fig)
    print(f"  Saved {PLOTS / outname}")


def plot_speedup_vs_fraction():
    print("Generating Extension: Speedup vs Fraction ...")
    ext = load_extension_data()
    if ext is None:
        print("  [SKIP] No extension data"); return
    avg = avg_extension(ext)
    _ext_plot(avg, "speedup", "Speedup (exact / KMV)", "ext_speedup_avg.png", log=True)


def plot_cka_vs_fraction():
    print("Generating Extension: CKA vs Fraction ...")
    ext = load_extension_data()
    if ext is None:
        print("  [SKIP] No extension data"); return
    avg = avg_extension(ext)
    _ext_plot(avg, "cka", "Linear CKA", "ext_cka_avg.png", hline=0.85)


def plot_edge_comparison():
    print("Generating Extension: Edge Count (exact vs KMV) ...")
    ext = load_extension_data()
    if ext is None:
        print("  [SKIP] No extension data"); return
    avg = avg_extension(ext)
    _ext_plot(avg, ("edges_exact", "edges_kmv"), "$|E^*|$ (edges)",
              "ext_edges_avg.png", log=True, paired=True)


def plot_f1_comparison():
    print("Generating Extension: F1 Exact vs F1 KMV ...")
    ext = load_extension_data()
    if ext is None:
        print("  [SKIP] No extension data"); return
    avg = avg_extension(ext)
    _ext_plot(avg, ("f1_exact", "f1_kmv"), "F1 (macro)",
              "ext_f1_avg.png", paired=True)


def plot_dirichlet():
    print("Generating Extension: Dirichlet Energy ...")
    ext = load_extension_data()
    if ext is None:
        print("  [SKIP] No extension data"); return
    avg = avg_extension(ext)
    _ext_plot(avg, ("dir_exact", "dir_kmv"), "Dirichlet Energy",
              "ext_dirichlet_avg.png", paired=True)


# ═══════════════════════════════════════════════════════════════════════════
# EXTENSION: Bench Scale Bar Chart (OGB_MAG, hardcoded)
# ═══════════════════════════════════════════════════════════════════════════
def plot_bench_scale():
    print("Generating Extension: Bench Scale (OGB-MAG) ...")
    csv_path = RESULTS / "OGB_MAG" / "bench_scale.csv"
    df = safe_read(csv_path)
    if df is None:
        print("  [SKIP] No bench_scale.csv — run scripts/_bench_scale.py first")
        return

    for c in ["fraction", "exactd_s", "glod_k8_s", "materialize_s", "sketch_k8_s"]:
        if c in df.columns:
            df[c] = to_numeric_safe(df[c])

    df = df.dropna(subset=["fraction"])
    fractions = [f"{int(f*100)}%" for f in df["fraction"]]

    methods = {
        "ExactD":      ("exactd_s",      "#1f77b4"),
        "GloD":        ("glod_k8_s",     "#2ca02c"),
        "Materialize": ("materialize_s", "#ff7f0e"),
        "Sketch":      ("sketch_k8_s",   "#d62728"),
    }

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(fractions))
    width = 0.18
    offsets = np.arange(len(methods)) - (len(methods) - 1) / 2

    for i, (label, (col, color)) in enumerate(methods.items()):
        vals = df[col].tolist() if col in df.columns else [np.nan] * len(fractions)
        ax.bar(x + offsets[i] * width, vals, width,
               label=label, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Graph Fraction")
    ax.set_ylabel("Time (s)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(fractions)
    ax.set_title("OGB-MAG: Scaling Benchmark")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS / "ext_bench_scale_ogb_mag.png", dpi=300)
    plt.close(fig)
    print(f"  Saved {PLOTS / 'ext_bench_scale_ogb_mag.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Output directory: {PLOTS}\n")

    # Part 1: Base paper reproduction
    plot_figure5()
    plot_figure6()
    plot_figure4()

    # Part 2: Extension
    plot_speedup_vs_fraction()
    plot_cka_vs_fraction()
    plot_edge_comparison()
    plot_f1_comparison()
    plot_dirichlet()
    plot_bench_scale()

    print(f"\nDone. All plots saved to {PLOTS}")
