"""
Experiment LP-Analyze — Aggregate master_lp_results.csv across seeds + datasets
into publication-ready tables and plots.

Reads all `results/<dataset>/master_lp_results.csv` files, aggregates (mean, std)
over seeds per (dataset, method, k/w), and produces:

  figures/lp/table1_mrr_auc.csv        Main quality table across datasets
  figures/lp/table2_scaling.csv        Memory + time scaling (incl. Exact OOM)
  figures/lp/table3_tail_mid_hub.csv   Per-degree-bin MRR breakdown
  figures/lp/plot_mrr_bars.pdf         Per-dataset MRR bar plot (Exact/KMV/MPRW)
  figures/lp/plot_mrr_vs_density.pdf   MRR vs edge count (density) — common axis
  figures/lp/plot_scaling.pdf          Quality vs graph size (killer plot)
  figures/lp/plot_perbin.pdf           Per-degree-bin MRR bars

Usage
-----
    python scripts/exp_lp_analyze.py                        # all datasets
    python scripts/exp_lp_analyze.py --datasets HGB_DBLP HGB_ACM
    python scripts/exp_lp_analyze.py --no-plots             # tables only
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Colors (consistent with prior exp4 convention)
COL = {"Exact": "#1f77b4", "KMV": "#2ca02c", "MPRW": "#d62728"}

METRIC_COLS = ["MRR", "ROC_AUC", "Hits_1", "Hits_10", "AP", "Recall_10",
               "MRR_tail", "MRR_mid", "MRR_hub",
               "ROC_AUC_tail", "ROC_AUC_mid", "ROC_AUC_hub"]


def _load_all(datasets: List[str]) -> pd.DataFrame:
    dfs = []
    for ds in datasets:
        p = Path("results") / ds / "master_lp_results.csv"
        if not p.exists():
            print(f"  [skip] {p} not found")
            continue
        df = pd.read_csv(p)
        df["_dataset"] = ds
        dfs.append(df)
    if not dfs:
        raise SystemExit("No master_lp_results.csv files found")
    return pd.concat(dfs, ignore_index=True)


def _agg_metric(df: pd.DataFrame, grouping: List[str],
                metrics: List[str]) -> pd.DataFrame:
    def _f(x):
        x = x.dropna().astype(float)
        if len(x) == 0:
            return pd.Series({"mean": float("nan"), "std": float("nan"), "n": 0})
        return pd.Series({"mean": float(x.mean()), "std": float(x.std()),
                          "n": int(len(x))})

    rows = []
    # For each group, compute mean/std over seeds
    for grp_key, grp_df in df.groupby(grouping, dropna=False):
        row = dict(zip(grouping, grp_key if isinstance(grp_key, tuple) else (grp_key,)))
        for m in metrics:
            if m not in grp_df.columns:
                continue
            s = _f(grp_df[m])
            row[f"{m}_mean"] = s["mean"]
            row[f"{m}_std"]  = s["std"]
            row[f"{m}_n"]    = int(s["n"])
        # Edge count mean (for density-axis plots)
        if "Edge_Count" in grp_df.columns:
            ec = grp_df["Edge_Count"].dropna().astype(float)
            row["Edge_Count_mean"] = float(ec.mean()) if len(ec) else float("nan")
        if "Materialization_Time" in grp_df.columns:
            mt = grp_df["Materialization_Time"].dropna().astype(float)
            row["Materialization_Time_mean"] = float(mt.mean()) if len(mt) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_mean_std(mean: float, std: float, n: int) -> str:
    if pd.isna(mean):
        return "—"
    if n <= 1 or pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def table1_quality(df_agg: pd.DataFrame, out_path: Path) -> None:
    """Main quality table — one row per (Dataset, Method, k/w)."""
    cols = ["_dataset", "Method", "k_value", "w_value",
            "MRR_mean", "MRR_std", "MRR_n",
            "ROC_AUC_mean", "ROC_AUC_std",
            "Hits_10_mean", "Hits_10_std",
            "AP_mean", "AP_std",
            "Edge_Count_mean"]
    df_agg = df_agg.reindex(columns=[c for c in cols if c in df_agg.columns])
    df_agg.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  wrote {out_path}")


def table2_scaling(df: pd.DataFrame, out_path: Path) -> None:
    """Scaling table — materialization time, memory, edges per (Dataset, Method)."""
    rows = []
    for (ds, method), g in df.groupby(["_dataset", "Method"]):
        row = {
            "Dataset": ds,
            "Method": method,
            "Edge_Count_mean": g["Edge_Count"].dropna().astype(float).mean() if len(g["Edge_Count"].dropna()) else float("nan"),
            "Mat_Time_mean":   g["Materialization_Time"].dropna().astype(float).mean() if len(g["Materialization_Time"].dropna()) else float("nan"),
            "Mat_RAM_mean":    g["Mat_RAM_MB"].dropna().astype(float).mean() if len(g["Mat_RAM_MB"].dropna()) else float("nan"),
            "Inf_Time_mean":   g["Inference_Time"].dropna().astype(float).mean() if len(g["Inference_Time"].dropna()) else float("nan"),
            "Status":          ", ".join(sorted(set(g["exact_status"].dropna().astype(str).tolist()))),
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.4f")
    print(f"  wrote {out_path}")


def table3_perbin(df_agg: pd.DataFrame, out_path: Path) -> None:
    cols = ["_dataset", "Method", "k_value", "w_value",
            "MRR_tail_mean", "MRR_tail_std",
            "MRR_mid_mean",  "MRR_mid_std",
            "MRR_hub_mean",  "MRR_hub_std"]
    df_agg = df_agg.reindex(columns=[c for c in cols if c in df_agg.columns])
    df_agg.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  wrote {out_path}")


def plot_mrr_bars(df_agg: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not available")
        return

    datasets = sorted(df_agg["_dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df_agg[df_agg["_dataset"] == ds].copy()
        # Format labels: "Exact", "KMV k=8", "MPRW w=8" etc.
        def _lbl(r):
            if r["Method"] == "Exact":
                return "Exact"
            if r["Method"] == "KMV":
                return f"KMV k={int(r['k_value']) if not pd.isna(r['k_value']) else '?'}"
            return f"MPRW w={int(r['w_value']) if not pd.isna(r['w_value']) else '?'}"
        sub["label"] = sub.apply(_lbl, axis=1)
        # Order: Exact first, then KMV by k, then MPRW by w
        def _sort_key(r):
            if r["Method"] == "Exact":
                return (0, 0)
            if r["Method"] == "KMV":
                return (1, r["k_value"] if not pd.isna(r["k_value"]) else 0)
            return (2, r["w_value"] if not pd.isna(r["w_value"]) else 0)
        sub["sortk"] = sub.apply(_sort_key, axis=1)
        sub = sub.sort_values("sortk")

        x = np.arange(len(sub))
        colors = [COL[m] for m in sub["Method"]]
        ax.bar(x, sub["MRR_mean"], yerr=sub["MRR_std"].fillna(0.0),
               color=colors, alpha=0.85, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=8)
        ax.set_title(ds)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("MRR")
    fig.suptitle("LP MRR — Exact vs KMV vs MPRW (mean ± std across seeds)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_mrr_vs_density(df_agg: pd.DataFrame, out_path: Path) -> None:
    """KMV and MPRW dots on common edge-count axis. The main fairness plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not available")
        return

    datasets = sorted(df_agg["_dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df_agg[df_agg["_dataset"] == ds]
        # Exact horizontal line
        ex = sub[sub["Method"] == "Exact"]
        if len(ex) and not pd.isna(ex["MRR_mean"].iloc[0]):
            ax.axhline(ex["MRR_mean"].iloc[0], ls="--", lw=1.5,
                       color=COL["Exact"], label="Exact")
        # KMV dots
        km = sub[sub["Method"] == "KMV"].sort_values("Edge_Count_mean")
        if len(km):
            ax.errorbar(km["Edge_Count_mean"], km["MRR_mean"],
                        yerr=km["MRR_std"].fillna(0.0),
                        marker="o", color=COL["KMV"], label="KMV",
                        lw=2, capsize=3)
        # MPRW dots
        mp = sub[sub["Method"] == "MPRW"].sort_values("Edge_Count_mean")
        if len(mp):
            ax.errorbar(mp["Edge_Count_mean"], mp["MRR_mean"],
                        yerr=mp["MRR_std"].fillna(0.0),
                        marker="s", color=COL["MPRW"], label="MPRW",
                        lw=2, capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("Edge count (log scale)")
        ax.set_title(ds)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("MRR")
    fig.suptitle("LP MRR vs Edge Density — common axis", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_perbin(df_agg: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not available")
        return

    datasets = sorted(df_agg["_dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    bins = ["tail", "mid", "hub"]

    for ax, ds in zip(axes, datasets):
        sub = df_agg[df_agg["_dataset"] == ds]
        # Pick representative: KMV k=32 (or max k), MPRW w=128 (or max w), Exact
        exact = sub[sub["Method"] == "Exact"]
        kmv = sub[sub["Method"] == "KMV"]
        if len(kmv):
            best_k = kmv.loc[kmv["k_value"].astype(float).idxmax()]
        else:
            best_k = None
        mprw = sub[sub["Method"] == "MPRW"]
        if len(mprw):
            best_w = mprw.loc[mprw["w_value"].astype(float).idxmax()]
        else:
            best_w = None

        x = np.arange(len(bins))
        width = 0.28
        if len(exact):
            r = exact.iloc[0]
            vals = [r.get(f"MRR_{b}_mean", float("nan")) for b in bins]
            ax.bar(x - width, vals, width, color=COL["Exact"], label="Exact", alpha=0.85)
        if best_k is not None:
            vals = [best_k.get(f"MRR_{b}_mean", float("nan")) for b in bins]
            errs = [best_k.get(f"MRR_{b}_std", 0.0) or 0.0 for b in bins]
            ax.bar(x, vals, width, yerr=errs, color=COL["KMV"],
                   label=f"KMV k={int(best_k['k_value'])}", alpha=0.85, capsize=3)
        if best_w is not None:
            vals = [best_w.get(f"MRR_{b}_mean", float("nan")) for b in bins]
            errs = [best_w.get(f"MRR_{b}_std", 0.0) or 0.0 for b in bins]
            ax.bar(x + width, vals, width, yerr=errs, color=COL["MPRW"],
                   label=f"MPRW w={int(best_w['w_value'])}", alpha=0.85, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_title(ds)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    axes[0].set_ylabel("MRR")
    fig.suptitle("LP MRR stratified by source-node degree bin", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+",
                        default=["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed"])
    parser.add_argument("--out-dir", default="figures/lp")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LP results from {len(args.datasets)} datasets...")
    df = _load_all(args.datasets)
    print(f"  {len(df)} total rows")

    df_agg = _agg_metric(
        df,
        grouping=["_dataset", "Method", "k_value", "w_value"],
        metrics=METRIC_COLS,
    )

    table1_quality(df_agg, out_dir / "table1_mrr_auc.csv")
    table2_scaling(df,    out_dir / "table2_scaling.csv")
    table3_perbin(df_agg, out_dir / "table3_tail_mid_hub.csv")

    if not args.no_plots:
        plot_mrr_bars(df_agg,       out_dir / "plot_mrr_bars.pdf")
        plot_mrr_vs_density(df_agg, out_dir / "plot_mrr_vs_density.pdf")
        plot_perbin(df_agg,         out_dir / "plot_perbin.pdf")

    print(f"\nDone. Outputs → {out_dir}")


if __name__ == "__main__":
    main()
