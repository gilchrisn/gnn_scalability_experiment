"""
Generate all paper figures and table summaries from experiment CSVs.

Reads:
    results/<DATASET>/table3.csv   — matching graph stats (|E*|, density)
    results/<DATASET>/table4.csv   — F1 + time per method per metapath
    results/<DATASET>/figure4.csv  — scatter: |E*| vs time (KMV methods)
    results/<DATASET>/figure5.csv  — F1 vs lambda sweep
    results/<DATASET>/figure6.csv  — F1 vs k sweep

Writes:
    results/figures/table3_summary.txt
    results/figures/table3.tex
    results/figures/table4_summary.txt
    results/figures/table4.tex
    results/figures/figure4.png    — scatter |E*| vs time, all datasets
    results/figures/figure5.png    — F1 vs lambda, all datasets
    results/figures/figure6.png    — F1 vs k, all datasets

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --results-dir results/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # non-interactive — safe for headless runs
import matplotlib.pyplot as plt
import matplotlib.ticker


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "lines.linewidth":  1.5,
    "lines.markersize": 5,
    "figure.dpi":       150,
})

DATASETS = ["HGB_ACM", "HGB_DBLP", "HGB_IMDB", "HGB_Freebase"]

DEGREE_METHODS = ["GloD",  "PerD",  "PerD+"]
HINDEX_METHODS = ["GloH",  "PerH",  "PerH+"]
ALL_METHODS    = DEGREE_METHODS + HINDEX_METHODS
BOOLAP_METHODS = ["BoolAP", "BoolAP+"]

_COLOR = {
    "GloD":    "#e41a1c",
    "PerD":    "#377eb8",
    "PerD+":   "#ff7f00",
    "GloH":    "#4daf4a",
    "PerH":    "#984ea3",
    "PerH+":   "#a65628",
    "BoolAP":  "#e6ab02",   # gold
    "BoolAP+": "#66a61e",   # green
}
_MARKER = {
    "GloD": "o", "PerD": "s", "PerD+": "^",
    "GloH": "D", "PerH": "v", "PerH+": "P",
    "BoolAP": "*", "BoolAP+": "X",
}
_LS = {
    "GloD": "-",  "PerD": "--", "PerD+": "-.",
    "GloH": "-",  "PerH": "--", "PerH+": "-.",
    "BoolAP": ":",  "BoolAP+": ":",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def load_all(results_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Return {dataset: {csv_name: DataFrame}}. Missing files silently skipped."""
    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for ds in DATASETS:
        ds_dir = results_dir / ds
        if not ds_dir.is_dir():
            continue
        entry: Dict[str, pd.DataFrame] = {}
        for name in ("table3", "table4", "figure4", "figure5", "figure6"):
            df = _load(ds_dir / f"{name}.csv")
            if df is not None:
                entry[name] = df
        if entry:
            data[ds] = entry
    return data


# ---------------------------------------------------------------------------
# Figure 4 — scatter: |E*| vs time (KMV methods + BoolAP from table4)
# ---------------------------------------------------------------------------

def _boolap_scatter_data(data: dict, ds: str) -> Optional[pd.DataFrame]:
    """
    BoolAP timing lives in table4.csv (avg_time_s), not figure4.csv.
    We reconstruct a figure4-style frame by joining with table3 avg_edges.
    Returns None if either table is missing.
    """
    if "table4" not in data[ds] or "table3" not in data[ds]:
        return None
    t4 = data[ds]["table4"].copy()
    t3 = data[ds]["table3"].copy()
    boolap = t4[t4["method"].isin(BOOLAP_METHODS)].copy()
    if boolap.empty:
        return None
    t3 = t3[["metapath", "avg_edges"]].copy()
    t3["avg_edges"] = pd.to_numeric(t3["avg_edges"], errors="coerce")
    boolap["avg_time_s"] = pd.to_numeric(boolap["avg_time_s"], errors="coerce")
    merged = boolap.merge(t3, on="metapath", how="left")
    merged = merged.rename(columns={"avg_edges": "edges", "avg_time_s": "time_s"})
    return merged[["method", "edges", "time_s"]].dropna()


def plot_figure4(data: dict, out_dir: Path) -> None:
    datasets = [ds for ds in DATASETS if ds in data and "figure4" in data[ds]]
    if not datasets:
        print("  [figure4] No data — skipped.")
        return

    n   = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), squeeze=False)
    fig.suptitle("Running Time vs |E*|  (Figure 4)", fontsize=10, y=1.01)

    for col, ds in enumerate(datasets):
        ax = axes[0][col]
        df = data[ds]["figure4"].copy()
        df["edges"]  = pd.to_numeric(df["edges"],  errors="coerce")
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
        df.dropna(subset=["edges", "time_s"], inplace=True)

        # Collect all values to decide on log scale
        all_edges = df["edges"].values.tolist()
        all_times = df["time_s"].values.tolist()

        # BoolAP supplemental data from table4
        boolap_df = _boolap_scatter_data(data, ds)
        if boolap_df is not None:
            all_edges += boolap_df["edges"].values.tolist()
            all_times += boolap_df["time_s"].values.tolist()

        # Decide log scale: use log if data spans > 2 orders of magnitude
        use_log_x = (max(all_edges) / max(min(all_edges), 1e-9)) > 100
        use_log_y = (max(all_times) / max(min(all_times), 1e-9)) > 100

        # KMV methods
        for method in ALL_METHODS:
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            x = sub["edges"].values.astype(float)
            y = sub["time_s"].values.astype(float)

            ax.scatter(x, y, color=_COLOR[method], marker=_MARKER[method],
                       alpha=0.65, s=20, label=method, zorder=3)

            if len(x) >= 2:
                if use_log_x and use_log_y:
                    # Fit in log space
                    log_x = np.log10(x)
                    log_y = np.log10(y)
                    coeffs = np.polyfit(log_x, log_y, 1)
                    x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
                    y_line = 10 ** np.polyval(coeffs, np.log10(x_line))
                    ss_res = np.sum((log_y - np.polyval(coeffs, log_x)) ** 2)
                    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
                else:
                    coeffs = np.polyfit(x, y, 1)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = np.polyval(coeffs, x_line)
                    ss_res = np.sum((y - np.polyval(coeffs, x)) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                ax.plot(x_line, y_line, color=_COLOR[method],
                        ls=_LS[method], lw=1.0, alpha=0.8,
                        label=f"_{method} R²={r2:.2f}")   # _ hides from legend

        # BoolAP scatter points (no trend line — usually too few points)
        if boolap_df is not None:
            for method in BOOLAP_METHODS:
                sub = boolap_df[boolap_df["method"] == method]
                if sub.empty:
                    continue
                x = sub["edges"].values.astype(float)
                y = sub["time_s"].values.astype(float)
                marker_size = 60 if method == "BoolAP" else 35
                ax.scatter(x, y, color=_COLOR[method], marker=_MARKER[method],
                           alpha=0.85, s=marker_size, label=method,
                           zorder=4, edgecolors="k", linewidths=0.4)

        if use_log_x:
            ax.set_xscale("log")
        if use_log_y:
            ax.set_yscale("log")

        ax.set_title(ds.replace("HGB_", ""))
        ax.set_xlabel("|E*|")
        ax.set_ylabel("Time (s)" if col == 0 else "")
        if not use_log_x:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.grid(True, ls=":", alpha=0.4)

    # Legend on rightmost axis — KMV methods + BoolAP
    kmv_handles = [
        plt.scatter([], [], color=_COLOR[m], marker=_MARKER[m], s=20, label=m)
        for m in ALL_METHODS
    ]
    boolap_handles = [
        plt.scatter([], [], color=_COLOR["BoolAP"], marker=_MARKER["BoolAP"],
                    s=60, label="BoolAP", edgecolors="k", linewidths=0.4),
        plt.scatter([], [], color=_COLOR["BoolAP+"], marker=_MARKER["BoolAP+"],
                    s=35, label="BoolAP+", edgecolors="k", linewidths=0.4),
    ]
    axes[0][-1].legend(handles=kmv_handles + boolap_handles,
                       loc="upper left", fontsize=6, framealpha=0.7)

    fig.tight_layout()
    out = out_dir / "figure4.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure4] Saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 5 — F1 vs lambda
# ---------------------------------------------------------------------------

def plot_figure5(data: dict, out_dir: Path) -> None:
    datasets = [ds for ds in DATASETS if ds in data and "figure5" in data[ds]]
    if not datasets:
        print("  [figure5] No data — run without --skip-sweeps.")
        return

    n   = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), squeeze=False)
    fig.suptitle("Effectiveness vs λ  (Figure 5)", fontsize=10, y=1.01)

    for col, ds in enumerate(datasets):
        ax = axes[0][col]
        df = data[ds]["figure5"].copy()
        df["lambda"]    = pd.to_numeric(df["lambda"],    errors="coerce")
        df["f1_or_acc"] = pd.to_numeric(df["f1_or_acc"], errors="coerce")
        df.dropna(inplace=True)

        for method in ALL_METHODS:
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            grouped = sub.groupby("lambda")["f1_or_acc"].mean().reset_index()
            ax.plot(grouped["lambda"], grouped["f1_or_acc"],
                    color=_COLOR[method], ls=_LS[method],
                    marker=_MARKER[method], ms=4, label=method)

        ax.set_title(ds.replace("HGB_", ""))
        ax.set_xlabel("λ")
        ax.set_ylabel("F1 / Accuracy" if col == 0 else "")
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0.02, 0.03, 0.04, 0.05])
        ax.grid(True, ls=":", alpha=0.4)

    handles = [
        plt.Line2D([0], [0], color=_COLOR[m], ls=_LS[m],
                   marker=_MARKER[m], ms=4, label=m)
        for m in ALL_METHODS
    ]
    axes[0][-1].legend(handles=handles, loc="lower right", fontsize=6, framealpha=0.7)

    fig.tight_layout()
    out = out_dir / "figure5.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure5] Saved -> {out}")


# ---------------------------------------------------------------------------
# Figure 6 — F1 vs k
# ---------------------------------------------------------------------------

def plot_figure6(data: dict, out_dir: Path) -> None:
    datasets = [ds for ds in DATASETS if ds in data and "figure6" in data[ds]]
    if not datasets:
        print("  [figure6] No data — run without --skip-sweeps.")
        return

    n   = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), squeeze=False)
    fig.suptitle("Effectiveness vs k  (Figure 6)", fontsize=10, y=1.01)

    for col, ds in enumerate(datasets):
        ax = axes[0][col]
        df = data[ds]["figure6"].copy()
        df["k"]         = pd.to_numeric(df["k"],         errors="coerce")
        df["f1_or_acc"] = pd.to_numeric(df["f1_or_acc"], errors="coerce")
        df.dropna(inplace=True)

        for method in ALL_METHODS:
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            grouped = sub.groupby("k")["f1_or_acc"].mean().reset_index()
            ax.plot(grouped["k"], grouped["f1_or_acc"],
                    color=_COLOR[method], ls=_LS[method],
                    marker=_MARKER[method], ms=4, label=method)

        ax.set_title(ds.replace("HGB_", ""))
        ax.set_xlabel("k")
        ax.set_ylabel("F1 / Accuracy" if col == 0 else "")
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, ls=":", alpha=0.4)

    handles = [
        plt.Line2D([0], [0], color=_COLOR[m], ls=_LS[m],
                   marker=_MARKER[m], ms=4, label=m)
        for m in ALL_METHODS
    ]
    axes[0][-1].legend(handles=handles, loc="lower right", fontsize=6, framealpha=0.7)

    fig.tight_layout()
    out = out_dir / "figure6.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure6] Saved -> {out}")


# ---------------------------------------------------------------------------
# Table III — plain-text + LaTeX
# ---------------------------------------------------------------------------

def _fmt_edges(val: float) -> str:
    """Format edge count in scientific notation, e.g. 4.02e5."""
    exp = int(np.floor(np.log10(abs(val)))) if val > 0 else 0
    mant = val / (10 ** exp)
    return f"{mant:.2f}e{exp}"


def summarize_table3(data: dict, out_dir: Path) -> None:
    rows = []
    for ds in DATASETS:
        if ds not in data or "table3" not in data[ds]:
            continue
        df = data[ds]["table3"]
        df["avg_edges"] = pd.to_numeric(df["avg_edges"], errors="coerce")
        df["density"]   = pd.to_numeric(df["density"],   errors="coerce")
        rows.append({
            "Dataset":     ds.replace("HGB_", ""),
            "N_metapaths": len(df),
            "avg|E*|":     f"{df['avg_edges'].mean():.1f}",
            "std|E*|":     f"{df['avg_edges'].std():.1f}",
            "avg_density": f"{df['density'].mean():.5f}",
            "std_density": f"{df['density'].std():.5f}",
        })

    if not rows:
        print("  [table3] No data.")
        return

    summary = pd.DataFrame(rows)
    lines = [
        "TABLE III — Matching Graph Statistics",
        "=" * 70,
        summary.to_string(index=False),
        "",
        "  avg|E*|  = average number of edges in matching graph (per metapath)",
        "  density  = average matching graph density (rho*)",
        "",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    out = out_dir / "table3_summary.txt"
    out.write_text(text, encoding="utf-8")
    print(f"  [table3] Saved -> {out}")

    latex_table3(data, out_dir)


def latex_table3(data: dict, out_dir: Path) -> None:
    """Write results/figures/table3.tex."""
    rows = []
    for ds in DATASETS:
        if ds not in data or "table3" not in data[ds]:
            continue
        df = data[ds]["table3"].copy()
        df["avg_edges"] = pd.to_numeric(df["avg_edges"], errors="coerce")
        df["density"]   = pd.to_numeric(df["density"],   errors="coerce")
        df.dropna(subset=["avg_edges", "density"], inplace=True)
        n_mp      = len(df)
        avg_edges = df["avg_edges"].mean()
        avg_dens  = df["density"].mean()
        rows.append((ds.replace("HGB_", ""), n_mp, avg_edges, avg_dens))

    if not rows:
        return

    body_lines = []
    for ds_short, n_mp, avg_edges, avg_dens in rows:
        e_str = _fmt_edges(avg_edges)
        d_str = f"{avg_dens:.5f}"
        body_lines.append(
            f"{ds_short} & {n_mp} & ${e_str}$ & {d_str} \\\\"
        )

    tex = r"""\begin{table}[t]
\setlength\tabcolsep{5pt}
\footnotesize
\centering
\caption{Matching graph statistics per dataset.
  $|\mathcal{M}|$ is the number of evaluated meta-paths;
  $|\mathcal{E}^*|$ and $\rho^*$ denote the average number of edges
  and density of matching graphs.}
\label{tab:datasets}
\vspace{-1em}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Dataset} & $|\mathcal{M}|$ & $|\mathcal{E}^*|$ & $\rho^*$ \\
\hline
"""
    tex += "\n".join(body_lines) + "\n"
    tex += r"""\hline
\end{tabular}
\vspace{-1em}
\end{table}
"""
    out = out_dir / "table3.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"  [table3] LaTeX saved -> {out}")


# ---------------------------------------------------------------------------
# Table IV — plain-text + LaTeX
# ---------------------------------------------------------------------------

def summarize_table4(data: dict, out_dir: Path) -> None:
    rows = []
    all_methods = ALL_METHODS + BOOLAP_METHODS
    for ds in DATASETS:
        if ds not in data or "table4" not in data[ds]:
            continue
        df = data[ds]["table4"]
        df["f1_or_acc"]  = pd.to_numeric(df["f1_or_acc"],  errors="coerce")
        df["avg_time_s"] = pd.to_numeric(df["avg_time_s"], errors="coerce")
        df["rule_count"] = pd.to_numeric(df["rule_count"], errors="coerce")

        for method in all_methods:
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            rows.append({
                "Dataset":    ds.replace("HGB_", ""),
                "Method":     method,
                "N_mp":       len(sub),
                "avg_F1/Acc": f"{sub['f1_or_acc'].mean():.4f}",
                "std_F1/Acc": f"{sub['f1_or_acc'].std():.4f}",
                "avg_time_s": f"{sub['avg_time_s'].mean():.6f}",
                "avg_rules":  f"{sub['rule_count'].mean():.1f}",
            })

    if not rows:
        print("  [table4] No data.")
        return

    summary = pd.DataFrame(rows)
    lines = [
        "TABLE IV — Efficiency + Effectiveness",
        "=" * 80,
        summary.to_string(index=False),
        "",
        "  avg_F1/Acc = mean F1 (GloD/GloH) or accuracy (PerD/PerH variants) across metapaths",
        "  avg_time_s = mean CPU time per query",
        "  avg_rules  = mean rule_count (should be > 0 for all rows)",
        "",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    out = out_dir / "table4_summary.txt"
    out.write_text(text, encoding="utf-8")
    print(f"  [table4] Saved -> {out}")

    latex_table4(data, out_dir)


def _fmt_time(t: float) -> str:
    """Format time: scientific notation if < 0.01, otherwise 4 decimal places."""
    if t < 0.01:
        exp = int(np.floor(np.log10(abs(t)))) if t > 0 else 0
        mant = t / (10 ** exp)
        return f"${mant:.2f}\\times10^{{{exp}}}$"
    return f"{t:.4f}"


def latex_table4(data: dict, out_dir: Path) -> None:
    """Write results/figures/table4.tex using booktabs style."""
    all_methods = ALL_METHODS + BOOLAP_METHODS
    # Collect per-dataset, per-method aggregates
    table_rows: List[dict] = []
    for ds in DATASETS:
        if ds not in data or "table4" not in data[ds]:
            continue
        df = data[ds]["table4"].copy()
        df["f1_or_acc"]  = pd.to_numeric(df["f1_or_acc"],  errors="coerce")
        df["avg_time_s"] = pd.to_numeric(df["avg_time_s"], errors="coerce")
        ds_short = ds.replace("HGB_", "")
        method_rows = []
        for method in all_methods:
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            avg_f1   = sub["f1_or_acc"].mean()
            avg_time = sub["avg_time_s"].mean()
            method_rows.append({
                "ds":      ds_short,
                "method":  method,
                "avg_f1":  avg_f1,
                "avg_t":   avg_time,
                "n_mp":    len(sub),
            })
        if method_rows:
            # Tag first row of each dataset for \multirow
            for i, r in enumerate(method_rows):
                r["first_in_ds"] = (i == 0)
                r["ds_nrows"]    = len(method_rows)
            table_rows.extend(method_rows)

    if not table_rows:
        return

    header = (
        r"\begin{table}[t]" + "\n"
        r"\setlength\tabcolsep{4pt}" + "\n"
        r"\footnotesize" + "\n"
        r"\centering" + "\n"
        r"\caption{Efficiency and effectiveness per method per dataset. "
        r"Avg.\ F1/Acc is averaged across all evaluated meta-paths; "
        r"Time (s) is the mean CPU time per query.}" + "\n"
        r"\label{tab:table4}" + "\n"
        r"\begin{tabular}{llccc}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Dataset} & \textbf{Method} & \textbf{Avg.\ F1/Acc} "
        r"& \textbf{Time (s)} & \textbf{\#Meta-paths} \\" + "\n"
        r"\midrule"
    )

    body_lines = [header]
    prev_ds = None
    for r in table_rows:
        if r["ds"] != prev_ds and prev_ds is not None:
            body_lines.append(r"\midrule")
        prev_ds = r["ds"]

        if r["first_in_ds"]:
            ds_cell = f"\\multirow{{{r['ds_nrows']}}}{{*}}{{{r['ds']}}}"
        else:
            ds_cell = ""

        f1_str = f"{r['avg_f1']:.4f}" if not np.isnan(r["avg_f1"]) else "--"
        t_str  = _fmt_time(r["avg_t"]) if not np.isnan(r["avg_t"]) else "--"

        body_lines.append(
            f"{ds_cell} & {r['method']} & {f1_str} & {t_str} & {r['n_mp']} \\\\"
        )

    footer = (
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )
    body_lines.append(footer)

    tex = "\n".join(body_lines) + "\n"
    out = out_dir / "table4.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"  [table4] LaTeX saved -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper figures from experiment CSVs."
    )
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing per-dataset result folders.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        print(f"[ERROR] Results directory not found: {results_dir}")
        sys.exit(1)

    out_dir = results_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    print(f"\nLoading results from {results_dir} ...")
    data = load_all(results_dir)

    if not data:
        print("[ERROR] No CSV files found in results/. Run experiments first.")
        sys.exit(1)

    found = list(data.keys())
    print(f"Datasets found : {found}")
    for ds in found:
        csvs = list(data[ds].keys())
        print(f"  {ds:20s}  CSVs: {csvs}")

    print("\nGenerating figures ...")
    plot_figure4(data, out_dir)
    plot_figure5(data, out_dir)
    plot_figure6(data, out_dir)

    print("\nGenerating table summaries ...")
    summarize_table3(data, out_dir)
    summarize_table4(data, out_dir)

    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
