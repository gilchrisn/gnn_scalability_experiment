"""aggregate_3way_l2.py — Merge KMV + KGRW + MPRW rows from kgrw_bench.csv at L=2
into a tidy summary + a 3-way scatter plot per dataset.

PURPOSE
-------
Each dataset's `kgrw_bench.csv` has one row per (method, k, w', seed) cell.
This script:
  1. Loads those rows.
  2. Aggregates seeds into mean ± std per cell.
  3. Writes a long-form CSV (one row per method-cell, all 4 datasets).
  4. Plots a 4×3 grid: rows = quality metrics (F1, CKA, PA);
                       cols = datasets;
                       all three methods on each panel, x = edge count.
  5. Writes a markdown table for slides.

THE PLOT IS THE HEADLINE — it answers:
  "At matched edge density, which method gives the best downstream quality?"
  Each method (KMV/green, KGRW/blue, MPRW/red) shows up as a curve sweeping
  its parameter; X is edge count so curves are directly comparable.

Outputs
-------
results/3way_l2_summary.csv    — long-form aggregated table
results/3way_l2_scatter.pdf    — 3 metrics × 4 datasets, methods overlaid
results/3way_l2_summary.md     — short markdown for slides
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed"]
METRICS  = ["Macro_F1", "CKA", "Pred_Agreement"]

# Method colour scheme — used everywhere downstream so plots are consistent.
# Green = KMV (deterministic), Blue = KGRW (hybrid), Red = MPRW (stochastic).
COLORS = {"KMV": "#2CA02C", "KGRW": "#1F77B4", "MPRW": "#D62728"}


# ────────────────────────────────────────────────────────────────────────────
# CLI — optional filters to restrict which (method, k, w') cells are plotted.
# Default (no flags) reproduces the canonical 3way_l2_* outputs unchanged.
# Use --tag to suffix output filenames so filtered runs don't overwrite them.
# ────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kmv-k",   nargs="+", type=int, default=None,
                   help="Restrict KMV cells to these k values.")
    p.add_argument("--mprw-w",  nargs="+", type=int, default=None,
                   help="Restrict MPRW cells to these w' values.")
    p.add_argument("--kgrw-k",  nargs="+", type=int, default=None,
                   help="Restrict KGRW cells to these k values.")
    p.add_argument("--kgrw-w",  nargs="+", type=int, default=None,
                   help="Restrict KGRW cells to these w' values.")
    p.add_argument("--tag",     type=str, default="",
                   help="Suffix appended to output filenames (e.g. '_uniform').")
    return p.parse_args()

ARGS = _parse_args()


def _accept(method: str, k: str, w: str) -> bool:
    """True iff the (method, k, w') cell passes the active CLI filters."""
    def _in(val: str, allowed):
        if allowed is None:
            return True
        try:
            return int(val) in allowed
        except (ValueError, TypeError):
            return False
    if method == "KMV":
        return _in(k, ARGS.kmv_k)
    if method == "MPRW":
        return _in(w, ARGS.mprw_w)
    if method == "KGRW":
        return _in(k, ARGS.kgrw_k) and _in(w, ARGS.kgrw_w)
    return True


# ────────────────────────────────────────────────────────────────────────────
# Helpers — load/flatten one dataset's rows and aggregate seed replicates.
# ────────────────────────────────────────────────────────────────────────────

def _load(ds: str) -> list[dict]:
    """Read the per-dataset CSV. Filter to SAGE-depth L=2 only (we run all
    benchmarks at L=2 SAGE; metapath length is set in the rule file)."""
    p = ROOT / "results" / ds / "kgrw_bench.csv"
    with open(p, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("L") == "2"]


def _flt(v):
    """Tolerant float conversion (CSV cells can be empty / NaN / 'OK')."""
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


def _agg(rows: list[dict], method: str, k: str, w: str) -> dict | None:
    """Aggregate seeds for ONE (method, k, w') cell.

    Each cell becomes one point on the scatter plot: mean across seeds,
    std for error bars. Empty cells return None and are filtered out.

    A 'cell' is a single experimental config:
      - KMV cell: (KMV, k=X, w='')         e.g. (KMV, 32, '')
      - MPRW cell: (MPRW, k='', w'=Y)      e.g. (MPRW, '', 64)
      - KGRW cell: (KGRW, k=X, w'=Y)       e.g. (KGRW, 16, 4)
    """
    sub = [r for r in rows
           if r["Method"] == method and r["k"] == k and r["w_prime"] == w]
    if not sub:
        return None
    out: dict = {"Method": method, "k": k, "w_prime": w, "n_seeds": len(sub)}
    edges = [_flt(r["Edge_Count"]) for r in sub]
    edges = [e for e in edges if e is not None]
    out["edges_mean"] = mean(edges) if edges else float("nan")
    # Per-metric mean and std across seeds.
    for col in METRICS:
        vals = [_flt(r[col]) for r in sub]
        vals = [v for v in vals if v is not None]
        out[f"{col}_mean"] = mean(vals) if vals else float("nan")
        out[f"{col}_std"]  = stdev(vals) if len(vals) > 1 else 0.0
    return out


# ────────────────────────────────────────────────────────────────────────────
# Stage 1 — Aggregate.  Loop datasets × cells, compute mean±std, build the
# `summary` list. Each entry is one point on the final plot.
# ────────────────────────────────────────────────────────────────────────────

summary: list[dict] = []
for ds in DATASETS:
    rows = _load(ds)
    # Group by (method, k, w') so we can iterate unique cells.
    cells = defaultdict(list)
    for r in rows:
        cells[(r["Method"], r["k"], r["w_prime"])].append(r)
    for (method, k, w), _ in cells.items():
        if not _accept(method, k, w):
            continue
        agg = _agg(rows, method, k, w)
        if agg is None: continue
        agg["Dataset"] = ds
        summary.append(agg)


# ────────────────────────────────────────────────────────────────────────────
# Stage 2 — CSV out (long form).  One row per (dataset, method, k, w') cell
# with mean and std for every metric. This is the data backing both the
# plot and the markdown table.
# ────────────────────────────────────────────────────────────────────────────

out_csv = ROOT / "results" / f"3way_l2_summary{ARGS.tag}.csv"
fields  = ["Dataset", "Method", "k", "w_prime", "n_seeds", "edges_mean",
           "Macro_F1_mean", "Macro_F1_std",
           "CKA_mean", "CKA_std",
           "Pred_Agreement_mean", "Pred_Agreement_std"]
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w_csv = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w_csv.writeheader()
    for row in sorted(summary, key=lambda r: (r["Dataset"], r["Method"],
                                                r.get("edges_mean", 0))):
        w_csv.writerow(row)
print(f"Wrote {out_csv} ({len(summary)} rows)")


# ────────────────────────────────────────────────────────────────────────────
# Stage 3 — The headline plot.  Grid: 3 metrics × 4 datasets.
#
# Each panel plots all three methods as curves on a common edge-count axis
# (log scale). This is the FAIRNESS question — at matched density, which
# method gives the best quality? You read the plot like:
#   - Same x (edges) → which y (quality) is highest? That method wins at
#     this density level on this dataset.
#   - Curves cross or stay apart? Tells you whether ranking depends on
#     budget.
# ────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, len(DATASETS), figsize=(5 * len(DATASETS), 11), sharex=False)
metric_labels = {"Macro_F1": "Macro F1", "CKA": "Output CKA",
                 "Pred_Agreement": "Pred. Agreement vs Exact"}

handles_for_legend = []
for col_idx, ds in enumerate(DATASETS):
    ds_short = ds.replace("HGB_", "")
    rows = [s for s in summary if s["Dataset"] == ds]
    for row_idx, metric in enumerate(METRICS):
        ax = axes[row_idx, col_idx]
        for method in ["KMV", "KGRW", "MPRW"]:
            sub = sorted([s for s in rows if s["Method"] == method],
                         key=lambda r: r["edges_mean"])
            if not sub: continue
            xs  = [s["edges_mean"] for s in sub]
            ys  = [s[f"{metric}_mean"] for s in sub]
            errs = [s[f"{metric}_std"] for s in sub]
            line = ax.errorbar(xs, ys, yerr=errs, fmt="o-", color=COLORS[method],
                               label=method, capsize=2, markersize=4, linewidth=1.0)
            if col_idx == 0 and row_idx == 0:
                handles_for_legend.append(line)
        ax.set_xlabel("Edge count (log)")
        ax.set_ylabel(metric_labels[metric])
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        if row_idx == 0:
            ax.set_title(ds_short, fontsize=12, fontweight="bold")
        # Per-panel legend, top-left so it doesn't overlap data
        ax.legend(loc="lower right", frameon=True, fontsize=9,
                  fancybox=True, framealpha=0.9)

fig.suptitle("3-way comparison at L=2 (5 seeds): KMV (green) vs KGRW (blue) vs MPRW (red)\n"
             "X = edge count after materialization (log scale). Higher Y = better.",
             fontsize=13, y=1.02)
fig.tight_layout()
out_pdf = ROOT / "results" / f"3way_l2_scatter{ARGS.tag}.pdf"
fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
print(f"Wrote {out_pdf}")


# ────────────────────────────────────────────────────────────────────────────
# Stage 4 — Markdown summary table.  Same content as the plot, in a form
# that copy-pastes into slides.  One section per dataset; sorted by method
# then edge count to make trends easy to read down the column.
# ────────────────────────────────────────────────────────────────────────────

md = [f"# 3-way L=2 summary  (KMV vs KGRW vs MPRW, 5 seeds)\n"]
for ds in DATASETS:
    rows = sorted([s for s in summary if s["Dataset"] == ds],
                  key=lambda r: (r["Method"], r["edges_mean"]))
    md.append(f"\n## {ds}\n")
    md.append("| Method | k | w' | edges | F1 | CKA | PA |")
    md.append("|---|---|---|---:|---:|---:|---:|")
    for r in rows:
        md.append(f"| {r['Method']} | {r['k'] or '—'} | {r['w_prime'] or '—'} "
                  f"| {r['edges_mean']:,.0f} "
                  f"| {r['Macro_F1_mean']:.3f}±{r['Macro_F1_std']:.3f} "
                  f"| {r['CKA_mean']:.3f}±{r['CKA_std']:.3f} "
                  f"| {r['Pred_Agreement_mean']:.3f}±{r['Pred_Agreement_std']:.3f} |")
out_md = ROOT / "results" / f"3way_l2_summary{ARGS.tag}.md"
out_md.write_text("\n".join(md), encoding="utf-8")
print(f"Wrote {out_md}")
