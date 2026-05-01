"""aggregate_fractions.py — combined plots over the fraction sweep.

Reads `results/<DATASET>/kgrw_bench_fractions.csv` (produced by
`bench_fraction_sweep.py`) for each requested dataset, aggregates seeds, and
writes two figures:

* `results/3way_l2_scatter_fractions.pdf`
    Rows = metrics (F1, CKA, PA) × datasets — so 3 × N_datasets rows.
    Cols = fractions (5).  All three methods overlaid on each panel,
    x = edge count (log).  Reads same color scheme as `aggregate_3way_l2.py`.
    F1 row is omitted for datasets where all F1 cells are blank (untrained).

* `results/method_saturation_fractions.pdf`
    Rows = datasets, cols = fractions.  Per panel: time vs edges (log-log),
    KMV / MPRW lines + KGRW scatter, same convention as
    `scripts/method_saturation.py`.

CLI
---
    python scripts/aggregate_fractions.py \\
        --datasets HNE_PubMed OGB_MAG \\
        [--csv-name kgrw_bench_fractions.csv]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"KMV": "#2CA02C", "KGRW": "#1F77B4", "MPRW": "#D62728",
          "Exact": "#1f77b4"}
METRICS = ["Macro_F1", "CKA", "Pred_Agreement"]
METRIC_LABELS = {"Macro_F1": "Macro F1", "CKA": "Output CKA",
                 "Pred_Agreement": "Pred. Agreement vs Reference"}


def _flt(v):
    try: return float(v) if v not in ("", None) else None
    except ValueError: return None


def _load(ds: str, csv_name: str, root_dir: str = "results") -> list[dict]:
    p = ROOT / root_dir / ds / csv_name
    if not p.exists():
        print(f"[skip] {p} not found")
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _agg_cells(rows: list[dict]) -> list[dict]:
    """Collapse seeds into one mean±std record per (Fraction, Method, k, w')."""
    cells = defaultdict(list)
    for r in rows:
        key = (r["Fraction"], r["Method"], r["k"], r["w_prime"])
        cells[key].append(r)

    out = []
    for (frac, method, k, w), sub in cells.items():
        rec = {"Fraction": float(frac), "Method": method, "k": k, "w_prime": w,
               "n_seeds": len(sub)}
        edges = [_flt(r["Edge_Count"]) for r in sub]
        edges = [e for e in edges if e is not None]
        rec["edges_mean"] = mean(edges) if edges else float("nan")
        dens = [_flt(r.get("Density", "")) for r in sub]
        dens = [d for d in dens if d is not None]
        rec["density_mean"] = mean(dens) if dens else float("nan")
        times = [_flt(r["Mat_Time_s"]) for r in sub]
        times = [t for t in times if t is not None]
        rec["time_mean"] = mean(times) if times else float("nan")
        rec["time_std"]  = stdev(times) if len(times) > 1 else 0.0
        rams = [_flt(r.get("Mat_Peak_RAM_MB", "")) for r in sub]
        rams = [v for v in rams if v is not None]
        rec["ram_mean"] = mean(rams) if rams else float("nan")
        for col in METRICS:
            vals = [_flt(r[col]) for r in sub]
            vals = [v for v in vals if v is not None]
            rec[f"{col}_mean"] = mean(vals) if vals else float("nan")
            rec[f"{col}_std"]  = stdev(vals) if len(vals) > 1 else 0.0
        out.append(rec)
    return out


# ─── Plot 1: 3-way scatter, fraction-faceted ──────────────────────────────

def plot_3way(agg: dict, fractions: list[float], out_pdf: Path) -> None:
    """Rows: metrics × datasets ; Cols: fractions ; methods overlaid."""
    datasets = list(agg.keys())
    # F1 row is dropped if all F1 cells in a dataset are NaN (untrained mode)
    rows: list[tuple[str, str, str]] = []  # (ds, metric, label)
    for ds in datasets:
        for metric in METRICS:
            has_data = any(
                rec[f"{metric}_mean"] == rec[f"{metric}_mean"]
                for rec in agg[ds]
            )
            if has_data:
                rows.append((ds, metric, f"{ds.replace('HNE_','').replace('HGB_','').replace('OGB_','')}\n{METRIC_LABELS[metric]}"))

    if not rows or not fractions:
        print(f"[plot_3way] no plottable rows or fractions"); return

    n_rows, n_cols = len(rows), len(fractions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows),
                             squeeze=False)

    for ri, (ds, metric, label) in enumerate(rows):
        for ci, frac in enumerate(fractions):
            ax = axes[ri][ci]
            cells = [r for r in agg[ds] if r["Fraction"] == frac
                     and r["Method"] in ("KMV", "KGRW", "MPRW")]
            for method in ("KMV", "KGRW", "MPRW"):
                sub = sorted([c for c in cells if c["Method"] == method],
                             key=lambda r: r["edges_mean"])
                if not sub: continue
                xs = [c["edges_mean"] for c in sub]
                ys = [c[f"{metric}_mean"] for c in sub]
                yerr = [c[f"{metric}_std"] for c in sub]
                ax.errorbar(xs, ys, yerr=yerr, fmt="o-",
                            color=COLORS[method], label=method,
                            capsize=2, markersize=4, linewidth=1.0)
            if ri == 0:
                ax.set_title(f"frac={frac:.4f}", fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.set_xscale("log"); ax.grid(alpha=0.3)
            if ri == n_rows - 1:
                ax.set_xlabel("edges in Ã (log)")
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle(f"3-way fraction sweep: KMV (green) / KGRW (blue) / MPRW (red).  "
                 f"X = edges, log scale.", fontsize=12, y=1.001)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_pdf}")


# ─── Plot 2: method saturation, fraction-faceted ──────────────────────────

def plot_saturation(agg: dict, fractions: list[float], out_pdf: Path) -> None:
    """Rows: datasets ; Cols: fractions. time vs edges, log-log per panel."""
    datasets = list(agg.keys())
    n_rows, n_cols = len(datasets), len(fractions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.4 * n_rows),
                             squeeze=False)

    for ri, ds in enumerate(datasets):
        ds_short = ds.replace("HNE_", "").replace("HGB_", "").replace("OGB_", "")
        for ci, frac in enumerate(fractions):
            ax = axes[ri][ci]
            cells = [c for c in agg[ds] if c["Fraction"] == frac]
            kmv  = sorted([(c["edges_mean"], c["time_mean"]) for c in cells if c["Method"] == "KMV"])
            mprw = sorted([(c["edges_mean"], c["time_mean"]) for c in cells if c["Method"] == "MPRW"])
            kgrw = [(c["edges_mean"], c["time_mean"]) for c in cells if c["Method"] == "KGRW"]
            if kmv:
                ax.plot([p[0] for p in kmv], [p[1] for p in kmv], "o-",
                        color=COLORS["KMV"], label="KMV (sweep k)",
                        markersize=5, linewidth=1.6)
            if mprw:
                ax.plot([p[0] for p in mprw], [p[1] for p in mprw], "s-",
                        color=COLORS["MPRW"], label="MPRW (sweep w)",
                        markersize=5, linewidth=1.6)
            if kgrw:
                ax.scatter([p[0] for p in kgrw], [p[1] for p in kgrw],
                           marker="^", color=COLORS["KGRW"], s=22, alpha=0.55,
                           label=f"KGRW (n={len(kgrw)})")
            ax.set_xscale("log"); ax.set_yscale("log")
            if ri == 0:
                ax.set_title(f"frac={frac:.4f}", fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{ds_short}\nmat time (s)", fontsize=10)
            if ri == n_rows - 1:
                ax.set_xlabel("edges in Ã")
            ax.grid(alpha=0.3, which="both")
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle("Method saturation: cost vs density per fraction.  "
                 "Log-log; slope = cost per edge.", fontsize=12, y=1.001)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_pdf}")


# ─── CSV summary ───────────────────────────────────────────────────────────

def write_summary_csv(agg: dict, out_csv: Path) -> None:
    fields = ["Dataset", "Fraction", "Method", "k", "w_prime", "n_seeds",
              "edges_mean", "density_mean",
              "time_mean", "time_std", "ram_mean",
              "Macro_F1_mean", "Macro_F1_std",
              "CKA_mean", "CKA_std",
              "Pred_Agreement_mean", "Pred_Agreement_std"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for ds, recs in agg.items():
            for rec in sorted(recs, key=lambda r: (r["Fraction"], r["Method"],
                                                   r.get("edges_mean", 0))):
                w.writerow({"Dataset": ds, **rec})
    print(f"Wrote {out_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+",
                   default=["HNE_PubMed", "OGB_MAG"])
    p.add_argument("--csv-name", default="kgrw_bench_fractions.csv")
    p.add_argument("--root-dir", default="results",
                   help="Directory under project root containing per-dataset "
                        "subfolders with the CSV (default: results)")
    p.add_argument("--out-dir", default=None,
                   help="Where to write figures and summary CSV "
                        "(default: same as --root-dir)")
    p.add_argument("--out-suffix", default="",
                   help="Optional suffix appended to output filenames "
                        "(e.g. '_transfer' → '3way_l2_scatter_fractions_transfer.pdf')")
    args = p.parse_args()

    agg: dict[str, list[dict]] = {}
    all_fracs: set[float] = set()
    for ds in args.datasets:
        rows = _load(ds, args.csv_name, args.root_dir)
        if not rows: continue
        agg[ds] = _agg_cells(rows)
        all_fracs.update(r["Fraction"] for r in agg[ds])

    fractions = sorted(all_fracs)
    if not fractions:
        print("No data to plot."); return

    out_dir = ROOT / (args.out_dir or args.root_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = args.out_suffix
    plot_3way(agg, fractions, out_dir / f"3way_l2_scatter_fractions{sfx}.pdf")
    plot_saturation(agg, fractions, out_dir / f"method_saturation_fractions{sfx}.pdf")
    write_summary_csv(agg, out_dir / f"fractions_summary{sfx}.csv")


if __name__ == "__main__":
    main()
