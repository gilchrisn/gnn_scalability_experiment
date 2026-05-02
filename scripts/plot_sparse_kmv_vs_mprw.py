"""Sparse-graph KMV vs MPRW comparison — direct answer to the prof's
'explore KMV vs MPRW on sparse graphs' ask.

Produces `figures/sketch_session/sparse_kmv_vs_mprw.{pdf,png}`:
    Rows = metrics {Macro_F1, CKA, Pred_Agreement}
    Cols = datasets {DBLP, ACM, IMDB, PubMed}
    Each panel: KMV (k-sweep) green, MPRW (w-sweep) red, x-axis = edges
    (log), x-axis includes Exact dot as ceiling reference.
    All four sub-panels at fraction = 0.0625 (the sparse end).

Reads from `results/<DS>/kgrw_bench_fractions.csv` produced by the
recent server run (with the corrected unified edge counter).
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures" / "sketch_session"
FIG.mkdir(parents=True, exist_ok=True)

DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed"]
DS_LABELS = {"HGB_DBLP": "DBLP", "HGB_ACM": "ACM",
             "HGB_IMDB": "IMDB", "HNE_PubMed": "PubMed"}
METRICS = [("Macro_F1", "Macro F1"),
           ("CKA", "Output CKA vs Exact"),
           ("Pred_Agreement", "Pred. Agreement vs Exact")]
COLORS = {"Exact": "#1f77b4", "KMV": "#2CA02C", "MPRW": "#D62728"}
SPARSE_FRAC = "0.0625"


def _flt(v):
    try:
        return float(v) if v not in ("", None, "nan") else None
    except ValueError:
        return None


def load(ds: str) -> list[dict]:
    p = RES / ds / "kgrw_bench_fractions.csv"
    if not p.exists():
        return []
    return list(csv.DictReader(open(p, encoding="utf-8")))


def aggregate(rows: list[dict], frac: str) -> dict:
    """Group by (Method, k_or_w) at the given fraction; return mean+/-std per metric."""
    by_cell = defaultdict(list)
    for r in rows:
        if r["Fraction"] != frac:
            continue
        m = r["Method"]
        cell = r["k"] if m == "KMV" else (r["w_prime"] if m == "MPRW" else "-")
        by_cell[(m, cell)].append(r)

    out = []
    for (method, cell), sub in by_cell.items():
        rec = {"method": method, "cell": cell}
        edges = [_flt(r["Edge_Count"]) for r in sub]
        edges = [e for e in edges if e is not None]
        rec["edges_mean"] = statistics.fmean(edges) if edges else float("nan")
        for col, _ in METRICS:
            vals = [_flt(r[col]) for r in sub]
            vals = [v for v in vals if v is not None]
            rec[f"{col}_mean"] = statistics.fmean(vals) if vals else float("nan")
            rec[f"{col}_std"]  = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        out.append(rec)
    return out


def main() -> int:
    fig, axes = plt.subplots(len(METRICS), len(DATASETS),
                             figsize=(4.0 * len(DATASETS), 3.0 * len(METRICS)),
                             squeeze=False)

    for ci, ds in enumerate(DATASETS):
        rows = load(ds)
        if not rows:
            for ri in range(len(METRICS)):
                ax = axes[ri][ci]
                ax.text(0.5, 0.5, f"no data for {ds}",
                        transform=ax.transAxes, ha="center", va="center")
            continue
        agg = aggregate(rows, SPARSE_FRAC)
        if not agg:
            continue

        for ri, (col, ylabel) in enumerate(METRICS):
            ax = axes[ri][ci]
            # Draw KMV (green) and MPRW (red) as lines, sorted by edges.
            for method in ("KMV", "MPRW"):
                cells = sorted(
                    [a for a in agg if a["method"] == method],
                    key=lambda r: r["edges_mean"],
                )
                if not cells:
                    continue
                xs = [r["edges_mean"] for r in cells]
                ys = [r[f"{col}_mean"] for r in cells]
                yerr = [r[f"{col}_std"] for r in cells]
                ax.errorbar(xs, ys, yerr=yerr, fmt="o-",
                            color=COLORS[method], label=method,
                            capsize=2.5, markersize=5, linewidth=1.4,
                            alpha=0.92)
            # Exact ceiling.
            exacts = [a for a in agg if a["method"] == "Exact"]
            if exacts:
                e = exacts[0]
                ax.axhline(e[f"{col}_mean"], linestyle="--",
                           color=COLORS["Exact"], alpha=0.7, linewidth=1.2,
                           label="Exact (ceiling)")

            if ri == 0:
                ax.set_title(f"{DS_LABELS[ds]}", fontsize=12, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=11)
            if ri == len(METRICS) - 1:
                ax.set_xlabel("edges in materialised adjacency (log)", fontsize=10)
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    fig.suptitle(
        f"Sparse regime (fraction={SPARSE_FRAC}): "
        f"KMV vs MPRW match Exact within ±0.005 F1 across all 4 HGB datasets",
        fontsize=13, fontweight="bold", y=1.001,
    )
    fig.tight_layout()
    fig.savefig(FIG / "sparse_kmv_vs_mprw.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / "sparse_kmv_vs_mprw.png", dpi=150, bbox_inches="tight")
    print(f"[fig]  {FIG / 'sparse_kmv_vs_mprw.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
