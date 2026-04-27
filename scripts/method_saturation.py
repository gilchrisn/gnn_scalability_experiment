"""method_saturation.py — saturation + marginal-cost analysis for KMV / KGRW / MPRW.

PURPOSE
-------
Tests the **Marginal Cost Theorem** (THEORY.md §5) empirically.  The theorem
says MPRW's cost-per-new-edge dt/dE diverges as edges → max-neighborhood
(coupon-collector effect), while KMV's stays bounded.  KGRW should sit
between, with a crossover.

We plot 3 rows, one column per dataset, all three methods on each panel:

    Row 1 — edges vs primary parameter
        Are we even reaching saturation?  How fast does each method "fill in"?
        KMV against k.  MPRW against w.  KGRW against k (fixed w') and
        against w' (fixed k).

    Row 2 — time vs edges
        Direct cost picture.  Slope of each curve = avg cost per edge.
        If methods overlap, the cheaper-per-edge wins for any density target.

    Row 3 — marginal cost dt/dE   ★ THE TALK SLIDE ★
        Computed numerically: between consecutive parameter steps,
        ΔTime / ΔEdges.  This is the dealbreaker plot.  Theorem prediction:
            - MPRW: dt/dE rises monotonically → coupon collector visible
            - KMV: roughly flat → bounded marginal cost
            - KGRW: between, with crossover

Inputs:  results/<DS>/kgrw_bench.csv    (L=2)
Outputs: results/method_saturation.pdf  (3-row × N-dataset grid)
         results/method_saturation.csv  (long-form per-cell table)
         results/method_saturation.md   (per-cell markdown table per dataset)
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed"]
COLORS   = {"KMV": "#2CA02C", "KGRW": "#1F77B4", "MPRW": "#D62728"}


# ────────────────────────────────────────────────────────────────────────────
# Loading & aggregation helpers.
# ────────────────────────────────────────────────────────────────────────────

def _load(ds: str) -> list[dict]:
    """Load per-dataset CSV, filter to SAGE-depth L=2 (the row's "L" column
    is GNN depth, not metapath length — we run all benches at L_sage=2)."""
    with open(ROOT / "results" / ds / "kgrw_bench.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("L") == "2"]


def _flt(v):
    """Tolerant float parse (CSV cells may be empty)."""
    try: return float(v) if v not in ("", None) else None
    except ValueError: return None


def _agg_method(rows: list[dict], method: str
               ) -> list[tuple]:
    """For one method, aggregate seeds into one (edges, time, time_std)
    point per cell.

    Returns list of (k, w, edges_mean, time_mean, time_std), sorted by edges.
    time_std is the across-seed std of materialization time — used both
    for error bars on the time-vs-edges plot and for diagnosing whether
    a negative dt/dE is just noise (yes, almost always).
    """
    cells: dict[tuple, list[tuple[float, float]]] = {}
    for r in rows:
        if r["Method"] != method: continue
        try:
            k = int(float(r["k"])) if r["k"] else None
            w = int(float(r["w_prime"])) if r["w_prime"] else None
        except ValueError:
            continue
        e = _flt(r["Edge_Count"])
        t = _flt(r["Mat_Time_s"])
        if e is None or t is None: continue
        cells.setdefault((k, w), []).append((e, t))
    out = []
    for (k, w), pairs in cells.items():
        es = [p[0] for p in pairs]
        ts = [p[1] for p in pairs]
        t_mean = mean(ts)
        t_std  = stdev(ts) if len(ts) > 1 else 0.0
        out.append((k, w, mean(es), t_mean, t_std))
    return sorted(out, key=lambda r: r[2])


def _linfit(xs, ys):
    """Linear least squares y = a*x + b. Returns (slope, intercept, r2)
    or None if degenerate."""
    n = len(xs)
    if n < 2: return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    if den == 0: return None
    a = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / den
    b = my - a * mx
    ss_res = sum((ys[i] - (a * xs[i] + b)) ** 2 for i in range(n))
    ss_tot = sum((ys[i] - my) ** 2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


def _densest(kgrw, idx: int):
    """In a 2D KGRW grid, return the value of axis `idx` (0=k, 1=w') that
    appears in the most cells — i.e. the slice with the most data points.

    Picking the *largest* parameter often gives a sparse slice (only 3-4
    points on DBLP/IMDB). Picking the densest slice gives more points
    for the marginal/fit, at the cost of not always being the "biggest k"."""
    if not kgrw: return None
    counts = Counter(p[idx] for p in kgrw)
    return max(counts, key=counts.get)


def _marginal(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Numeric derivative dt/dE between consecutive (edges, time) points.

    For two adjacent parameter steps producing (E1, T1) and (E2, T2):
        marginal_cost = (T2 - T1) / (E2 - E1)
    Plotted at the midpoint of [E1, E2] so the curve sits sensibly on x.

    THIS IS THE METRIC FOR THE TALK SLIDE.  Rising = coupon collector
    (MPRW) or sketch-merge cost growing (KMV at very high k).  Flat =
    bounded (KMV in normal regime).
    """
    out = []
    for i in range(1, len(points)):
        e1, t1 = points[i - 1]
        e2, t2 = points[i]
        de = e2 - e1
        if de <= 0: continue
        out.append(((e1 + e2) / 2, (t2 - t1) / de))
    return out


# ────────────────────────────────────────────────────────────────────────────
# Stage 1 — Aggregate per dataset.  For each dataset, build the three
# per-method curves we'll plot in 3 rows below.
# ────────────────────────────────────────────────────────────────────────────

agg: dict = {}
for ds in DATASETS:
    rows = _load(ds)
    kmv  = _agg_method(rows, "KMV")     # (k, None, edges, time)
    mprw = _agg_method(rows, "MPRW")    # (None, w, edges, time)
    kgrw = _agg_method(rows, "KGRW")    # (k, w, edges, time)
    agg[ds] = {"KMV":  kmv, "MPRW": mprw, "KGRW": kgrw}


# ────────────────────────────────────────────────────────────────────────────
# Stage 2 — The plot.
#
# ONE row × N datasets.  Each panel: time vs edges, log-log.
#   KMV  (green line)   — 1D sweep over k, sorted by edges produced.
#   MPRW (red line)     — 1D sweep over w, sorted by edges produced.
#   KGRW (blue scatter) — 2D grid over (k, w'), one dot per cell.
#
# How to read:
#   X = how many edges the materialised graph Ã ended up with (density).
#   Y = wall-clock cost to produce those edges.
#   Local slope of a curve = cost per edge at that density.
#   Saturation = the rightmost x value a method's curve reaches.  KMV
#     stops well before MPRW because at large k it can't add edges that
#     don't exist; MPRW keeps trying with more walks.
#
# What we want to see in the picture:
#   - KMV sits in the lower-left and plateaus on x: cheap, capped density.
#   - MPRW reaches further right on x but bends up on y near the right
#     edge: each extra edge costs more (coupon collector).
#   - KGRW (blue dots) — does the cloud sit between the two lines, on
#     KMV's, or above MPRW's?  That's the answer to "does KGRW work".
# ────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(DATASETS),
                         figsize=(4.5 * len(DATASETS), 4.2))
if len(DATASETS) == 1:
    axes = [axes]

for col, ds in enumerate(DATASETS):
    ds_short = ds.replace("HGB_", "").replace("HNE_", "")
    ax = axes[col]

    # KMV: 1D sweep over k → connected line sorted by edges.
    kmv = sorted([(p[2], p[3]) for p in agg[ds]["KMV"]])
    if kmv:
        ax.plot([p[0] for p in kmv], [p[1] for p in kmv],
                "o-", color=COLORS["KMV"], label="KMV (sweep k)",
                markersize=5, linewidth=1.6)
    # MPRW: 1D sweep over w → connected line sorted by edges.
    mprw = sorted([(p[2], p[3]) for p in agg[ds]["MPRW"]])
    if mprw:
        ax.plot([p[0] for p in mprw], [p[1] for p in mprw],
                "s-", color=COLORS["MPRW"], label="MPRW (sweep w)",
                markersize=5, linewidth=1.6)
    # KGRW: 2D grid → scatter cloud, one dot per (k, w') cell.
    kgrw = [(p[2], p[3]) for p in agg[ds]["KGRW"]]
    if kgrw:
        ax.scatter([p[0] for p in kgrw], [p[1] for p in kgrw],
                   marker="^", color=COLORS["KGRW"], s=22, alpha=0.55,
                   label=f"KGRW (k × w', n={len(kgrw)})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("edges in Ã")
    if col == 0:
        ax.set_ylabel("materialisation time (s)")
    ax.set_title(ds_short, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)

fig.suptitle("Cost vs density at L=2 — slope on log-log = cost per edge",
             fontsize=12, y=1.02)
fig.tight_layout()
out_pdf = ROOT / "results" / "method_saturation.pdf"
fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
print(f"Wrote {out_pdf}")


# ────────────────────────────────────────────────────────────────────────────
# Stage 3 — Per-cell CSV + Markdown.  One row per (dataset, method, k, w')
# cell with mean edges, mean materialization time, and marginal dt/dE
# (= cost of the *next* edge between this cell and the previous one in the
# same method's sweep, sorted by edge count). Sweeps are intra-method:
# the marginal compares cell i to cell i-1 within the same method only.
# ────────────────────────────────────────────────────────────────────────────

def _emit_rows(ds: str, method: str):
    """Yield dicts for every cell in this (ds, method) sweep, with
    marginal_dtdE relative to the previous cell in this method's curve.
    Includes time std across seeds — useful for telling apart real
    saturation jumps from timer noise."""
    pts = agg[ds][method]   # already sorted by edges_mean
    prev_e = prev_t = None
    for k, w, e, t, t_std in pts:
        if prev_e is not None and e > prev_e:
            dtde = (t - prev_t) / (e - prev_e)
        else:
            dtde = None
        yield {
            "Dataset":      ds,
            "Method":       method,
            "k":            k if k is not None else "",
            "w_prime":      w if w is not None else "",
            "edges_mean":   f"{e:.0f}",
            "mat_time_s":   f"{t:.4f}",
            "mat_time_std": f"{t_std:.4f}",
            "marginal_dtdE": "" if dtde is None else f"{dtde:.6e}",
        }
        prev_e, prev_t = e, t

out_csv = ROOT / "results" / "method_saturation.csv"
fields = ["Dataset", "Method", "k", "w_prime",
          "edges_mean", "mat_time_s", "mat_time_std", "marginal_dtdE"]
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w_csv = csv.DictWriter(f, fieldnames=fields)
    w_csv.writeheader()
    for ds in DATASETS:
        for method in ["KMV", "KGRW", "MPRW"]:
            for row in _emit_rows(ds, method):
                w_csv.writerow(row)
print(f"Wrote {out_csv}")


# Markdown — every cell, grouped by dataset, sorted by Method then edges.

md = ["# Method saturation @ L=2 — KMV / KGRW / MPRW\n",
      "Every (method, k, w') cell. Sorted by Method then edge count.",
      "`dt/dE` = marginal cost of the next edge vs the previous cell in",
      "the same method's sweep (empty for the smallest cell of each method).\n"]

for ds in DATASETS:
    md.append(f"\n## {ds}\n")
    md.append("| Method | k | w' | edges | mat time mean ±std (s) | dt/dE (s/edge) |")
    md.append("|---|---|---|---:|---:|---:|")
    for method in ["KMV", "KGRW", "MPRW"]:
        for row in _emit_rows(ds, method):
            k_disp = row["k"] or "—"
            w_disp = row["w_prime"] or "—"
            e_disp = f"{int(row['edges_mean']):,}"
            t_disp = f"{row['mat_time_s']} ± {row['mat_time_std']}"
            dtde_disp = "—" if row["marginal_dtdE"] == "" \
                            else f"{float(row['marginal_dtdE']):.2e}"
            md.append(f"| {method} | {k_disp} | {w_disp} | {e_disp} "
                      f"| {t_disp} | {dtde_disp} |")

out_md = ROOT / "results" / "method_saturation.md"
out_md.write_text("\n".join(md), encoding="utf-8")
print(f"Wrote {out_md}")
