"""method_saturation.py — saturation + marginal-cost analysis for KMV / KGRW / MPRW.

Three lenses (1 row each, 3 columns one per dataset):
    (1) Edges vs primary parameter (k for KMV, w for MPRW, k×w for KGRW)
    (2) Time vs edges                — slope = cost per new edge
    (3) Marginal cost dt/dE          — does it diverge (coupon collector) or plateau?

Inputs:  results/<DS>/kgrw_bench.csv    (L=2)
Outputs: results/method_saturation.pdf
         results/method_saturation.md
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed"]
COLORS   = {"KMV": "#2CA02C", "KGRW": "#1F77B4", "MPRW": "#D62728"}


def _load(ds: str) -> list[dict]:
    with open(ROOT / "results" / ds / "kgrw_bench.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("L") == "2"]


def _flt(v):
    try: return float(v) if v not in ("", None) else None
    except ValueError: return None


def _agg_method(rows: list[dict], method: str
               ) -> list[tuple[int, int, float, float]]:
    """Return list of (k_or_None, w_or_None, edges_mean, time_mean) sorted by edges.
    For KMV: k set, w None.  For MPRW: w set, k None.  For KGRW: both set."""
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
        out.append((k, w, mean(es), mean(ts)))
    return sorted(out, key=lambda r: r[2])


def _marginal(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Given list of (edges, time) sorted by edges, return [(edges, dt/dE)]."""
    out = []
    for i in range(1, len(points)):
        e1, t1 = points[i - 1]
        e2, t2 = points[i]
        de = e2 - e1
        if de <= 0: continue
        out.append(((e1 + e2) / 2, (t2 - t1) / de))
    return out


# ─── Aggregate per dataset ─────────────────────────────────────────────────

agg: dict = {}
for ds in DATASETS:
    rows = _load(ds)
    kmv  = _agg_method(rows, "KMV")     # (k, None, edges, time)
    mprw = _agg_method(rows, "MPRW")    # (None, w, edges, time)
    kgrw = _agg_method(rows, "KGRW")    # (k, w, edges, time)

    agg[ds] = {"KMV":  kmv, "MPRW": mprw, "KGRW": kgrw}


# ─── Plot ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, len(DATASETS), figsize=(5 * len(DATASETS), 13))

for col, ds in enumerate(DATASETS):
    ds_short = ds.replace("HGB_", "")

    # Row 1 — edges vs primary param
    ax = axes[0, col]
    kmv_pts  = [(p[0], p[2]) for p in agg[ds]["KMV"]]   # (k, edges)
    mprw_pts = [(p[1], p[2]) for p in agg[ds]["MPRW"]]  # (w, edges)
    if kmv_pts:
        kmv_pts.sort()
        ax.plot([p[0] for p in kmv_pts], [p[1] for p in kmv_pts],
                "o-", color=COLORS["KMV"], label="KMV  (x = sketch size k)",
                markersize=6, linewidth=2)
    if mprw_pts:
        mprw_pts.sort()
        ax.plot([p[0] for p in mprw_pts], [p[1] for p in mprw_pts],
                "s-", color=COLORS["MPRW"], label="MPRW (x = walks per node w)",
                markersize=6, linewidth=2)
    # KGRW: pick k-sweep at largest w', and w-sweep at largest k
    kgrw = agg[ds]["KGRW"]
    if kgrw:
        ks = sorted({p[0] for p in kgrw})
        ws = sorted({p[1] for p in kgrw})
        # k-sweep at largest w'
        if ws:
            wmax = ws[-1]
            sub = sorted([(p[0], p[2]) for p in kgrw if p[1] == wmax])
            ax.plot([p[0] for p in sub], [p[1] for p in sub],
                    "^--", color=COLORS["KGRW"],
                    label=f"KGRW (x = k, fixed w'={wmax})",
                    markersize=6, linewidth=1.5, alpha=0.9)
        # w-sweep at largest k
        if ks:
            kmax = ks[-1]
            sub = sorted([(p[1], p[2]) for p in kgrw if p[0] == kmax])
            ax.plot([p[0] for p in sub], [p[1] for p in sub],
                    "v:", color=COLORS["KGRW"],
                    label=f"KGRW (x = w', fixed k={kmax})",
                    markersize=6, linewidth=1.5, alpha=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Method primary parameter (k or w, log₂)")
    ax.set_ylabel("Edge count")
    ax.set_title(f"{ds_short}: how edges grow with parameter",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)

    # Row 2 — time vs edges (slope = cost per edge)
    ax = axes[1, col]
    for label, pts in [("KMV",  [(p[2], p[3]) for p in agg[ds]["KMV"]]),
                       ("MPRW", [(p[2], p[3]) for p in agg[ds]["MPRW"]]),
                       ("KGRW", [(p[2], p[3]) for p in agg[ds]["KGRW"]])]:
        if not pts: continue
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                "o", color=COLORS[label], label=label, markersize=5, alpha=0.7)
    ax.set_xlabel("Edge count")
    ax.set_ylabel("Materialization time (s)")
    ax.set_title(f"{ds_short}: total time vs edges produced",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)

    # Row 3 — marginal dt/dE
    ax = axes[2, col]
    for label in ["KMV", "MPRW", "KGRW"]:
        pts = sorted([(p[2], p[3]) for p in agg[ds][label]])
        marg = _marginal(pts)
        if marg:
            ax.plot([p[0] for p in marg], [p[1] for p in marg],
                    "o-", color=COLORS[label], label=label,
                    markersize=5, linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Edge count (midpoint of interval)")
    ax.set_ylabel("Marginal cost  dt / dE  (s per new edge)")
    ax.set_title(f"{ds_short}: cost per next edge\n"
                 f"(rising = coupon-collector / saturation)",
                 fontsize=11, fontweight="bold")
    ax.set_yscale("symlog", linthresh=1e-7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)

fig.suptitle("Method saturation at L=2:  KMV (green) vs KGRW (blue) vs MPRW (red)\n"
             "Row 1 = edge growth | Row 2 = total time | Row 3 = marginal cost (dt/dE)",
             fontsize=13, y=1.0, fontweight="bold")
fig.tight_layout()
out_pdf = ROOT / "results" / "method_saturation.pdf"
fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
print(f"Wrote {out_pdf}")


# ─── Markdown — saturation point + marginal-cost regime per method ────────

md = ["# Method saturation @ L=2 — KMV / KGRW / MPRW\n",
      "Saturation = first parameter step where edges grow < 5% (per doubling).",
      "Marginal cost = dt/dE; rising = coupon collector (MPRW) or capacity exhaustion (KMV).\n"]

def _saturation_param(pts: list[tuple[int | None, float]],
                       thresh: float = 0.05) -> str:
    """Smallest parameter where Δedges/edges < thresh on next doubling."""
    pts = [p for p in pts if p[0] is not None]
    pts.sort()
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if a[1] > 0 and (b[1] - a[1]) / a[1] < thresh:
            return str(a[0])
    return f">{pts[-1][0]}" if pts else "—"

for ds in DATASETS:
    md.append(f"\n## {ds}\n")
    md.append("| Method | sweep | sat. param | edge plateau | mat time at plateau |")
    md.append("|---|---|---:|---:|---:|")
    kmv  = [(p[0], p[2], p[3]) for p in agg[ds]["KMV"]]
    mprw = [(p[1], p[2], p[3]) for p in agg[ds]["MPRW"]]
    kgrw = agg[ds]["KGRW"]

    if kmv:
        sat = _saturation_param([(x[0], x[1]) for x in kmv])
        e   = max(x[1] for x in kmv)
        t   = next(x[2] for x in kmv if x[1] == e)
        md.append(f"| KMV  | k          | {sat} | {e:,.0f} | {t:.4f}s |")
    if mprw:
        sat = _saturation_param([(x[0], x[1]) for x in mprw])
        e   = max(x[1] for x in mprw)
        t   = next(x[2] for x in mprw if x[1] == e)
        md.append(f"| MPRW | w          | {sat} | {e:,.0f} | {t:.4f}s |")
    if kgrw:
        # KGRW k-sweep at largest w'
        ws = sorted({p[1] for p in kgrw})
        if ws:
            wmax = ws[-1]
            sub = sorted([(p[0], p[2], p[3]) for p in kgrw if p[1] == wmax])
            sat = _saturation_param([(x[0], x[1]) for x in sub])
            e   = max(x[1] for x in sub)
            t   = next(x[2] for x in sub if x[1] == e)
            md.append(f"| KGRW | k @ w'={wmax} | {sat} | {e:,.0f} | {t:.4f}s |")
        # KGRW w-sweep at largest k
        ks = sorted({p[0] for p in kgrw})
        if ks:
            kmax = ks[-1]
            sub = sorted([(p[1], p[2], p[3]) for p in kgrw if p[0] == kmax])
            sat = _saturation_param([(x[0], x[1]) for x in sub])
            e   = max(x[1] for x in sub)
            t   = next(x[2] for x in sub if x[1] == e)
            md.append(f"| KGRW | w' @ k={kmax} | {sat} | {e:,.0f} | {t:.4f}s |")

out_md = ROOT / "results" / "method_saturation.md"
out_md.write_text("\n".join(md), encoding="utf-8")
print(f"Wrote {out_md}")
