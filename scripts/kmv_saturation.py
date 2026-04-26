"""kmv_saturation.py — analyse how fast KMV's edge count saturates with k.

For each dataset, reads kgrw_bench.csv, extracts KMV rows at L=2, and:
  - Plots edge_count vs k (log scale)
  - Estimates saturation k* (smallest k with d(edges)/dlog(k) < 5% of plateau)
  - Compares KMV plateau to MPRW asymptote (largest w in CSV)

Outputs
-------
results/kmv_saturation.pdf   — 1x3 panel, edge_count vs k per dataset
results/kmv_saturation.md    — table of saturation k* + plateau ratios
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed"]


def _load(ds: str) -> list[dict]:
    with open(ROOT / "results" / ds / "kgrw_bench.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("L") == "2"]


def _flt(v):
    try: return float(v) if v not in ("", None) else None
    except ValueError: return None


def _avg_edges(rows: list[dict], method: str, k_or_w: str, key: str) -> dict[int, float]:
    """Returns {param_value: mean_edges_across_seeds}."""
    out: dict[int, list[float]] = {}
    for r in rows:
        if r["Method"] != method: continue
        kv = r[key]
        if not kv: continue
        try: param = int(float(kv))
        except ValueError: continue
        e = _flt(r["Edge_Count"])
        if e is None: continue
        out.setdefault(param, []).append(e)
    return {k: mean(v) for k, v in sorted(out.items())}


def _saturation_k(kmv_curve: dict[int, float], thresh_frac: float = 0.02) -> int | None:
    """Smallest k where (edges[k+1] - edges[k]) / edges[k] < thresh_frac."""
    keys = sorted(kmv_curve.keys())
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        ea, eb = kmv_curve[a], kmv_curve[b]
        if ea > 0 and (eb - ea) / ea < thresh_frac:
            return a
    return None


# ─── Aggregate ──────────────────────────────────────────────────────────────

results = {}
for ds in DATASETS:
    rows = _load(ds)
    kmv_curve  = _avg_edges(rows, "KMV",  "k_value", "k")           if False else _avg_edges(rows, "KMV",  "k", "k")
    mprw_curve = _avg_edges(rows, "MPRW", "w",       "w_prime")
    kgrw_at_k_max_w = {}
    for r in rows:
        if r["Method"] != "KGRW": continue
        try:
            k  = int(float(r["k"])); wp = int(float(r["w_prime"]))
        except (ValueError, TypeError):
            continue
        e = _flt(r["Edge_Count"])
        if e is None: continue
        # Aggregate KGRW max edges per k (= largest w' shows the saturation potential)
        kgrw_at_k_max_w.setdefault(k, []).append((wp, e))
    kgrw_max = {}
    for k, lst in kgrw_at_k_max_w.items():
        lst.sort()
        # Pick edges at largest w' for that k
        kgrw_max[k] = lst[-1][1]

    sat_k = _saturation_k(kmv_curve, thresh_frac=0.02)
    kmv_plateau  = max(kmv_curve.values())  if kmv_curve  else 0
    mprw_plateau = max(mprw_curve.values()) if mprw_curve else 0
    kgrw_plateau = max(kgrw_max.values())   if kgrw_max   else 0

    results[ds] = {
        "kmv_curve":     kmv_curve,
        "mprw_curve":    mprw_curve,
        "kgrw_max":      kgrw_max,
        "sat_k":         sat_k,
        "kmv_plateau":   kmv_plateau,
        "mprw_plateau":  mprw_plateau,
        "kgrw_plateau":  kgrw_plateau,
    }


# ─── Plot ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.3 * len(DATASETS), 5), sharey=False)
for ax, ds in zip(axes, DATASETS):
    R = results[ds]
    if R["kmv_curve"]:
        ks = sorted(R["kmv_curve"].keys())
        ax.plot(ks, [R["kmv_curve"][k] for k in ks],
                "o-", color="#2CA02C", label="KMV edge count",
                markersize=7, linewidth=2)
    if R["sat_k"] is not None:
        ax.axvline(R["sat_k"], color="#2CA02C", linestyle=":", alpha=0.8, linewidth=2,
                   label=f"KMV saturation k* = {R['sat_k']}")
    if R["mprw_plateau"] > 0:
        ax.axhline(R["mprw_plateau"], color="#D62728", linestyle="--", alpha=0.7,
                   linewidth=1.8,
                   label=f"MPRW plateau (max w): {R['mprw_plateau']:,.0f} edges")
    if R["kgrw_plateau"] > 0:
        ax.axhline(R["kgrw_plateau"], color="#1F77B4", linestyle="--", alpha=0.7,
                   linewidth=1.8,
                   label=f"KGRW plateau (max k,w): {R['kgrw_plateau']:,.0f} edges")
    ax.set_title(ds.replace("HGB_", ""), fontsize=12, fontweight="bold")
    ax.set_xlabel("k = sketch size  (log₂)")
    ax.set_ylabel("Edge count (mean over 5 seeds)")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, frameon=True, framealpha=0.95, loc="lower right")

fig.suptitle("KMV edge-count saturation at L=2  (green dashed line = saturation k*)",
             fontsize=13, y=1.03, fontweight="bold")
fig.tight_layout()
out_pdf = ROOT / "results" / "kmv_saturation.pdf"
fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
print(f"Wrote {out_pdf}")


# ─── Markdown summary ───────────────────────────────────────────────────────

md = ["# KMV saturation analysis @ L=2\n",
      "Saturation k* = smallest k where adding 1 doubling of k yields < 2% more edges.",
      "Plateau values from largest available k/w in CSV.\n",
      "| Dataset | sat k* | KMV plateau | MPRW plateau | KGRW plateau | KMV/MPRW |",
      "|---|---:|---:|---:|---:|---:|"]
for ds in DATASETS:
    R = results[ds]
    sat_str = str(R["sat_k"]) if R["sat_k"] is not None else "not reached"
    ratio   = R["kmv_plateau"] / R["mprw_plateau"] if R["mprw_plateau"] > 0 else 0
    md.append(f"| {ds.replace('HGB_','')} | {sat_str} "
              f"| {R['kmv_plateau']:,.0f} "
              f"| {R['mprw_plateau']:,.0f} "
              f"| {R['kgrw_plateau']:,.0f} "
              f"| {ratio:.2f} |")

md.append("\n## Per-dataset KMV edge count vs k\n")
for ds in DATASETS:
    R = results[ds]
    if not R["kmv_curve"]: continue
    md.append(f"### {ds}\n")
    md.append("| k | edges | Δ vs prev | Δ% |")
    md.append("|---:|---:|---:|---:|")
    keys = sorted(R["kmv_curve"].keys())
    prev = None
    for k in keys:
        e = R["kmv_curve"][k]
        if prev is None:
            md.append(f"| {k} | {e:,.0f} | — | — |")
        else:
            d = e - prev
            dp = (d / prev * 100) if prev > 0 else 0
            md.append(f"| {k} | {e:,.0f} | +{d:,.0f} | +{dp:.1f}% |")
        prev = e
    md.append("")

out_md = ROOT / "results" / "kmv_saturation.md"
out_md.write_text("\n".join(md), encoding="utf-8")
print(f"Wrote {out_md}")
