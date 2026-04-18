"""gen_kgrw_plots.py — Generate all plots for KGRW supervisor report.

Reads:
  - results/<DS>/kgrw_bench.csv         (5 seeds × L={1..4} × methods × (k,w'))
  - results/tail_fraction.csv            (5 metapaths × 5 k values)

Writes to figures/kgrw/:
  fig1_marginal_cost.pdf     — MPRW edges/ms vs w (coupon-collector)
  fig2_cumulative_edges.pdf  — cumulative edges vs w, KGRW overlays
  fig3_quality_vs_density.pdf— 3×3 grid: {CKA, F1, PA} × 3 datasets
  fig4_time_vs_density.pdf   — time (ms) vs edges
  fig5_depth_sweep.pdf       — L-sweep CKA for minimal budget
  fig6_tail_fraction.pdf     — |T|/|V_L| @ k, 5 metapaths
  fig7_rdeg_cdf.pdf          — reverse-degree CDF, 5 metapaths
  fig8_tail_vs_advantage.pdf — theory-empirics bridge
"""
from __future__ import annotations

import csv
import io
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "figures" / "kgrw"
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB"]
METAPATH_LABEL = {"HGB_DBLP": "APAPA", "HGB_ACM": "PAPAP", "HGB_IMDB": "MDMDM"}
COL_MPRW = "#d62728"
COL_KGRW = "#2ca02c"
plt.rcParams.update({"font.size": 10, "figure.dpi": 110})


def _f(s):
    try: return float(s) if s else float("nan")
    except ValueError: return float("nan")


def _norm(x: str) -> str:
    if not x: return ""
    try: return str(int(float(x)))
    except ValueError: return x


def _load(ds: str, L: int = 2) -> list[dict]:
    p = ROOT / "results" / ds / "kgrw_bench.csv"
    rows = []
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("L", "")) != str(L): continue
            rows.append(r)
    return rows


def _agg(rows: list[dict], method: str, k: str, w: str) -> dict | None:
    m = [r for r in rows if r["Method"] == method
         and _norm(r["k"]) == _norm(k) and _norm(r["w_prime"]) == _norm(w)]
    if not m: return None
    def ms(v):
        v = [x for x in v if not np.isnan(x)]
        if not v: return (float("nan"), float("nan"))
        return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
    return {
        "edges": ms([_f(r["Edge_Count"]) for r in m]),
        "time":  ms([_f(r["Mat_Time_s"]) for r in m]),
        "f1":    ms([_f(r["Macro_F1"])   for r in m if r.get("Macro_F1")]),
        "cka":   ms([_f(r["CKA"])        for r in m if r.get("CKA")]),
        "pa":    ms([_f(r["Pred_Agreement"]) for r in m if r.get("Pred_Agreement")]),
    }


def _mprw_series(rows: list[dict]) -> list[tuple]:
    """Returns list of (w, agg_dict) sorted by w."""
    ws = sorted({int(float(r["w_prime"])) for r in rows
                 if r["Method"] == "MPRW" and r["w_prime"]})
    out = []
    for w in ws:
        a = _agg(rows, "MPRW", "", str(w))
        if a: out.append((w, a))
    return out


def _kgrw_grid(rows: list[dict]) -> dict[tuple[int, int], dict]:
    """Returns {(k, w'): agg_dict}."""
    ks = sorted({int(float(r["k"])) for r in rows
                 if r["Method"] == "KGRW" and r["k"]})
    wps = sorted({int(float(r["w_prime"])) for r in rows
                  if r["Method"] == "KGRW" and r["w_prime"]})
    grid = {}
    for k in ks:
        for wp in wps:
            a = _agg(rows, "KGRW", str(k), str(wp))
            if a: grid[(k, wp)] = a
    return grid


# ── Figure 1: MPRW marginal cost ─────────────────────────────────────────
def fig1_marginal_cost():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharex=False)
    for ax, ds in zip(axes, DATASETS):
        rows = _load(ds)
        series = _mprw_series(rows)
        if not series: continue
        ws, edges, times = [], [], []
        for w, a in series:
            ws.append(w); edges.append(a["edges"][0]); times.append(a["time"][0] * 1000)
        prev_e = 0
        ratios = []
        for e, t in zip(edges, times):
            de = e - prev_e
            ratios.append(de / t if t > 0 else 0)
            prev_e = e
        ax.plot(ws, ratios, "o-", color=COL_MPRW, linewidth=2, markersize=7)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("walk budget w")
        ax.set_ylabel("marginal Δedges / ms" if ax == axes[0] else "")
        ax.set_title(f"{ds.replace('HGB_','')} {METAPATH_LABEL[ds]}")
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle("MPRW coupon-collector: marginal edge discovery rate decays with w", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_marginal_cost.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig1_marginal_cost.pdf")


# ── Figure 2: cumulative edges vs w ──────────────────────────────────────
def fig2_cumulative_edges():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharex=False)
    for ax, ds in zip(axes, DATASETS):
        rows = _load(ds)
        mprw = _mprw_series(rows)
        ws = [w for w, _ in mprw]; es = [a["edges"][0] for _, a in mprw]
        ax.plot(ws, es, "o-", color=COL_MPRW, linewidth=2, label="MPRW",
                markersize=7)
        # KGRW scatter — pick a few k values across w' sweep
        grid = _kgrw_grid(rows)
        for k, marker in [(4, "s"), (16, "^"), (32, "D")]:
            ks_wps = sorted([(kk, w) for (kk, w) in grid if kk == k])
            if not ks_wps: continue
            xs = [w for _, w in ks_wps]
            ys = [grid[(k, w)]["edges"][0] for _, w in ks_wps]
            ax.plot(xs, ys, marker + "--", color=COL_KGRW, alpha=0.6,
                    markersize=6, label=f"KGRW k={k}")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("walk budget (w or w')")
        if ax == axes[0]: ax.set_ylabel("unique edges discovered")
        ax.set_title(f"{ds.replace('HGB_','')} {METAPATH_LABEL[ds]}")
        ax.grid(True, alpha=0.3, which="both")
        if ax == axes[-1]: ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Cumulative edges vs budget: KGRW jumps to higher density at same w'", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_cumulative_edges.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig2_cumulative_edges.pdf")


# ── Figure 3: quality vs edge density ────────────────────────────────────
def fig3_quality_vs_density():
    metrics = [("cka", "CKA"), ("f1", "Macro-F1"), ("pa", "Pred. Agreement")]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=False)
    for col, ds in enumerate(DATASETS):
        rows = _load(ds)
        mprw = _mprw_series(rows)
        grid = _kgrw_grid(rows)
        for row, (mk, mlabel) in enumerate(metrics):
            ax = axes[row, col]
            xm = [a["edges"][0] for _, a in mprw]
            ym = [a[mk][0] for _, a in mprw]
            em = [a[mk][1] for _, a in mprw]
            ax.errorbar(xm, ym, yerr=em, fmt="o-", color=COL_MPRW,
                        linewidth=1.5, markersize=6, capsize=2, label="MPRW")
            xs = [v["edges"][0] for v in grid.values()]
            ys = [v[mk][0] for v in grid.values()]
            es = [v[mk][1] for v in grid.values()]
            ax.errorbar(xs, ys, yerr=es, fmt="s", color=COL_KGRW,
                        markersize=5, capsize=1.5, alpha=0.6, label="KGRW (all k,w')")
            ax.set_xscale("log")
            if row == 0:
                ax.set_title(f"{ds.replace('HGB_','')} {METAPATH_LABEL[ds]}")
            if col == 0: ax.set_ylabel(mlabel)
            if row == 2: ax.set_xlabel("edge count (density)")
            ax.grid(True, alpha=0.3, which="both")
            if row == 0 and col == 2: ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Quality vs density: matched-density comparison across 3 datasets", y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_quality_vs_density.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig3_quality_vs_density.pdf")


# ── Figure 4: time vs density ────────────────────────────────────────────
def fig4_time_vs_density():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharex=False)
    for ax, ds in zip(axes, DATASETS):
        rows = _load(ds)
        mprw = _mprw_series(rows); grid = _kgrw_grid(rows)
        ax.plot([a["edges"][0] for _, a in mprw],
                [a["time"][0] * 1000 for _, a in mprw],
                "o-", color=COL_MPRW, linewidth=1.8, markersize=6, label="MPRW")
        ax.plot([v["edges"][0] for v in grid.values()],
                [v["time"][0] * 1000 for v in grid.values()],
                "s", color=COL_KGRW, markersize=5, alpha=0.6, label="KGRW")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("edge count");
        if ax == axes[0]: ax.set_ylabel("materialization time (ms)")
        ax.set_title(f"{ds.replace('HGB_','')} {METAPATH_LABEL[ds]}")
        ax.grid(True, alpha=0.3, which="both")
        if ax == axes[-1]: ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Time vs edges: KGRW spends KMV-Phase-1 overhead at low densities", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_time_vs_density.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig4_time_vs_density.pdf")


# ── Figure 5: depth sweep ────────────────────────────────────────────────
def fig5_depth_sweep():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharex=False)
    for ax, ds in zip(axes, DATASETS):
        mprw_pts, kgrw_pts = [], []
        for L in [1, 2, 3, 4]:
            rows = _load(ds, L=L)
            am = _agg(rows, "MPRW", "", "1")
            ak = _agg(rows, "KGRW", "4", "1")
            if am: mprw_pts.append((L, am["cka"]))
            if ak: kgrw_pts.append((L, ak["cka"]))
        if mprw_pts:
            xs, ys = zip(*mprw_pts)
            m = [y[0] for y in ys]; s = [y[1] for y in ys]
            ax.errorbar(xs, m, yerr=s, fmt="o-", color=COL_MPRW,
                        linewidth=1.8, markersize=7, capsize=3, label="MPRW w=1")
        if kgrw_pts:
            xs, ys = zip(*kgrw_pts)
            m = [y[0] for y in ys]; s = [y[1] for y in ys]
            ax.errorbar(xs, m, yerr=s, fmt="s-", color=COL_KGRW,
                        linewidth=1.8, markersize=7, capsize=3, label="KGRW k=4 w'=1")
        ax.set_xlabel("SAGE depth L");
        if ax == axes[0]: ax.set_ylabel("CKA vs Exact")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_title(f"{ds.replace('HGB_','')} {METAPATH_LABEL[ds]}")
        ax.grid(True, alpha=0.3)
        if ax == axes[-1]: ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Depth sweep at minimal budget: KGRW's KMV Phase-1 reduces IS-bias drift",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_depth_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig5_depth_sweep.pdf")


# ── Figure 6: tail fraction bar ──────────────────────────────────────────
def fig6_tail_fraction():
    p = ROOT / "results" / "tail_fraction.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    ks = [4, 8, 16, 32, 64]
    labels = [f"{r['dataset'].replace('HGB_','')}\n{r['metapath']}" for r in rows]
    x = np.arange(len(rows))
    w = 0.16
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ks)))
    for i, k in enumerate(ks):
        ys = [_f(r[f"tail_frac_k{k}"]) for r in rows]
        ax.bar(x + (i - 2) * w, ys, w, color=colors[i], label=f"k={k}")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("|T| / |V_L|   (fraction of endpoints with r(w) ≤ k)")
    ax.set_title("Tail fraction by dataset × metapath × sketch size k")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(True, alpha=0.3, axis="y"); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_tail_fraction.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig6_tail_fraction.pdf")


# ── Figure 7: reverse-degree CDF ─────────────────────────────────────────
def fig7_rdeg_cdf():
    from collections import Counter
    TARGETS = [
        ("HGB_DBLP", "apapa",  "exact_apapa.adj"),
        ("HGB_ACM",  "papap",  "exact_papap.adj"),
        ("HGB_IMDB", "mdmdm",  "exact_mdmdm.adj"),
        ("HGB_IMDB", "mam",    "exact_movie_to_actor_actor_to_movie.adj"),
        ("HGB_IMDB", "mdm",    "exact_movie_to_director_director_to_movie.adj"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(TARGETS)))
    for (ds, mp, fname), c in zip(TARGETS, colors):
        adj = ROOT / "results" / ds / fname
        if not adj.exists(): continue
        rdeg: Counter[int] = Counter()
        with open(adj) as f:
            for line in f:
                toks = line.split()
                if len(toks) < 2: continue
                for t in set(int(t) for t in toks[1:]):
                    rdeg[t] += 1
        vals = sorted(rdeg.values())
        xs = np.array(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, "-", color=c, linewidth=1.8,
                label=f"{ds.replace('HGB_','')} {mp} (|V_L|={len(xs)})")
    for k in [4, 8, 16, 32, 64]:
        ax.axvline(k, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        ax.text(k, 0.02, f"k={k}", rotation=90, fontsize=7,
                ha="right", va="bottom", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("reverse degree r(w)")
    ax.set_ylabel("cumulative fraction of endpoints")
    ax.set_title("Reverse-degree CDF: r(w) distribution differs sharply across metapaths")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_rdeg_cdf.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig7_rdeg_cdf.pdf")


# ── Figure 8: tail-fraction vs KGRW advantage ────────────────────────────
def fig8_tail_vs_advantage():
    # Compute KGRW advantage at lowest MPRW budget (w=1)
    points = []  # (tail_frac_k4, kgrw_dCKA_low_budget, label)
    tail = {r["dataset"]: _f(r["tail_frac_k4"])
            for r in csv.DictReader(open(ROOT / "results" / "tail_fraction.csv",
                                         encoding="utf-8"))
            if r["metapath"] in ["apapa", "papap", "mdmdm"]}
    for ds in DATASETS:
        rows = _load(ds)
        am = _agg(rows, "MPRW", "", "1")
        if not am: continue
        # Find KGRW config with closest edges to am
        grid = _kgrw_grid(rows)
        target = am["edges"][0]
        best = min(grid.items(), key=lambda kv: abs(kv[1]["edges"][0] - target))
        (k, wp), a_k = best
        d = a_k["cka"][0] - am["cka"][0]
        points.append((tail[ds], d, f"{ds.replace('HGB_','')} {METAPATH_LABEL[ds]}",
                       am["edges"][0], a_k["edges"][0], k, wp))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for tf, d, lbl, em, ek, k, wp in points:
        color = COL_KGRW if d > 0 else COL_MPRW
        ax.scatter(tf, d, s=120, color=color, edgecolor="black", linewidth=1, zorder=3)
        ax.annotate(f"{lbl}\n(MPRW={em:.0f}e, KGRW k={k} w'={wp}={ek:.0f}e)",
                    (tf, d), xytext=(8, 8), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("|T|/|V_L| at k=4   (fraction of endpoints KMV captures exactly)")
    ax.set_ylabel("ΔCKA (KGRW − MPRW) at lowest budget")
    ax.set_title("Theory-empirics bridge: tail fraction predicts KGRW advantage")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.75); ax.set_ylim(-0.05, 0.12)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_tail_vs_advantage.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig8_tail_vs_advantage.pdf")


if __name__ == "__main__":
    fig1_marginal_cost()
    fig2_cumulative_edges()
    fig3_quality_vs_density()
    fig4_time_vs_density()
    fig5_depth_sweep()
    fig6_tail_fraction()
    fig7_rdeg_cdf()
    fig8_tail_vs_advantage()
    print(f"\nAll plots written to {OUT}")
