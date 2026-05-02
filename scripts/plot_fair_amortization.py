"""Plot the fair multi-task SHGN amortization comparison.

Headline figure for the prof meeting: KMV total Q=2 (NC + Sim) vs MT-SHGN
total Q=2 (joint NC+LP train + exact Jaccard). Generated as
`figures/sketch_session/fair_amortization_q2.{pdf,png}`.

Reads from the JSONs already on disk:
  - results/<DS>/multi_query_amortization_k32_seed*.json   (KMV side)
  - results/<DS>/simple_hgn_multitask_seed*.json           (MT-SHGN train_time)
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures" / "sketch_session"
FIG.mkdir(parents=True, exist_ok=True)

DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed"]
COLOR_KMV = "#2CA02C"
COLOR_MT = "#9467BD"


def _load_amort(ds: str) -> dict | None:
    paths = sorted((RES / ds).glob("multi_query_amortization_k32_seed*.json"))
    if not paths:
        return None
    return json.loads(paths[0].read_text())


def _load_mt_train(ds: str) -> float | None:
    times = []
    for p in sorted((RES / ds).glob("simple_hgn_multitask_seed*.json")):
        if "seed99" in p.name:
            continue
        try:
            d = json.loads(p.read_text())
            t = d.get("train_time_s")
            if t is not None:
                times.append(float(t))
        except Exception:
            pass
    return statistics.fmean(times) if times else None


def main() -> int:
    rows = []
    for ds in DATASETS:
        a = _load_amort(ds)
        mt = _load_mt_train(ds)
        if not a or mt is None:
            print(f"[skip] {ds}: amort={bool(a)} mt={mt}")
            continue
        kmv_pre = a["kmv"]["precompute_s"]
        kmv_nc  = a["kmv"]["nc_consume_s"]
        kmv_sim = a["kmv"]["sim_consume_s"]
        kmv_total = a["totals_q2_nc_sim"]["kmv"]
        exact_sim = a["baseline"]["exact_sim_s"]
        mt_total = mt + exact_sim
        rows.append({
            "dataset": ds.replace("HNE_", "").replace("HGB_", ""),
            "kmv_pre": kmv_pre,
            "kmv_nc":  kmv_nc,
            "kmv_sim": kmv_sim,
            "kmv_total": kmv_total,
            "mt_train": mt,
            "exact_sim": exact_sim,
            "mt_total": mt_total,
            "speedup": mt_total / kmv_total,
        })

    if not rows:
        print("No rows to plot.")
        return 1

    # Stacked bar plot, log scale on y because ACM is 200× DBLP.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(rows))
    width = 0.36

    kmv_pre  = np.array([r["kmv_pre"]  for r in rows])
    kmv_nc   = np.array([r["kmv_nc"]   for r in rows])
    kmv_sim  = np.array([r["kmv_sim"]  for r in rows])
    mt_train = np.array([r["mt_train"] for r in rows])
    mt_sim   = np.array([r["exact_sim"] for r in rows])

    # KMV stacked bar (left of each pair).
    ax.bar(x - width / 2, kmv_pre, width, color=COLOR_KMV,
           alpha=0.95, label="KMV: precompute")
    ax.bar(x - width / 2, kmv_nc, width, bottom=kmv_pre,
           color=COLOR_KMV, alpha=0.65, label="KMV: NC consume")
    ax.bar(x - width / 2, kmv_sim, width, bottom=kmv_pre + kmv_nc,
           color=COLOR_KMV, alpha=0.4, label="KMV: Sim consume")

    # MT-SHGN stacked bar (right of each pair).
    ax.bar(x + width / 2, mt_train, width, color=COLOR_MT,
           alpha=0.95, label="MT-SHGN: joint NC+LP train")
    ax.bar(x + width / 2, mt_sim, width, bottom=mt_train,
           color=COLOR_MT, alpha=0.4, label="MT-SHGN: exact Sim")

    # Speedup annotations above bars.
    for i, r in enumerate(rows):
        h = max(r["kmv_total"], r["mt_total"])
        ax.annotate(f"{r['speedup']:.2f}×",
                    xy=(x[i], h),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                    color="#444")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r["dataset"] for r in rows], fontsize=11)
    ax.set_ylabel("Wall-clock seconds (log scale)", fontsize=11)
    ax.set_title("Fair multi-query amortization at Q=2 (NC + Similarity):  "
                 "KMV vs multi-task Simple-HGN",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(FIG / "fair_amortization_q2.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / "fair_amortization_q2.png", dpi=150, bbox_inches="tight")
    print(f"[fig]  {FIG / 'fair_amortization_q2.png'}")

    # Summary line for the report.
    print()
    print("Speedups (KMV vs MT-SHGN, Q=2):")
    for r in rows:
        print(f"  {r['dataset']:<10} {r['speedup']:>7.2f}×  "
              f"(KMV {r['kmv_total']:.2f}s vs MT-SHGN {r['mt_total']:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
