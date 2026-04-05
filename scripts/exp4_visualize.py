"""
Experiment 4 — Visualize master_results.csv.

Reads results/<dataset>/master_results.csv (or a custom path) and produces
three plot types:

  1. stability.pdf   — CKA convergence vs k (log₂ scale), one subplot per L.
                       Solid line = KMV.  Dashed horizontal = Exact asymptote.
                       MPRW rows (exact_status=MPRW_PENDING) are skipped.

  2. pareto.pdf      — Pareto frontier: RAM vs Macro_F1 per dataset.
                       KMV curve at each k; Exact marker at the left edge.

  3. scalability.pdf — Log-log: Edge_Count vs Peak_RAM_MB.
                       All datasets on same axes; Exact line ends at OOM marker.

Usage
-----
    python scripts/exp4_visualize.py
    python scripts/exp4_visualize.py --results results/HGB_ACM/master_results.csv \\
        --output-dir figures/
    python scripts/exp4_visualize.py --results results/all/master_results.csv \\
        --datasets HGB_ACM HGB_DBLP --depth 2 3
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> List[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _float(val: str) -> Optional[float]:
    try:
        return float(val) if val.strip() != "" else None
    except ValueError:
        return None


def _int(val: str) -> Optional[int]:
    try:
        v = float(val)
        return int(v) if val.strip() != "" else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Plot 1: Stability (CKA vs k)
# ---------------------------------------------------------------------------

def plot_stability(rows: List[dict], out_dir: Path,
                   datasets: Optional[List[str]],
                   depths: Optional[List[int]]) -> None:
    """
    For each (dataset, metapath) combination: one figure, subplots = L values.
    x = k (log₂), y = CKA (averaged over available CKA_L* columns).
    Solid = KMV.  Dashed horizontal = Exact (mean CKA across layers = 1.0 by
    definition, but we show the L-specific final-layer CKA for KMV vs Exact
    when available — actually we just use mean of available CKA_Lx columns).

    If only one CKA column is populated (e.g. L=2 has CKA_L1, CKA_L2), we
    plot all populated columns as separate lines; otherwise average them.
    """
    # Group rows: (dataset, metapath, L) → {k: avg_cka, ...}
    # We plot mean of non-empty CKA_L1..CKA_L4 per row.
    def _avg_cka(row):
        vals = [_float(row.get(f"CKA_L{i}", "")) for i in range(1, 5)]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    # Collect KMV data
    kmv_data: Dict[Tuple, Dict[int, float]] = defaultdict(dict)
    for row in rows:
        if row.get("Method") != "KMV":
            continue
        ds = row.get("Dataset", "")
        mp = row.get("MetaPath", "")
        L  = _int(row.get("L", ""))
        k  = _int(row.get("k_value", ""))
        if datasets and ds not in datasets:
            continue
        if depths and L not in depths:
            continue
        if L is None or k is None:
            continue
        cka = _avg_cka(row)
        if cka is not None:
            kmv_data[(ds, mp, L)][k] = cka

    if not kmv_data:
        print("[exp4] No KMV CKA data to plot (stability).")
        return

    # One figure per (dataset, metapath)
    grouped: Dict[Tuple, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for (ds, mp, L), kmap in kmv_data.items():
        grouped[(ds, mp)][L] = kmap

    for (ds, mp), l_data in grouped.items():
        L_vals = sorted(l_data.keys())
        fig, axes = plt.subplots(1, len(L_vals), figsize=(5 * len(L_vals), 4),
                                 sharey=True, squeeze=False)
        fig.suptitle(f"{ds} — {mp}", fontsize=9)

        for ax, L in zip(axes[0], L_vals):
            kmap = l_data[L]
            ks   = sorted(kmap.keys())
            ckas = [kmap[k] for k in ks]
            log_ks = [np.log2(k) for k in ks]
            ax.plot(log_ks, ckas, "o-", label="KMV", linewidth=1.5)
            ax.set_title(f"L={L}", fontsize=9)
            ax.set_xlabel("log₂(k)", fontsize=8)
            ax.set_ylabel("CKA (avg layers)", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)

        plt.tight_layout()
        safe_mp = mp.replace(",", "_").replace("/", "_")[:40]
        fname = out_dir / f"stability_{ds}_{safe_mp}.pdf"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"[exp4] Saved {fname}")


# ---------------------------------------------------------------------------
# Plot 2: Pareto frontier (RAM vs F1)
# ---------------------------------------------------------------------------

def plot_pareto(rows: List[dict], out_dir: Path,
                datasets: Optional[List[str]],
                depths: Optional[List[int]]) -> None:
    """
    Per (dataset, L): scatter of (Peak_RAM_MB, Macro_F1) for KMV at each k.
    Exact is plotted as a star marker.
    """
    # Collect data: (dataset, L) → [(ram, f1, label)]
    data: Dict[Tuple, List[Tuple[float, float, str]]] = defaultdict(list)
    exact_data: Dict[Tuple, Tuple[float, float]] = {}

    for row in rows:
        ds  = row.get("Dataset", "")
        L   = _int(row.get("L", ""))
        mth = row.get("Method", "")
        ram = _float(row.get("Peak_RAM_MB", ""))
        f1  = _float(row.get("Macro_F1", ""))
        if datasets and ds not in datasets:
            continue
        if depths and L not in depths:
            continue
        if ram is None or f1 is None or L is None:
            continue
        key = (ds, L)
        if mth == "KMV":
            k = row.get("k_value", "?")
            data[key].append((ram, f1, f"k={k}"))
        elif mth == "Exact":
            exact_data[key] = (ram, f1)

    if not data and not exact_data:
        print("[exp4] No data to plot (pareto).")
        return

    all_keys = sorted(set(list(data.keys()) + list(exact_data.keys())))
    fig, axes = plt.subplots(1, len(all_keys),
                             figsize=(5 * len(all_keys), 4), squeeze=False)
    for ax, key in zip(axes[0], all_keys):
        ds, L = key
        ax.set_title(f"{ds}  L={L}", fontsize=9)
        ax.set_xlabel("Peak RAM (MB)", fontsize=8)
        ax.set_ylabel("Macro F1", fontsize=8)
        ax.tick_params(labelsize=7)

        pts = data.get(key, [])
        if pts:
            rams, f1s, labels = zip(*pts)
            ax.scatter(rams, f1s, zorder=3, s=50)
            for r, f, lbl in zip(rams, f1s, labels):
                ax.annotate(lbl, (r, f), fontsize=6, textcoords="offset points",
                            xytext=(3, 3))

        if key in exact_data:
            r, f = exact_data[key]
            ax.scatter([r], [f], marker="*", s=120, color="red",
                       zorder=4, label="Exact")
            ax.legend(fontsize=7)

    plt.tight_layout()
    fname = out_dir / "pareto.pdf"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp4] Saved {fname}")


# ---------------------------------------------------------------------------
# Plot 3: Scalability (Edge_Count vs Peak_RAM)
# ---------------------------------------------------------------------------

def plot_scalability(rows: List[dict], out_dir: Path,
                     datasets: Optional[List[str]],
                     depths: Optional[List[int]]) -> None:
    """
    Log-log plot: Edge_Count (x) vs Peak_RAM_MB (y).
    Each dataset is a separate colour; KMV and Exact are different markers.
    Exact rows that OOM (exact_status != OK) are shown as 'X' markers at the
    last edge count where they did not OOM.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlabel("Edge Count (log)", fontsize=9)
    ax.set_ylabel("Peak RAM (MB, log)", fontsize=9)
    ax.set_title("Scalability: Edge Count vs Peak RAM", fontsize=10)

    colors = plt.cm.tab10.colors
    ds_color: Dict[str, str] = {}
    c_idx = 0

    for row in rows:
        ds  = row.get("Dataset", "")
        mth = row.get("Method", "")
        L   = _int(row.get("L", ""))
        ec  = _float(row.get("Edge_Count", ""))
        ram = _float(row.get("Peak_RAM_MB", ""))
        if datasets and ds not in datasets:
            continue
        if depths and L not in depths:
            continue
        if ec is None or ram is None or mth == "MPRW":
            continue

        if ds not in ds_color:
            ds_color[ds] = colors[c_idx % len(colors)]
            c_idx += 1
        color = ds_color[ds]
        status = row.get("exact_status", "OK")

        if mth == "Exact":
            if status == "OK":
                ax.scatter([ec], [ram], marker="^", color=color, s=80, zorder=4)
            else:
                ax.scatter([ec], [ram], marker="X", color=color, s=80,
                           zorder=4, alpha=0.5)
        elif mth == "KMV":
            ax.scatter([ec], [ram], marker="o", color=color, s=30, alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Legend: one entry per dataset
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=col, markersize=8, label=ds)
               for ds, col in ds_color.items()]
    handles += [
        plt.Line2D([0], [0], marker="^", color="grey", markersize=8,
                   linestyle="None", label="Exact (OK)"),
        plt.Line2D([0], [0], marker="X", color="grey", markersize=8,
                   linestyle="None", label="Exact (OOM)"),
        plt.Line2D([0], [0], marker="o", color="grey", markersize=6,
                   linestyle="None", label="KMV"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    fname = out_dir / "scalability.pdf"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp4] Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results",    type=str, default=None,
                        help="Path to master_results.csv. "
                             "Default: results/master_results.csv")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="Directory to write PDF figures (default: figures/)")
    parser.add_argument("--datasets",   type=str, nargs="+", default=None,
                        help="Filter to specific datasets (default: all)")
    parser.add_argument("--depth",      type=int, nargs="+", default=None,
                        help="Filter to specific L values (default: all)")
    args = parser.parse_args()

    results_path = Path(args.results) if args.results else Path("results") / "master_results.csv"
    if not results_path.exists():
        print(f"[exp4] ERROR: results file not found: {results_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_csv(results_path)
    print(f"[exp4] Loaded {len(rows)} rows from {results_path}")

    plot_stability(rows, out_dir, args.datasets, args.depth)
    plot_pareto(rows, out_dir, args.datasets, args.depth)
    plot_scalability(rows, out_dir, args.datasets, args.depth)

    print(f"\n[exp4] All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
