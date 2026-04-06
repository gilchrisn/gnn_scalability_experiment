"""
exp4_visualize.py — Generate paper figures from master_results.csv files.

Outputs
-------
  Table 2 — Main Performance Matrix (CSV + formatted text)
  Fig 1   — Peak Memory vs. Graph Size (log-log, all datasets)
  Fig 2   — Macro-F1 vs. Materialization Time (1×3 grid: ACM, DBLP, IMDB)
  Fig 3   — Output CKA vs. Network Depth (1×3 grid)
  Fig 4   — Dirichlet Energy vs. Network Depth (1×3 grid)

Aggregation rules
-----------------
Speedup and RAM-reduction ratios: Geometric Mean across meta-paths.
Bounded quality metrics (F1, CKA):  Arithmetic Mean across meta-paths.
Dirichlet Energy: NEVER aggregated across meta-paths (topology-specific).

Variance protocol
-----------------
If the CSV contains a 'Seed' column (multi-seed run), every metric is
reported as mean ± std across seeds (shaded fill_between for line plots,
error bars for scatter).  Without a Seed column values are plotted as-is.

Usage
-----
    python scripts/exp4_visualize.py
    python scripts/exp4_visualize.py --results-dir results --out-dir figures
    python scripts/exp4_visualize.py --datasets HGB_ACM HGB_DBLP HGB_IMDB
    python scripts/exp4_visualize.py --metapath "author_to_paper,paper_to_author"
    python scripts/exp4_visualize.py --fig 1 2        # specific figures only
    python scripts/exp4_visualize.py --table 2        # Table 2 only
    python scripts/exp4_visualize.py --k-fixed 32
    python scripts/exp4_visualize.py --depth 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib not installed: pip install matplotlib")
    sys.exit(1)


# ── Dataset metadata ─────────────────────────────────────────────────────────

# Fallback target-node counts if partition.json is absent.
_FALLBACK_N: dict[str, int] = {
    "HGB_ACM":   4_019,
    "HGB_DBLP":  4_057,
    "HGB_IMDB":  4_278,
    "OGB_MAG":  736_389,
    "OAG_CS":   546_440,
}

_LABEL: dict[str, str] = {
    "HGB_ACM":  "ACM",
    "HGB_DBLP": "DBLP",
    "HGB_IMDB": "IMDB",
    "OGB_MAG":  "OGB-MAG",
    "OAG_CS":   "OAG-CS",
}

# Datasets shown in Figs 2/3/4 (small, fully runnable on all methods).
_SMALL = ["HGB_ACM", "HGB_DBLP", "HGB_IMDB"]

# ── Colours / markers ─────────────────────────────────────────────────────────

_C = {"Exact": "#1f77b4", "KMV": "#2ca02c", "MPRW": "#d62728"}
_M = {"Exact": "*",       "KMV": "o",        "MPRW": "s"}


# ── Data loading ──────────────────────────────────────────────────────────────

_NUM_COLS = [
    "L", "k_value", "Density_Matched_w",
    "Materialization_Time", "Inference_Time",
    "Mat_RAM_MB", "Inf_RAM_MB", "Edge_Count",
    "CKA_L1", "CKA_L2", "CKA_L3", "CKA_L4",
    "Pred_Similarity", "Macro_F1", "Dirichlet_Energy",
]


def _load(results_dir: Path, datasets: list[str]) -> pd.DataFrame:
    frames = []
    for ds in datasets:
        csv = results_dir / ds / "master_results.csv"
        if not csv.exists():
            warnings.warn(f"No CSV for {ds}: {csv}")
            continue
        df = pd.read_csv(csv, dtype=str)
        df["_DS"] = ds
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No master_results.csv found under {results_dir} "
            f"for any of: {datasets}"
        )
    df = pd.concat(frames, ignore_index=True)
    df["Dataset"] = df["_DS"]   # canonical dataset key
    df.drop(columns=["_DS"], inplace=True)

    for c in _NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Seed" in df.columns:
        df["Seed"] = pd.to_numeric(df["Seed"], errors="coerce")
    if "L" in df.columns:
        df["L"] = df["L"].astype("Int64")
    if "k_value" in df.columns:
        df["k_value"] = df["k_value"].astype("Int64")
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _n_targets(ds: str, results_dir: Path) -> Optional[int]:
    """Target node count from partition.json, fallback to lookup table."""
    pj = results_dir / ds / "partition.json"
    if pj.exists():
        with open(pj) as f:
            p = json.load(f)
        n = len(p.get("train_node_ids", [])) + len(p.get("test_node_ids", []))
        if n > 0:
            return n
    return _FALLBACK_N.get(ds)


def _pick_mp(df: pd.DataFrame, ds: str, prefer: Optional[str]) -> str:
    """Return one canonical metapath for this dataset."""
    avail = df.loc[df["Dataset"] == ds, "MetaPath"].dropna().unique().tolist()
    if not avail:
        raise ValueError(f"No rows for dataset {ds}")
    if prefer and prefer in avail:
        return prefer
    if prefer:
        warnings.warn(f"Metapath '{prefer}' absent for {ds}; using '{avail[0]}'")
    return avail[0]


def _is_oom(status) -> bool:
    if pd.isna(status):
        return False
    s = str(status).upper()
    return "OOM" in s or "TIMEOUT" in s


def _ms(sub: pd.DataFrame, col: str) -> tuple[float, float]:
    """Return (mean, std) of col over sub rows. std=0 for single row."""
    if col not in sub.columns:
        return float("nan"), 0.0
    v = sub[col].dropna()
    if v.empty:
        return float("nan"), 0.0
    return float(v.mean()), float(v.std()) if len(v) > 1 else 0.0


def _eb(ax, x, y, yerr, **kw):
    """errorbar wrapper that omits yerr=0 to avoid tiny caps."""
    ebar = yerr if (yerr and yerr > 0) else None
    ax.errorbar(x, y, yerr=ebar, **kw)


def _geomean(vals: list[float]) -> float:
    """Geometric mean of strictly positive values."""
    pos = [v for v in vals if v > 0 and not np.isnan(v)]
    if not pos:
        return float("nan")
    return float(np.exp(np.mean(np.log(pos))))


def _hop_count(mp: str) -> int:
    """Number of edges (hops) in a comma-separated metapath string."""
    return len(mp.strip().split(","))


def _any_mprw_saturates(df: pd.DataFrame, ds: str, mp: str, L: int) -> bool:
    """True if any MPRW k-value has Mat_RAM_MB > Exact Mat_RAM_MB.

    Indicates the 'small graph saturation anomaly': MPRW was forced to a
    near-exact materialization, destroying its efficiency claims.
    """
    base      = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp) & (df["L"] == L)]
    exact_sub = base[base["Method"] == "Exact"]
    mprw_sub  = base[base["Method"] == "MPRW"]
    if exact_sub.empty or mprw_sub.empty:
        return False
    exact_ram, _ = _ms(exact_sub, "Mat_RAM_MB")
    if np.isnan(exact_ram) or exact_ram <= 0:
        return False
    mprw_rams = mprw_sub["Mat_RAM_MB"].dropna()
    return bool((mprw_rams > exact_ram).any())


# ── Plot style ─────────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        10,
        "axes.titlesize":   11,
        "axes.labelsize":   10,
        "legend.fontsize":  8,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linestyle":   "--",
    })


def _save(fig, path: Path, dpi: int = 300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)


# ── Figure 1: Peak Memory vs. Graph Size (log-log) ────────────────────────────

def fig1(df: pd.DataFrame, datasets: list[str], results_dir: Path,
         out_dir: Path, prefer_mp: Optional[str], L_fixed: int) -> None:
    print(f"\n[Fig 1] Peak Memory vs. Graph Size (L={L_fixed}) ...")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    exact_pts:  list[tuple[int, float, str]] = []   # (n_nodes, GB, ds)
    kmv_pts:    list[tuple[int, float]]      = []   # (n_nodes, GB)
    oom_n:      list[int]                    = []

    for ds in datasets:
        n = _n_targets(ds, results_dir)
        if n is None:
            warnings.warn(f"[Fig1] No node count for {ds}; skipping")
            continue
        try:
            mp = _pick_mp(df, ds, prefer_mp)
        except ValueError:
            continue

        base = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp)
                  & (df["L"] == L_fixed)]

        # Exact
        esub = base[base["Method"] == "Exact"]
        if not esub.empty:
            status = esub["exact_status"].iloc[0] if "exact_status" in esub.columns else "OK"
            if _is_oom(status):
                oom_n.append(n)
            else:
                ram, _ = _ms(esub, "Mat_RAM_MB")
                if np.isnan(ram):
                    ram, _ = _ms(esub, "Inf_RAM_MB")
                if not np.isnan(ram):
                    exact_pts.append((n, ram / 1024.0, ds))

        # KMV k=32
        ksub = base[(base["Method"] == "KMV") & (base["k_value"] == 32)]
        if not ksub.empty:
            ram, _ = _ms(ksub, "Mat_RAM_MB")
            if np.isnan(ram):
                ram, _ = _ms(ksub, "Inf_RAM_MB")
            if not np.isnan(ram):
                kmv_pts.append((n, ram / 1024.0))

    # Plot Exact line
    if exact_pts:
        exact_pts.sort(key=lambda t: t[0])
        ex_n, ex_r, ex_ds = zip(*exact_pts)
        ax.plot(ex_n, ex_r, "o-", color=_C["Exact"], label="Exact",
                linewidth=2, markersize=7, zorder=3)
        for n_i, r_i, ds_i in exact_pts:
            ax.annotate(_LABEL.get(ds_i, ds_i), xy=(n_i, r_i),
                        xytext=(5, 3), textcoords="offset points",
                        fontsize=7.5, color=_C["Exact"])

    # Plot KMV line
    if kmv_pts:
        kmv_pts.sort(key=lambda t: t[0])
        kv_n, kv_r = zip(*kmv_pts)
        ax.plot(kv_n, kv_r, "s--", color=_C["KMV"], label=r"KMV ($k=32$)",
                linewidth=2, markersize=7, zorder=3)

    # OOM markers — red X above the plot area
    if oom_n:
        all_r = ([r for _, r, _ in exact_pts] if exact_pts else []) + \
                ([r for _, r in kmv_pts] if kmv_pts else [])
        y_oom = (max(all_r) * 8) if all_r else 1.0
        for n_oom in oom_n:
            ax.plot(n_oom, y_oom, "rx", markersize=14, markeredgewidth=3,
                    zorder=5, clip_on=False,
                    label="Exact OOM" if n_oom == oom_n[0] else "_")
            ax.annotate("OOM", xy=(n_oom, y_oom),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=8, color="red", fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\mathcal{V}_0|$ — Target Nodes (log scale)")
    ax.set_ylabel("Materialization Peak RAM (GB, log scale)")
    ax.set_title("Memory Scalability: Exact vs.\ KMV")
    ax.legend(loc="upper left")
    _save(fig, out_dir / "fig1_memory_vs_scale.pdf")


# ── Table 2: Main Performance Matrix ─────────────────────────────────────────

def _print_table2(sdf: pd.DataFrame) -> None:
    W = 96
    print("\n" + "=" * W)
    print("TABLE 2 — Main Performance Matrix  (L=2, k=32 — geo-means across meta-paths)")
    print("=" * W)
    hdr = (f"{'Dataset':8}  {'Hop Group':16}  {'Method':5}  "
           f"{'GeoSpeedup':>11}  {'GeoRAM-Red':>11}  "
           f"{'F1 (mean+/-std)':>17}  {'CKA (mean+/-std)':>17}")
    print(hdr)
    print("-" * W)
    for _, row in sdf.iterrows():
        sp  = f"{row['GeoMean_Speedup']:.2f}x"  if not np.isnan(row["GeoMean_Speedup"]) else "n/a"
        rr  = f"{row['GeoMean_RAM_Red']:.2f}x"  if not np.isnan(row["GeoMean_RAM_Red"]) else "n/a"
        f1s = f"{row['F1_mean']:.4f}+/-{row['F1_std']:.4f}"
        cka = (f"{row['CKA_mean']:.4f}+/-{row['CKA_std']:.4f}"
               if not np.isnan(row["CKA_mean"]) else "n/a")
        print(f"{row['Dataset']:8}  {row['Hop_Group']:16}  {row['Method']:5}  "
              f"{sp:>11}  {rr:>11}  {f1s:>17}  {cka:>17}")
    print("=" * W)
    print("Speedup/RAM-Red: geometric mean across meta-paths in each hop-length group.")
    print("F1/CKA:          arithmetic mean (bounded quality metrics, mean is valid).")
    print("Dirichlet Energy is NOT shown here — never aggregated across meta-paths.")


def table2(df: pd.DataFrame, datasets: list[str], out_dir: Path,
           L_fixed: int = 2, k_fixed: int = 32) -> None:
    """Generate Table 2: geometric-mean speedup + RAM reduction, arith-mean F1/CKA."""
    print(f"\n[Table 2] Main Performance Matrix (L={L_fixed}, k={k_fixed}) ...")
    small_ds = [d for d in _SMALL if d in datasets]
    records: list[dict] = []

    for ds in small_ds:
        all_mps = df[df["Dataset"] == ds]["MetaPath"].dropna().unique()
        for mp in all_mps:
            hop       = _hop_count(mp)
            hop_group = "Short (<=3-hop)" if hop <= 3 else "Deep (>=4-hop)"
            base      = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp)
                           & (df["L"] == L_fixed)]

            esub = base[base["Method"] == "Exact"]
            if esub.empty:
                continue
            t_exact,   _ = _ms(esub, "Materialization_Time")
            ram_exact, _ = _ms(esub, "Mat_RAM_MB")
            if np.isnan(t_exact):
                continue

            for method in ("KMV", "MPRW"):
                if method == "KMV":
                    msub = base[(base["Method"] == "KMV") & (base["k_value"] == k_fixed)]
                else:
                    msub = base[(base["Method"] == "MPRW") & (base["k_value"] == k_fixed)]
                if msub.empty:
                    continue

                t_m,   _       = _ms(msub, "Materialization_Time")
                ram_m, _       = _ms(msub, "Mat_RAM_MB")
                f1_m,  f1_s    = _ms(msub, "Macro_F1")
                cka_m, cka_s   = _ms(msub, f"CKA_L{L_fixed}")

                speedup = (t_exact / t_m
                           if not np.isnan(t_m) and t_m > 0 else float("nan"))
                ram_red = (ram_exact / ram_m
                           if not (np.isnan(ram_exact) or np.isnan(ram_m)) and ram_m > 0
                           else float("nan"))

                records.append({
                    "Dataset":   ds,
                    "MetaPath":  mp,
                    "Hop_Group": hop_group,
                    "Method":    method,
                    "Speedup":   speedup,
                    "RAM_Red":   ram_red,
                    "F1_mean":   f1_m,
                    "F1_std":    f1_s,
                    "CKA_mean":  cka_m,
                    "CKA_std":   cka_s,
                })

    if not records:
        print("  [skip] No data for Table 2 (need Exact + KMV/MPRW rows at "
              f"L={L_fixed}, k={k_fixed})")
        return

    rdf = pd.DataFrame(records)

    # Aggregate per (dataset, hop_group, method)
    summary: list[dict] = []
    for (ds, hop_group, method), grp in rdf.groupby(
        ["Dataset", "Hop_Group", "Method"], sort=False
    ):
        summary.append({
            "Dataset":          _LABEL.get(ds, ds),
            "Hop_Group":        hop_group,
            "Method":           method,
            "GeoMean_Speedup":  _geomean(grp["Speedup"].dropna().tolist()),
            "GeoMean_RAM_Red":  _geomean(grp["RAM_Red"].dropna().tolist()),
            # Arith mean for bounded quality metrics
            "F1_mean":          float(grp["F1_mean"].mean()),
            "F1_std":           float(grp["F1_std"].mean()),   # avg std across metapaths
            "CKA_mean":         float(grp["CKA_mean"].mean()),
            "CKA_std":          float(grp["CKA_std"].mean()),
        })

    sdf = pd.DataFrame(summary)
    # Sort for readable output: dataset order, Short before Deep, KMV before MPRW
    hop_order    = {"Short (<=3-hop)": 0, "Deep (>=4-hop)": 1}
    method_order = {"KMV": 0, "MPRW": 1}
    ds_order     = {_LABEL.get(d, d): i for i, d in enumerate(small_ds)}
    sdf["_ds_ord"]  = sdf["Dataset"].map(ds_order).fillna(99)
    sdf["_hop_ord"] = sdf["Hop_Group"].map(hop_order).fillna(99)
    sdf["_mth_ord"] = sdf["Method"].map(method_order).fillna(99)
    sdf.sort_values(["_ds_ord", "_hop_ord", "_mth_ord"], inplace=True)
    sdf.drop(columns=["_ds_ord", "_hop_ord", "_mth_ord"], inplace=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "table2.csv"
    sdf.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved -> {csv_path}")
    _print_table2(sdf)


# ── Figure 2: F1 vs. Materialization Cost (1×3) ──────────────────────────────

def fig2(df: pd.DataFrame, datasets: list[str], out_dir: Path,
         prefer_mp: Optional[str], L_fixed: int) -> None:
    print(f"\n[Fig 2] F1 vs. Materialization Cost (L={L_fixed}) ...")
    plot_ds = [d for d in _SMALL if d in datasets]
    if not plot_ds:
        print("  [skip] ACM / DBLP / IMDB not found in loaded results")
        return

    fig, axes = plt.subplots(1, len(plot_ds),
                             figsize=(4.2 * len(plot_ds), 4.2), sharey=False)
    if len(plot_ds) == 1:
        axes = [axes]

    all_k = sorted(df.loc[df["Method"] == "KMV", "k_value"].dropna().unique())

    for ax, ds in zip(axes, plot_ds):
        mp   = _pick_mp(df, ds, prefer_mp)
        base = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp)
                  & (df["L"] == L_fixed)]

        # ── Exact star ───────────────────────────────────────────────────────
        esub = base[base["Method"] == "Exact"]
        if not esub.empty:
            t,  _  = _ms(esub, "Materialization_Time")
            f1, _  = _ms(esub, "Macro_F1")
            if not (np.isnan(t) or np.isnan(f1)):
                ax.plot(t, f1, marker="*", color=_C["Exact"], markersize=20,
                        linestyle="none", zorder=6, label="Exact", clip_on=False)

        # ── KMV k-sweep ──────────────────────────────────────────────────────
        kmv_base = base[base["Method"] == "KMV"].dropna(subset=["k_value"])
        if not kmv_base.empty:
            alphas  = np.linspace(0.30, 1.0, max(len(all_k), 1))
            k_alpha = {int(k): float(a) for k, a in zip(sorted(all_k), alphas)}
            xs, ys  = [], []
            for k in sorted(kmv_base["k_value"].unique()):
                ksub = kmv_base[kmv_base["k_value"] == k]
                t,   _   = _ms(ksub, "Materialization_Time")
                f1, f1e  = _ms(ksub, "Macro_F1")
                if np.isnan(t) or np.isnan(f1):
                    continue
                lbl = f"KMV ($k={k}$)" if k == max(all_k) else "_"
                _eb(ax, t, f1, f1e, fmt="o", color=_C["KMV"],
                    alpha=k_alpha.get(int(k), 1.0),
                    markersize=6, capsize=3, label=lbl)
                xs.append(t); ys.append(f1)
            if len(xs) > 1:
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                ax.plot([xs[i] for i in order], [ys[i] for i in order],
                        "-", color=_C["KMV"], alpha=0.30, zorder=1, linewidth=1.2)

        # ── MPRW density-matched sweep ────────────────────────────────────────
        mprw_base = base[base["Method"] == "MPRW"].dropna(subset=["k_value"])
        if not mprw_base.empty:
            k_sorted = sorted(mprw_base["k_value"].unique())
            alphas2  = np.linspace(0.30, 1.0, max(len(k_sorted), 1))
            xs2, ys2 = [], []
            for i, k in enumerate(k_sorted):
                msub = mprw_base[mprw_base["k_value"] == k]
                t,   _   = _ms(msub, "Materialization_Time")
                f1, f1e  = _ms(msub, "Macro_F1")
                if np.isnan(t) or np.isnan(f1):
                    continue
                lbl = "MPRW" if i == len(k_sorted) - 1 else "_"
                _eb(ax, t, f1, f1e, fmt="s", color=_C["MPRW"],
                    alpha=alphas2[i], markersize=6, capsize=3, label=lbl)
                xs2.append(t); ys2.append(f1)
            if len(xs2) > 1:
                order2 = sorted(range(len(xs2)), key=lambda i: xs2[i])
                ax.plot([xs2[i] for i in order2], [ys2[i] for i in order2],
                        "-", color=_C["MPRW"], alpha=0.30, zorder=1, linewidth=1.2)

        ax.set_xscale("log")
        ax.set_xlabel("Materialization Time (s, log scale)")
        ax.set_ylabel("Macro-F1")
        ax.set_title(_LABEL.get(ds, ds))
        ax.legend(loc="lower right", fontsize=7)

        # ── Saturation anomaly annotation ─────────────────────────────────
        # On small/dense graphs MPRW may hit the w-ceiling, forcing a near-exact
        # materialization whose RAM cost exceeds the Exact baseline.  Flag this
        # visually so reviewers see the efficiency claim is self-defeating there.
        if _any_mprw_saturates(df, ds, mp, L_fixed):
            ax.text(0.04, 0.97,
                    "† MPRW Mat. RAM > Exact\n  (saturation regime)",
                    transform=ax.transAxes, fontsize=6.5, style="italic",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#ffe8c0", edgecolor="#cc8800", alpha=0.85))

    fig.suptitle(rf"Efficiency–Fidelity Trade-off  ($L={L_fixed}$)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "fig2_f1_vs_cost.pdf")


# ── Figure 3: Output CKA vs. Network Depth (1×3) ─────────────────────────────

def fig3(df: pd.DataFrame, datasets: list[str], out_dir: Path,
         prefer_mp: Optional[str], k_fixed: int, depths: list[int]) -> None:
    print(f"\n[Fig 3] Output CKA vs. Network Depth (k={k_fixed}) ...")
    plot_ds = [d for d in _SMALL if d in datasets]
    if not plot_ds:
        print("  [skip] ACM / DBLP / IMDB not found")
        return

    fig, axes = plt.subplots(1, len(plot_ds),
                             figsize=(4.2 * len(plot_ds), 3.8), sharey=True)
    if len(plot_ds) == 1:
        axes = [axes]

    for ax, ds in zip(axes, plot_ds):
        mp = _pick_mp(df, ds, prefer_mp)

        for method, label in [
            ("KMV",  f"KMV ($k={k_fixed}$)"),
            ("MPRW", "MPRW"),
        ]:
            ys, errs = [], []
            for L in depths:
                cka_col = f"CKA_L{L}"
                if method == "KMV":
                    sub = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp)
                             & (df["L"] == L) & (df["Method"] == "KMV")
                             & (df["k_value"] == k_fixed)]
                else:
                    sub = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp)
                             & (df["L"] == L) & (df["Method"] == "MPRW")
                             & (df["k_value"] == k_fixed)]
                m, s = _ms(sub, cka_col)
                ys.append(m); errs.append(s)

            valid = [(d, y, e) for d, y, e in zip(depths, ys, errs)
                     if not np.isnan(y)]
            if not valid:
                continue
            vd, vy, ve = zip(*valid)
            color = _C[method]
            ax.errorbar(vd, vy,
                        yerr=[e if e > 0 else np.nan for e in ve],
                        fmt=f"{_M[method]}-", color=color, label=label,
                        linewidth=2, markersize=7, capsize=4)
            if any(e > 0 for e in ve):
                lo = [y - e for y, e in zip(vy, ve)]
                hi = [y + e for y, e in zip(vy, ve)]
                ax.fill_between(vd, lo, hi, alpha=0.12, color=color)

        # Exact flat reference
        ax.axhline(1.0, color=_C["Exact"], linestyle="--",
                   linewidth=1.5, label="Exact (ref.)", alpha=0.75)

        ax.set_xticks(depths)
        ax.set_xlabel("Network Depth $L$")
        ax.set_ylabel("Output CKA")
        ax.set_title(_LABEL.get(ds, ds))
        ax.set_ylim(0.0, 1.10)
        ax.legend(fontsize=7)

        # ── Saturation caveat annotation ──────────────────────────────────
        # On small/dense datasets MPRW avoids CKA collapse ONLY because its
        # walk count w saturates the graph (near-exact materialization).
        # This must be called out explicitly so the reviewer does not
        # misread the high MPRW CKA as evidence of algorithmic robustness.
        if _any_mprw_saturates(df, ds, mp, max(depths)):
            ax.text(0.04, 0.04,
                    "Saturation caveat:\nMPRW CKA stable only\nbecause Mat. RAM > Exact\n"
                    "(w-ceiling reached)",
                    transform=ax.transAxes, fontsize=6, style="italic",
                    verticalalignment="bottom",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#ffe8c0", edgecolor="#cc8800", alpha=0.85))

    fig.suptitle(rf"Representational Fidelity at Depth  ($k={k_fixed}$)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "fig3_cka_vs_depth.pdf")


# ── Figure 4: Dirichlet Energy vs. Network Depth (1×3) ───────────────────────

def fig4(df: pd.DataFrame, datasets: list[str], out_dir: Path,
         prefer_mp: Optional[str], k_fixed: int, depths: list[int]) -> None:
    print(f"\n[Fig 4] Dirichlet Energy vs. Network Depth (k={k_fixed}) ...")
    plot_ds = [d for d in _SMALL if d in datasets]
    if not plot_ds:
        print("  [skip] ACM / DBLP / IMDB not found")
        return

    fig, axes = plt.subplots(1, len(plot_ds),
                             figsize=(4.2 * len(plot_ds), 3.8), sharey=False)
    if len(plot_ds) == 1:
        axes = [axes]

    for ax, ds in zip(axes, plot_ds):
        mp = _pick_mp(df, ds, prefer_mp)

        for method, label, ls in [
            ("Exact", "Exact",                "--"),
            ("KMV",   f"KMV ($k={k_fixed}$)", "-"),
            ("MPRW",  "MPRW",                 "-"),
        ]:
            ys, errs = [], []
            for L in depths:
                if method == "KMV":
                    sub = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp)
                             & (df["L"] == L) & (df["Method"] == "KMV")
                             & (df["k_value"] == k_fixed)]
                elif method == "MPRW":
                    sub = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp)
                             & (df["L"] == L) & (df["Method"] == "MPRW")
                             & (df["k_value"] == k_fixed)]
                else:
                    sub = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp)
                             & (df["L"] == L) & (df["Method"] == "Exact")]
                m, s = _ms(sub, "Dirichlet_Energy")
                ys.append(m); errs.append(s)

            valid = [(d, y, e) for d, y, e in zip(depths, ys, errs)
                     if not np.isnan(y)]
            if not valid:
                continue
            vd, vy, ve = zip(*valid)
            color = _C[method]
            ms    = 11 if method == "Exact" else 7
            ax.errorbar(vd, vy,
                        yerr=[e if e > 0 else np.nan for e in ve],
                        fmt=f"{_M[method]}{ls}", color=color, label=label,
                        linewidth=2, markersize=ms, capsize=4)
            if any(e > 0 for e in ve):
                lo = [y - e for y, e in zip(vy, ve)]
                hi = [y + e for y, e in zip(vy, ve)]
                ax.fill_between(vd, lo, hi, alpha=0.12, color=color)

        ax.set_xticks(depths)
        ax.set_xlabel("Network Depth $L$")
        ax.set_ylabel("Dirichlet Energy")
        ax.set_title(_LABEL.get(ds, ds))
        ax.legend(fontsize=7)

    fig.suptitle(
        rf"Embedding Smoothness (Dirichlet Energy) vs. Depth  ($k={k_fixed}$)",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out_dir / "fig4_dirichlet_vs_depth.pdf")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results-dir", default="results",
        help="Root dir containing <dataset>/master_results.csv (default: results/)",
    )
    p.add_argument(
        "--out-dir", default="figures",
        help="Output directory for PDF figures (default: figures/)",
    )
    p.add_argument(
        "--datasets", nargs="+",
        default=["HGB_ACM", "HGB_DBLP", "HGB_IMDB", "OGB_MAG", "OAG_CS"],
        help="Datasets to include (default: all 5)",
    )
    p.add_argument(
        "--metapath", default=None,
        help="Preferred metapath string (uses first available per dataset if absent)",
    )
    p.add_argument(
        "--fig", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4],
        help="Which figures to generate (default: 1 2 3 4)",
    )
    p.add_argument(
        "--table", nargs="+", type=int, choices=[2], default=[2],
        help="Which tables to generate (default: 2)",
    )
    p.add_argument(
        "--no-table", action="store_true",
        help="Skip table generation (figures only)",
    )
    p.add_argument(
        "--k-fixed", type=int, default=32,
        help="k value used as the fixed parameter in Figs 3 & 4 (default: 32)",
    )
    p.add_argument(
        "--depth", nargs="+", type=int, default=[2, 3, 4],
        help="Network depths shown in Figs 3 & 4 (default: 2 3 4)",
    )
    p.add_argument(
        "--l-fixed", type=int, default=2,
        help="Network depth fixed in Figs 1 & 2 (default: 2)",
    )
    return p.parse_args()


def main():
    args = _parse()
    _style()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)

    # Restrict to datasets that actually have a CSV
    datasets = [
        d for d in args.datasets
        if (results_dir / d / "master_results.csv").exists()
    ]
    if not datasets:
        print(
            f"[ERROR] No master_results.csv found under {results_dir} "
            f"for any of: {args.datasets}"
        )
        sys.exit(1)

    print(f"Loading CSVs for: {datasets}")
    df = _load(results_dir, datasets)
    print(f"  {len(df)} total rows loaded")

    # Variance warning
    if "Seed" not in df.columns or df["Seed"].isna().all():
        print(
            "[WARN] No 'Seed' column — plotting single-seed values, no variance bands.\n"
            "       Run exp3 with 5 seeds and include a Seed column for the full "
            "variance protocol."
        )

    tables = set() if args.no_table else set(args.table)
    if 2 in tables:
        table2(df, datasets, out_dir, L_fixed=args.l_fixed, k_fixed=args.k_fixed)

    figs = set(args.fig)
    if 1 in figs:
        fig1(df, datasets, results_dir, out_dir, args.metapath, args.l_fixed)
    if 2 in figs:
        fig2(df, datasets, out_dir, args.metapath, args.l_fixed)
    if 3 in figs:
        fig3(df, datasets, out_dir, args.metapath, args.k_fixed, args.depth)
    if 4 in figs:
        fig4(df, datasets, out_dir, args.metapath, args.k_fixed, args.depth)

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
