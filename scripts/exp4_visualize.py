"""
exp4_visualize.py — Generate paper figures from master_results.csv files.

Outputs (15 total)
------------------
Systems (Phase I)
  plot1a  — Peak Memory Footprint         (stacked bar: Exact vs KMV, all datasets)
  plot1b  — End-to-End Execution Time     (stacked bar: Exact vs KMV, all datasets)
  plot1c  — Preprocessing Latency         (bar: KMV vs MPRW, all datasets)

Quality gate
  table1  — Information Gain Validation   (MLP F1 | Exact GNN F1 | Delta, all datasets)

Fidelity — Pareto
  plot2   — Cost-Quality Pareto Frontier  (x=Mat RAM, y=F1, k-sweep, all datasets)

Fidelity — L-sweep (k fixed)
  plot3   — Dirichlet Energy vs Depth     (Exact / KMV / MPRW, all datasets)
  plot_f1_vs_depth       — Macro-F1 vs Depth L
  plot_cka_vs_depth      — Output CKA vs Depth L
  plot_predsim_vs_depth  — Prediction Similarity vs Depth L

Fidelity — k-sweep (L fixed)
  plot_f1_vs_k           — Macro-F1 vs k
  plot_cka_vs_k          — Output CKA vs k
  plot_predsim_vs_k      — Prediction Similarity vs k

Fidelity — within-model CKA trajectory
  plot_cka_per_layer     — CKA at each GNN layer (L1..L_max) for fixed k

Summary tables
  table2  — Downstream Integrity (Exact F1 | KMV F1+-std | MPRW F1+-std | PA%)
  app_a   — Appendix: Meta-Path Generalization (all metapaths)

Multi-seed loading
------------------
Scans --transfer-dir for every results_<seed>/ subdirectory.
Assigns a Seed column.  With N>1 seeds: bounded metrics are reported as
mean +/- std (shaded fill_between for line plots, error bars for scatter).

Metapath selection
------------------
Without --metapath: auto-selects the densest metapath per dataset
(highest Edge_Count at Method=Exact, L=2).  Override with --metapath.

Usage
-----
    python scripts/exp4_visualize.py
    python scripts/exp4_visualize.py --transfer-dir transfer --out-dir figures
    python scripts/exp4_visualize.py --datasets HGB_ACM HGB_DBLP HGB_IMDB
    python scripts/exp4_visualize.py --metapath "paper_to_author,author_to_paper"
    python scripts/exp4_visualize.py --k-fixed 32 --l-fixed 2
    python scripts/exp4_visualize.py --depth 1 2 3 4
    python scripts/exp4_visualize.py --plot plot1a plot2 plot_cka_vs_k
    python scripts/exp4_visualize.py --table table1 table2
"""
from __future__ import annotations

import argparse
import re
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
    from matplotlib.patches import Patch
except ImportError:
    print("[ERROR] matplotlib not installed: pip install matplotlib")
    sys.exit(1)


# ── Dataset metadata ──────────────────────────────────────────────────────────

_LABEL: dict[str, str] = {
    "HGB_ACM":  "ACM",
    "HGB_DBLP": "DBLP",
    "HGB_IMDB": "IMDB",
    "HNE_PubMed": "PubMed",
    "OGB_MAG":  "OGB-MAG",
    "OAG_CS":   "OAG-CS",
}

# ── Colours / markers / hatches ───────────────────────────────────────────────

_C = {"Exact": "#1f77b4", "KMV": "#2ca02c", "MPRW": "#d62728"}
_M = {"Exact": "*",       "KMV": "o",        "MPRW": "s"}
_HATCH_MAT = ""
_HATCH_INF = "///"

# ── Numeric columns ───────────────────────────────────────────────────────────

_NUM_COLS = [
    "L", "k_value", "Density_Matched_w",
    "Materialization_Time", "Inference_Time",
    "Mat_RAM_MB", "Inf_RAM_MB", "Edge_Count", "Graph_Density",
    "CKA_L1", "CKA_L2", "CKA_L3", "CKA_L4",
    "Pred_Similarity", "Macro_F1", "Dirichlet_Energy",
]


# ── Multi-seed data loader ────────────────────────────────────────────────────

def _load_multiseed(transfer_dir: Path, datasets: list[str]) -> pd.DataFrame:
    """Scan transfer_dir/results_<seed>/<dataset>/master_results.csv.

    Adds a Seed column from the directory name.
    """
    transfer_dir = Path(transfer_dir)
    seed_dirs = sorted(
        d for d in transfer_dir.iterdir()
        if d.is_dir() and re.match(r"results_\d+", d.name)
    )
    if not seed_dirs:
        raise FileNotFoundError(
            f"No results_<seed>/ directories found under {transfer_dir}"
        )

    frames: list[pd.DataFrame] = []
    for sd in seed_dirs:
        seed_val = int(sd.name.split("_", 1)[1])
        for ds in datasets:
            csv = sd / ds / "master_results.csv"
            if not csv.exists():
                warnings.warn(f"[load] Missing: {csv}")
                continue
            df = pd.read_csv(csv, dtype=str)
            df["_DS"]  = ds
            df["Seed"] = seed_val
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No master_results.csv found in {transfer_dir} for: {datasets}"
        )

    df = pd.concat(frames, ignore_index=True)
    df["Dataset"] = df["_DS"]
    df.drop(columns=["_DS"], inplace=True)

    for c in _NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Seed"]    = pd.to_numeric(df["Seed"],    errors="coerce").astype("Int64")
    df["L"]       = pd.to_numeric(df["L"],       errors="coerce").astype("Int64")
    df["k_value"] = pd.to_numeric(df["k_value"], errors="coerce").astype("Int64")
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ms(sub: pd.DataFrame, col: str) -> tuple[float, float]:
    """(mean, std); std=0 for a single row."""
    if col not in sub.columns:
        return float("nan"), 0.0
    v = sub[col].dropna()
    if v.empty:
        return float("nan"), 0.0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def _geomean(vals: list[float]) -> float:
    pos = [v for v in vals if v > 0 and not np.isnan(v)]
    if not pos:
        return float("nan")
    return float(np.exp(np.mean(np.log(pos))))


def _hop_count(mp: str) -> int:
    return len(mp.strip().split(","))


def _is_oom(status) -> bool:
    if pd.isna(status):
        return False
    return "OOM" in str(status).upper() or "TIMEOUT" in str(status).upper()


def _pick_densest_mp(df: pd.DataFrame, ds: str, L_ref: int = 2) -> str:
    """Metapath with the highest Edge_Count at Exact, L=L_ref."""
    sub = df[(df["Dataset"] == ds) & df["MetaPath"].notna()]
    exact_sub = sub[sub["Method"] == "Exact"]
    ref = exact_sub[exact_sub["L"] == L_ref] if not exact_sub.empty else exact_sub
    if ref.empty:
        ref = exact_sub
    if ref.empty:
        ref = sub
    if ref.empty:
        raise ValueError(f"No rows for dataset {ds}")
    by_mp = (
        ref.groupby("MetaPath")["Edge_Count"]
        .max().dropna().sort_values(ascending=False)
    )
    if by_mp.empty:
        return ref["MetaPath"].dropna().iloc[0]
    return str(by_mp.index[0])


def _pick_mp(df: pd.DataFrame, ds: str, prefer: Optional[str],
             L_ref: int = 2) -> str:
    avail = df.loc[df["Dataset"] == ds, "MetaPath"].dropna().unique().tolist()
    if not avail:
        raise ValueError(f"No rows for dataset {ds}")
    if prefer and prefer in avail:
        return prefer
    if prefer:
        warnings.warn(f"MetaPath '{prefer}' absent for {ds}; auto-selecting densest")
    try:
        return _pick_densest_mp(df, ds, L_ref=L_ref)
    except ValueError:
        return avail[0]


def _get_mlp_f1(df: pd.DataFrame, ds: str) -> float:
    sub = df[(df["Dataset"] == ds) & (df["Method"] == "mlp") & df["MetaPath"].isna()]
    m, _ = _ms(sub, "Macro_F1")
    return m


def _any_mprw_saturates(df: pd.DataFrame, ds: str, mp: str, L: int) -> bool:
    base      = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp) & (df["L"] == L)]
    exact_ram, _ = _ms(base[base["Method"] == "Exact"], "Mat_RAM_MB")
    if np.isnan(exact_ram) or exact_ram <= 0:
        return False
    mprw_rams = base[base["Method"] == "MPRW"]["Mat_RAM_MB"].dropna()
    return bool((mprw_rams > exact_ram).any())


def _label(ds: str) -> str:
    return _LABEL.get(ds, ds)


# ── Shared line-plot helper ───────────────────────────────────────────────────

def _line_with_band(ax, xs, ys, errs, color, marker, ls, label, ms=7):
    """Plot a line + optional shaded std band."""
    yerr_arr = [e if (e and e > 0) else np.nan for e in errs]
    has_err  = any(not np.isnan(e) for e in yerr_arr)
    ax.errorbar(xs, ys,
                yerr=yerr_arr if has_err else None,
                fmt=f"{marker}{ls}", color=color, label=label,
                linewidth=2, markersize=ms, capsize=4)
    if has_err:
        lo = [y - e for y, e in zip(ys, errs)]
        hi = [y + e for y, e in zip(ys, errs)]
        ax.fill_between(xs, lo, hi, alpha=0.12, color=color)


# ── Plot style ────────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":       10,
        "axes.titlesize":  11,
        "axes.labelsize":  10,
        "legend.fontsize":  8,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linestyle":   "--",
        "axes.axisbelow":   True,
    })


def _save(fig, path: Path, dpi: int = 300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)


def _make_grid(n: int, row_h: float = 3.8) -> tuple:
    """Return (fig, axes_list) for a 1xN subplot row."""
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, row_h), sharey=False)
    return fig, ([axes] if n == 1 else list(axes))


# ── PHASE I — Systems ────────────────────────────────────────────────────────

def plot1a(df, datasets, out_dir, prefer_mp, L_fixed, k_fixed):
    """Stacked bar: peak RAM (Mat + Inf), Exact vs KMV."""
    print(f"\n[plot1a] Peak Memory Footprint — stacked bar (L={L_fixed}, k={k_fixed})")
    x, width = np.arange(len(datasets)), 0.32
    fig, ax  = plt.subplots(figsize=(max(5.5, 1.8 * len(datasets)), 4.2))

    for method, offset in [("Exact", -width / 2), ("KMV", width / 2)]:
        mat_m, mat_s, inf_m, inf_s = [], [], [], []
        for ds in datasets:
            mp   = _pick_mp(df, ds, prefer_mp, L_ref=L_fixed)
            base = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp) & (df["L"] == L_fixed)]
            sub  = base[base["Method"] == "Exact"] if method == "Exact" else \
                   base[(base["Method"] == "KMV") & (base["k_value"] == k_fixed)]
            m, s = _ms(sub, "Mat_RAM_MB"); mat_m.append(m); mat_s.append(s)
            m2, s2 = _ms(sub, "Inf_RAM_MB"); inf_m.append(m2); inf_s.append(s2)

        color = _C[method]
        ax.bar(x + offset, mat_m, width, color=color, alpha=0.85,
               yerr=mat_s if any(s > 0 for s in mat_s) else None,
               capsize=4, error_kw={"elinewidth": 1.2})
        valid = [(i, inf_m[i], inf_s[i], mat_m[i]) for i in range(len(datasets))
                 if not np.isnan(inf_m[i])]
        if valid:
            xi   = np.array([x[i] + offset for i, _, _, _ in valid])
            yi   = [v for _, v, _, _ in valid]
            si   = [s for _, _, s, _ in valid]
            bi   = [b for _, _, _, b in valid]
            ax.bar(xi, yi, width, bottom=bi,
                   color=color, alpha=0.45, hatch=_HATCH_INF,
                   yerr=si if any(s > 0 for s in si) else None,
                   capsize=4, error_kw={"elinewidth": 1.0})

    ax.set_xticks(x); ax.set_xticklabels([_label(d) for d in datasets])
    ax.set_ylabel("Peak RAM (MB)")
    ax.set_title(f"Peak Memory Footprint (L={L_fixed}, k={k_fixed})\n"
                 "Solid = Materialization, Hatched = Inference")

    all_vals = df[df["Dataset"].isin(datasets)][["Mat_RAM_MB", "Inf_RAM_MB"]] \
                 .stack().dropna()
    if not all_vals.empty and (all_vals.max() / max(all_vals.min(), 1e-3)) > 50:
        ax.set_yscale("log"); ax.set_ylabel("Peak RAM (MB, log scale)")

    ax.legend(handles=[
        Patch(facecolor=_C["Exact"], label="Exact",              alpha=0.85),
        Patch(facecolor=_C["KMV"],   label=f"KMV (k={k_fixed})", alpha=0.85),
        Patch(facecolor="grey",      label="Materialization", hatch=_HATCH_MAT, alpha=0.6),
        Patch(facecolor="grey",      label="Inference",       hatch=_HATCH_INF, alpha=0.6),
    ], fontsize=7.5)
    fig.tight_layout()
    _save(fig, out_dir / "plot1a_memory_stacked.pdf")


def plot1b(df, datasets, out_dir, prefer_mp, L_fixed, k_fixed):
    """Stacked bar: E2E time (Mat + Inf), Exact vs KMV."""
    print(f"\n[plot1b] E2E Execution Time — stacked bar (L={L_fixed}, k={k_fixed})")
    x, width = np.arange(len(datasets)), 0.32
    fig, ax  = plt.subplots(figsize=(max(5.5, 1.8 * len(datasets)), 4.2))

    for method, offset in [("Exact", -width / 2), ("KMV", width / 2)]:
        mat_m, mat_s, inf_m, inf_s = [], [], [], []
        for ds in datasets:
            mp   = _pick_mp(df, ds, prefer_mp, L_ref=L_fixed)
            base = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp) & (df["L"] == L_fixed)]
            sub  = base[base["Method"] == "Exact"] if method == "Exact" else \
                   base[(base["Method"] == "KMV") & (base["k_value"] == k_fixed)]
            m, s = _ms(sub, "Materialization_Time"); mat_m.append(m); mat_s.append(s)
            m2, s2 = _ms(sub, "Inference_Time"); inf_m.append(m2); inf_s.append(s2)

        color = _C[method]
        ax.bar(x + offset, mat_m, width, color=color, alpha=0.85,
               yerr=mat_s if any(s > 0 for s in mat_s) else None,
               capsize=4, error_kw={"elinewidth": 1.2})
        valid = [(i, inf_m[i], inf_s[i], mat_m[i]) for i in range(len(datasets))
                 if not np.isnan(inf_m[i])]
        if valid:
            xi  = np.array([x[i] + offset for i, _, _, _ in valid])
            yi  = [v for _, v, _, _ in valid]
            si  = [s for _, _, s, _ in valid]
            bi  = [b for _, _, _, b in valid]
            ax.bar(xi, yi, width, bottom=bi,
                   color=color, alpha=0.45, hatch=_HATCH_INF,
                   yerr=si if any(s > 0 for s in si) else None,
                   capsize=4, error_kw={"elinewidth": 1.0})

    ax.set_xticks(x); ax.set_xticklabels([_label(d) for d in datasets])
    ax.set_ylabel("Wall-clock Time (s)")
    ax.set_title(f"End-to-End Execution Time (L={L_fixed}, k={k_fixed})\n"
                 "Solid = Materialization, Hatched = Inference")
    ax.legend(handles=[
        Patch(facecolor=_C["Exact"], label="Exact",              alpha=0.85),
        Patch(facecolor=_C["KMV"],   label=f"KMV (k={k_fixed})", alpha=0.85),
        Patch(facecolor="grey",      label="Materialization", hatch=_HATCH_MAT, alpha=0.6),
        Patch(facecolor="grey",      label="Inference",       hatch=_HATCH_INF, alpha=0.6),
    ], fontsize=7.5)
    fig.tight_layout()
    _save(fig, out_dir / "plot1b_time_stacked.pdf")


def plot1c(df, datasets, out_dir, prefer_mp, k_fixed):
    """Bar: preprocessing latency (Mat time only), KMV vs MPRW."""
    print(f"\n[plot1c] Preprocessing Latency — KMV vs MPRW (k={k_fixed})")
    x, width = np.arange(len(datasets)), 0.32
    fig, ax  = plt.subplots(figsize=(max(5.0, 1.8 * len(datasets)), 4.2))

    for method, offset in [("KMV", -width / 2), ("MPRW", width / 2)]:
        means, stds = [], []
        for ds in datasets:
            mp   = _pick_mp(df, ds, prefer_mp)
            base = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp) & df["L"].notna()]
            sub  = base[(base["Method"] == method) & (base["k_value"] == k_fixed)]
            m, s = _ms(sub, "Materialization_Time"); means.append(m); stds.append(s)
        ax.bar(x + offset, means, width, color=_C[method], alpha=0.85,
               label=f"{method} (k={k_fixed})",
               yerr=stds if any(s > 0 for s in stds) else None,
               capsize=4, error_kw={"elinewidth": 1.2})

    ax.set_xticks(x); ax.set_xticklabels([_label(d) for d in datasets])
    ax.set_ylabel("Graph Generation Time (s)")
    ax.set_title(f"Preprocessing Latency: KMV vs. MPRW (k={k_fixed})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / "plot1c_preprocessing_latency.pdf")


# ── Table 1: Information Gain Validation ──────────────────────────────────────

def table1(df, datasets, out_dir, L_fixed, k_fixed):
    print(f"\n[table1] Information Gain Validation (L={L_fixed})")
    records = []
    for ds in datasets:
        mlp_f1  = _get_mlp_f1(df, ds)
        mp      = _pick_mp(df, ds, None, L_ref=L_fixed)
        base    = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp)
                     & (df["L"] == L_fixed) & (df["Method"] == "Exact")]
        exact_f1, _ = _ms(base, "Macro_F1")
        delta = (exact_f1 - mlp_f1
                 if not (np.isnan(exact_f1) or np.isnan(mlp_f1))
                 else float("nan"))
        records.append({
            "Dataset":              _label(ds),
            "MetaPath":             mp,
            "Feature-Only F1 (MLP)": mlp_f1,
            "Exact GNN F1":          exact_f1,
            "Delta (Info Gain)":     delta,
            "Signal": "PASS" if (not np.isnan(delta) and delta >= 0) else "FAIL",
        })

    rdf = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "table1_info_gain.csv"
    rdf.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved -> {csv_path}")

    W = 92
    print("\n" + "=" * W)
    print("TABLE 1 -- Information Gain Validation")
    print("=" * W)
    print(f"{'Dataset':8}  {'MLP F1':>10}  {'Exact F1':>10}  {'Delta':>10}  {'Signal':>6}")
    print("-" * W)
    for _, row in rdf.iterrows():
        mlp = f"{row['Feature-Only F1 (MLP)']:.4f}" if not np.isnan(row["Feature-Only F1 (MLP)"]) else "n/a"
        ex  = f"{row['Exact GNN F1']:.4f}"           if not np.isnan(row["Exact GNN F1"]) else "n/a"
        dt  = f"{row['Delta (Info Gain)']:+.4f}"     if not np.isnan(row["Delta (Info Gain)"]) else "n/a"
        print(f"{row['Dataset']:8}  {mlp:>10}  {ex:>10}  {dt:>10}  {row['Signal']:>6}")
    print("=" * W)
    print("PASS = topology adds signal (Exact > MLP).")


# ── Plot 2: Pareto Frontier (k-sweep, x=RAM, y=F1) ───────────────────────────

def plot2(df, datasets, out_dir, prefer_mp, L_fixed):
    print(f"\n[plot2] Cost-Quality Pareto Frontier (L={L_fixed})")
    fig, axes = _make_grid(len(datasets))
    all_k = sorted(df.loc[df["Method"] == "KMV", "k_value"].dropna().unique())

    for ax, ds in zip(axes, datasets):
        mp   = _pick_mp(df, ds, prefer_mp, L_ref=L_fixed)
        base = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp) & (df["L"] == L_fixed)]

        # Exact star
        esub = base[base["Method"] == "Exact"]
        if not esub.empty:
            ram, _ = _ms(esub, "Mat_RAM_MB"); f1, _ = _ms(esub, "Macro_F1")
            if not (np.isnan(ram) or np.isnan(f1)):
                ax.plot(ram, f1, marker="*", color=_C["Exact"], markersize=20,
                        linestyle="none", zorder=6, label="Exact")

        # KMV k-sweep
        kmv_base = base[base["Method"] == "KMV"].dropna(subset=["k_value"])
        if not kmv_base.empty:
            alphas  = np.linspace(0.30, 1.0, max(len(all_k), 1))
            k_alpha = {int(k): float(a) for k, a in zip(sorted(all_k), alphas)}
            xs, ys = [], []
            for k in sorted(kmv_base["k_value"].unique()):
                ksub = kmv_base[kmv_base["k_value"] == k]
                ram, _ = _ms(ksub, "Mat_RAM_MB"); f1, f1e = _ms(ksub, "Macro_F1")
                if np.isnan(ram) or np.isnan(f1):
                    continue
                ax.errorbar(ram, f1, yerr=(f1e if f1e > 0 else None),
                            fmt="o", color=_C["KMV"], alpha=k_alpha.get(int(k), 1.0),
                            markersize=6, capsize=3,
                            label=f"KMV ($k={int(k)}$)" if k == max(all_k) else "_")
                xs.append(ram); ys.append(f1)
            if len(xs) > 1:
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                ax.plot([xs[i] for i in order], [ys[i] for i in order],
                        "-", color=_C["KMV"], alpha=0.30, linewidth=1.2)

        # MPRW k-sweep
        mprw_base = base[base["Method"] == "MPRW"].dropna(subset=["k_value"])
        if not mprw_base.empty:
            k_sorted = sorted(mprw_base["k_value"].unique())
            alphas2  = np.linspace(0.30, 1.0, max(len(k_sorted), 1))
            xs2, ys2 = [], []
            for i, k in enumerate(k_sorted):
                msub = mprw_base[mprw_base["k_value"] == k]
                ram, _ = _ms(msub, "Mat_RAM_MB"); f1, f1e = _ms(msub, "Macro_F1")
                if np.isnan(ram) or np.isnan(f1):
                    continue
                ax.errorbar(ram, f1, yerr=(f1e if f1e > 0 else None),
                            fmt="s", color=_C["MPRW"], alpha=alphas2[i],
                            markersize=6, capsize=3,
                            label="MPRW" if i == len(k_sorted) - 1 else "_")
                xs2.append(ram); ys2.append(f1)
            if len(xs2) > 1:
                order2 = sorted(range(len(xs2)), key=lambda i: xs2[i])
                ax.plot([xs2[i] for i in order2], [ys2[i] for i in order2],
                        "-", color=_C["MPRW"], alpha=0.30, linewidth=1.2)

        ax.set_xlabel("Peak Materialization RAM (MB)")
        ax.set_ylabel("Macro-F1")
        ax.set_title(_label(ds))
        ax.legend(loc="lower right", fontsize=7)
        if _any_mprw_saturates(df, ds, mp, L_fixed):
            ax.text(0.04, 0.97, "MPRW Mat.RAM > Exact\n(saturation regime)",
                    transform=ax.transAxes, fontsize=6.5, style="italic",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#ffe8c0", edgecolor="#cc8800", alpha=0.85))

    fig.suptitle(rf"Cost-Quality Pareto Frontier  ($L={L_fixed}$)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "plot2_pareto_frontier.pdf")


# ── Plot 3: Dirichlet Energy vs Depth ────────────────────────────────────────

def plot3(df, datasets, out_dir, prefer_mp, k_fixed, depths):
    print(f"\n[plot3] Dirichlet Energy vs Depth (k={k_fixed})")
    fig, axes = _make_grid(len(datasets))
    for ax, ds in zip(axes, datasets):
        mp = _pick_mp(df, ds, prefer_mp)
        for method, label, ls in [
            ("Exact", "Exact",                "--"),
            ("KMV",   f"KMV (k={k_fixed})",   "-"),
            ("MPRW",  "MPRW",                  "-"),
        ]:
            xs, ys, errs = [], [], []
            for L in depths:
                if L == 0:
                    continue
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
                if not np.isnan(m):
                    xs.append(L); ys.append(m); errs.append(s)
            if not xs:
                continue
            ms = 11 if method == "Exact" else 7
            _line_with_band(ax, xs, ys, errs, _C[method], _M[method], ls, label, ms=ms)

        ax.set_xticks([d for d in depths if d > 0])
        ax.set_xlabel("Network Depth L")
        ax.set_ylabel("Dirichlet Energy")
        ax.set_title(_label(ds)); ax.legend(fontsize=7)

    fig.suptitle(f"Embedding Smoothness vs. Depth  (k={k_fixed})", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "plot3_dirichlet_vs_depth.pdf")


# ── k-sweep suite: 3-panel (F1 / CKA / Pred-Sim) vs k, one column per dataset ─

def _sub_query(df, ds, mp, L, method, k=None):
    """Return the filtered sub-DataFrame for one (ds, mp, L, method[, k]) cell."""
    base = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp) & (df["L"] == L)
              & (df["Method"] == method)]
    if k is not None:
        base = base[base["k_value"] == k]
    return base


def plot_k_sweep_suite(df, datasets, out_dir, prefer_mp, L_fixed):
    """3-row × N-col figure: rows = (F1, CKA, Pred-Agreement), cols = datasets.

    Blueprint:
      Panel A — Downstream Quality:    Macro-F1 vs k
      Panel B — Structural Alignment:  Output CKA vs k
      Panel C — Behavioral Alignment:  Prediction Agreement vs k

    Series: Exact (horizontal ceiling), KMV (mean ± std), MPRW (mean ± std).
    X-axis: k budget.  All seeds aggregated via mean ± std.
    """
    print(f"\n[plot_k_sweep_suite] 3-panel k-sweep (L={L_fixed})")
    n = len(datasets)
    all_k = sorted(df.loc[df["Method"] == "KMV", "k_value"].dropna().unique())

    fig, axes = plt.subplots(3, n, figsize=(4.5 * n, 10.5), squeeze=False)
    row_titles = [
        "Panel A — Downstream Quality (Macro-F1)",
        "Panel B — Structural Alignment (Output CKA)",
        "Panel C — Behavioral Alignment (Prediction Agreement)",
    ]
    ylabels = ["Macro-F1", "Output CKA", "Prediction Agreement"]

    for col, ds in enumerate(datasets):
        mp   = _pick_mp(df, ds, prefer_mp, L_ref=L_fixed)
        base = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp) & (df["L"] == L_fixed)]

        for row, (metric_fn, ylabel) in enumerate(zip(
            [
                lambda sub, L: _ms(sub, "Macro_F1"),
                lambda sub, L: _ms(sub, f"CKA_L{L}"),
                lambda sub, L: _ms(sub, "Pred_Similarity"),
            ],
            ylabels,
        )):
            ax = axes[row][col]

            # Exact ceiling line
            esub = base[base["Method"] == "Exact"]
            if not esub.empty:
                ev, _ = metric_fn(esub, L_fixed)
                if not np.isnan(ev):
                    ax.axhline(ev, color=_C["Exact"], linestyle="--",
                               linewidth=1.8, alpha=0.85, label="Exact")

            # KMV and MPRW k-sweep
            for method in ("KMV", "MPRW"):
                xs, ys, errs = [], [], []
                for k in all_k:
                    msub = base[(base["Method"] == method) & (base["k_value"] == k)]
                    m, s = metric_fn(msub, L_fixed)
                    if not np.isnan(m):
                        xs.append(int(k)); ys.append(m); errs.append(s)
                if xs:
                    lbl = f"KMV" if method == "KMV" else "MPRW"
                    _line_with_band(ax, xs, ys, errs, _C[method], _M[method], "-", lbl)

            if row == 1:      # CKA panel: fix y-range
                ax.set_ylim(0.0, 1.10)
            if all_k:
                ax.set_xticks([int(k) for k in all_k])
                ax.tick_params(axis="x", labelrotation=45)
            ax.set_xlabel("Sketch Budget k")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)

            # Column header on top row only
            if row == 0:
                ax.set_title(_label(ds), fontweight="bold")

    for row, rt in enumerate(row_titles):
        axes[row][0].annotate(
            rt, xy=(0, 0.5), xytext=(-axes[row][0].yaxis.labelpad - 38, 0),
            xycoords=axes[row][0].yaxis.label, textcoords="offset points",
            ha="right", va="center", fontsize=8, style="italic",
            rotation=90,
        )

    fig.suptitle(f"Budget Scaling Proof: k-Sweep Suite  (L={L_fixed})", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=[0.04, 0, 1, 0.97])
    _save(fig, out_dir / "plot_k_sweep_suite.pdf")


# ── L-sweep suite: 3-panel (Dirichlet / F1 / CKA) vs L, one column per dataset ─

def plot_l_sweep_suite(df, datasets, out_dir, prefer_mp, k_fixed, depths):
    """3-row × N-col figure: rows = (Dirichlet Energy, F1, CKA), cols = datasets.

    Blueprint:
      Panel A — Over-smoothing Check:  Dirichlet Energy vs L
      Panel B — Accuracy Check:        Macro-F1 vs L
      Panel C — Alignment Check:       Output CKA vs L

    Dirichlet Energy uses all three series (Exact, KMV, MPRW) to prove
    that the diffusion trajectory is preserved.  CKA and F1 include Exact
    as a ceiling reference.  CKA is read as CKA_L{L} (the deepest layer
    CKA for each model depth L), so it tracks the final representation.
    """
    print(f"\n[plot_l_sweep_suite] 3-panel L-sweep (k={k_fixed})")
    n = len(datasets)
    valid_depths = [d for d in depths if d > 0]

    fig, axes = plt.subplots(3, n, figsize=(4.5 * n, 10.5), squeeze=False)
    row_titles = [
        "Panel A — Over-smoothing Check (Dirichlet Energy)",
        "Panel B — Accuracy Check (Macro-F1)",
        "Panel C — Alignment Check (Output CKA)",
    ]
    ylabels = ["Dirichlet Energy", "Macro-F1", "Output CKA"]

    for col, ds in enumerate(datasets):
        mp = _pick_mp(df, ds, prefer_mp)

        for row in range(3):
            ax = axes[row][col]

            for method, label, ls in [
                ("Exact", "Exact",            "--"),
                ("KMV",   f"KMV (k={k_fixed})", "-"),
                ("MPRW",  "MPRW",               "-"),
            ]:
                xs, ys, errs = [], [], []
                for L in valid_depths:
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

                    if row == 0:
                        m, s = _ms(sub, "Dirichlet_Energy")
                    elif row == 1:
                        m, s = _ms(sub, "Macro_F1")
                    else:  # row == 2: CKA at the final layer of this model depth
                        # CKA_L{L} is the last-layer CKA for the L-layer model
                        if method == "Exact":
                            # Exact has no CKA (reference); skip
                            continue
                        m, s = _ms(sub, f"CKA_L{L}")

                    if not np.isnan(m):
                        xs.append(L); ys.append(m); errs.append(s)

                if not xs:
                    continue
                ms_size = 11 if method == "Exact" else 7
                _line_with_band(ax, xs, ys, errs, _C[method], _M[method], ls,
                                label, ms=ms_size)

            # CKA panel: add Exact ref line
            if row == 2:
                ax.axhline(1.0, color=_C["Exact"], linestyle="--",
                           linewidth=1.5, alpha=0.75, label="Exact (ref.)")
                ax.set_ylim(0.0, 1.10)

            ax.set_xticks(valid_depths)
            ax.set_xlabel("Network Depth L")
            ax.set_ylabel(ylabels[row])
            ax.legend(fontsize=7)

            if row == 0:
                ax.set_title(_label(ds), fontweight="bold")

    for row, rt in enumerate(row_titles):
        axes[row][0].annotate(
            rt, xy=(0, 0.5), xytext=(-axes[row][0].yaxis.labelpad - 38, 0),
            xycoords=axes[row][0].yaxis.label, textcoords="offset points",
            ha="right", va="center", fontsize=8, style="italic",
            rotation=90,
        )

    fig.suptitle(f"Depth Robustness Proof: L-Sweep Suite  (k={k_fixed})", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=[0.04, 0, 1, 0.97])
    _save(fig, out_dir / "plot_l_sweep_suite.pdf")


# ── CKA per-layer trajectory (within a fixed-depth run) ──────────────────────

def plot_cka_per_layer(df, datasets, out_dir, prefer_mp, k_fixed,
                       L_for_trajectory: int = 4):
    """For a model of depth L_for_trajectory, plot CKA_L1..CKA_L{L} vs layer index.

    Shows how representational fidelity evolves *within* one model as activations
    propagate layer-by-layer — complementary to the across-L suite which tracks
    final-layer CKA only.
    """
    print(f"\n[plot_cka_per_layer] Within-model CKA trajectory "
          f"(L={L_for_trajectory}, k={k_fixed})")
    n_layers   = L_for_trajectory
    layer_cols = [f"CKA_L{i}" for i in range(1, n_layers + 1)]

    fig, axes = _make_grid(len(datasets))
    for ax, ds in zip(axes, datasets):
        mp = _pick_mp(df, ds, prefer_mp)
        for method, label, ls in [
            ("KMV",  f"KMV (k={k_fixed})", "-"),
            ("MPRW", "MPRW",               "-"),
        ]:
            sub = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp)
                     & (df["L"] == L_for_trajectory) & (df["Method"] == method)]
            if method == "KMV":
                sub = sub[sub["k_value"] == k_fixed]
            elif method == "MPRW":
                sub = sub[sub["k_value"] == k_fixed]

            xs, ys, errs = [], [], []
            for i, col in enumerate(layer_cols, start=1):
                if col not in sub.columns:
                    continue
                m, s = _ms(sub, col)
                if not np.isnan(m):
                    xs.append(i); ys.append(m); errs.append(s)
            if xs:
                _line_with_band(ax, xs, ys, errs, _C[method], _M[method], ls, label)

        ax.axhline(1.0, color=_C["Exact"], linestyle="--",
                   linewidth=1.5, alpha=0.75, label="Exact (ref.)")
        ax.set_xlim(0.5, n_layers + 0.5)
        ax.set_xticks(range(1, n_layers + 1))
        ax.set_ylim(0.0, 1.10)
        ax.set_xlabel("GNN Layer Index")
        ax.set_ylabel("CKA Similarity")
        ax.set_title(_label(ds))
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Within-Model CKA Trajectory  (L={L_for_trajectory}, k={k_fixed})",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out_dir / "plot_cka_per_layer.pdf")


# ── Table 2: Downstream Integrity ────────────────────────────────────────────

def table2_integrity(df, datasets, out_dir, L_fixed, k_fixed):
    print(f"\n[table2] Downstream Integrity (L={L_fixed}, k={k_fixed})")
    records = []
    for ds in datasets:
        mp   = _pick_mp(df, ds, None, L_ref=L_fixed)
        base = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp) & (df["L"] == L_fixed)]
        exact_f1, _ = _ms(base[base["Method"] == "Exact"], "Macro_F1")

        for method in ("KMV", "MPRW"):
            msub = base[(base["Method"] == method) & (base["k_value"] == k_fixed)]
            if msub.empty:
                continue
            f1_m, f1_s = _ms(msub, "Macro_F1")
            pa_m, pa_s = _ms(msub, "Pred_Similarity")
            records.append({
                "Dataset":                 _label(ds),
                "MetaPath":                mp,
                "Method":                  method,
                "Exact F1":                exact_f1,
                "Method F1 (mean)":        f1_m,
                "Method F1 (std)":         f1_s,
                "Pred. Agreement (mean)":  pa_m * 100 if not np.isnan(pa_m) else float("nan"),
                "Pred. Agreement (std)":   pa_s * 100 if not np.isnan(pa_s) else float("nan"),
            })

    if not records:
        print("  [skip] No data for Table 2"); return

    rdf = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "table2_integrity.csv"
    rdf.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved -> {csv_path}")

    W = 102
    print("\n" + "=" * W)
    print(f"TABLE 2 -- Downstream Integrity  (L={L_fixed}, k={k_fixed})")
    print("=" * W)
    print(f"{'Dataset':8}  {'Method':5}  {'Exact F1':>10}  "
          f"{'Method F1 (m+/-s)':>20}  {'Pred. Agree. (m+/-s %)':>24}")
    print("-" * W)
    for _, row in rdf.iterrows():
        ex  = f"{row['Exact F1']:.4f}" if not np.isnan(row["Exact F1"]) else "n/a"
        f1s = (f"{row['Method F1 (mean)']:.4f}+/-{row['Method F1 (std)']:.4f}"
               if not np.isnan(row["Method F1 (mean)"]) else "n/a")
        pas = (f"{row['Pred. Agreement (mean)']:.1f}+/-{row['Pred. Agreement (std)']:.1f}"
               if not np.isnan(row["Pred. Agreement (mean)"]) else "n/a")
        print(f"{row['Dataset']:8}  {row['Method']:5}  {ex:>10}  {f1s:>20}  {pas:>24}")
    print("=" * W)


# ── Appendix Table A: Meta-Path Generalization ───────────────────────────────

def appendix_table_a(df, datasets, out_dir, L_fixed, k_fixed):
    print(f"\n[app_a] Meta-Path Generalization (L={L_fixed}, k={k_fixed})")
    records = []
    for ds in datasets:
        all_mps = df[df["Dataset"] == ds]["MetaPath"].dropna().unique()
        for mp in sorted(all_mps):
            base = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp) & (df["L"] == L_fixed)]
            esub = base[base["Method"] == "Exact"]
            if esub.empty:
                continue
            ksub = base[(base["Method"] == "KMV") & (base["k_value"] == k_fixed)]
            exact_ram, _ = _ms(esub, "Mat_RAM_MB")
            exact_f1,  _ = _ms(esub, "Macro_F1")
            kmv_ram,   _ = _ms(ksub, "Mat_RAM_MB")
            kmv_f1, kmv_s = _ms(ksub, "Macro_F1")
            retention = (kmv_f1 / exact_f1 * 100
                         if not (np.isnan(kmv_f1) or np.isnan(exact_f1) or exact_f1 == 0)
                         else float("nan"))
            records.append({
                "Dataset":              _label(ds),
                "MetaPath":             mp,
                "Hops":                 _hop_count(mp),
                "Exact RAM (MB)":       exact_ram,
                "KMV RAM (MB)":         kmv_ram,
                "Exact F1":             exact_f1,
                f"KMV F1 mean (k={k_fixed})": kmv_f1,
                f"KMV F1 std  (k={k_fixed})": kmv_s,
                "Relative Retention (%)":      retention,
            })

    if not records:
        print("  [skip] No data"); return

    rdf = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "appendix_table_a_metapath_generalization.csv"
    rdf.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved -> {csv_path}")

    W = 118
    print("\n" + "=" * W)
    print(f"APPENDIX TABLE A -- Meta-Path Generalization  (L={L_fixed}, k={k_fixed})")
    print("=" * W)
    print(f"{'Dataset':8}  {'Hops':>4}  {'Exact RAM':>10}  {'KMV RAM':>10}  "
          f"{'Exact F1':>9}  {'KMV F1 m+/-s':>22}  {'Retention':>10}  MetaPath")
    print("-" * W)
    for _, row in rdf.iterrows():
        er  = f"{row['Exact RAM (MB)']:.1f}"           if not np.isnan(row["Exact RAM (MB)"]) else "n/a"
        kr  = f"{row['KMV RAM (MB)']:.1f}"             if not np.isnan(row["KMV RAM (MB)"]) else "n/a"
        ef1 = f"{row['Exact F1']:.4f}"                 if not np.isnan(row["Exact F1"]) else "n/a"
        km  = row[f"KMV F1 mean (k={k_fixed})"]
        ks  = row[f"KMV F1 std  (k={k_fixed})"]
        kf1 = f"{km:.4f}+/-{ks:.4f}" if not np.isnan(km) else "n/a"
        ret = f"{row['Relative Retention (%)']:.1f}%" if not np.isnan(row["Relative Retention (%)"]) else "n/a"
        print(f"{row['Dataset']:8}  {int(row['Hops']):>4}  {er:>10}  {kr:>10}  "
              f"{ef1:>9}  {kf1:>22}  {ret:>10}  {row['MetaPath']}")
    print("=" * W)


# ── Master Numbers Table ─────────────────────────────────────────────────────

def table_master_numbers(df, datasets, out_dir, depths, k_values):
    """Exhaustive CSV: every (dataset, metapath, L, method, k) cell, all metrics.

    Columns (mean +/- std across seeds):
        Dataset, MetaPath, Hops, L, Method, k_value,
        Mat_RAM_MB (mean), Mat_RAM_MB (std),
        Inf_RAM_MB (mean), Inf_RAM_MB (std),
        Mat_Time   (mean), Mat_Time   (std),
        Inf_Time   (mean), Inf_Time   (std),
        Edge_Count (mean),
        Macro_F1   (mean), Macro_F1   (std),
        CKA_final  (mean), CKA_final  (std),   ← CKA_L{L} for each row
        Pred_Sim   (mean), Pred_Sim   (std),
        Dirichlet  (mean), Dirichlet  (std),
    """
    print(f"\n[table_master] Exhaustive master numbers table ...")
    records = []

    for ds in datasets:
        all_mps = sorted(df[df["Dataset"] == ds]["MetaPath"].dropna().unique())
        for mp in all_mps:
            hops = _hop_count(mp)
            for L in depths:
                base = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp)
                          & (df["L"] == L)]
                if base.empty:
                    continue

                # Exact (no k)
                esub = base[base["Method"] == "Exact"]
                if not esub.empty:
                    cka_col = f"CKA_L{L}"
                    row = {"Dataset": _label(ds), "MetaPath": mp, "Hops": hops,
                           "L": L, "Method": "Exact", "k_value": ""}
                    for col, out in [
                        ("Mat_RAM_MB",          "Mat_RAM_MB"),
                        ("Inf_RAM_MB",          "Inf_RAM_MB"),
                        ("Materialization_Time","Mat_Time"),
                        ("Inference_Time",      "Inf_Time"),
                        ("Macro_F1",            "Macro_F1"),
                        ("Pred_Similarity",     "Pred_Sim"),
                        ("Dirichlet_Energy",    "Dirichlet"),
                    ]:
                        m, s = _ms(esub, col)
                        row[f"{out} (mean)"] = round(m, 6) if not np.isnan(m) else ""
                        row[f"{out} (std)"]  = round(s, 6) if not np.isnan(m) else ""
                    ec, _ = _ms(esub, "Edge_Count")
                    row["Edge_Count"] = int(ec) if not np.isnan(ec) else ""
                    # Exact has no CKA (it's the reference)
                    row["CKA_final (mean)"] = ""
                    row["CKA_final (std)"]  = ""
                    records.append(row)

                # KMV and MPRW — per k
                for method in ("KMV", "MPRW"):
                    msub_all = base[base["Method"] == method]
                    for k in sorted(msub_all["k_value"].dropna().unique()):
                        if int(k) not in [int(kv) for kv in k_values]:
                            continue
                        msub = msub_all[msub_all["k_value"] == k]
                        if msub.empty:
                            continue
                        cka_col = f"CKA_L{L}"
                        row = {"Dataset": _label(ds), "MetaPath": mp, "Hops": hops,
                               "L": L, "Method": method, "k_value": int(k)}
                        for col, out in [
                            ("Mat_RAM_MB",          "Mat_RAM_MB"),
                            ("Inf_RAM_MB",          "Inf_RAM_MB"),
                            ("Materialization_Time","Mat_Time"),
                            ("Inference_Time",      "Inf_Time"),
                            ("Macro_F1",            "Macro_F1"),
                            ("Pred_Similarity",     "Pred_Sim"),
                            ("Dirichlet_Energy",    "Dirichlet"),
                        ]:
                            m, s = _ms(msub, col)
                            row[f"{out} (mean)"] = round(m, 6) if not np.isnan(m) else ""
                            row[f"{out} (std)"]  = round(s, 6) if not np.isnan(m) else ""
                        ec, _ = _ms(msub, "Edge_Count")
                        row["Edge_Count"] = int(ec) if not np.isnan(ec) else ""
                        cka_m, cka_s = _ms(msub, cka_col)
                        row["CKA_final (mean)"] = round(cka_m, 6) if not np.isnan(cka_m) else ""
                        row["CKA_final (std)"]  = round(cka_s, 6) if not np.isnan(cka_m) else ""
                        records.append(row)

    if not records:
        print("  [skip] No data"); return

    col_order = [
        "Dataset", "MetaPath", "Hops", "L", "Method", "k_value", "Edge_Count",
        "Mat_RAM_MB (mean)", "Mat_RAM_MB (std)",
        "Inf_RAM_MB (mean)", "Inf_RAM_MB (std)",
        "Mat_Time (mean)",   "Mat_Time (std)",
        "Inf_Time (mean)",   "Inf_Time (std)",
        "Macro_F1 (mean)",   "Macro_F1 (std)",
        "CKA_final (mean)",  "CKA_final (std)",
        "Pred_Sim (mean)",   "Pred_Sim (std)",
        "Dirichlet (mean)",  "Dirichlet (std)",
    ]
    rdf = pd.DataFrame(records)[col_order]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "table_master_numbers.csv"
    rdf.to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}  ({len(rdf)} rows)")


# ── CLI ───────────────────────────────────────────────────────────────────────

_ALL_PLOTS  = [
    "plot1a", "plot1b", "plot1c",
    "plot2", "plot3",
    "plot_k_sweep_suite",
    "plot_l_sweep_suite",
    "plot_cka_per_layer",
]
_ALL_TABLES = ["table1", "table2", "app_a", "master"]


def _parse():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--transfer-dir", default="transfer",
                   help="Root dir with results_<seed>/ subdirs (default: transfer/)")
    p.add_argument("--out-dir", default="figures",
                   help="Output directory for PDFs and CSVs (default: figures/)")
    p.add_argument("--datasets", nargs="+",
                   default=["HGB_ACM", "HGB_DBLP", "HGB_IMDB", "HNE_PubMed"],
                   help="Datasets to include (default: HGB_ACM HGB_DBLP HGB_IMDB HNE_PubMed)")
    p.add_argument("--metapath", default=None,
                   help="Override metapath string (applied to all datasets). "
                        "Default: auto-select densest per dataset.")
    p.add_argument("--k-fixed", type=int, default=32,
                   help="Fixed k budget for L-sweep plots and tables (default: 32)")
    p.add_argument("--l-fixed", type=int, default=2,
                   help="Fixed depth for k-sweep plots and stacked bar (default: 2)")
    p.add_argument("--depth", nargs="+", type=int, default=[1, 2, 3, 4],
                   help="Depth values for L-sweep plots (default: 1 2 3 4)")
    p.add_argument("--cka-trajectory-l", type=int, default=4,
                   help="Model depth to use for plot_cka_per_layer (default: 4)")
    p.add_argument("--plot", nargs="+", default=_ALL_PLOTS, metavar="PLOT",
                   help=f"Plots to generate (default: all). Choices: {_ALL_PLOTS}")
    p.add_argument("--table", nargs="+", default=_ALL_TABLES, metavar="TABLE",
                   help=f"Tables to generate (default: all). Choices: {_ALL_TABLES}")
    return p.parse_args()


def main():
    args = _parse()
    _style()

    transfer_dir = Path(args.transfer_dir)
    out_dir      = Path(args.out_dir)

    # Discover which datasets actually have data
    seed_dirs = sorted(
        d for d in transfer_dir.iterdir()
        if d.is_dir() and re.match(r"results_\d+", d.name)
    ) if transfer_dir.exists() else []

    datasets = [
        d for d in args.datasets
        if any((sd / d / "master_results.csv").exists() for sd in seed_dirs)
    ]
    missing = [d for d in args.datasets if d not in datasets]
    if missing:
        warnings.warn(f"No results found for: {missing}; skipping")
    if not datasets:
        print(f"[ERROR] No data found under {transfer_dir} for {args.datasets}")
        sys.exit(1)

    print(f"Loading multi-seed CSVs for: {datasets}")
    df = _load_multiseed(transfer_dir, datasets)
    seeds = sorted(df["Seed"].dropna().unique().tolist())
    print(f"  {len(df)} total rows — seeds: {seeds}")

    if len(seeds) == 1:
        print(f"[WARN] Only 1 seed ({int(seeds[0])}) — no variance bands. "
              "Add more results_<seed>/ dirs for 5-seed protocol.")

    print("\n  Auto-detected densest metapaths:")
    for ds in datasets:
        try:
            mp = _pick_densest_mp(df, ds, L_ref=args.l_fixed)
            n_edges_rows = df[(df["Dataset"] == ds) & (df["MetaPath"] == mp)
                              & (df["Method"] == "Exact") & (df["L"] == args.l_fixed)]
            edge_count, _ = _ms(n_edges_rows, "Edge_Count")
            print(f"    {_label(ds)}: {mp}  ({int(edge_count):,} edges)" if not np.isnan(edge_count)
                  else f"    {_label(ds)}: {mp}")
        except ValueError:
            pass

    plots  = set(args.plot)
    tables = set(args.table)

    # Tables
    if "table1" in tables:
        table1(df, datasets, out_dir, L_fixed=args.l_fixed, k_fixed=args.k_fixed)
    if "table2" in tables:
        table2_integrity(df, datasets, out_dir, L_fixed=args.l_fixed, k_fixed=args.k_fixed)
    if "app_a" in tables:
        appendix_table_a(df, datasets, out_dir, L_fixed=args.l_fixed, k_fixed=args.k_fixed)
    if "master" in tables:
        k_vals = sorted(df["k_value"].dropna().unique().tolist())
        table_master_numbers(df, datasets, out_dir, depths=args.depth, k_values=k_vals)

    # Systems plots
    if "plot1a" in plots:
        plot1a(df, datasets, out_dir, args.metapath, args.l_fixed, args.k_fixed)
    if "plot1b" in plots:
        plot1b(df, datasets, out_dir, args.metapath, args.l_fixed, args.k_fixed)
    if "plot1c" in plots:
        plot1c(df, datasets, out_dir, args.metapath, args.k_fixed)

    # Pareto
    if "plot2" in plots:
        plot2(df, datasets, out_dir, args.metapath, args.l_fixed)

    # L-sweep (standalone Dirichlet + 3-panel suite)
    if "plot3" in plots:
        plot3(df, datasets, out_dir, args.metapath, args.k_fixed, args.depth)
    if "plot_l_sweep_suite" in plots:
        plot_l_sweep_suite(df, datasets, out_dir, args.metapath, args.k_fixed, args.depth)

    # k-sweep (3-panel suite)
    if "plot_k_sweep_suite" in plots:
        plot_k_sweep_suite(df, datasets, out_dir, args.metapath, args.l_fixed)

    # Per-layer CKA trajectory
    if "plot_cka_per_layer" in plots:
        plot_cka_per_layer(df, datasets, out_dir, args.metapath,
                           args.k_fixed, L_for_trajectory=args.cka_trajectory_l)

    print(f"\nDone. {len(plots)} plots + {len(tables)} tables -> {out_dir}/")


if __name__ == "__main__":
    main()
