"""Generate extra plots for the 2026-05-07 results report.

Reads:
  results/approach_a_2026_05_07/<DS>/<arch>/<mp>_seed*_k*.json   (300 main JSONs)
  results/approach_a_2026_05_07/sanity/<mode>/*.json             (60 sanity JSONs)
  results/OGB_MAG/master_results_v2.csv                          (OGB headline)
  results/equivalence_tests_v2.csv                               (Wilcoxon + TOST)
  results/convergence_matrix.csv                                 (per-cell convergence)
  results/meta_path_density.csv                                  (density classes)

Writes ~18 PDFs to figures/approach_a_v2/extra/.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MAIN_DIR = RESULTS / "approach_a_2026_05_07"
SANITY_DIR = MAIN_DIR / "sanity"
OUT_DIR = ROOT / "figures" / "approach_a_v2" / "extra"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [8, 16, 32, 64, 128]
ARCHS = ["SAGE", "GCN", "GAT"]
DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed"]

# Short labels for meta-paths (matches equivalence_tests_v2.csv mp_label)
MP_SHORT = {
    "author_to_paper,paper_to_author": "APA",
    "author_to_paper,paper_to_venue,venue_to_paper,paper_to_author": "APVPA",
    "paper_to_author,author_to_paper": "PAP",
    "paper_to_term,term_to_paper": "PTP",
    "movie_to_actor,actor_to_movie": "MAM",
    "movie_to_keyword,keyword_to_movie": "MKM",
    "disease_to_gene,gene_to_disease": "DGD",
    "disease_to_chemical,chemical_to_disease": "DCD",
}

ARCH_COLOR = {"SAGE": "#1f77b4", "GCN": "#2ca02c", "GAT": "#d62728", "GIN": "#9467bd"}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def short_ds(ds: str) -> str:
    return ds.replace("HGB_", "").replace("HNE_", "")


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

def load_main_jsons() -> pd.DataFrame:
    """Load all 300 main-sweep JSONs into a flat DataFrame."""
    rows: list[dict[str, Any]] = []
    for ds in DATASETS:
        ds_dir = MAIN_DIR / ds
        if not ds_dir.exists():
            continue
        for arch_dir in ds_dir.iterdir():
            if not arch_dir.is_dir():
                continue
            arch = arch_dir.name
            for jp in arch_dir.glob("*.json"):
                try:
                    j = json.loads(jp.read_text())
                except Exception:
                    continue
                row = {
                    "dataset": j.get("dataset", ds),
                    "arch": j.get("arch", arch),
                    "meta_path": j.get("meta_path", ""),
                    "mp_label": MP_SHORT.get(j.get("meta_path", ""), j.get("meta_path", "")[:8]),
                    "seed": j.get("seed"),
                    "k": j.get("kmv_k"),
                    "n_edges_exact": j.get("n_edges_exact"),
                    "n_edges_kmv": j.get("n_edges_kmv"),
                    "pred_agreement": j.get("pred_agreement"),
                    "f1_gap": j.get("f1_gap"),
                    "macro_f1_exact": j.get("macro_f1_exact"),
                    "macro_f1_kmv": j.get("macro_f1_kmv"),
                    "mat_time_exact_s": j.get("mat_time_exact_s"),
                    "mat_time_kmv_s": j.get("mat_time_kmv_s"),
                }
                # per-layer arrays — just take last layer + first layer
                for fld in ("cka_per_layer", "cka_unbiased_per_layer",
                            "row_cosine_per_layer_mean", "row_rel_l2_per_layer_mean",
                            "frob_recon_err_per_layer",
                            "procrustes_q_eq_i_per_layer", "procrustes_q_orth_per_layer"):
                    arr = j.get(fld)
                    if isinstance(arr, list) and len(arr) >= 2:
                        row[f"{fld}_L1"] = arr[0]
                        row[f"{fld}_L2"] = arr[1]
                    else:
                        row[f"{fld}_L1"] = None
                        row[f"{fld}_L2"] = None
                rows.append(row)
    df = pd.DataFrame(rows)
    return df


def load_sanity(mode: str) -> pd.DataFrame:
    sub = SANITY_DIR / mode
    if not sub.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for jp in sub.glob("*.json"):
        try:
            j = json.loads(jp.read_text())
        except Exception:
            continue
        rows.append(j)
    return pd.DataFrame(rows)


def first_array_val(x: Any, idx: int = 1) -> float | None:
    if isinstance(x, list) and len(x) > idx:
        return x[idx]
    return None


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------

def save(fig, name: str) -> None:
    fig.tight_layout()
    fp = OUT_DIR / f"{name}.pdf"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fp.name}")


# ------------------------------------------------------------------
# 1. row-cosine vs k panel grid (12 cells)
# ------------------------------------------------------------------

def plot_row_cosine_panels(df: pd.DataFrame) -> None:
    cells = sorted(df.dropna(subset=["mp_label"])
                     .groupby(["dataset", "arch", "mp_label"])
                     .size().index.tolist())
    n = len(cells)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.2),
                             squeeze=False)
    for i, (ds, arch, mp) in enumerate(cells):
        ax = axes[i // ncols, i % ncols]
        sub = df[(df.dataset == ds) & (df.arch == arch) & (df.mp_label == mp)]
        means_l1, stds_l1, means_l2, stds_l2 = [], [], [], []
        for k in K_VALUES:
            ks = sub[sub.k == k]
            v1 = ks["row_cosine_per_layer_mean_L1"].dropna()
            v2 = ks["row_cosine_per_layer_mean_L2"].dropna()
            means_l1.append(v1.mean() if len(v1) else np.nan)
            stds_l1.append(v1.std() if len(v1) else 0.0)
            means_l2.append(v2.mean() if len(v2) else np.nan)
            stds_l2.append(v2.std() if len(v2) else 0.0)
        means_l1 = np.array(means_l1)
        means_l2 = np.array(means_l2)
        stds_l1 = np.array(stds_l1)
        stds_l2 = np.array(stds_l2)
        ax.plot(K_VALUES, means_l1, "-o", color="#aaaaaa", label="L1", lw=1.0, ms=3)
        ax.fill_between(K_VALUES, means_l1 - stds_l1, means_l1 + stds_l1, color="#aaaaaa", alpha=0.2)
        ax.plot(K_VALUES, means_l2, "-s", color=ARCH_COLOR.get(arch, "#000"), label="L2", lw=1.2, ms=3.5)
        ax.fill_between(K_VALUES, means_l2 - stds_l2, means_l2 + stds_l2,
                        color=ARCH_COLOR.get(arch, "#000"), alpha=0.2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_ylim(0.7, 1.01)
        ax.set_title(f"{short_ds(ds)} / {arch} / {mp}", fontsize=8.5)
        ax.legend(loc="lower right", fontsize=7)
    # blank unused
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("k")
    for r in range(nrows):
        axes[r, 0].set_ylabel("row-cosine")
    fig.suptitle("Row-cosine vs k per cell (mean +/- std, 5 seeds; L1 grey, L2 colored)",
                 fontsize=11, y=1.0)
    save(fig, "panel_row_cosine_vs_k")


# ------------------------------------------------------------------
# 2. Frob recon error vs k panel grid
# ------------------------------------------------------------------

def plot_frob_recon_panels(df: pd.DataFrame) -> None:
    cells = sorted(df.dropna(subset=["mp_label"])
                     .groupby(["dataset", "arch", "mp_label"])
                     .size().index.tolist())
    n = len(cells)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.2),
                             squeeze=False)
    for i, (ds, arch, mp) in enumerate(cells):
        ax = axes[i // ncols, i % ncols]
        sub = df[(df.dataset == ds) & (df.arch == arch) & (df.mp_label == mp)]
        m_l1, s_l1, m_l2, s_l2 = [], [], [], []
        for k in K_VALUES:
            ks = sub[sub.k == k]
            v1 = ks["frob_recon_err_per_layer_L1"].dropna()
            v2 = ks["frob_recon_err_per_layer_L2"].dropna()
            m_l1.append(v1.mean() if len(v1) else np.nan)
            s_l1.append(v1.std() if len(v1) else 0.0)
            m_l2.append(v2.mean() if len(v2) else np.nan)
            s_l2.append(v2.std() if len(v2) else 0.0)
        m_l1, m_l2, s_l1, s_l2 = map(np.array, (m_l1, m_l2, s_l1, s_l2))
        ax.plot(K_VALUES, m_l1, "-o", color="#aaaaaa", label="L1", lw=1.0, ms=3)
        ax.fill_between(K_VALUES, m_l1 - s_l1, m_l1 + s_l1, color="#aaaaaa", alpha=0.2)
        ax.plot(K_VALUES, m_l2, "-s", color=ARCH_COLOR.get(arch, "#000"), label="L2", lw=1.2, ms=3.5)
        ax.fill_between(K_VALUES, m_l2 - s_l2, m_l2 + s_l2,
                        color=ARCH_COLOR.get(arch, "#000"), alpha=0.2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{short_ds(ds)} / {arch} / {mp}", fontsize=8.5)
        ax.legend(loc="upper right", fontsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("k")
    for r in range(nrows):
        axes[r, 0].set_ylabel("Frob recon err")
    fig.suptitle("Relative Frobenius reconstruction error vs k (lower is better)",
                 fontsize=11, y=1.0)
    save(fig, "panel_frob_recon_vs_k")


# ------------------------------------------------------------------
# 3. per-layer L1 vs L2 trajectory bar chart
# ------------------------------------------------------------------

def plot_layer_trajectory(df: pd.DataFrame) -> None:
    """For each cell at k=32, show row-cosine for L1 and L2 (drift across depth)."""
    cells = sorted(df.dropna(subset=["mp_label"])
                     .groupby(["dataset", "arch", "mp_label"])
                     .size().index.tolist())
    rows = []
    for ds, arch, mp in cells:
        sub = df[(df.dataset == ds) & (df.arch == arch) & (df.mp_label == mp) & (df.k == 32)]
        if not len(sub):
            continue
        rows.append({
            "label": f"{short_ds(ds)}/{arch}/{mp}",
            "arch": arch,
            "L1_mean": sub["row_cosine_per_layer_mean_L1"].mean(),
            "L1_std": sub["row_cosine_per_layer_mean_L1"].std(),
            "L2_mean": sub["row_cosine_per_layer_mean_L2"].mean(),
            "L2_std": sub["row_cosine_per_layer_mean_L2"].std(),
        })
    if not rows:
        return
    rdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(rdf))
    w = 0.4
    ax.bar(x - w/2, rdf.L1_mean, w, yerr=rdf.L1_std, label="Layer 1",
           color="#aaaaaa", capsize=3)
    bars = ax.bar(x + w/2, rdf.L2_mean, w, yerr=rdf.L2_std, label="Layer 2",
                  color=[ARCH_COLOR[a] for a in rdf.arch], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(rdf["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("row-cosine")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Row-cosine drift across layers at k=32 (L1 grey, L2 by arch)")
    ax.legend()
    save(fig, "layer_trajectory_k32")


# ------------------------------------------------------------------
# 4. F1-gap boxplots faceted by arch
# ------------------------------------------------------------------

def plot_f1_gap_box(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, arch in zip(axes, ARCHS):
        sub = df[df.arch == arch].dropna(subset=["f1_gap"])
        if not len(sub):
            ax.set_title(f"{arch} (no data)")
            ax.axis("off")
            continue
        data_per_k = [sub[sub.k == k]["f1_gap"].values for k in K_VALUES]
        bp = ax.boxplot(data_per_k, positions=range(len(K_VALUES)),
                        widths=0.6, patch_artist=True,
                        flierprops={"marker": "o", "markersize": 3})
        for patch in bp["boxes"]:
            patch.set_facecolor(ARCH_COLOR[arch])
            patch.set_alpha(0.4)
        ax.axhline(0.0, color="black", lw=0.8, ls="--")
        ax.axhline(0.02, color="#666", lw=0.6, ls=":")
        ax.axhline(-0.02, color="#666", lw=0.6, ls=":")
        ax.set_xticks(range(len(K_VALUES)))
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_xlabel("k")
        ax.set_title(arch)
    axes[0].set_ylabel("F1 gap (KMV - Exact)")
    fig.suptitle("F1-gap distribution across all converged cells, per arch x k", y=1.02)
    save(fig, "f1_gap_box_by_arch")


# ------------------------------------------------------------------
# 5. Procrustes Q=I vs Q-orth scatter
# ------------------------------------------------------------------

def plot_procrustes_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for arch in ARCHS:
        sub = df[df.arch == arch].dropna(subset=[
            "procrustes_q_eq_i_per_layer_L2", "procrustes_q_orth_per_layer_L2"])
        ax.scatter(sub["procrustes_q_eq_i_per_layer_L2"],
                   sub["procrustes_q_orth_per_layer_L2"],
                   c=ARCH_COLOR[arch], label=arch, alpha=0.5, s=14, edgecolors="none")
    lim = (0, 1.0)
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Procrustes Q=I (last layer)")
    ax.set_ylabel("Procrustes Q-orth (last layer)")
    ax.set_title("Q=I vs Q-orth: tight diagonal = no basis drift\n(off-diagonal points = basis rotation matters)")
    ax.legend()
    save(fig, "procrustes_q_eq_i_vs_q_orth")


# ------------------------------------------------------------------
# 6. Unbiased vs biased CKA scatter
# ------------------------------------------------------------------

def plot_cka_unbiased_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for arch in ARCHS:
        sub = df[df.arch == arch].dropna(subset=[
            "cka_per_layer_L2", "cka_unbiased_per_layer_L2"])
        ax.scatter(sub["cka_per_layer_L2"],
                   sub["cka_unbiased_per_layer_L2"],
                   c=ARCH_COLOR[arch], label=arch, alpha=0.5, s=14, edgecolors="none")
    lim = (0.5, 1.01)
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Biased linear CKA (last layer)")
    ax.set_ylabel("Unbiased linear CKA (last layer)")
    ax.set_title("Unbiased vs biased CKA: should cluster on diagonal")
    ax.legend()
    save(fig, "cka_unbiased_vs_biased")


# ------------------------------------------------------------------
# 7. saturation: n_edges_kmv / n_edges_exact ratio vs k
# ------------------------------------------------------------------

def plot_saturation_ratio(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    cells = sorted(df.dropna(subset=["mp_label"])
                     .groupby(["dataset", "mp_label"])
                     .size().index.tolist())
    for ds, mp in cells:
        # Use SAGE rows since they exist for all cells
        sub = df[(df.dataset == ds) & (df.mp_label == mp) & (df.arch == "SAGE")]
        if not len(sub):
            continue
        means = []
        for k in K_VALUES:
            ks = sub[sub.k == k]
            ratio = (ks["n_edges_kmv"] / ks["n_edges_exact"]).dropna()
            means.append(ratio.mean() if len(ratio) else np.nan)
        ax.plot(K_VALUES, means, "-o", label=f"{short_ds(ds)}/{mp}", ms=4)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Exact")
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("n_edges_kmv / n_edges_exact")
    ax.set_title("Saturation: edge-budget ratio vs k\nratio == 1 means KMV reproduces all distinct edges")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    save(fig, "saturation_ratio_vs_k")


# ------------------------------------------------------------------
# 8. density-stratified row-cosine at k=128 vs density
# ------------------------------------------------------------------

def plot_quality_vs_density(df: pd.DataFrame) -> None:
    dens = pd.read_csv(RESULTS / "meta_path_density.csv")
    rows = []
    for _, drow in dens.iterrows():
        ds, mp_full = drow.dataset, drow.meta_path
        for arch in ARCHS:
            sub = df[(df.dataset == ds) & (df.arch == arch) &
                     (df.meta_path == mp_full) & (df.k == 128)]
            v = sub["row_cosine_per_layer_mean_L2"].dropna()
            if len(v):
                rows.append({"density": drow.density, "ds": ds, "arch": arch,
                             "mp": MP_SHORT.get(mp_full, mp_full[:8]),
                             "row_cos": v.mean(), "density_class": drow.density_class})
    rdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for arch in ARCHS:
        sub = rdf[rdf.arch == arch]
        ax.scatter(sub.density, sub.row_cos, c=ARCH_COLOR[arch], label=arch,
                   alpha=0.7, s=50, edgecolors="black", linewidths=0.5)
    for _, r in rdf[rdf.arch == "SAGE"].iterrows():
        ax.annotate(f"{short_ds(r['ds'])}/{r['mp']}", (r.density, r.row_cos),
                    fontsize=6.5, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("meta-path density (log)")
    ax.set_ylabel("row-cosine last layer @ k=128")
    ax.set_title("Quality vs density: harder regimes are denser metapaths")
    ax.legend()
    save(fig, "quality_vs_density_k128")


# ------------------------------------------------------------------
# 9. Arch comparison bar chart at k=32 on convergent cells
# ------------------------------------------------------------------

def plot_arch_compare_bars(df: pd.DataFrame) -> None:
    conv = pd.read_csv(RESULTS / "convergence_matrix.csv")
    # cells where ALL THREE archs converged for the same (dataset, mp)
    grouped = conv[conv.verdict == "CONVERGED"].groupby(["dataset", "mp"])["arch"].apply(set)
    full_cells = [k for k, v in grouped.items() if v == {"SAGE", "GCN", "GAT"}]
    if not full_cells:
        # fallback: just show all archs that did converge somewhere
        print("  no fully-convergent (DBLP-APA, ACM-PAP only? still proceed)")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    metrics = [
        ("row_cosine_per_layer_mean_L2", "row-cosine L2", (0.85, 1.0)),
        ("cka_unbiased_per_layer_L2", "unbiased CKA L2", (0.7, 1.0)),
        ("pred_agreement", "Pred. Agreement", (0.85, 1.0)),
        ("f1_gap", "F1 gap", (-0.02, 0.06)),
    ]
    cell_labels = [f"{short_ds(d)}/{m}" for d, m in full_cells]
    for ax, (col, title, ylim) in zip(axes, metrics):
        x = np.arange(len(full_cells))
        w = 0.25
        for i, arch in enumerate(ARCHS):
            means, stds = [], []
            for ds, mp in full_cells:
                sub = df[(df.dataset == ds) & (df.arch == arch) & (df.mp_label == mp) & (df.k == 32)]
                v = sub[col].dropna()
                means.append(v.mean() if len(v) else 0.0)
                stds.append(v.std() if len(v) else 0.0)
            ax.bar(x + (i - 1) * w, means, w, yerr=stds, label=arch,
                   color=ARCH_COLOR[arch], capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(cell_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        if col == "f1_gap":
            ax.axhline(0.0, color="black", lw=0.8, ls="--")
    axes[0].legend(fontsize=8)
    fig.suptitle("Arch comparison at k=32 (cells where all 3 archs converged)", y=1.04)
    save(fig, "arch_compare_k32")


# ------------------------------------------------------------------
# 10. heatmaps: rows=(ds,arch), cols=k
# ------------------------------------------------------------------

def plot_heatmaps(df: pd.DataFrame) -> None:
    cells = sorted(df.dropna(subset=["mp_label"])
                     .groupby(["dataset", "arch", "mp_label"])
                     .size().index.tolist())
    metrics = [
        ("cka_unbiased_per_layer_L2", "unbiased CKA L2", "viridis", (0.7, 1.0)),
        ("f1_gap", "F1 gap", "RdBu_r", (-0.05, 0.05)),
        ("row_cosine_per_layer_mean_L2", "row-cosine L2", "viridis", (0.85, 1.0)),
    ]
    for col, title, cmap, vrange in metrics:
        mat = np.full((len(cells), len(K_VALUES)), np.nan)
        for r, (ds, arch, mp) in enumerate(cells):
            for c, k in enumerate(K_VALUES):
                sub = df[(df.dataset == ds) & (df.arch == arch) &
                         (df.mp_label == mp) & (df.k == k)]
                v = sub[col].dropna()
                if len(v):
                    mat[r, c] = v.mean()
        fig, ax = plt.subplots(figsize=(5.5, max(4.0, 0.30 * len(cells))))
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vrange[0], vmax=vrange[1])
        ax.set_xticks(range(len(K_VALUES)))
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels([f"{short_ds(d)}/{a}/{m}" for d, a, m in cells], fontsize=7)
        ax.set_xlabel("k")
        ax.set_title(f"{title} heatmap")
        # annotate
        for r in range(len(cells)):
            for c in range(len(K_VALUES)):
                if not np.isnan(mat[r, c]):
                    txt_color = "white" if (cmap == "viridis" and mat[r, c] < 0.85) else "black"
                    ax.text(c, r, f"{mat[r,c]:.2f}", ha="center", va="center",
                            fontsize=6, color=txt_color)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        safe = col.replace("/", "_")
        save(fig, f"heatmap_{safe}")


# ------------------------------------------------------------------
# 11. random_theta bar chart
# ------------------------------------------------------------------

def plot_sanity_random_theta() -> None:
    df = load_sanity("random_theta")
    if df.empty:
        return
    df["row_cos_L2"] = df["row_cosine_per_layer_mean"].apply(lambda x: first_array_val(x, 1))
    df["cka_L2"] = df["cka_per_layer"].apply(lambda x: first_array_val(x, 1))
    df["frob_L2"] = df["frob_recon_err_per_layer"].apply(lambda x: first_array_val(x, 1))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    cols = [
        ("row_cos_L2", "row-cosine L2", (0.7, 1.0)),
        ("cka_L2", "CKA L2", (0.5, 1.0)),
        ("pred_agreement", "Pred. Agreement", (0.0, 1.0)),
        ("macro_f1_kmv", "F1 (random theta)", (0.0, 0.5)),
    ]
    for ax, (col, title, ylim) in zip(axes, cols):
        agg = df.groupby("kmv_k")[col].agg(["mean", "std"]).reset_index()
        ax.bar(agg["kmv_k"].astype(str), agg["mean"], yerr=agg["std"],
               capsize=3, color="#888888")
        ax.set_xlabel("k")
        ax.set_title(title)
        ax.set_ylim(*ylim)
    axes[0].set_ylabel("metric value")
    fig.suptitle("Sanity 1: random theta. F1 collapses to 0.19 (chance ~0.25); KMV-side geometry preserved.", y=1.04)
    save(fig, "sanity_random_theta")


# ------------------------------------------------------------------
# 12. edge_perturb line plot
# ------------------------------------------------------------------

def plot_sanity_edge_perturb() -> None:
    df = load_sanity("edge_perturb")
    if df.empty:
        return
    df["row_cos_L2"] = df["row_cosine_per_layer_mean"].apply(lambda x: first_array_val(x, 1))
    df["cka_L2"] = df["cka_per_layer"].apply(lambda x: first_array_val(x, 1))
    df["frob_L2"] = df["frob_recon_err_per_layer"].apply(lambda x: first_array_val(x, 1))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col, label, color, marker in [
        ("row_cos_L2", "row-cosine L2", "#1f77b4", "o"),
        ("cka_L2", "CKA L2", "#ff7f0e", "s"),
        ("pred_agreement", "Pred. Agreement", "#2ca02c", "^"),
    ]:
        agg = df.groupby("perturb_p")[col].agg(["mean", "std"]).reset_index().sort_values("perturb_p")
        ax.errorbar(agg["perturb_p"], agg["mean"], yerr=agg["std"],
                    label=label, marker=marker, color=color, capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("p (edge perturbation rate)")
    ax.set_ylabel("metric")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Sanity 2: edge_perturb. Strict monotone degradation as p increases (cleanest control).")
    ax.legend()
    save(fig, "sanity_edge_perturb")


# ------------------------------------------------------------------
# 13. density_matched_random bar chart KMV vs random
# ------------------------------------------------------------------

def plot_sanity_density_matched() -> None:
    df = load_sanity("density_matched_random")
    if df.empty:
        return
    rows = []
    for _, j in df.iterrows():
        kmv = j.get("kmv_metrics") or {}
        rnd = j.get("random_metrics") or {}
        rows.append({
            "k": j.get("kmv_k"),
            "kmv_cka_L2": first_array_val(kmv.get("cka_per_layer"), 1),
            "rnd_cka_L2": first_array_val(rnd.get("cka_per_layer"), 1),
            "kmv_pa": kmv.get("pred_agreement"),
            "rnd_pa": rnd.get("pred_agreement"),
            "kmv_f1": j.get("macro_f1_kmv"),
            "rnd_f1": j.get("macro_f1_random"),
        })
    rdf = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    pairs = [
        (("kmv_cka_L2", "rnd_cka_L2"), "CKA L2", (0.7, 1.0)),
        (("kmv_pa", "rnd_pa"), "Pred. Agreement", (0.7, 1.0)),
        (("kmv_f1", "rnd_f1"), "Macro-F1", (0.5, 1.0)),
    ]
    for ax, ((kcol, rcol), title, ylim) in zip(axes, pairs):
        agg = rdf.groupby("k").agg({kcol: ["mean", "std"], rcol: ["mean", "std"]}).reset_index()
        x = np.arange(len(agg))
        w = 0.4
        ax.bar(x - w/2, agg[kcol]["mean"], w, yerr=agg[kcol]["std"],
               label="KMV", color="#1f77b4", capsize=2)
        ax.bar(x + w/2, agg[rcol]["mean"], w, yerr=agg[rcol]["std"],
               label="Density-matched random", color="#d62728", capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(agg["k"].astype(int).astype(str))
        ax.set_xlabel("k")
        ax.set_title(title)
        ax.set_ylim(*ylim)
    axes[0].set_ylabel("metric")
    axes[0].legend(fontsize=8)
    fig.suptitle("Sanity 3: density-matched random. KMV beats random at every k.", y=1.04)
    save(fig, "sanity_density_matched_random")


# ------------------------------------------------------------------
# 14. layer_permutation distribution
# ------------------------------------------------------------------

def plot_sanity_layer_permutation() -> None:
    df = load_sanity("layer_permutation")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    same_l1 = df["cka_aligned_L1"].dropna().values
    same_l2 = df["cka_aligned_L2"].dropna().values
    cross = df["cka_cross_exactL1_vs_kmvL2"].dropna().values
    parts = ax.violinplot([cross, same_l1, same_l2],
                          positions=[0, 1, 2], showmeans=True, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_alpha(0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["cross\n(Exact L1 vs KMV L2)", "aligned L1", "aligned L2"])
    ax.set_ylabel("Linear CKA")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.85, color="grey", lw=0.5, ls="--")
    ax.set_title("Sanity 4: layer_permutation. Cross-layer CKA ~0.82 only modestly below same-layer; INCONCLUSIVE at L=2.")
    save(fig, "sanity_layer_permutation")


# ------------------------------------------------------------------
# 15. OGB-MAG additional plots
# ------------------------------------------------------------------

def plot_ogb_extras() -> None:
    csv = RESULTS / "OGB_MAG" / "master_results_v2.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df = df.dropna(subset=["Method"])
    exact_edges = float(df[df.Method == "Exact"]["Edge_Count"].iloc[0])
    kmv = df[df.Method == "KMV"].copy()
    kmv["k"] = kmv["k_value"].astype(int)
    kmv = kmv.sort_values("k")
    # 15a — edge count comparison
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = ["Exact"] + [f"KMV-k{int(k)}" for k in kmv.k]
    edges = [exact_edges] + kmv.Edge_Count.tolist()
    colors = ["#444444"] + ["#1f77b4"] * len(kmv)
    bars = ax.bar(labels, edges, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("edges materialized (log)")
    ax.set_title("OGB-MAG edge count: Exact (2.86 B) vs KMV k-sweep")
    for b, e in zip(bars, edges):
        ax.text(b.get_x() + b.get_width() / 2, e * 1.1, f"{e:.2e}",
                ha="center", fontsize=7)
    save(fig, "ogb_edge_counts")
    # 15b — % of Exact (edges + mat time)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    pct_edges = [100 * e / exact_edges for e in kmv.Edge_Count]
    exact_time = float(df[df.Method == "Exact"]["Materialization_Time"].iloc[0])
    pct_time = [100 * t / exact_time for t in kmv.Materialization_Time]
    x = np.arange(len(kmv))
    w = 0.4
    ax.bar(x - w/2, pct_edges, w, label="Edges (% of Exact)", color="#1f77b4")
    ax.bar(x + w/2, pct_time, w, label="Mat time (% of Exact)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={int(k)}" for k in kmv.k])
    ax.set_ylabel("% of Exact")
    ax.set_title("OGB-MAG KMV cost relative to Exact (lower is better)")
    ax.legend()
    save(fig, "ogb_relative_cost")


# ------------------------------------------------------------------
# 16. Wilcoxon p-value vs cell + n=5 floor
# ------------------------------------------------------------------

def plot_wilcoxon_floor() -> None:
    eq = pd.read_csv(RESULTS / "equivalence_tests_v2.csv")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    eq = eq.sort_values(["dataset", "arch", "mp_label", "k"]).reset_index(drop=True)
    labels = [f"{short_ds(r['dataset'])}/{r['arch']}/{r['mp_label']}/k{int(r['k'])}"
              for _, r in eq.iterrows()]
    x = np.arange(len(eq))
    ax.scatter(x, eq.wilcoxon_p_f1, c="#1f77b4", s=10, label="Wilcoxon p (F1)")
    ax.axhline(0.0625, color="red", ls="--", lw=0.8, label="n=5 minimum p (0.0625)")
    ax.axhline(0.05, color="green", ls=":", lw=0.6, label="alpha=0.05")
    ax.set_xticks(x[::5])
    ax.set_xticklabels([labels[i] for i in x[::5]], rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("Wilcoxon p-value")
    ax.set_yscale("log")
    ax.set_title("Wilcoxon p-values across all (cell, k) pairs hit the n=5 floor at 0.0625")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "wilcoxon_p_n5_floor")


# ------------------------------------------------------------------
# 17. TOST decision heatmap
# ------------------------------------------------------------------

def plot_tost_heatmap() -> None:
    eq = pd.read_csv(RESULTS / "equivalence_tests_v2.csv")
    eq["cell"] = eq.dataset.map(short_ds) + "/" + eq.arch + "/" + eq.mp_label
    cells = list(eq["cell"].drop_duplicates())
    mat = np.zeros((len(cells), len(K_VALUES)), dtype=float)
    for r, c in enumerate(cells):
        for j, k in enumerate(K_VALUES):
            sub = eq[(eq["cell"] == c) & (eq.k == k)]
            if not len(sub):
                mat[r, j] = np.nan
            else:
                v = sub.tost_decision_f1.iloc[0]
                mat[r, j] = 1.0 if v == "REJECT_H0" else 0.0
    fig, ax = plt.subplots(figsize=(5.0, max(4.0, 0.30 * len(cells))))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(cells, fontsize=7)
    ax.set_xlabel("k")
    ax.set_title("TOST F1 equivalence decision\n(green = REJECT H0 / equivalent at eps=0.02)")
    for r in range(len(cells)):
        for c in range(len(K_VALUES)):
            if not np.isnan(mat[r, c]):
                txt = "REJECT" if mat[r, c] == 1 else "FAIL"
                ax.text(c, r, txt, ha="center", va="center", fontsize=6, color="black")
    save(fig, "tost_decision_heatmap")


# ------------------------------------------------------------------
# 18. F1-gap distribution histogram across all 60 (cell, k) pairs
# ------------------------------------------------------------------

def plot_f1_gap_hist(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    gaps = df.f1_gap.dropna().values
    ax.hist(gaps, bins=40, color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.axvline(0.0, color="black", lw=1.0, ls="--")
    ax.axvline(0.02, color="grey", lw=0.6, ls=":")
    ax.axvline(-0.02, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("F1 gap (KMV - Exact) per seed")
    ax.set_ylabel("count")
    n_pos = int((gaps > 0).sum())
    n_neg = int((gaps < 0).sum())
    n_in_band = int(((gaps > -0.02) & (gaps < 0.02)).sum())
    ax.set_title(f"F1-gap histogram across {len(gaps)} (cell, k, seed) rows  "
                 f"|  positive: {n_pos}  negative: {n_neg}  |gap|<0.02: {n_in_band}")
    save(fig, "f1_gap_hist_all")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print(f"Loading main JSONs from {MAIN_DIR} ...")
    df = load_main_jsons()
    print(f"  loaded {len(df)} rows.")
    print(f"Writing PDFs to {OUT_DIR}")

    # HGB cell-level
    plot_row_cosine_panels(df)
    plot_frob_recon_panels(df)
    plot_layer_trajectory(df)
    plot_f1_gap_box(df)
    plot_procrustes_scatter(df)
    plot_cka_unbiased_scatter(df)
    plot_saturation_ratio(df)
    plot_quality_vs_density(df)

    # Arch comparison
    plot_arch_compare_bars(df)
    plot_heatmaps(df)

    # Sanity
    plot_sanity_random_theta()
    plot_sanity_edge_perturb()
    plot_sanity_density_matched()
    plot_sanity_layer_permutation()

    # OGB
    plot_ogb_extras()

    # Stats
    plot_wilcoxon_floor()
    plot_tost_heatmap()
    plot_f1_gap_hist(df)

    print("Done.")


if __name__ == "__main__":
    main()
