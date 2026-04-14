"""
exp6_is_vs_us.py — Test the "MPRW ≈ importance sampling, KMV ≈ uniform sampling" hypothesis.

Consumes z-files already written by exp3_inference.py (no C++ re-runs).

Part (a) — per-class F1 on labeled test nodes.
    If MPRW is importance-sampling in disguise, it should over-perform on classes
    whose members are hub-ish and under-perform on classes whose members are tail-ish.
    KMV is uniform → should be flatter across classes.

Part (b) — Prediction-Agreement-vs-Exact per degree bin (Tail/Mid/Hub).
    Uses *all* test nodes (not just labeled ones) → 12,103 samples, 50× more than (a).
    If MPRW is hub-biased and KMV is uniform, MPRW PA-vs-Exact should be higher on Hub
    and collapse on Tail; KMV should be flat.

Both analyses done at density-matched (KMV k_j ↔ MPRW w_j) pairs picked from CSV.

Outputs
-------
    figures/exp6/per_class_f1_k{k}.pdf
    figures/exp6/degree_bin_pa_k{k}.pdf
    results/HNE_PubMed/exp6_summary.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

_ROOT = Path(__file__).resolve().parent.parent

_KMV_COLOR  = "#2CA02C"
_MPRW_COLOR = "#D62728"


# ─────────────────────────────────────────────────────────────────────────────
# File I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_base(inf_dir: Path):
    labels = torch.load(inf_dir / "labels.pt", weights_only=False).numpy()
    mask = torch.load(inf_dir / "mask.pt", weights_only=False).numpy()
    z_exact = torch.load(inf_dir / "z_exact_L2.pt", weights_only=False).numpy()
    return labels, mask, z_exact


def _load_preds(inf_dir: Path, method: str, value: int, seed: int, L: int = 2) -> np.ndarray:
    """Load logits and return argmax predictions [n_total]."""
    if method == "kmv":
        fname = f"z_kmv_{value}_s{seed}_L{L}.pt"
    elif method == "mprw":
        fname = f"z_mprw_w{value}_s{seed}_L{L}.pt"
    else:
        raise ValueError(method)
    z = torch.load(inf_dir / fname, weights_only=False).numpy()
    return z.argmax(axis=1)


def _compute_degrees(adj_path: Path, n_total: int, target_offset: int, n_target: int) -> np.ndarray:
    """Parse C++ adjacency list → per-target-node degree (no self-loops)."""
    degrees = np.zeros(n_target, dtype=np.int64)
    with open(adj_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u_global = int(parts[0])
            u_local = u_global - target_offset
            if 0 <= u_local < n_target:
                # Count neighbours excluding self
                neighbours = [v for v in parts[1:] if int(v) - target_offset != u_local]
                degrees[u_local] = len(neighbours)
    return degrees


# ─────────────────────────────────────────────────────────────────────────────
# Density matching
# ─────────────────────────────────────────────────────────────────────────────

def _density_match(csv_path: Path, metapath: str, L: int = 2):
    """
    For each KMV k, find MPRW w (both at seed 0) whose edge count is closest to
    that KMV run's edge count.

    Returns dict: {k_value: (matched_w, kmv_edges, mprw_edges_at_matched_w)}
    """
    df = pd.read_csv(csv_path)
    df = df[(df["MetaPath"] == metapath) & (df["L"] == L) & (df["Seed"] == 0)]
    kmv = df[df["Method"] == "KMV"][["k_value", "Edge_Count"]].dropna()
    mprw = df[df["Method"] == "MPRW"][["w_value", "Edge_Count"]].dropna()

    matches = {}
    for _, row in kmv.iterrows():
        k = int(row["k_value"])
        target = int(row["Edge_Count"])
        best_row = mprw.iloc[(mprw["Edge_Count"] - target).abs().argsort()[:1]]
        w = int(best_row["w_value"].values[0])
        w_edges = int(best_row["Edge_Count"].values[0])
        matches[k] = (w, target, w_edges)
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Part (a): per-class F1
# ─────────────────────────────────────────────────────────────────────────────

def _per_class_f1(
    inf_dir: Path,
    labels: np.ndarray,
    mask: np.ndarray,
    k: int,
    matched_w: int,
    n_seeds: int,
    log: logging.Logger,
) -> pd.DataFrame:
    """
    For each class, compute per-seed F1 for KMV(k) and MPRW(matched_w).
    Returns long-form DataFrame: [class, method, seed, f1].
    """
    # Restrict to labeled test nodes
    labeled_mask = mask & (labels >= 0)
    y_true = labels[labeled_mask]
    classes = sorted(set(y_true.tolist()))
    n_classes = int(labels.max() + 1)  # logits have n_classes outputs

    rows = []
    for seed in range(n_seeds):
        for method, value in [("kmv", k), ("mprw", matched_w)]:
            try:
                preds_full = _load_preds(inf_dir, method, value, seed)
            except FileNotFoundError:
                log.warning("Missing %s value=%d seed=%d", method, value, seed)
                continue
            y_pred = preds_full[labeled_mask]
            # per-class F1 (one-vs-rest)
            per_class = f1_score(
                y_true, y_pred,
                labels=list(range(n_classes)),
                average=None,
                zero_division=0,
            )
            for cls in classes:
                rows.append({
                    "class": cls,
                    "method": method.upper(),
                    "seed": seed,
                    "f1": per_class[cls],
                })
    return pd.DataFrame(rows)


def _plot_per_class_f1(df: pd.DataFrame, out_path: Path, k: int, matched_w: int):
    """Grouped bar chart: per-class F1 mean ± SD, KMV vs MPRW."""
    agg = df.groupby(["class", "method"]).agg(mean=("f1", "mean"),
                                                std=("f1", "std")).reset_index()
    classes = sorted(agg["class"].unique())
    x = np.arange(len(classes))
    width = 0.36

    kmv = agg[agg["method"] == "KMV"].set_index("class").loc[classes]
    mprw = agg[agg["method"] == "MPRW"].set_index("class").loc[classes]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width/2, kmv["mean"], width, yerr=kmv["std"], capsize=3,
           color=_KMV_COLOR, label=f"KMV (k={k})")
    ax.bar(x + width/2, mprw["mean"], width, yerr=mprw["std"], capsize=3,
           color=_MPRW_COLOR, label=f"MPRW (w={matched_w}, density-matched)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{c}" for c in classes])
    ax.set_xlabel("Class (HNE_PubMed disease categories)")
    ax.set_ylabel("F1 score (per-class, mean ± SD over seeds)")
    ax.set_title(f"Per-class F1 — KMV k={k} vs MPRW w={matched_w}  (labeled test n=226)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Part (b): PA-vs-Exact per degree bin
# ─────────────────────────────────────────────────────────────────────────────

def _pa_vs_exact_per_bin(
    inf_dir: Path,
    z_exact: np.ndarray,
    mask: np.ndarray,
    degrees: np.ndarray,
    k: int,
    matched_w: int,
    n_seeds: int,
    log: logging.Logger,
) -> pd.DataFrame:
    """
    For each degree bin × method × seed, compute PA-vs-Exact on test nodes.
    Returns long-form DataFrame: [bin, method, seed, pa, n_nodes_in_bin].
    """
    preds_exact_full = z_exact.argmax(axis=1)
    test_idx = np.where(mask)[0]
    test_degrees = degrees[test_idx]

    p50 = np.percentile(test_degrees, 50)
    p90 = np.percentile(test_degrees, 90)
    bins = {
        "Tail (0–50%)":  np.where(test_degrees <= p50)[0],
        "Mid (50–90%)":  np.where((test_degrees > p50) & (test_degrees <= p90))[0],
        "Hub (top 10%)": np.where(test_degrees > p90)[0],
    }
    log.info("Degree percentiles on test: p50=%.1f p90=%.1f", p50, p90)
    for bname, bidx in bins.items():
        log.info("  %s: n=%d, deg range=[%d, %d]", bname, len(bidx),
                 int(test_degrees[bidx].min()) if len(bidx) else 0,
                 int(test_degrees[bidx].max()) if len(bidx) else 0)

    rows = []
    for seed in range(n_seeds):
        for method, value in [("kmv", k), ("mprw", matched_w)]:
            try:
                preds_full = _load_preds(inf_dir, method, value, seed)
            except FileNotFoundError:
                log.warning("Missing %s value=%d seed=%d", method, value, seed)
                continue
            preds_test = preds_full[test_idx]
            preds_exact_test = preds_exact_full[test_idx]
            agree = (preds_test == preds_exact_test).astype(float)
            for bname, bidx in bins.items():
                if len(bidx) == 0:
                    continue
                pa = float(agree[bidx].mean())
                rows.append({
                    "bin": bname,
                    "method": method.upper(),
                    "seed": seed,
                    "pa": pa,
                    "n_nodes": len(bidx),
                })
    return pd.DataFrame(rows)


def _plot_degree_bins(df: pd.DataFrame, out_path: Path, k: int, matched_w: int):
    """Grouped bar chart: mean PA-vs-Exact per degree bin, KMV vs MPRW."""
    agg = df.groupby(["bin", "method"]).agg(
        mean=("pa", "mean"),
        std=("pa", "std"),
        n_seeds=("pa", "count"),
    ).reset_index()
    agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["n_seeds"].clip(lower=1))

    bin_order = ["Tail (0–50%)", "Mid (50–90%)", "Hub (top 10%)"]
    x = np.arange(len(bin_order))
    width = 0.36

    kmv = agg[agg["method"] == "KMV"].set_index("bin").reindex(bin_order)
    mprw = agg[agg["method"] == "MPRW"].set_index("bin").reindex(bin_order)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, kmv["mean"], width, yerr=kmv["ci"], capsize=5,
           color=_KMV_COLOR, label=f"KMV (k={k})")
    ax.bar(x + width/2, mprw["mean"], width, yerr=mprw["ci"], capsize=5,
           color=_MPRW_COLOR, label=f"MPRW (w={matched_w}, density-matched)")

    for i, (k_mu, m_mu) in enumerate(zip(kmv["mean"], mprw["mean"])):
        ax.text(i - width/2, k_mu + 0.01, f"{k_mu:.3f}",
                ha="center", va="bottom", fontsize=9, color="#1a6e1a", fontweight="bold")
        ax.text(i + width/2, m_mu + 0.01, f"{m_mu:.3f}",
                ha="center", va="bottom", fontsize=9, color="#8b1a1a", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_order)
    ax.set_xlabel("Degree bin (in H_exact, test nodes)")
    ax.set_ylabel("Prediction Agreement vs Exact (mean ± 95% CI)")
    ax.set_title(f"Degree-stratified fidelity — KMV k={k} vs MPRW w={matched_w}\n"
                 f"(all 12,103 test nodes; label-free metric)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HNE_PubMed")
    p.add_argument("--metapath", default="disease_to_chemical,chemical_to_disease")
    p.add_argument("--adj-file", default=None,
                   help="Path to exact adjacency file. Auto-derived if omitted.")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--target-offset", type=int, default=None,
                   help="Global-ID offset for target type. HNE_PubMed disease=0.")
    p.add_argument("--n-target", type=int, default=None,
                   help="Count of target-type nodes (full graph). HNE_PubMed disease=20163.")
    p.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128])
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("exp6")

    results_dir = _ROOT / "results" / args.dataset
    mp_safe = args.metapath.replace(",", "_")
    inf_dir = results_dir / "inf_scratch" / mp_safe
    csv_path = results_dir / "master_results.csv"

    if args.adj_file is None:
        args.adj_file = str(results_dir / f"exact_{mp_safe}.adj")

    # Base data
    labels, mask, z_exact = _load_base(inf_dir)
    n_total = int(labels.shape[0])
    log.info("Loaded base: n_total=%d  labels-valid=%d  test-mask=%d",
             n_total, int((labels >= 0).sum()), int(mask.sum()))

    # Target geometry (for degree parsing). HNE_PubMed: disease first alphabetically → offset=0.
    if args.target_offset is None:
        args.target_offset = 0
        log.info("Assuming target_offset=0 (HNE_PubMed disease is first alphabetically).")
    if args.n_target is None:
        args.n_target = n_total  # target-type-only dataset: n_target == n_total
        log.info("Assuming n_target=n_total=%d.", args.n_target)

    # Degrees from exact adjacency
    log.info("Computing degrees from %s ...", args.adj_file)
    degrees = _compute_degrees(
        Path(args.adj_file),
        n_total=n_total,
        target_offset=args.target_offset,
        n_target=args.n_target,
    )
    log.info("  mean deg=%.1f  max=%d  non-zero=%d/%d",
             degrees.mean(), degrees.max(), int((degrees > 0).sum()), args.n_target)

    # Density matching
    log.info("Density-matching MPRW w to each KMV k ...")
    matches = _density_match(csv_path, args.metapath, L=2)
    for k_val, (w_match, k_e, w_e) in matches.items():
        log.info("  k=%d (edges=%d) ↔ w=%d (edges=%d) %+.1f%%",
                 k_val, k_e, w_match, w_e, 100 * (w_e - k_e) / max(k_e, 1))

    # Analyses — one pair per k in args.ks
    summary_rows = []
    for k in args.ks:
        if k not in matches:
            log.warning("k=%d not in CSV, skipping", k)
            continue
        matched_w = matches[k][0]
        log.info("=== Analysing k=%d vs w=%d ===", k, matched_w)

        # Part (a)
        df_f1 = _per_class_f1(inf_dir, labels, mask, k, matched_w, args.n_seeds, log)
        out_f1 = _ROOT / "figures" / "exp6" / f"per_class_f1_k{k}.pdf"
        _plot_per_class_f1(df_f1, out_f1, k, matched_w)
        log.info("  per-class F1 → %s", out_f1)

        # Part (b)
        df_pa = _pa_vs_exact_per_bin(
            inf_dir, z_exact, mask, degrees, k, matched_w, args.n_seeds, log,
        )
        out_pa = _ROOT / "figures" / "exp6" / f"degree_bin_pa_k{k}.pdf"
        _plot_degree_bins(df_pa, out_pa, k, matched_w)
        log.info("  degree bin PA → %s", out_pa)

        # Summary numbers
        for method in ["KMV", "MPRW"]:
            f1_sub = df_f1[df_f1["method"] == method]
            class_means = f1_sub.groupby("seed")["f1"].mean()
            macro_f1_mean = float(class_means.mean())
            macro_f1_std = float(class_means.std())
            for bname in ["Tail (0–50%)", "Mid (50–90%)", "Hub (top 10%)"]:
                pa_sub = df_pa[(df_pa["method"] == method) & (df_pa["bin"] == bname)]
                if pa_sub.empty:
                    continue
                summary_rows.append({
                    "k": k,
                    "matched_w": matched_w,
                    "method": method,
                    "bin": bname,
                    "pa_mean": float(pa_sub["pa"].mean()),
                    "pa_std": float(pa_sub["pa"].std()),
                    "macro_f1_mean": macro_f1_mean,
                    "macro_f1_std": macro_f1_std,
                    "n_seeds": int(pa_sub["seed"].nunique()),
                })

    summary_df = pd.DataFrame(summary_rows)
    out_csv = results_dir / "exp6_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    log.info("Summary → %s", out_csv)

    # Console table for quick eyeballing
    print("\n" + "="*80)
    print("EXP6 SUMMARY — degree-bin PA-vs-Exact at each density-matched (k, w)")
    print("="*80)
    for k in args.ks:
        if k not in matches:
            continue
        w = matches[k][0]
        print(f"\n  KMV k={k}  vs  MPRW w={w}")
        sub = summary_df[(summary_df["k"] == k)]
        for _, r in sub.iterrows():
            print(f"    [{r['method']:4s}] {r['bin']:16s}  "
                  f"PA={r['pa_mean']:.4f} ± {r['pa_std']:.4f}  "
                  f"MacroF1={r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}")


if __name__ == "__main__":
    _main()
