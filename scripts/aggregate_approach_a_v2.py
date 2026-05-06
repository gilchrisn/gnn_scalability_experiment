"""Aggregate Approach A v2 fidelity results.

Spec: final_report/research_notes/SPEC_approach_a_2026_05_07.md (sections 4, 6, 9, 10).

Reads:
    results/approach_a_2026_05_07/<dataset>/<arch>/*.json
Optionally:
    results/approach_a_2026_05_07/sanity/<mode>/*.json

Writes:
    results/master_table_approach_a_v2.md
    results/equivalence_tests_v2.csv
    figures/approach_a_v2/cka_vs_k.pdf
    figures/approach_a_v2/row_cosine_vs_k.pdf
    figures/approach_a_v2/f1_gap_distribution.pdf
    figures/approach_a_v2/sanity_controls.pdf      (only if --include-sanity)
    figures/approach_a_v2/wallclock_vs_density.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed", "OGB_MAG"]
DS_LABELS = {
    "HGB_DBLP": "DBLP",
    "HGB_ACM": "ACM",
    "HGB_IMDB": "IMDB",
    "HNE_PubMed": "PubMed",
    "OGB_MAG": "OGB-MAG",
}
ARCHS = ["SAGE", "GCN", "GAT"]
ARCH_COLOR = {"SAGE": "#2CA02C", "GCN": "#1F77B4", "GAT": "#9467BD"}

MP_LABELS = {
    "author_to_paper_paper_to_author": "APA",
    "author_to_paper_paper_to_venue_venue_to_paper_paper_to_author": "APVPA",
    "paper_to_author_author_to_paper": "PAP",
    "paper_to_term_term_to_paper": "PTP",
    "movie_to_actor_actor_to_movie": "MAM",
    "movie_to_keyword_keyword_to_movie": "MKM",
    "disease_to_gene_gene_to_disease": "DGD",
    "disease_to_chemical_chemical_to_disease": "DCD",
}

K_VALUES = [8, 16, 32, 64, 128]

EPS_F1 = 0.02
EPS_PA = 0.05
TOST_ALPHA = 0.05
MIN_SEEDS_FOR_TESTS = 5

CI95_Z = 1.959963984540054  # not used directly; we use t-distribution

# Filename pattern: <mp_safe>_seed<seed>_k<k>.json
JSON_RE = re.compile(r"^(?P<mp>.+)_seed(?P<seed>\d+)_k(?P<k>\d+)\.json$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def parse_filename(name: str) -> Optional[Tuple[str, int, int]]:
    m = JSON_RE.match(name)
    if not m:
        return None
    return m.group("mp"), int(m.group("seed")), int(m.group("k"))


def load_main_jsons(input_dir: Path) -> pd.DataFrame:
    """Walk results/approach_a_2026_05_07/<ds>/<arch>/*.json -> long DataFrame.

    Excludes sanity/ subtree. One row per JSON.
    """
    rows: List[Dict[str, Any]] = []
    if not input_dir.exists():
        return pd.DataFrame(rows)
    for ds_dir in sorted(input_dir.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name == "sanity":
            continue
        ds = ds_dir.name
        for arch_dir in sorted(ds_dir.iterdir()):
            if not arch_dir.is_dir():
                continue
            arch = arch_dir.name
            for jp in sorted(arch_dir.glob("*.json")):
                parsed = parse_filename(jp.name)
                if parsed is None:
                    continue
                mp_safe, seed, k = parsed
                try:
                    obj = json.loads(jp.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"[load] skip {jp}: {exc}")
                    continue
                row: Dict[str, Any] = {
                    "dataset": ds,
                    "arch": arch,
                    "mp_safe": mp_safe,
                    "mp_label": MP_LABELS.get(mp_safe, mp_safe),
                    "seed": seed,
                    "k": k,
                    "path": str(jp),
                }
                # Pull the per-layer last entries; v2 schema uses arrays.
                row["row_cos_last"] = _last(obj.get("row_cosine_per_layer_mean"))
                row["row_rel_l2_last"] = _last(obj.get("row_rel_l2_per_layer_mean"))
                row["frob_recon_last"] = _last(obj.get("frob_recon_err_per_layer"))
                row["unbiased_cka_last"] = _last(obj.get("cka_unbiased_per_layer"))
                row["procrustes_q_eq_i_last"] = _last(
                    obj.get("procrustes_q_eq_i_per_layer")
                )
                row["procrustes_q_orth_last"] = _last(
                    obj.get("procrustes_q_orth_per_layer")
                )
                row["pred_agreement"] = obj.get("pred_agreement")
                row["macro_f1_exact"] = obj.get("macro_f1_exact")
                row["macro_f1_kmv"] = obj.get("macro_f1_kmv")
                row["macro_f1_gap"] = obj.get("macro_f1_gap")
                row["micro_f1_exact"] = obj.get("micro_f1_exact")
                row["micro_f1_kmv"] = obj.get("micro_f1_kmv")
                row["micro_f1_gap"] = obj.get("micro_f1_gap")
                row["inference_time_exact_s"] = obj.get("inference_time_exact_s")
                row["inference_time_kmv_s"] = obj.get("inference_time_kmv_s")
                row["mat_time_exact_s"] = obj.get("mat_time_exact_s")
                row["mat_time_kmv_s"] = obj.get("mat_time_kmv_s")
                row["n_edges_exact"] = obj.get("n_edges_exact")
                row["n_edges_kmv"] = obj.get("n_edges_kmv")
                row["arch_status"] = obj.get("arch_status", "OK")
                rows.append(row)
    return pd.DataFrame(rows)


def load_sanity_jsons(input_dir: Path) -> pd.DataFrame:
    """Walk results/approach_a_2026_05_07/sanity/<mode>/*.json -> long DataFrame."""
    rows: List[Dict[str, Any]] = []
    sroot = input_dir / "sanity"
    if not sroot.exists():
        return pd.DataFrame(rows)
    for mode_dir in sorted(sroot.iterdir()):
        if not mode_dir.is_dir():
            continue
        mode = mode_dir.name
        for jp in sorted(mode_dir.glob("*.json")):
            try:
                obj = json.loads(jp.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[sanity] skip {jp}: {exc}")
                continue
            row = {
                "mode": mode,
                "path": str(jp),
                "k": obj.get("kmv_k"),
                "seed": obj.get("seed"),
                "perturb_p": obj.get("perturb_p"),
                "row_cos_last": _last(obj.get("row_cosine_per_layer_mean")),
                "unbiased_cka_last": _last(obj.get("cka_unbiased_per_layer")),
                "macro_f1_gap": obj.get("macro_f1_gap"),
                "pred_agreement": obj.get("pred_agreement"),
                "is_kmv": obj.get("is_kmv", True),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _last(v: Any) -> Optional[float]:
    if isinstance(v, list) and v:
        try:
            return float(v[-1])
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mean_ci95(arr: Iterable[float]) -> Tuple[Optional[float], Optional[float], int]:
    """Return (mean, half-width of 95% CI using t-distribution, n)."""
    a = np.asarray([x for x in arr if x is not None and not _is_nan(x)], dtype=float)
    n = len(a)
    if n == 0:
        return None, None, 0
    if n == 1:
        return float(a[0]), 0.0, 1
    mu = float(a.mean())
    sd = float(a.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mu, tcrit * se, n


def _is_nan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def wilcoxon_safe(x: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Paired Wilcoxon vs zero. Returns (statistic, p-value) or (None, None)."""
    a = np.asarray([v for v in x if v is not None and not _is_nan(v)], dtype=float)
    if len(a) < MIN_SEEDS_FOR_TESTS:
        return None, None
    if np.allclose(a, 0.0):
        return 0.0, 1.0
    try:
        stat, p = stats.wilcoxon(a, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(p)
    except ValueError:
        return None, None


def tost_paired(x: np.ndarray, eps: float, alpha: float = TOST_ALPHA) -> Dict[str, Any]:
    """Two One-Sided t-Tests for equivalence of x to 0 within margin eps.

    H0_lower: mean(x) <= -eps   (one-sided, alternative='greater')
    H0_upper: mean(x) >=  eps   (one-sided, alternative='less')
    Reject both at alpha -> equivalence.

    Schuirmann 1987. Returns dict with the two p-values, the larger one, and the decision.
    """
    a = np.asarray([v for v in x if v is not None and not _is_nan(v)], dtype=float)
    if len(a) < MIN_SEEDS_FOR_TESTS:
        return {"p_lower": None, "p_upper": None, "p_max": None, "decision": "INSUFFICIENT_SEEDS"}
    n = len(a)
    mu = a.mean()
    sd = a.std(ddof=1)
    if sd == 0.0:
        within = abs(mu) < eps
        return {
            "p_lower": 0.0 if within else 1.0,
            "p_upper": 0.0 if within else 1.0,
            "p_max": 0.0 if within else 1.0,
            "decision": "REJECT_H0" if within else "FAIL",
        }
    se = sd / math.sqrt(n)
    df = n - 1
    t_lower = (mu - (-eps)) / se
    t_upper = (mu - eps) / se
    p_lower = 1.0 - float(stats.t.cdf(t_lower, df=df))
    p_upper = float(stats.t.cdf(t_upper, df=df))
    p_max = max(p_lower, p_upper)
    decision = "REJECT_H0" if p_max < alpha else "FAIL"
    return {"p_lower": p_lower, "p_upper": p_upper, "p_max": p_max, "decision": decision}


def cohen_dz(x: np.ndarray) -> Optional[float]:
    a = np.asarray([v for v in x if v is not None and not _is_nan(v)], dtype=float)
    if len(a) < 2:
        return None
    sd = a.std(ddof=1)
    if sd == 0.0:
        return 0.0
    return float(a.mean() / sd)


# ---------------------------------------------------------------------------
# Master table
# ---------------------------------------------------------------------------


def fmt_mean_ci(mu: Optional[float], hw: Optional[float], digits: int = 3) -> str:
    if mu is None:
        return "---"
    if hw is None:
        return f"{mu:.{digits}f}"
    return f"{mu:.{digits}f} ± {hw:.{digits}f}"


def write_master_table(df: pd.DataFrame, eq_df: pd.DataFrame, out_path: Path) -> None:
    """One row per (dataset, arch, mp, k) cell."""
    if df.empty:
        out_path.write_text("# Approach A v2 — no data\n", encoding="utf-8")
        print(f"[md] {out_path} (empty)")
        return

    lines: List[str] = [
        "# Approach A v2 — Master Results",
        "",
        "Train SAGE/GCN/GAT on H_exact, freeze theta*, infer on Exact + KMV. "
        "Per-row cosine and rel-L2 are headline; unbiased CKA, F1 gap, PA support. "
        f"5+ seeds per cell. Equivalence margin: F1 epsilon={EPS_F1}, PA epsilon={EPS_PA}.",
        "",
        "| Dataset | Arch | MP | k | n_seeds | row_cos_last | row_rel_l2_last | frob_recon_last | "
        "unbiased_CKA_last | PA | Macro-F1 gap | Micro-F1 gap | Wilcoxon p (F1) | TOST decision (F1) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    eq_lookup: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    if not eq_df.empty:
        for _, r in eq_df.iterrows():
            key = (r["dataset"], r["arch"], r["mp_label"], int(r["k"]))
            eq_lookup[key] = r.to_dict()

    grouped = df.groupby(["dataset", "arch", "mp_label", "k"], dropna=False)
    for (ds, arch, mp_label, k), g in grouped:
        row_cos = mean_ci95(g["row_cos_last"])
        row_l2 = mean_ci95(g["row_rel_l2_last"])
        frob = mean_ci95(g["frob_recon_last"])
        cka = mean_ci95(g["unbiased_cka_last"])
        pa = mean_ci95(g["pred_agreement"])
        f1_gap = mean_ci95(g["macro_f1_gap"])
        micro_gap = mean_ci95(g["micro_f1_gap"])
        n_seeds = len(g)

        eq_row = eq_lookup.get((ds, arch, mp_label, int(k)), {})
        wp = eq_row.get("wilcoxon_p_f1")
        tost = eq_row.get("tost_decision_f1", "")
        wp_str = "---" if wp is None or _is_nan(wp) else f"{wp:.3g}"

        lines.append(
            "| {ds} | {arch} | {mp} | {k} | {n} | {rcos} | {rl2} | {frob} | "
            "{cka} | {pa} | {f1g} | {mig} | {wp} | {tost} |".format(
                ds=DS_LABELS.get(ds, ds),
                arch=arch,
                mp=mp_label,
                k=k,
                n=n_seeds,
                rcos=fmt_mean_ci(*row_cos[:2]),
                rl2=fmt_mean_ci(*row_l2[:2]),
                frob=fmt_mean_ci(*frob[:2]),
                cka=fmt_mean_ci(*cka[:2]),
                pa=fmt_mean_ci(*pa[:2]),
                f1g=fmt_mean_ci(*f1_gap[:2]),
                mig=fmt_mean_ci(*micro_gap[:2]),
                wp=wp_str,
                tost=tost or "---",
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[md] {out_path}")


# ---------------------------------------------------------------------------
# Equivalence tests CSV
# ---------------------------------------------------------------------------


def build_equivalence_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, arch, mp_label, k) with Wilcoxon + TOST on F1 and PA gaps."""
    if df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for (ds, arch, mp_label, k), g in df.groupby(
        ["dataset", "arch", "mp_label", "k"], dropna=False
    ):
        f1_arr = np.asarray(g["macro_f1_gap"].dropna().values, dtype=float)
        pa_kmv = g["pred_agreement"].dropna().values.astype(float)
        # PA-gap = pred_agreement - 1 (exact-vs-exact would be 1.0)
        pa_arr = pa_kmv - 1.0 if len(pa_kmv) > 0 else np.array([])
        n = len(g)

        f1_mean, f1_hw, _ = mean_ci95(f1_arr)
        f1_median = float(np.median(f1_arr)) if len(f1_arr) else None
        f1_dz = cohen_dz(f1_arr)
        f1_w_stat, f1_w_p = wilcoxon_safe(f1_arr)
        f1_tost = tost_paired(f1_arr, EPS_F1)

        pa_mean, pa_hw, _ = mean_ci95(pa_arr)
        pa_median = float(np.median(pa_arr)) if len(pa_arr) else None
        pa_dz = cohen_dz(pa_arr)
        pa_w_stat, pa_w_p = wilcoxon_safe(pa_arr)
        pa_tost = tost_paired(pa_arr, EPS_PA)

        rows.append(
            {
                "dataset": ds,
                "arch": arch,
                "mp_label": mp_label,
                "k": int(k),
                "n_seeds": n,
                "f1_gap_mean": f1_mean,
                "f1_gap_ci95_hw": f1_hw,
                "f1_gap_median": f1_median,
                "f1_gap_dz": f1_dz,
                "wilcoxon_stat_f1": f1_w_stat,
                "wilcoxon_p_f1": f1_w_p,
                "tost_p_lower_f1": f1_tost["p_lower"],
                "tost_p_upper_f1": f1_tost["p_upper"],
                "tost_p_max_f1": f1_tost["p_max"],
                "tost_decision_f1": f1_tost["decision"],
                "pa_gap_mean": pa_mean,
                "pa_gap_ci95_hw": pa_hw,
                "pa_gap_median": pa_median,
                "pa_gap_dz": pa_dz,
                "wilcoxon_stat_pa": pa_w_stat,
                "wilcoxon_p_pa": pa_w_p,
                "tost_p_lower_pa": pa_tost["p_lower"],
                "tost_p_upper_pa": pa_tost["p_upper"],
                "tost_p_max_pa": pa_tost["p_max"],
                "tost_decision_pa": pa_tost["decision"],
                "eps_f1": EPS_F1,
                "eps_pa": EPS_PA,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _agg_by_k(g: pd.DataFrame, col: str) -> Tuple[List[int], List[float], List[float]]:
    ks: List[int] = []
    means: List[float] = []
    hws: List[float] = []
    for k, gg in g.groupby("k"):
        mu, hw, n = mean_ci95(gg[col])
        if mu is None:
            continue
        ks.append(int(k))
        means.append(mu)
        hws.append(hw if hw is not None else 0.0)
    order = np.argsort(ks)
    ks_sorted = [ks[i] for i in order]
    means_sorted = [means[i] for i in order]
    hws_sorted = [hws[i] for i in order]
    return ks_sorted, means_sorted, hws_sorted


def plot_cka_vs_k(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    ds_arch_pairs = sorted({(d, a) for d, a in zip(df["dataset"], df["arch"])})
    if not ds_arch_pairs:
        return
    ncols = min(4, len(ds_arch_pairs))
    nrows = math.ceil(len(ds_arch_pairs) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows), squeeze=False, sharey=True
    )
    for idx, (ds, arch) in enumerate(ds_arch_pairs):
        r, c = idx // ncols, idx % ncols
        ax = axes[r][c]
        sub = df[(df["dataset"] == ds) & (df["arch"] == arch)]
        for mp_label, gmp in sub.groupby("mp_label"):
            ks, means, hws = _agg_by_k(gmp, "unbiased_cka_last")
            if not ks:
                continue
            ax.errorbar(ks, means, yerr=hws, marker="o", capsize=3, label=str(mp_label))
        ax.axhline(1.0, linestyle="--", color="black", alpha=0.4, linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_xlabel("k (KMV budget)")
        if c == 0:
            ax.set_ylabel("Unbiased CKA (last layer)")
        ax.set_title(f"{DS_LABELS.get(ds, ds)} / {arch}", fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    for idx in range(len(ds_arch_pairs), nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r][c].axis("off")
    fig.suptitle("Approach A v2: unbiased CKA vs k (Exact baseline = 1.0)", fontsize=12)
    fig.tight_layout()
    out = out_dir / "cka_vs_k.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def plot_row_cosine_vs_k(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    ds_arch_pairs = sorted({(d, a) for d, a in zip(df["dataset"], df["arch"])})
    ncols = min(4, len(ds_arch_pairs))
    nrows = math.ceil(len(ds_arch_pairs) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows), squeeze=False, sharey=True
    )
    for idx, (ds, arch) in enumerate(ds_arch_pairs):
        r, c = idx // ncols, idx % ncols
        ax = axes[r][c]
        sub = df[(df["dataset"] == ds) & (df["arch"] == arch)]
        for mp_label, gmp in sub.groupby("mp_label"):
            ks, means, hws = _agg_by_k(gmp, "row_cos_last")
            if not ks:
                continue
            ax.errorbar(ks, means, yerr=hws, marker="s", capsize=3, label=str(mp_label))
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_xlabel("k (KMV budget)")
        if c == 0:
            ax.set_ylabel("Mean per-row cosine (last layer)")
        ax.set_title(f"{DS_LABELS.get(ds, ds)} / {arch}", fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    for idx in range(len(ds_arch_pairs), nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r][c].axis("off")
    fig.suptitle(
        "Approach A v2: per-row cosine vs k (mean ± 95% CI across seeds)", fontsize=12
    )
    fig.tight_layout()
    out = out_dir / "row_cosine_vs_k.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def plot_f1_gap_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    archs_present = [a for a in ARCHS if a in set(df["arch"].unique())]
    if not archs_present:
        return
    ncols = len(archs_present)
    fig, axes = plt.subplots(
        1, ncols, figsize=(4.5 * ncols, 4.2), squeeze=False, sharey=True
    )
    for ai, arch in enumerate(archs_present):
        ax = axes[0][ai]
        sub = df[df["arch"] == arch].dropna(subset=["macro_f1_gap"])
        data_per_k: List[List[float]] = []
        labels: List[str] = []
        for k in K_VALUES:
            vals = sub[sub["k"] == k]["macro_f1_gap"].astype(float).tolist()
            if vals:
                data_per_k.append(vals)
                labels.append(str(k))
        if not data_per_k:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.boxplot(data_per_k, labels=labels, showmeans=True)
        ax.axhline(0, linestyle="-", color="black", alpha=0.5, linewidth=0.7)
        ax.axhline(EPS_F1, linestyle="--", color="red", alpha=0.4, linewidth=1)
        ax.axhline(-EPS_F1, linestyle="--", color="red", alpha=0.4, linewidth=1)
        ax.set_xlabel("k (KMV budget)")
        if ai == 0:
            ax.set_ylabel("Macro-F1 gap (KMV - Exact)")
        ax.set_title(arch, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"Approach A v2: F1 gap distribution per k (red dashes = ±{EPS_F1})", fontsize=12
    )
    fig.tight_layout()
    out = out_dir / "f1_gap_distribution.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def plot_sanity_controls(sanity_df: pd.DataFrame, out_dir: Path) -> None:
    if sanity_df.empty:
        print("[fig] sanity_controls.pdf: no sanity data, skipping")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), squeeze=False)

    # (a) random_theta floor
    ax = axes[0][0]
    rt = sanity_df[sanity_df["mode"] == "random_theta"]
    if not rt.empty:
        ks, means, hws = _agg_by_k(rt, "unbiased_cka_last")
        if ks:
            ax.errorbar(ks, means, yerr=hws, marker="o", capsize=3, label="random theta")
        ks, means, hws = _agg_by_k(rt, "row_cos_last")
        if ks:
            ax.errorbar(ks, means, yerr=hws, marker="s", capsize=3, label="row cos (random)")
    ax.axhline(0.0, linestyle="--", color="grey", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("k")
    ax.set_ylabel("Metric value")
    ax.set_title("(a) random-theta floor", fontsize=10)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (b) edge_perturb monotone degradation
    ax = axes[0][1]
    ep = sanity_df[sanity_df["mode"] == "edge_perturb"].dropna(subset=["perturb_p"])
    if not ep.empty:
        ep = ep.copy()
        ep["perturb_p"] = ep["perturb_p"].astype(float)
        for metric, mlabel, marker in [
            ("unbiased_cka_last", "CKA", "o"),
            ("row_cos_last", "row cos", "s"),
        ]:
            xs: List[float] = []
            ys: List[float] = []
            es: List[float] = []
            for p, gp in ep.groupby("perturb_p"):
                mu, hw, n = mean_ci95(gp[metric])
                if mu is None:
                    continue
                xs.append(float(p))
                ys.append(mu)
                es.append(hw if hw is not None else 0.0)
            if xs:
                order = np.argsort(xs)
                xs_s = [xs[i] for i in order]
                ys_s = [ys[i] for i in order]
                es_s = [es[i] for i in order]
                ax.errorbar(xs_s, ys_s, yerr=es_s, marker=marker, capsize=3, label=mlabel)
    ax.set_xlabel("Perturb fraction p")
    ax.set_ylabel("Metric value")
    ax.set_title("(b) edge-perturb monotone degradation", fontsize=10)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (c) KMV vs density-matched random
    ax = axes[0][2]
    dm = sanity_df[sanity_df["mode"] == "density_matched_random"]
    if not dm.empty:
        dm_kmv = dm[dm["is_kmv"] == True]
        dm_rand = dm[dm["is_kmv"] == False]
        for sub, lbl, marker in [(dm_kmv, "KMV", "o"), (dm_rand, "random", "x")]:
            if sub.empty:
                continue
            ks, means, hws = _agg_by_k(sub, "row_cos_last")
            if ks:
                ax.errorbar(ks, means, yerr=hws, marker=marker, capsize=3, label=lbl)
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("k (matched edge count)")
    ax.set_ylabel("Mean per-row cosine")
    ax.set_title("(c) KMV vs density-matched random", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Approach A v2: sanity controls", fontsize=12)
    fig.tight_layout()
    out = out_dir / "sanity_controls.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def plot_wallclock_vs_density(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), squeeze=False)

    # Mat time vs edges
    ax = axes[0][0]
    sub_e = df.dropna(subset=["n_edges_exact", "mat_time_exact_s"])
    sub_k = df.dropna(subset=["n_edges_kmv", "mat_time_kmv_s"])
    if not sub_e.empty:
        ax.scatter(
            sub_e["n_edges_exact"],
            sub_e["mat_time_exact_s"],
            c="#1f77b4",
            alpha=0.6,
            s=22,
            label="Exact",
        )
    if not sub_k.empty:
        ax.scatter(
            sub_k["n_edges_kmv"],
            sub_k["mat_time_kmv_s"],
            c="#2ca02c",
            alpha=0.6,
            s=22,
            label="KMV",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Edge count")
    ax.set_ylabel("Materialisation time (s)")
    ax.set_title("Materialisation wall-clock vs density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # Inf time vs edges
    ax = axes[0][1]
    sub_e = df.dropna(subset=["n_edges_exact", "inference_time_exact_s"])
    sub_k = df.dropna(subset=["n_edges_kmv", "inference_time_kmv_s"])
    if not sub_e.empty:
        ax.scatter(
            sub_e["n_edges_exact"],
            sub_e["inference_time_exact_s"],
            c="#1f77b4",
            alpha=0.6,
            s=22,
            label="Exact",
        )
    if not sub_k.empty:
        ax.scatter(
            sub_k["n_edges_kmv"],
            sub_k["inference_time_kmv_s"],
            c="#2ca02c",
            alpha=0.6,
            s=22,
            label="KMV",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Edge count")
    ax.set_ylabel("Inference time (s)")
    ax.set_title("Inference wall-clock vs density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("Approach A v2: wall-clock vs edge count", fontsize=12)
    fig.tight_layout()
    out = out_dir / "wallclock_vs_density.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/approach_a_2026_05_07"),
        help="Directory containing v2 JSON outputs.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/approach_a_v2"),
        help="Directory for figure PDFs.",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Where master_table_approach_a_v2.md and equivalence_tests_v2.csv land.",
    )
    ap.add_argument(
        "--include-sanity",
        action="store_true",
        help="Render figures/approach_a_v2/sanity_controls.pdf if sanity JSONs exist.",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading from {args.input_dir}")
    df = load_main_jsons(args.input_dir)
    print(f"[load] main JSONs: {len(df)} rows")

    eq_df = build_equivalence_table(df)
    eq_path = args.results_dir / "equivalence_tests_v2.csv"
    eq_df.to_csv(eq_path, index=False)
    print(f"[csv] {eq_path} ({len(eq_df)} rows)")

    md_path = args.results_dir / "master_table_approach_a_v2.md"
    write_master_table(df, eq_df, md_path)

    plot_cka_vs_k(df, args.output_dir)
    plot_row_cosine_vs_k(df, args.output_dir)
    plot_f1_gap_distribution(df, args.output_dir)
    plot_wallclock_vs_density(df, args.output_dir)

    if args.include_sanity:
        sanity_df = load_sanity_jsons(args.input_dir)
        print(f"[load] sanity JSONs: {len(sanity_df)} rows")
        plot_sanity_controls(sanity_df, args.output_dir)


if __name__ == "__main__":
    main()
