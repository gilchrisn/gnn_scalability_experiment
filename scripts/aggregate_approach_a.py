"""Aggregate Approach A results: master table + CKA bar chart.

Reads results/approach_a_2026_05_05/<DS>/<arch>/<mp>_seed<seed>.json (72 cells).
Writes:
  - results/master_table_approach_a.md
  - figures/approach_a/cka_summary.png  (per-arch CKA bar chart)
  - figures/approach_a/f1_gap.png       (per-arch F1 gap)
"""
import json
import statistics
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "approach_a_2026_05_05"
FIG = ROOT / "figures" / "approach_a"
FIG.mkdir(parents=True, exist_ok=True)

DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed"]
DS_LABELS = {"HGB_DBLP": "DBLP", "HGB_ACM": "ACM",
             "HGB_IMDB": "IMDB", "HNE_PubMed": "PubMed"}
ARCHS = ["SAGE", "GCN", "GAT"]
ARCH_COLOR = {"SAGE": "#2CA02C", "GCN": "#1F77B4", "GAT": "#9467BD"}
META_PATHS = {
    "HGB_DBLP": ["author_to_paper_paper_to_author",
                 "author_to_paper_paper_to_venue_venue_to_paper_paper_to_author"],
    "HGB_ACM": ["paper_to_author_author_to_paper",
                "paper_to_term_term_to_paper"],
    "HGB_IMDB": ["movie_to_actor_actor_to_movie",
                 "movie_to_keyword_keyword_to_movie"],
    "HNE_PubMed": ["disease_to_gene_gene_to_disease",
                   "disease_to_chemical_chemical_to_disease"],
}
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


def load_cells():
    """Return dict: (ds, arch, mp) -> list of seed JSONs."""
    cells = defaultdict(list)
    for ds in DATASETS:
        for arch in ARCHS:
            for mp in META_PATHS[ds]:
                for seed in (42, 43, 44):
                    p = RES / ds / arch / f"{mp}_seed{seed}.json"
                    if p.exists():
                        try:
                            cells[(ds, arch, mp)].append(json.loads(p.read_text()))
                        except Exception:
                            pass
    return cells


def agg(cells, key, depth=None):
    """Mean+std over seeds. depth=None for scalar key, int for cka_per_layer[depth]."""
    vals = []
    for c in cells:
        if depth is not None:
            v = c.get(key, [None])
            v = v[depth] if v and len(v) > depth else None
        else:
            v = c.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None, None, 0
    return (statistics.fmean(vals),
            statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            len(vals))


def write_master_table(cells):
    out = ROOT / "results" / "master_table_approach_a.md"
    lines = ["# Approach A — Master Results (2026-05-06)", "",
             "Train on Exact materialised H, freeze theta*, infer on Exact and KMV (k=32). "
             "Per-layer Linear CKA, Prediction Agreement, Macro-F1 gap. 3 seeds, L=2.", ""]
    lines.append("## Final-layer CKA (KMV vs Exact)")
    lines.append("")
    header = "| Dataset | Meta-path | " + " | ".join(ARCHS) + " |"
    sep = "|---|---|" + "---|" * len(ARCHS)
    lines += [header, sep]
    for ds in DATASETS:
        for mp in META_PATHS[ds]:
            row = [DS_LABELS[ds], MP_LABELS[mp]]
            for arch in ARCHS:
                items = cells.get((ds, arch, mp), [])
                # final-layer CKA = cka_per_layer[-1]
                vals = [c["cka_per_layer"][-1] for c in items
                        if isinstance(c.get("cka_per_layer"), list) and c["cka_per_layer"]]
                if vals:
                    mu = statistics.fmean(vals)
                    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                    row.append(f"{mu:.3f} ± {sd:.3f}")
                else:
                    row.append("---")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Macro-F1 gap (KMV - Exact)")
    lines += ["", header, sep]
    for ds in DATASETS:
        for mp in META_PATHS[ds]:
            row = [DS_LABELS[ds], MP_LABELS[mp]]
            for arch in ARCHS:
                items = cells.get((ds, arch, mp), [])
                vals = [c.get("f1_gap") for c in items if isinstance(c.get("f1_gap"), (int, float))]
                if vals:
                    mu = statistics.fmean(vals)
                    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                    sign = "+" if mu >= 0 else ""
                    row.append(f"{sign}{mu:.3f} ± {sd:.3f}")
                else:
                    row.append("---")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Prediction Agreement (KMV vs Exact predictions on V_test)")
    lines += ["", header, sep]
    for ds in DATASETS:
        for mp in META_PATHS[ds]:
            row = [DS_LABELS[ds], MP_LABELS[mp]]
            for arch in ARCHS:
                items = cells.get((ds, arch, mp), [])
                vals = [c.get("pred_agreement") for c in items
                        if isinstance(c.get("pred_agreement"), (int, float))]
                if vals:
                    mu = statistics.fmean(vals)
                    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                    row.append(f"{mu:.3f} ± {sd:.3f}")
                else:
                    row.append("---")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Per-(Arch) summary (avg across 8 dataset×meta-path cells)")
    lines.append("")
    lines.append("| Arch | CKA | F1 gap | PA | Cells passing gate (CKA≥0.80, |F1 gap|≤0.03, PA≥0.85) |")
    lines.append("|---|---|---|---|---|")
    for arch in ARCHS:
        all_cka, all_gap, all_pa, pass_count, total = [], [], [], 0, 0
        for ds in DATASETS:
            for mp in META_PATHS[ds]:
                items = cells.get((ds, arch, mp), [])
                cka_v = [c["cka_per_layer"][-1] for c in items
                         if isinstance(c.get("cka_per_layer"), list) and c["cka_per_layer"]]
                gap_v = [c.get("f1_gap") for c in items if isinstance(c.get("f1_gap"), (int, float))]
                pa_v = [c.get("pred_agreement") for c in items if isinstance(c.get("pred_agreement"), (int, float))]
                if cka_v and gap_v and pa_v:
                    cka_m = statistics.fmean(cka_v)
                    gap_m = statistics.fmean(gap_v)
                    pa_m = statistics.fmean(pa_v)
                    all_cka.append(cka_m); all_gap.append(gap_m); all_pa.append(pa_m)
                    if cka_m >= 0.80 and abs(gap_m) <= 0.03 and pa_m >= 0.85:
                        pass_count += 1
                    total += 1
        if all_cka:
            cka_str = f"{statistics.fmean(all_cka):.3f}"
            gap_str = f"{statistics.fmean(all_gap):+.3f}"
            pa_str = f"{statistics.fmean(all_pa):.3f}"
            lines.append(f"| {arch} | {cka_str} | {gap_str} | {pa_str} | {pass_count}/{total} |")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[md] {out}")


def plot_cka_summary():
    cells = load_cells()
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.0 * len(DATASETS), 4.2),
                              squeeze=False, sharey=True)
    for ci, ds in enumerate(DATASETS):
        ax = axes[0][ci]
        mps = META_PATHS[ds]
        x = np.arange(len(mps))
        w = 0.27
        for ai, arch in enumerate(ARCHS):
            means, stds = [], []
            for mp in mps:
                items = cells.get((ds, arch, mp), [])
                vals = [c["cka_per_layer"][-1] for c in items
                        if isinstance(c.get("cka_per_layer"), list) and c["cka_per_layer"]]
                if vals:
                    means.append(statistics.fmean(vals))
                    stds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
                else:
                    means.append(0); stds.append(0)
            offset = (ai - 1) * w
            ax.bar(x + offset, means, w, yerr=stds, capsize=3,
                   color=ARCH_COLOR[arch], alpha=0.92, label=arch,
                   edgecolor="white", linewidth=0.4)
            for j, m in enumerate(means):
                if m > 0:
                    ax.text(x[j] + offset, m + 0.015, f"{m:.2f}",
                            ha="center", fontsize=8)
        ax.axhline(0.85, linestyle="--", color="red", alpha=0.5, linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels([MP_LABELS[mp] for mp in mps], fontsize=10)
        ax.set_title(DS_LABELS[ds], fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        if ci == 0:
            ax.set_ylabel("Final-layer CKA(KMV, Exact)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if ci == len(DATASETS) - 1:
            ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    fig.suptitle("Approach A: KMV-vs-Exact embedding fidelity (3 seeds, L=2, k=32)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG / "cka_summary.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[fig] {out}")


def plot_f1_gap():
    cells = load_cells()
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.0 * len(DATASETS), 4.2),
                              squeeze=False, sharey=True)
    for ci, ds in enumerate(DATASETS):
        ax = axes[0][ci]
        mps = META_PATHS[ds]
        x = np.arange(len(mps))
        w = 0.27
        for ai, arch in enumerate(ARCHS):
            means, stds = [], []
            for mp in mps:
                items = cells.get((ds, arch, mp), [])
                vals = [c.get("f1_gap") for c in items
                        if isinstance(c.get("f1_gap"), (int, float))]
                if vals:
                    means.append(statistics.fmean(vals))
                    stds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
                else:
                    means.append(0); stds.append(0)
            offset = (ai - 1) * w
            ax.bar(x + offset, means, w, yerr=stds, capsize=3,
                   color=ARCH_COLOR[arch], alpha=0.92, label=arch,
                   edgecolor="white", linewidth=0.4)
        ax.axhline(0, linestyle="-", color="black", alpha=0.5, linewidth=0.7)
        ax.axhline(0.03, linestyle="--", color="red", alpha=0.4, linewidth=1)
        ax.axhline(-0.03, linestyle="--", color="red", alpha=0.4, linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels([MP_LABELS[mp] for mp in mps], fontsize=10)
        ax.set_title(DS_LABELS[ds], fontsize=11, fontweight="bold")
        if ci == 0:
            ax.set_ylabel("F1 gap (KMV - Exact)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if ci == len(DATASETS) - 1:
            ax.legend(fontsize=9, loc="best", framealpha=0.95)
    fig.suptitle("Approach A: F1 gap on V_test (3 seeds, L=2, k=32). Red dashes = +/-0.03 acceptance.",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG / "f1_gap.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"[fig] {out}")


if __name__ == "__main__":
    cells = load_cells()
    print(f"Loaded {sum(len(v) for v in cells.values())} JSONs across {len(cells)} cells")
    write_master_table(cells)
    plot_cka_summary()
    plot_f1_gap()
