"""Empirical convergence matrix: for every (arch, dataset, meta-path), did training
actually improve past the epoch-1 (random-init) val_F1?

Reads results/approach_a_2026_05_05/<DS>/<arch>/<mp>_seed<seed>.json (already exist).
For each (arch, dataset, meta-path), aggregates over seeds:
  - converged_seeds = count of seeds with convergence_epoch > 1
  - mean_conv_epoch
  - mean_f1_exact
  - mean_final_cka

Cell verdict:
  - CONVERGED  : >= 2/3 seeds had convergence_epoch > 1
  - PARTIAL    : 1/3 seeds converged
  - FAILED     : 0/3 seeds (all stuck at epoch-1 baseline)

Writes results/convergence_matrix.csv and prints a markdown matrix.
"""
import csv, json, statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "approach_a_2026_05_05"

DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed"]
DS_LABEL = {"HGB_DBLP":"DBLP","HGB_ACM":"ACM","HGB_IMDB":"IMDB","HNE_PubMed":"PubMed"}
ARCHS = ["SAGE", "GCN", "GAT"]
MPS = {
    "HGB_DBLP": [("APA","author_to_paper_paper_to_author"),
                 ("APVPA","author_to_paper_paper_to_venue_venue_to_paper_paper_to_author")],
    "HGB_ACM": [("PAP","paper_to_author_author_to_paper"),
                ("PTP","paper_to_term_term_to_paper")],
    "HGB_IMDB": [("MAM","movie_to_actor_actor_to_movie"),
                 ("MKM","movie_to_keyword_keyword_to_movie")],
    "HNE_PubMed":[("DGD","disease_to_gene_gene_to_disease"),
                  ("DCD","disease_to_chemical_chemical_to_disease")],
}


def cell_summary(ds, arch, mp_safe):
    seeds = []
    for s in (42, 43, 44):
        p = RES / ds / arch / f"{mp_safe}_seed{s}.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
            seeds.append({
                "conv_epoch": d.get("convergence_epoch", 1),
                "f1_exact":   d.get("macro_f1_exact"),
                "f1_kmv":     d.get("macro_f1_kmv"),
                "cka_last":   (d.get("cka_per_layer") or [None])[-1],
                "device":     d.get("device_used"),
            })
        except Exception:
            pass
    if not seeds:
        return None

    # Strict criterion: model produced meaningful embeddings AND learned a useful classifier
    # (CKA > 0.5: encoder is stable across Exact/KMV substrate)
    # (F1 > 0.30: classifier learned beyond random baseline)
    def really_converged(s):
        ce = s.get("conv_epoch") or 1
        cka = s.get("cka_last")
        f1 = s.get("f1_exact")
        return (ce > 1 and isinstance(cka,(int,float)) and cka > 0.5
                and isinstance(f1,(int,float)) and f1 > 0.30)
    conv = sum(1 for s in seeds if really_converged(s))
    if conv >= 2:        verdict = "CONVERGED"
    elif conv == 1:      verdict = "PARTIAL"
    else:                verdict = "FAILED"

    def avg(k):
        v = [s[k] for s in seeds if isinstance(s.get(k),(int,float))]
        return statistics.fmean(v) if v else None

    return {
        "n_seeds":   len(seeds),
        "conv_seeds":conv,
        "verdict":   verdict,
        "mean_conv_epoch": avg("conv_epoch"),
        "mean_f1_exact":   avg("f1_exact"),
        "mean_cka_last":   avg("cka_last"),
    }


def main():
    rows = []
    for ds in DATASETS:
        for arch in ARCHS:
            for mp_short, mp_safe in MPS[ds]:
                s = cell_summary(ds, arch, mp_safe)
                if not s:
                    rows.append({"dataset":ds,"arch":arch,"mp":mp_short,
                                 "verdict":"NO_DATA"})
                    continue
                rows.append({"dataset":ds,"arch":arch,"mp":mp_short, **s})

    csv_path = ROOT / "results" / "convergence_matrix.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        cols = ["dataset","arch","mp","verdict","conv_seeds","n_seeds",
                "mean_conv_epoch","mean_f1_exact","mean_cka_last"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"[csv] {csv_path}")
    print()

    # Pretty markdown matrix
    print("# Convergence matrix (seeds {42,43,44}; verdict = #seeds with conv_epoch>1)")
    print()
    header = "| Dataset | Meta-path | " + " | ".join(ARCHS) + " |"
    sep    = "|---|---|" + "---|"*len(ARCHS)
    print(header); print(sep)
    for ds in DATASETS:
        for mp_short, mp_safe in MPS[ds]:
            cells = []
            for arch in ARCHS:
                hit = next((r for r in rows
                            if r["dataset"]==ds and r["arch"]==arch and r["mp"]==mp_short), None)
                if not hit or hit.get("verdict")=="NO_DATA":
                    cells.append("---")
                else:
                    v = hit["verdict"]
                    cs = hit["conv_seeds"]
                    cka = hit.get("mean_cka_last")
                    f1 = hit.get("mean_f1_exact")
                    icon = {"CONVERGED":"✓","PARTIAL":"~","FAILED":"✗"}[v]
                    cka_s = f"{cka:.2f}" if isinstance(cka,(int,float)) else "?"
                    f1_s  = f"{f1:.2f}" if isinstance(f1,(int,float)) else "?"
                    cells.append(f"{icon} {cs}/{hit['n_seeds']} (CKA={cka_s} F1={f1_s})")
            print(f"| {DS_LABEL[ds]} | {mp_short} | " + " | ".join(cells) + " |")
    print()
    print("Legend: ✓=converged (≥2 seeds), ~=partial (1 seed), ✗=failed (0 seeds)")

    # Summary per-arch
    print()
    print("## Per-arch summary")
    print()
    print("| Arch | CONVERGED | PARTIAL | FAILED | NO_DATA | Of 8 cells |")
    print("|---|---|---|---|---|---|")
    for arch in ARCHS:
        sub = [r for r in rows if r["arch"]==arch]
        cnt = {"CONVERGED":0,"PARTIAL":0,"FAILED":0,"NO_DATA":0}
        for r in sub:
            cnt[r.get("verdict","NO_DATA")] += 1
        print(f"| {arch} | {cnt['CONVERGED']} | {cnt['PARTIAL']} | {cnt['FAILED']} | {cnt['NO_DATA']} | {sum(cnt.values())} |")


if __name__ == "__main__":
    main()
