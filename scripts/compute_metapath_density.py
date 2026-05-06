"""Per-(dataset, meta-path) density manifest for Approach A.

For each (dataset, meta-path) cell, materialize the Exact homogeneous adjacency
once via bin/graph_prep, count unique undirected edges, and compute:

    density       = E / (V * (V-1) / 2)            in [0, 1]
    avg_degree    = 2 * E / V

Writes:
    results/meta_path_density.csv   with columns:
        dataset, meta_path, target_type, n_target, n_edges,
        density, avg_degree, density_class
    where density_class ∈ {sparse, medium, dense, saturated}:
        sparse     : density < 0.05
        medium     : 0.05 <= density < 0.20
        dense      : 0.20 <= density < 0.50
        saturated  : density >= 0.50   (skip GCN/GAT — Laplacian collapse)

Usage:
    python scripts/compute_metapath_density.py
"""
from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Pull edge counts from existing master_results.csv (no need to re-run C++).
DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB", "HNE_PubMed"]
N_TARGET = {  # number of target-type nodes per dataset (from prior runs)
    "HGB_DBLP": 4057,
    "HGB_ACM": 3025,
    "HGB_IMDB": 4932,
    "HNE_PubMed": 13561,
}
META_PATHS = {
    "HGB_DBLP": [
        ("APA", "author_to_paper,paper_to_author"),
        ("APVPA", "author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"),
    ],
    "HGB_ACM": [
        ("PAP", "paper_to_author,author_to_paper"),
        ("PSP", "paper_to_subject,subject_to_paper"),
        ("PTP", "paper_to_term,term_to_paper"),
    ],
    "HGB_IMDB": [
        ("MAM", "movie_to_actor,actor_to_movie"),
        ("MDM", "movie_to_director,director_to_movie"),
        ("MKM", "movie_to_keyword,keyword_to_movie"),
    ],
    "HNE_PubMed": [
        ("DGD", "disease_to_gene,gene_to_disease"),
        ("DCD", "disease_to_chemical,chemical_to_disease"),
    ],
}


def classify(d: float) -> str:
    if d < 0.05: return "sparse"
    if d < 0.20: return "medium"
    if d < 0.50: return "dense"
    return "saturated"


def lookup_edges(ds: str, mp: str) -> int | None:
    """Find the Exact edge count for (ds, mp) in master_results.csv."""
    p = ROOT / "results" / ds / "master_results.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        if (r.get("Method") == "Exact"
            and r.get("MetaPath") == mp
            and r.get("Edge_Count")):
            try:
                return int(float(r["Edge_Count"]))
            except (ValueError, TypeError):
                pass
    return None


def main():
    out = []
    for ds in DATASETS:
        n = N_TARGET[ds]
        max_undirected = n * (n - 1) // 2
        for mp_short, mp in META_PATHS[ds]:
            e = lookup_edges(ds, mp)
            if e is None:
                print(f"[skip] no Edge_Count for {ds}/{mp_short} in master_results.csv")
                continue
            density = e / max_undirected
            avg_deg = 2 * e / n
            out.append({
                "dataset": ds,
                "meta_path": mp,
                "meta_path_short": mp_short,
                "n_target": n,
                "n_edges_exact": e,
                "density": round(density, 6),
                "avg_degree": round(avg_deg, 2),
                "density_class": classify(density),
                "skip_for_gcn_gat": density >= 0.50,
            })

    if not out:
        print("No cells computed; check master_results.csv presence")
        sys.exit(1)

    csv_path = ROOT / "results" / "meta_path_density.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for row in out:
            w.writerow(row)
    print(f"[csv] {csv_path}")

    # Pretty-print
    print()
    print(f'{"Dataset":<12} {"MP":<8} {"n_target":>10} {"edges":>12} {"density":>10} {"avg_deg":>9} {"class":<10} {"skip?":<7}')
    print("-" * 84)
    for r in out:
        print(f'{r["dataset"]:<12} {r["meta_path_short"]:<8} {r["n_target"]:>10,} '
              f'{r["n_edges_exact"]:>12,} {r["density"]:>10.4f} {r["avg_degree"]:>9.1f} '
              f'{r["density_class"]:<10} {"YES" if r["skip_for_gcn_gat"] else "no":<7}')


if __name__ == "__main__":
    main()
