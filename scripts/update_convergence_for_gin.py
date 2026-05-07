"""Append GIN rows to results/convergence_matrix.csv based on v2 smoke-test JSONs.

For each (dataset, mp) cell, reads results/approach_a_2026_05_07/<ds>/GIN/
<mp_safe>_seed42_k32.json (the smoke-test produced by phase 2 of run_gin_pipeline.sh)
and applies the same convergence criterion as compute_convergence_matrix.py:
  CONVERGED if cka_last > 0.5 AND macro_f1_exact > 0.30 AND convergence_epoch > 1.

Appends rows to convergence_matrix.csv with verdict CONVERGED / FAILED / NO_DATA.

Idempotent: if a (dataset, GIN, mp) row already exists, it's overwritten.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "approach_a_2026_05_07"
CSV = ROOT / "results" / "convergence_matrix.csv"

CELLS = [
    ("HGB_DBLP",   "APA",   "author_to_paper_paper_to_author"),
    ("HGB_DBLP",   "APVPA", "author_to_paper_paper_to_venue_venue_to_paper_paper_to_author"),
    ("HGB_ACM",    "PAP",   "paper_to_author_author_to_paper"),
    ("HGB_ACM",    "PTP",   "paper_to_term_term_to_paper"),
    ("HGB_IMDB",   "MAM",   "movie_to_actor_actor_to_movie"),
    ("HGB_IMDB",   "MKM",   "movie_to_keyword_keyword_to_movie"),
    ("HNE_PubMed", "DGD",   "disease_to_gene_gene_to_disease"),
    ("HNE_PubMed", "DCD",   "disease_to_chemical_chemical_to_disease"),
]


def gin_verdict(ds: str, mp_safe: str) -> dict:
    p = RES / ds / "GIN" / f"{mp_safe}_seed42_k32.json"
    if not p.exists():
        return {"verdict": "NO_DATA", "conv_seeds": 0, "n_seeds": 0,
                "mean_conv_epoch": "", "mean_f1_exact": "", "mean_cka_last": ""}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {"verdict": "NO_DATA", "conv_seeds": 0, "n_seeds": 1,
                "mean_conv_epoch": "", "mean_f1_exact": "", "mean_cka_last": ""}
    cka = d.get("cka_per_layer") or [None]
    cka_last = cka[-1] if cka else None
    f1 = d.get("macro_f1_exact")
    ce = d.get("convergence_epoch", 1)
    converged = (
        isinstance(ce, (int, float)) and ce > 1
        and isinstance(cka_last, (int, float)) and cka_last > 0.5
        and isinstance(f1, (int, float)) and f1 > 0.30
    )
    return {
        "verdict": "CONVERGED" if converged else "FAILED",
        "conv_seeds": 1 if converged else 0,
        "n_seeds": 1,
        "mean_conv_epoch": ce or "",
        "mean_f1_exact": f1 or "",
        "mean_cka_last": cka_last or "",
    }


def main():
    rows = []
    if CSV.exists():
        with open(CSV, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    rows = [r for r in rows if r.get("arch") != "GIN"]
    for ds, mp_short, mp_safe in CELLS:
        s = gin_verdict(ds, mp_safe)
        rows.append({
            "dataset": ds, "arch": "GIN", "mp": mp_short,
            **s,
        })

    cols = ["dataset", "arch", "mp", "verdict", "conv_seeds", "n_seeds",
            "mean_conv_epoch", "mean_f1_exact", "mean_cka_last"]
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print(f"[csv] {CSV}")
    print()
    print("# GIN convergence (smoke-test seed=42 k=32, criterion: cka_last>0.5 AND macro_f1_exact>0.30)")
    print("| Dataset | MP | Verdict | CKA | F1 |")
    print("|---|---|---|---|---|")
    for r in rows:
        if r.get("arch") == "GIN":
            cka = r.get("mean_cka_last", "")
            f1 = r.get("mean_f1_exact", "")
            cka_s = f"{float(cka):.3f}" if cka not in ("", None) else "?"
            f1_s = f"{float(f1):.3f}" if f1 not in ("", None) else "?"
            print(f"| {r['dataset']} | {r['mp']} | {r['verdict']} | {cka_s} | {f1_s} |")


if __name__ == "__main__":
    main()
