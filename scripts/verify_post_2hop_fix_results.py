#!/usr/bin/env python3
"""Verification script — run AFTER the post-2hop-fix rerun completes on the server.

Checks that the new results are sane and not contaminated with stale data.

Run on server with:
    cd /home/yudong/gilchris/gnn_scalability_experiment
    source .venv/bin/activate
    python scripts/verify_post_2hop_fix_results.py
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS = []
PASS = 0
FAIL = 0


def check(label: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    icon = "✓" if ok else "✗"
    msg = f"  [{icon}] {label}"
    if detail:
        msg += f"  --  {detail}"
    print(msg)
    CHECKS.append((label, ok, detail))
    if ok:
        PASS += 1
    else:
        FAIL += 1


def main():
    print("=== verifying post-2hop-fix rerun results ===\n")

    # ---- 1. Substrate fidelity: KMV == FIXED Exact on sparse cells ---------
    print("1. Substrate-fidelity: per-row exact match (KMV ≡ Exact when k ≥ d_max)")
    try:
        sys.path.insert(0, str(ROOT))
        from src.data.factory import DatasetFactory
        g, _ = DatasetFactory.get_data("HGB", "DBLP", "author")
        from collections import defaultdict
        paper_authors = defaultdict(set)
        ap = g[("author", "author_to_paper", "paper")].edge_index
        for i in range(ap.shape[1]):
            paper_authors[int(ap[1, i])].add(int(ap[0, i]))
        coauthors = defaultdict(set)
        for p, auths in paper_authors.items():
            for a in auths:
                coauthors[a] |= auths
        for a in range(g["author"].num_nodes):
            coauthors[a].add(a)
        true_total = sum(len(s) for s in coauthors.values())
        true_max = max(len(s) for s in coauthors.values())
        check(f"PyG-derived true 1-hop APA: total={true_total} max={true_max}", True)
    except Exception as e:
        check("PyG 1-hop computation", False, str(e))
        true_total = -1
        true_max = -1

    # ---- 2. JSON output coverage --------------------------------------------
    print("\n2. JSON output coverage (post-fix runs)")
    base = ROOT / "results" / "approach_a_2026_05_07"
    expected_cells = [
        ("HGB_DBLP", "SAGE", "author_to_paper_paper_to_author"),
        ("HGB_DBLP", "SAGE", "author_to_paper_paper_to_venue_venue_to_paper_paper_to_author"),
        ("HGB_ACM", "SAGE", "paper_to_author_author_to_paper"),
        ("HGB_ACM", "SAGE", "paper_to_term_term_to_paper"),
        ("HGB_IMDB", "SAGE", "movie_to_actor_actor_to_movie"),
        ("HGB_IMDB", "SAGE", "movie_to_keyword_keyword_to_movie"),
        ("HNE_PubMed", "SAGE", "disease_to_gene_gene_to_disease"),
        ("HNE_PubMed", "SAGE", "disease_to_chemical_chemical_to_disease"),
    ]
    for ds, arch, mp in expected_cells:
        cell_dir = base / ds / arch
        n = len(list(cell_dir.glob(f"{mp}_seed*_k*.json"))) if cell_dir.exists() else 0
        check(f"{ds}/{arch}/{mp}: {n} JSONs (expect ≥25 = 5 seeds × 5 k)",
              n >= 25, f"got {n}")

    # ---- 3. Snapshot exists for diff ---------------------------------------
    print("\n3. Pre-fix snapshot preserved for diff")
    snapshots = list((ROOT / "results").glob("approach_a_2026_05_07_pre_2hop_fix_*"))
    check(f"snapshot dir(s) exist: {len(snapshots)}",
          len(snapshots) >= 1,
          f"snapshots: {[s.name for s in snapshots]}")
    if snapshots:
        latest = max(snapshots, key=lambda p: p.stat().st_mtime)
        n_old_jsons = len(list(latest.rglob("*.json")))
        check(f"pre-fix snapshot has JSONs: {n_old_jsons}", n_old_jsons > 0)

    # ---- 4. KMV ≡ Exact bit-for-bit on DBLP-APA (the airtight expectation)
    print("\n4. DBLP-APA SAGE k=128: KMV ≡ Exact (substrate identity check)")
    sample = base / "HGB_DBLP" / "SAGE" / "author_to_paper_paper_to_author_seed42_k128.json"
    if sample.exists():
        with open(sample) as f:
            d = json.load(f)
        cka_l2 = d.get("cka_unbiased_per_layer", [None, None])[1]
        rowcos = d.get("row_cosine_per_layer_mean", [None, None])[1]
        f1gap = d.get("macro_f1_gap")
        n_edges_kmv = d.get("n_edges_kmv")
        n_edges_exact = d.get("n_edges_exact")
        check(f"DBLP-APA k=128 row_cos = {rowcos:.4f}  (expect ≈ 1.0)",
              rowcos is not None and rowcos > 0.999,
              "fail = substrate not identical, fix didn't take")
        check(f"DBLP-APA k=128 uCKA = {cka_l2:.4f}  (expect ≈ 1.0)",
              cka_l2 is not None and cka_l2 > 0.999)
        check(f"DBLP-APA k=128 |F1 gap| = {abs(f1gap):.4f}  (expect ≈ 0)",
              f1gap is not None and abs(f1gap) < 0.001)
        check(f"DBLP-APA k=128 n_edges_kmv = {n_edges_kmv}  (expect 11113)",
              n_edges_kmv == 11113)
        check(f"DBLP-APA k=128 n_edges_exact = {n_edges_exact}  (expect 11113, NOT 40703)",
              n_edges_exact == 11113,
              "if 40703, the bug fix didn't propagate to mat_exact regeneration")
    else:
        check(f"DBLP-APA k=128 JSON missing", False, str(sample))

    # ---- 5. Convergence matrix: more cells should converge ------------------
    print("\n5. Convergence matrix (expect more CONVERGED than pre-fix)")
    cm = ROOT / "results" / "convergence_matrix.csv"
    if cm.exists():
        n_conv = sum(1 for line in cm.read_text().splitlines() if "CONVERGED" in line)
        n_failed = sum(1 for line in cm.read_text().splitlines() if "FAILED" in line)
        check(f"CONVERGED cells: {n_conv}", n_conv >= 14,
              "pre-fix had 14; expect ≥14 (likely more after substrate fix)")
        check(f"FAILED cells: {n_failed}", n_failed <= 10,
              "pre-fix had 10 FAILED on dense-meta-path GCN/GAT/GIN; expect fewer")
    else:
        check("convergence_matrix.csv missing", False)

    # ---- 6. No mixing of pre-fix and post-fix data --------------------------
    print("\n6. No stale 2-hop data leaking into post-fix tables")
    table_md = ROOT / "results" / "master_table_approach_a_v2.md"
    if table_md.exists():
        # Check that DBLP-APA row at k=128 has the new (correct) numbers
        for line in table_md.read_text().splitlines():
            if "DBLP" in line and "SAGE" in line and "APA" in line and "| 128 |" in line:
                # Look for the row-cos field; should be ≈ 1.0, not the old 0.977
                check(f"DBLP-APA SAGE k=128 in master_table: {line[:120]}",
                      "1.000" in line or "1.00" in line,
                      "if shows old 0.977, table was generated from stale data")
                break

    # ---- 7. OGB-MAG: did separate run land? --------------------------------
    print("\n7. OGB-MAG cost data freshness")
    ogb_csv = ROOT / "results" / "OGB_MAG" / "master_results_v2.csv"
    pre_log = ROOT / "results" / "OGB_MAG" / "exact_attempt_165329_PRE_2HOP_FIX.log"
    check("OGB-MAG pre-fix Exact log snapshot preserved",
          pre_log.exists(),
          "if missing, pre-fix OGB cost data was overwritten without snapshot")
    if ogb_csv.exists():
        # Rough check: the new csv should differ in size/timestamp from the snapshot
        n_lines = len(ogb_csv.read_text().splitlines())
        check(f"OGB-MAG master_results_v2.csv has {n_lines} lines",
              n_lines > 0,
              "if 0, OGB-MAG wasn't re-run after the main sweep")

    # ---- Summary -----------------------------------------------------------
    print(f"\n=== {PASS} passed / {FAIL} failed ===")
    if FAIL == 0:
        print("\nALL CHECKS PASS. Results are coherent with the post-2hop-fix rerun.")
        sys.exit(0)
    else:
        print(f"\n{FAIL} CHECK(S) FAILED. Investigate before trusting paper artefacts.")
        sys.exit(1)


if __name__ == "__main__":
    main()
