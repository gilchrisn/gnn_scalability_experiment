#!/usr/bin/env bash
# run_post_2hop_fix_rerun.sh
#
# One-shot full rerun after the 2026-05-12 hidden_graph_construction bug fix.
# Wipes stale 2-hop artefacts, re-trains all (dataset, arch, mp) cells on the
# corrected 1-hop substrate, recomputes the convergence matrix, runs the v2
# inference sweep, then re-aggregates everything.
#
# Run inside tmux on the yudong server:
#   tmux new -s post_2hop_fix
#   bash scripts/run_post_2hop_fix_rerun.sh 2>&1 | tee results/post_2hop_fix_rerun.log
#
# Resume-safe: each stage individually checks for stale outputs and skips
# if --resume is passed.
#
# Estimated wall-time on yudong (2x A5000): ~30-40 hours full sweep including
# sanity controls.

set -euo pipefail
cd "$(dirname "$0")/.."

RESUME=0
if [[ "${1:-}" == "--resume" ]]; then RESUME=1; fi

# ---- 0. Sanity: confirm HUB has the 2-hop fix -----------------------------
if ! grep -q "BUG FIX: the relational graph H_P" HUB/hg.cpp; then
    echo "ERROR: HUB/hg.cpp does not contain the 2-hop-fix marker comment." >&2
    echo "Pull origin/main and ensure hidden_graph_construction is fixed before rerunning." >&2
    exit 2
fi

# ---- 1. Activate venv + rebuild C++ binary --------------------------------
source .venv/bin/activate 2>/dev/null || source venv/bin/activate
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1

echo "[$(date '+%H:%M:%S')] Stage 1/6 — rebuild C++ binary"
make clean
make
test -x bin/graph_prep || { echo "graph_prep build failed"; exit 1; }

# ---- 2. Snapshot pre-fix artefacts and wipe stale state -------------------
if [[ $RESUME -eq 0 ]]; then
    echo "[$(date '+%H:%M:%S')] Stage 2/6 — snapshot + wipe stale state"
    SNAP_DIR="results/approach_a_2026_05_07_pre_2hop_fix_$(date +%Y%m%d_%H%M%S)"
    if [[ -d "results/approach_a_2026_05_07" ]]; then
        mv "results/approach_a_2026_05_07" "$SNAP_DIR"
        echo "  snapshotted: results/approach_a_2026_05_07 -> $SNAP_DIR"
    fi
    # Wipe stale weights + mat_exact files + training-log DONE rows + caches
    for ds in HGB_DBLP HGB_ACM HGB_IMDB HNE_PubMed OGB_MAG; do
        rm -f "results/$ds/weights/"*.pt 2>/dev/null || true
        rm -f "results/$ds/weights/"*.meta.json 2>/dev/null || true
        rm -rf "results/$ds/inf_scratch" 2>/dev/null || true
        rm -f "results/$ds/training_log.csv" 2>/dev/null || true
    done
    for stg in staging/HGBn-DBLP staging/HGBn-ACM staging/HGBn-IMDB staging/HNE_PubMed staging/OGB_MAG; do
        rm -f "$stg/mat_exact.adj" "$stg/mat_sketch" "$stg/mat_sketch_"* 2>/dev/null || true
    done
    rm -f results/master_table_approach_a_v2.md
    rm -f results/master_results.csv results/master_results_similarity.csv
    rm -f results/equivalence_tests_v2.csv
    rm -f results/convergence_matrix.csv
    echo "  wipe complete"
fi

mkdir -p results/approach_a_2026_05_07

# ---- 3. Re-train all cells (SAGE + GCN/GAT/GIN on every cell) -------------
# Cell roster — the union of v1+v2 cells; the new convergence matrix will
# decide which to keep at inference time.
declare -a SAGE_CELLS=(
    "HGB_DBLP|author|author_to_paper,paper_to_author"
    "HGB_DBLP|author|author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"
    "HGB_ACM|paper|paper_to_author,author_to_paper"
    "HGB_ACM|paper|paper_to_term,term_to_paper"
    "HGB_IMDB|movie|movie_to_actor,actor_to_movie"
    "HGB_IMDB|movie|movie_to_keyword,keyword_to_movie"
    "HNE_PubMed|disease|disease_to_gene,gene_to_disease"
    "HNE_PubMed|disease|disease_to_chemical,chemical_to_disease"
)
declare -a ARCH_CELLS=(
    "HGB_DBLP|author|author_to_paper,paper_to_author"
    "HGB_DBLP|author|author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"
    "HGB_ACM|paper|paper_to_author,author_to_paper"
    "HGB_ACM|paper|paper_to_term,term_to_paper"
    "HGB_IMDB|movie|movie_to_actor,actor_to_movie"
    "HGB_IMDB|movie|movie_to_keyword,keyword_to_movie"
    "HNE_PubMed|disease|disease_to_gene,gene_to_disease"
    "HNE_PubMed|disease|disease_to_chemical,chemical_to_disease"
)

echo "[$(date '+%H:%M:%S')] Stage 3/6 — re-train all cells on fixed substrate"

for cell in "${SAGE_CELLS[@]}"; do
    IFS='|' read -r ds tgt mp <<<"$cell"
    part="results/$ds/partition.json"
    if [[ ! -f "$part" ]]; then
        echo "  [skip $ds $mp] no partition.json"; continue
    fi
    echo "  [train SAGE] $ds $mp"
    python scripts/exp2_train.py "$ds" \
        --metapath "$mp" --depth 2 --arch SAGE \
        --partition-json "$part" --epochs 100 --seed 42 \
        >> results/post_2hop_fix_train.log 2>&1 || echo "    SAGE train failed for $ds $mp"
done

for cell in "${ARCH_CELLS[@]}"; do
    IFS='|' read -r ds tgt mp <<<"$cell"
    part="results/$ds/partition.json"
    if [[ ! -f "$part" ]]; then continue; fi
    for arch in GCN GAT GIN; do
        echo "  [train $arch] $ds $mp"
        python scripts/exp2_train.py "$ds" \
            --metapath "$mp" --depth 2 --arch "$arch" \
            --partition-json "$part" --epochs 100 --seed 42 \
            >> results/post_2hop_fix_train.log 2>&1 || echo "    $arch train failed for $ds $mp"
    done
done

# ---- 4. Recompute convergence matrix --------------------------------------
echo "[$(date '+%H:%M:%S')] Stage 4/6 — recompute convergence matrix"
python scripts/compute_convergence_matrix.py > results/convergence_matrix_post_fix.log 2>&1 || true
cat results/convergence_matrix.csv | head -30

# ---- 5. v2 main sweep (KMV inference across 5 k values, 5 seeds, sanity) --
echo "[$(date '+%H:%M:%S')] Stage 5/6 — v2 inference sweep"
bash scripts/run_approach_a_sweep_v2.sh --include-sanity 2>&1 | tee -a results/post_2hop_fix_sweep.log

# ---- 6. Headline cells: extend to 10 seeds --------------------------------
echo "[$(date '+%H:%M:%S')] Stage 6/6 — headline 10-seed extension"
bash scripts/run_approach_a_sweep_v2.sh --seeds-headline 2>&1 | tee -a results/post_2hop_fix_sweep.log

# ---- 7. Aggregate and regenerate paper artefacts --------------------------
echo "[$(date '+%H:%M:%S')] Stage 7 — aggregate + regenerate tables/figures"
python scripts/aggregate_approach_a_v2.py 2>&1 | tee -a results/post_2hop_fix_agg.log
python scripts/emit_v2_tables.py 2>&1 | tee -a results/post_2hop_fix_agg.log
python scripts/plot_v2_paper_figures.py 2>&1 | tee -a results/post_2hop_fix_agg.log

echo "[$(date '+%H:%M:%S')] DONE. Review:"
echo "  results/convergence_matrix.csv"
echo "  results/master_table_approach_a_v2.md"
echo "  results/equivalence_tests_v2.csv"
echo "  springer/tables/tab_sage_8cell.tex"
echo "  springer/tables/tab_arch_2cell.tex"
echo "  springer/figure/exp/hgnn_v2/fig{1,2,4,5}*.pdf"
