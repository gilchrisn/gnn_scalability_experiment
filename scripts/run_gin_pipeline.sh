#!/usr/bin/env bash
# GIN end-to-end pipeline (2026-05-07): train GIN on the 8 (dataset, mp) cells
# where SAGE converged in v2, run convergence test, append GIN rows to
# convergence_matrix.csv, then run the k-sweep on GIN-convergent cells.
#
# Each (dataset, mp) gets seed=42 training, then a smoke-test inference at k=32
# to determine convergence (cka_last > 0.5 AND macro_f1_exact > 0.30).
# Convergent cells are added to convergence_matrix.csv; the existing
# run_approach_a_sweep_v2.sh launcher then sweeps them at 5 seeds × 5 k.
set -uo pipefail
cd ~/gilchris/gnn_scalability_experiment
source .venv/bin/activate

LOG=results/gin_pipeline_$(date +%Y%m%d_%H%M%S).log
echo "[$(date)] === GIN pipeline starting ===" | tee -a $LOG

declare -A MP_FULL=(
  ["HGB_DBLP/APA"]="author_to_paper,paper_to_author"
  ["HGB_DBLP/APVPA"]="author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"
  ["HGB_ACM/PAP"]="paper_to_author,author_to_paper"
  ["HGB_ACM/PTP"]="paper_to_term,term_to_paper"
  ["HGB_IMDB/MAM"]="movie_to_actor,actor_to_movie"
  ["HGB_IMDB/MKM"]="movie_to_keyword,keyword_to_movie"
  ["HNE_PubMed/DGD"]="disease_to_gene,gene_to_disease"
  ["HNE_PubMed/DCD"]="disease_to_chemical,chemical_to_disease"
)
declare -A TARGET=(
  ["HGB_DBLP"]="author"
  ["HGB_ACM"]="paper"
  ["HGB_IMDB"]="movie"
  ["HNE_PubMed"]="disease"
)

# Phase 1: train GIN weights for each cell (skip if .pt already exists)
echo "[$(date)] === PHASE 1: train GIN weights ===" | tee -a $LOG
for cell_key in "${!MP_FULL[@]}"; do
  ds=${cell_key%/*}
  mp_short=${cell_key#*/}
  mp=${MP_FULL[$cell_key]}
  mp_safe=${mp//,/_}
  WTS=results/$ds/weights/${mp_safe}_L2_GIN.pt
  if [ -f "$WTS" ]; then
    echo "[$(date)] [SKIP] $cell_key — weights exist" | tee -a $LOG
    continue
  fi
  echo "[$(date)] [TRAIN] $cell_key (arch=GIN)" | tee -a $LOG
  timeout 1800 python scripts/exp2_train.py $ds \
    --metapath "$mp" \
    --depth 2 \
    --epochs 100 \
    --partition-json results/$ds/partition.json \
    --seed 42 \
    --arch GIN 2>&1 | tail -10 | tee -a $LOG
done

# Phase 2: convergence smoke-test (1 seed, 1 k)
echo "[$(date)] === PHASE 2: GIN convergence test ===" | tee -a $LOG
for cell_key in "${!MP_FULL[@]}"; do
  ds=${cell_key%/*}
  mp_short=${cell_key#*/}
  mp=${MP_FULL[$cell_key]}
  mp_safe=${mp//,/_}
  JSON=results/approach_a_2026_05_07/$ds/GIN/${mp_safe}_seed42_k32.json
  if [ -f "$JSON" ]; then
    echo "[$(date)] [SKIP-CONV] $cell_key" | tee -a $LOG
    continue
  fi
  echo "[$(date)] [CONV-TEST] $cell_key" | tee -a $LOG
  timeout 1800 python scripts/exp3_inference.py $ds \
    --metapath "$mp" \
    --depth 2 \
    --k-values 32 \
    --partition-json results/$ds/partition.json \
    --weights-dir results/$ds/weights \
    --output results/$ds/master_results_v2.csv \
    --arch GIN \
    --skip-mprw \
    --run-id 42 \
    --hash-seed 42 \
    --timeout 1500 2>&1 | tail -10 | tee -a $LOG
done

# Phase 3: append GIN rows to convergence_matrix.csv based on smoke-test JSONs
echo "[$(date)] === PHASE 3: update convergence_matrix.csv with GIN rows ===" | tee -a $LOG
python scripts/update_convergence_for_gin.py 2>&1 | tee -a $LOG

# Phase 4: full GIN k-sweep (5 seeds × 5 k on convergent cells)
# The existing launcher reads convergence_matrix.csv and runs only CONVERGED rows.
echo "[$(date)] === PHASE 4: GIN full k-sweep ===" | tee -a $LOG
bash scripts/run_approach_a_sweep_v2.sh 2>&1 | tail -50 | tee -a $LOG

# Phase 5: re-aggregate
echo "[$(date)] === PHASE 5: re-aggregate ===" | tee -a $LOG
python scripts/aggregate_approach_a_v2.py --include-sanity 2>&1 | tail -10 | tee -a $LOG

echo "[$(date)] === GIN pipeline DONE ===" | tee -a $LOG
