#!/usr/bin/env bash
# Approach A full HGB sweep: 3 archs × 4 datasets × 2 metapaths × 3 seeds = 72 runs.
# Sequential (no GPU contention). Resume-safe via JSON file existence + CSV resume.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG="results/approach_a_sweep_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results
echo "[$(date)] starting Approach A sweep, log=$LOG" | tee -a "$LOG"

# (dataset, target, mp_short, mp_full)
CELLS=(
  "HGB_DBLP|author|APA|author_to_paper,paper_to_author"
  "HGB_DBLP|author|APVPA|author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"
  "HGB_ACM|paper|PAP|paper_to_author,author_to_paper"
  "HGB_ACM|paper|PTP|paper_to_term,term_to_paper"
  "HGB_IMDB|movie|MAM|movie_to_actor,actor_to_movie"
  "HGB_IMDB|movie|MKM|movie_to_keyword,keyword_to_movie"
  "HNE_PubMed|disease|DGD|disease_to_gene,gene_to_disease"
  "HNE_PubMed|disease|DCD|disease_to_chemical,chemical_to_disease"
)
ARCHS=(SAGE GCN GAT)
SEEDS=(42 43 44)

mp_safe() { echo "$1" | tr ',' '_'; }

for cell in "${CELLS[@]}"; do
  IFS='|' read -r ds tgt short mp <<< "$cell"
  mps=$(mp_safe "$mp")
  for arch in "${ARCHS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out_json="results/approach_a_2026_05_05/$ds/$arch/${mps}_seed${seed}.json"
      if [ -f "$out_json" ]; then
        echo "[$(date)] SKIP (exists): $ds/$arch/$short/seed=$seed" | tee -a "$LOG"
        continue
      fi
      echo "[$(date)] RUN: $ds/$arch/$short/seed=$seed" | tee -a "$LOG"

      # Train (Exact materialization + GNN training, frozen θ*)
      python scripts/exp2_train.py "$ds" \
        --metapath "$mp" \
        --arch "$arch" \
        --depth 2 \
        --epochs 100 \
        --partition-json "results/$ds/partition.json" \
        --seed "$seed" >>"$LOG" 2>&1 || {
          echo "[$(date)] TRAIN_FAIL: $ds/$arch/$short/seed=$seed" | tee -a "$LOG"
          continue
        }

      # Inference (Exact + KMV at k=32, no MPRW)
      python scripts/exp3_inference.py "$ds" \
        --metapath "$mp" \
        --arch "$arch" \
        --partition-json "results/$ds/partition.json" \
        --depth 2 \
        --k-values 32 \
        --skip-mprw \
        --run-id "$seed" \
        --hash-seed "$seed" >>"$LOG" 2>&1 || {
          echo "[$(date)] INFER_FAIL: $ds/$arch/$short/seed=$seed" | tee -a "$LOG"
          continue
        }

      if [ -f "$out_json" ]; then
        echo "[$(date)] OK: $out_json" | tee -a "$LOG"
      else
        echo "[$(date)] WARN no JSON: $out_json" | tee -a "$LOG"
      fi
    done
  done
done

# Final tally
echo "" | tee -a "$LOG"
echo "[$(date)] FINAL TALLY:" | tee -a "$LOG"
for arch in "${ARCHS[@]}"; do
  total=$(find results/approach_a_2026_05_05 -path "*/$arch/*.json" 2>/dev/null | wc -l)
  echo "  $arch: $total/24 JSONs" | tee -a "$LOG"
done
echo "[$(date)] DONE" | tee -a "$LOG"
