#!/usr/bin/env bash
# v2 follow-up runs — 2026-05-07.
# 1. 10-seed headline re-run on 4 cells (DBLP-APA-SAGE, ACM-PAP × {SAGE, GCN, GAT}).
#    Uses the existing run_approach_a_sweep_v2.sh --seeds-headline flag; 200 runs.
# 2. Sanity re-run with the kmv_hash_seed bug fixed (--vary-hash-seed default).
#    Modes random_theta, edge_perturb, density_matched_random, layer_permutation.

set -uo pipefail
cd ~/gilchris/gnn_scalability_experiment
source .venv/bin/activate

LOG=results/v2_followups_$(date +%Y%m%d_%H%M%S).log
mkdir -p results
echo "[$(date)] === v2 follow-up runs ===" | tee -a $LOG

echo "[$(date)] === STAGE 1: 10-seed headline ===" | tee -a $LOG
bash scripts/run_approach_a_sweep_v2.sh --seeds-headline 2>&1 | tee -a $LOG

echo "[$(date)] === STAGE 2: sanity re-run with varied hash seed ===" | tee -a $LOG
PART=results/HGB_DBLP/partition.json
WTS=results/HGB_DBLP/weights
MP=author_to_paper,paper_to_author
for mode in random_theta edge_perturb density_matched_random layer_permutation; do
  echo "[$(date)] sanity mode=$mode" | tee -a $LOG
  python scripts/exp_sanity_controls.py \
    --mode $mode \
    --dataset HGB_DBLP \
    --metapath $MP \
    --target-type author \
    --arch SAGE \
    --depth 2 \
    --k-values 8 16 32 64 128 \
    --seeds 42 43 44 \
    --partition-json $PART \
    --weights-dir $WTS 2>&1 | tail -10 | tee -a $LOG
  echo "[$(date)] sanity mode=$mode DONE" | tee -a $LOG
done

echo "[$(date)] === v2 follow-ups DONE ===" | tee -a $LOG
