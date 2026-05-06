#!/usr/bin/env bash
# OGB-MAG v2 pipeline (revised 2026-05-07): partition -> train -> k-sweep.
#
# Calls exp3 ONCE per seed with all 5 k values so Exact materialises only once
# per seed (instead of 5x). Sets --timeout 3600 to handle OGB-MAG's ~21 min Exact
# materialisation. Phase 2 training falls back to CPU on GPU OOM.
set -uo pipefail
cd ~/gilchris/gnn_scalability_experiment
source .venv/bin/activate

LOG=results/OGB_MAG/ogb_mag_v2_$(date +%Y%m%d_%H%M%S).log
mkdir -p results/OGB_MAG
echo "[$(date)] OGB-MAG v2 pipeline starting" | tee -a $LOG

# Phase 1: partition (cheap; ~1 min)
if [ ! -f results/OGB_MAG/partition.json ]; then
  echo "[$(date)] === PHASE 1: partition ===" | tee -a $LOG
  python scripts/exp1_partition.py --dataset OGB_MAG --target-type paper --train-frac 0.4 --seed 42 2>&1 | tee -a $LOG
fi

# Phase 2: train SAGE on full V_train
WEIGHTS=results/OGB_MAG/weights/rev_writes_writes_L2.pt
if [ ! -f $WEIGHTS ]; then
  echo "[$(date)] === PHASE 2: train SAGE-PAP ===" | tee -a $LOG
  ulimit -v 200000000  # 200 GB virtual mem cap
  timeout 7200 python scripts/exp2_train.py OGB_MAG --metapath rev_writes,writes --depth 2 --epochs 100 --partition-json results/OGB_MAG/partition.json --seed 42 --arch SAGE 2>&1 | tee -a $LOG
  if [ ! -f $WEIGHTS ]; then
    echo "[$(date)] PHASE 2 FAILED — no weights file. Phase 3 ABORTED." | tee -a $LOG
    exit 0
  fi
fi

# Phase 3: full k-sweep per seed.
# One exp3 invocation per seed with all 5 k values, so Exact materialises once
# per seed instead of 5 times. --timeout 3600 (1 h) covers OGB-MAG's ~21 min
# Exact materialise.
echo "[$(date)] === PHASE 3: k-sweep ===" | tee -a $LOG
for seed in 42 43 44 45 46; do
  # Skip if all 5 k JSONs for this seed already exist
  count=0
  for k in 8 16 32 64 128; do
    JSON=results/approach_a_2026_05_07/OGB_MAG/SAGE/rev_writes_writes_seed${seed}_k${k}.json
    [ -f $JSON ] && count=$((count+1))
  done
  if [ $count -eq 5 ]; then
    echo "[$(date)] [SKIP] seed=$seed (all 5 k JSONs exist)" | tee -a $LOG
    continue
  fi
  echo "[$(date)] [RUN] seed=$seed (all k in one invocation, count_existing=$count)" | tee -a $LOG
  timeout 5400 python scripts/exp3_inference.py OGB_MAG --metapath rev_writes,writes --depth 2 --k-values 8 16 32 64 128 --partition-json results/OGB_MAG/partition.json --weights-dir results/OGB_MAG/weights --output results/OGB_MAG/master_results_v2.csv --arch SAGE --skip-mprw --run-id $seed --hash-seed $seed --timeout 3600 2>&1 | tail -50 | tee -a $LOG
done

echo "[$(date)] OGB-MAG v2 pipeline DONE" | tee -a $LOG
