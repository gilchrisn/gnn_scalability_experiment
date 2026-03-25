#!/usr/bin/env bash
# Server equivalent of run_exhaustive.ps1 + run_exhaustive_extension.ps1
#
# Part 1: Base paper reproduction (Table IV, Figure 4) on HGB datasets
# Part 2: Extension experiments on OGB_MAG + OAG_CS
#
# Resume-safe: interrupted runs pick up where they left off.

set -uo pipefail
# NOTE: no set -e — individual datasets can fail without stopping the whole run
source .venv/bin/activate
export PYTHONUTF8=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="results/server_run_${TIMESTAMP}.log"
mkdir -p results

echo "==================================================" | tee -a "$LOG"
echo "  Full Server Run — $TIMESTAMP"                      | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

# =====================================================
# Part 1: Base paper reproduction (HGB datasets)
# =====================================================
HGB_DATASETS=(HGB_ACM HGB_DBLP HGB_IMDB HGB_Freebase)

echo "" | tee -a "$LOG"
echo "=== Part 1: Base Paper Reproduction ===" | tee -a "$LOG"
echo "  Datasets: ${HGB_DATASETS[*]}" | tee -a "$LOG"
echo "  Mode: Table IV + Figure 4 (skip sweeps)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

for DS in "${HGB_DATASETS[@]}"; do
    echo "--- $DS ---" | tee -a "$LOG"
    python scripts/run_paper_experiments.py "$DS" \
        --max-metapaths 10 \
        --mining-timeout 10 \
        --timeout 600 \
        --skip-sweeps \
        2>&1 | tee -a "$LOG" || echo "  [WARN] $DS failed, continuing..." | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

# =====================================================
# Part 2: Extension experiments (large datasets)
# =====================================================
echo "" | tee -a "$LOG"
echo "=== Part 2: Extension Experiments ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# OGB_MAG: 40% as full dataset
echo "--- OGB_MAG (40% as full, 5 snapshots) ---" | tee -a "$LOG"
python scripts/run_extension_experiments.py OGB_MAG \
    --fractions 0.08 0.16 0.24 0.32 0.40 \
    --epochs 50 --k 32 --timeout 1800 \
    --max-adj-mb 5000 --num-cpu-threads 2 \
    --cpu \
    2>&1 | tee -a "$LOG" || echo "  [WARN] OGB_MAG failed" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# OAG_CS: 20% as full dataset
echo "--- OAG_CS (20% as full, 5 snapshots) ---" | tee -a "$LOG"
python scripts/run_extension_experiments.py OAG_CS \
    --fractions 0.04 0.08 0.12 0.16 0.20 \
    --epochs 50 --k 32 --timeout 1800 \
    --max-adj-mb 5000 --num-cpu-threads 2 \
    --cpu \
    2>&1 | tee -a "$LOG" || echo "  [WARN] OAG_CS failed" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "==================================================" | tee -a "$LOG"
echo "  All done. $(date)" | tee -a "$LOG"
echo "  Base paper:  results/<DATASET>/table4.csv, figure4.csv" | tee -a "$LOG"
echo "  Extension:   results/<DATASET>/extension.csv" | tee -a "$LOG"
echo "  Full log:    $LOG" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
