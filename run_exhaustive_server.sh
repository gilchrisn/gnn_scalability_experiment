#!/usr/bin/env bash
# Full paper reproduction + extension experiments on the server.
#
# Part 1: Base paper reproduction (Table III, IV, Figures 4, 5, 6) on ALL datasets
# Part 2: Extension experiments (exact vs KMV + GNN) on ALL datasets
#
# Resume-safe: interrupted runs pick up where they left off.
# No set -e: individual datasets can fail without stopping the whole run.

set -uo pipefail
source .venv/bin/activate
export PYTHONUTF8=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

TIMEOUT=1800
MAX_METAPATHS=10
EPOCHS=50
K=32
MAX_ADJ_MB=5000

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="results/server_run_${TIMESTAMP}.log"
mkdir -p results

echo "==================================================" | tee -a "$LOG"
echo "  Full Server Run — $TIMESTAMP"                      | tee -a "$LOG"
echo "  timeout=${TIMEOUT}s  max_metapaths=${MAX_METAPATHS}" | tee -a "$LOG"
echo "  epochs=${EPOCHS}  k=${K}  max_adj_mb=${MAX_ADJ_MB}" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

# =====================================================
# Part 1: Base paper reproduction (ALL datasets)
#   Table III, IV, Figures 4, 5, 6
#   ExactD may timeout on large datasets — that's the point
# =====================================================
echo "" | tee -a "$LOG"
echo "=== Part 1: Base Paper Reproduction ===" | tee -a "$LOG"

for DS in HGB_ACM HGB_DBLP HGB_IMDB OGB_MAG OAG_CS; do
    echo "" | tee -a "$LOG"
    echo "--- $DS (Table III/IV, Figures 4-6) ---" | tee -a "$LOG"
    python scripts/run_paper_experiments.py "$DS" \
        --max-metapaths "$MAX_METAPATHS" \
        --mining-timeout 10 \
        --timeout "$TIMEOUT" \
        2>&1 | tee -a "$LOG" || echo "  [WARN] $DS failed, continuing..." | tee -a "$LOG"
done

# =====================================================
# Part 2: Extension experiments (ALL datasets)
#   HGB: fractions 0.2-1.0
#   OGB_MAG: 40% as full dataset
#   OAG_CS: 20% as full dataset
# =====================================================
echo "" | tee -a "$LOG"
echo "=== Part 2: Extension Experiments ===" | tee -a "$LOG"

for DS in HGB_ACM HGB_DBLP HGB_IMDB; do
    echo "" | tee -a "$LOG"
    echo "--- $DS (extension, fractions 0.2-1.0) ---" | tee -a "$LOG"
    python scripts/run_extension_experiments.py "$DS" \
        --fractions 0.2 0.4 0.6 0.8 1.0 \
        --epochs "$EPOCHS" --k "$K" --timeout "$TIMEOUT" \
        --max-adj-mb "$MAX_ADJ_MB" --num-cpu-threads 2 \
        --cpu \
        2>&1 | tee -a "$LOG" || echo "  [WARN] $DS extension failed, continuing..." | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "--- OGB_MAG (40% as full, 5 snapshots) ---" | tee -a "$LOG"
python scripts/run_extension_experiments.py OGB_MAG \
    --fractions 0.08 0.16 0.24 0.32 0.40 \
    --epochs "$EPOCHS" --k "$K" --timeout "$TIMEOUT" \
    --max-adj-mb "$MAX_ADJ_MB" --num-cpu-threads 2 \
    --cpu \
    2>&1 | tee -a "$LOG" || echo "  [WARN] OGB_MAG extension failed" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- OAG_CS (20% as full, 5 snapshots) ---" | tee -a "$LOG"
python scripts/run_extension_experiments.py OAG_CS \
    --fractions 0.04 0.08 0.12 0.16 0.20 \
    --epochs "$EPOCHS" --k "$K" --timeout "$TIMEOUT" \
    --max-adj-mb "$MAX_ADJ_MB" --num-cpu-threads 2 \
    --cpu \
    2>&1 | tee -a "$LOG" || echo "  [WARN] OAG_CS extension failed" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
echo "  All done. $(date)" | tee -a "$LOG"
echo "  Part 1: results/<DATASET>/table4.csv, figure4-6.csv" | tee -a "$LOG"
echo "  Part 2: results/<DATASET>/extension.csv" | tee -a "$LOG"
echo "  Full log: $LOG" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
