#!/usr/bin/env bash
# Run all extension experiments on the server.
#
# Treats 40% OGB_MAG and 20% OAG_CS as the "full" datasets.
# Trains on the largest fraction, infers on all fractions.
# Dense metapaths that OOM/timeout on exact are handled gracefully
# (KMV still runs, exact fields left blank in CSV).
#
# Results saved to: results/<DATASET>/extension.csv
# Resume-safe: interrupted runs pick up where they left off.

set -euo pipefail
source .venv/bin/activate
export PYTHONUTF8=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

EPOCHS=50
K=32
TIMEOUT=1800
MAX_ADJ_MB=5000       # 5 GB — enough for PAP at 40% OGB_MAG (2 GB)
NUM_CPU_THREADS=2

# Clean old results so the mask fix takes effect
rm -f results/OGB_MAG/extension.csv results/OAG_CS/extension.csv

echo "=================================================="
echo "  Extension Experiments — Server Overnight Run"
echo "  $(date)"
echo "=================================================="
echo ""

# --- OGB_MAG: treat 40% as full dataset ---
# Fractions relative to full graph: 0.08, 0.16, 0.24, 0.32, 0.40
# = 20%, 40%, 60%, 80%, 100% of the 40% subgraph
echo "=========================================="
echo "  OGB_MAG (40% as full, 5 snapshots)"
echo "=========================================="
python scripts/run_extension_experiments.py OGB_MAG \
    --fractions 0.08 0.16 0.24 0.32 0.40 \
    --epochs "$EPOCHS" --k "$K" --timeout "$TIMEOUT" \
    --max-adj-mb "$MAX_ADJ_MB" --num-cpu-threads "$NUM_CPU_THREADS" \
    --cpu 2>&1 | tee -a results/server_run.log
echo ""

# --- OAG_CS: treat 20% as full dataset ---
# Fractions: 0.04, 0.08, 0.12, 0.16, 0.20
# = 20%, 40%, 60%, 80%, 100% of the 20% subgraph
echo "=========================================="
echo "  OAG_CS (20% as full, 5 snapshots)"
echo "=========================================="
python scripts/run_extension_experiments.py OAG_CS \
    --fractions 0.04 0.08 0.12 0.16 0.20 \
    --epochs "$EPOCHS" --k "$K" --timeout "$TIMEOUT" \
    --max-adj-mb "$MAX_ADJ_MB" --num-cpu-threads "$NUM_CPU_THREADS" \
    --cpu 2>&1 | tee -a results/server_run.log
echo ""

echo "=================================================="
echo "  All done. $(date)"
echo "  Results: results/OGB_MAG/extension.csv"
echo "           results/OAG_CS/extension.csv"
echo "=================================================="
