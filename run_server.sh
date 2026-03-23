#!/usr/bin/env bash
# Run all extension experiments on the server.
# Must run setup_server.sh first.
#
# Usage:
#   ./run_server.sh                         # full run, all datasets
#   ./run_server.sh --max-metapaths 3       # quick test (3 metapaths per dataset)
#   ./run_server.sh --epochs 50 --k 16      # custom SAGE / sketch settings
#
# Results saved to: results/<DATASET>/extension.csv
# Resume-safe: interrupted runs pick up where they left off.

set -e
source .venv/bin/activate
export PYTHONUTF8=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # override with e.g. CUDA_VISIBLE_DEVICES=1

# --- Defaults (override via flags below) ---
MAX_METAPATHS=500
EPOCHS=50
K=32
FRACTIONS="0.05 0.1 0.2 0.3 0.4"
MIN_CONF=0.1
TIMEOUT=1800
MAX_ADJ_MB=50000      # 50 GB cap — prevent OOM-killing on shared server
NUM_CPU_THREADS=2

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-metapaths) MAX_METAPATHS="$2"; shift 2 ;;
        --epochs)        EPOCHS="$2";        shift 2 ;;
        --k)             K="$2";             shift 2 ;;
        --min-conf)      MIN_CONF="$2";      shift 2 ;;
        --timeout)       TIMEOUT="$2";       shift 2 ;;
        --threads)       NUM_CPU_THREADS="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Datasets to run (OAG_CS requires: pip install H2GB — first load downloads ~2GB)
DATASETS=(OAG_CS OGB_MAG)

COMMON_ARGS=(
    --max-metapaths "$MAX_METAPATHS"
    --min-conf      "$MIN_CONF"
    --fractions     $FRACTIONS
    --epochs        "$EPOCHS"
    --k             "$K"
    --timeout       "$TIMEOUT"
    --max-adj-mb    "$MAX_ADJ_MB"
    --num-cpu-threads "$NUM_CPU_THREADS"
    --cpu
)

echo "=================================================="
echo "  Extension Experiments — Server Run"
echo "  datasets:      ${DATASETS[*]}"
echo "  max-metapaths: $MAX_METAPATHS"
echo "  epochs:        $EPOCHS   k: $K"
echo "  fractions:     $FRACTIONS"
echo "  timeout:       ${TIMEOUT}s per C++ call"
echo "=================================================="
echo ""

for DATASET in "${DATASETS[@]}"; do
    echo "--------------------------------------------------"
    echo "  Dataset: $DATASET"
    echo "--------------------------------------------------"
    python scripts/run_extension_experiments.py "$DATASET" "${COMMON_ARGS[@]}"
    echo ""
done

echo "=================================================="
echo "  All done. Results in results/<DATASET>/extension.csv"
echo "=================================================="
