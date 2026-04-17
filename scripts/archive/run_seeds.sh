#!/usr/bin/env bash
# run_seeds.sh — 100-seed KMV/MPRW inference sweep on HGB_DBLP.
#
# Exp1 + Exp2 run once (fixed partition + weights).
# Exp3 runs 100 times with --hash-seed 0..99.
# Each seed appends to master_results.csv — resume-safe.
#
# Usage:
#   bash run_seeds.sh
#   bash run_seeds.sh --n-seeds 50
#   bash run_seeds.sh --depths "2 3" --k-values "8 16 32"

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DS="HGB_DBLP"
MP="author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"
TARGET_TYPE="author"

TRAIN_FRAC=0.4
DEPTHS="2 3 4"
K_VALUES="2 4 8 16 32"
EPOCHS=100
PARTITION_SEED=42
N_SEEDS=100
MAX_RSS_GB=32

# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-seeds)     N_SEEDS="$2";     shift 2 ;;
        --epochs)      EPOCHS="$2";      shift 2 ;;
        --depths)      DEPTHS="$2";      shift 2 ;;
        --k-values)    K_VALUES="$2";    shift 2 ;;
        --train-frac)  TRAIN_FRAC="$2";  shift 2 ;;
        --max-rss-gb)  MAX_RSS_GB="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PART_JSON="results/${DS}/partition.json"
MASTER_CSV="results/${DS}/master_results.csv"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "================================================================"
log "  ${DS}  |  metapath=${MP}"
log "  ${N_SEEDS} hash-seeds  |  depths=${DEPTHS}  |  k=${K_VALUES}"
log "================================================================"

# ---------------------------------------------------------------------------
# Exp 1 — Partition (once)
# ---------------------------------------------------------------------------
log "=== EXP1: partition ==="
if [[ -f "${PART_JSON}" ]]; then
    log "  [SKIP] ${PART_JSON} exists"
else
    python scripts/exp1_partition.py \
        --dataset "${DS}" \
        --target-type "${TARGET_TYPE}" \
        --train-frac "${TRAIN_FRAC}" \
        --seed "${PARTITION_SEED}"
    log "  [OK] ${PART_JSON}"
fi

# ---------------------------------------------------------------------------
# Exp 2 — Train (once per depth)
# ---------------------------------------------------------------------------
log "=== EXP2: train ==="
python scripts/exp2_train.py "${DS}" \
    --metapath "${MP}" \
    --depth ${DEPTHS} \
    --epochs "${EPOCHS}" \
    --partition-json "${PART_JSON}" \
    --seed "${PARTITION_SEED}"
log "  [OK] weights saved"

# ---------------------------------------------------------------------------
# Exp 3 — Inference × N_SEEDS hash-seeds
# ---------------------------------------------------------------------------
log "=== EXP3: inference x ${N_SEEDS} seeds ==="

FAILED=0
for HSEED in $(seq 0 $((N_SEEDS - 1))); do
    log "--- seed ${HSEED}/${N_SEEDS} ---"
    python scripts/exp3_inference.py "${DS}" \
        --metapath "${MP}" \
        --depth ${DEPTHS} \
        --k-values ${K_VALUES} \
        --partition-json "${PART_JSON}" \
        --weights-dir "results/${DS}/weights" \
        --hash-seed "${HSEED}" \
        --max-rss-gb "${MAX_RSS_GB}" \
    || { log "WARNING: seed ${HSEED} failed (exit $?) — continuing"; FAILED=$((FAILED + 1)); continue; }
done

# ---------------------------------------------------------------------------
# Exp 4 — Visualize
# ---------------------------------------------------------------------------
log "=== EXP4: visualize ==="
if [[ -f "${MASTER_CSV}" ]]; then
    python scripts/exp4_visualize.py \
        --results "${MASTER_CSV}" \
        --output-dir figures/
else
    log "WARNING: no master_results.csv to visualize"
fi

log "================================================================"
log "  DONE — ${N_SEEDS} seeds, ${FAILED} failed"
log "  Results -> ${MASTER_CSV}"
log "  Figures -> figures/"
log "================================================================"
