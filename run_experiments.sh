#!/usr/bin/env bash
# run_experiments.sh
# Full 4-script pipeline for HGB_ACM, HGB_DBLP, HGB_IMDB.
#
# Usage:
#   bash run_experiments.sh               # all defaults
#   bash run_experiments.sh --epochs 50   # override epochs for exp2
#   bash run_experiments.sh --k-values 8 16 32 64 128
#
# Steps per dataset:
#   1. exp1_inductive_split.py  → results/<DS>/partition.json
#   2. exp2_train.py            → results/<DS>/weights/  (one .pt per metapath×L)
#   3. exp3_inference.py        → results/<DS>/master_results.csv
#   4. exp4_visualize.py        → figures/
#
# Metapaths are read from config.suggested_paths via Python one-liner.
# Extra args forwarded to exp2 and exp3 via EXTRA_TRAIN_ARGS / EXTRA_INFER_ARGS.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults — override via positional env or CLI style args
# ---------------------------------------------------------------------------
# DATASETS=("HGB_ACM" "HGB_DBLP" "HGB_IMDB" "OAG_CS" "OGB_MAG")
DATASETS=("HNE_PubMed")
TRAIN_FRAC=0.4
DEPTHS="2 3 4"
K_VALUES="2 4 8 16 32"
EPOCHS=100
SEED=42
MAX_RSS_GB=32        # e.g. 100  — leave empty to disable RSS guard

EXTRA_TRAIN_ARGS=""
EXTRA_INFER_ARGS=""

# Parse remaining args as --key value pairs forwarded to sub-scripts
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)      EPOCHS="$2";           shift 2 ;;
        --k-values)    K_VALUES="${@:2}";     break   ;;
        --train-frac)  TRAIN_FRAC="$2";       shift 2 ;;
        --seed)        SEED="$2";             shift 2 ;;
        --max-rss-gb)  MAX_RSS_GB="$2";       shift 2 ;;
        *)             EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS $1"; shift ;;
    esac
done

# Thread --max-rss-gb into exp3 if set
if [[ -n "${MAX_RSS_GB}" ]]; then
    EXTRA_INFER_ARGS="${EXTRA_INFER_ARGS} --max-rss-gb ${MAX_RSS_GB}"
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for DS in "${DATASETS[@]}"; do
    PART_JSON="results/${DS}/partition.json"

    # -----------------------------------------------------------------------
    # Exp 1 — Temporal / stratified partition
    # -----------------------------------------------------------------------
    log "=== EXP1: ${DS} ==="

    # Read target node type and metapaths from config (no hardcoding)
    TARGET_TYPE=$(python -c "from src.config import config; print(config.get_dataset_config('${DS}').target_node)")
    METAPATHS=$(python -c "
from src.config import config
cfg = config.get_dataset_config('${DS}')
for p in cfg.suggested_paths:
    print(p)
")

    python scripts/exp1_partition.py \
        --dataset "${DS}" \
        --target-type "${TARGET_TYPE}" \
        --train-frac "${TRAIN_FRAC}" \
        --seed "${SEED}"

    [[ -f "${PART_JSON}" ]] || { echo "ERROR: partition.json missing for ${DS}"; exit 1; }
    log "partition.json ready: ${PART_JSON}"

    # -----------------------------------------------------------------------
    # Dummy + MLP baselines (once per dataset, no metapath needed)
    # -----------------------------------------------------------------------
    log "--- BASELINES: ${DS} ---"
    python scripts/dummy_baseline.py "${DS}" \
        --strategies most_frequent prior mlp \
        --epochs 200 \
        --seed "${SEED}" \
        --master-csv "results/${DS}/master_results.csv" \
    || log "WARNING: dummy_baseline failed for ${DS} (exit $?) — continuing"

    # -----------------------------------------------------------------------
    # Exp 2 + 3 — per metapath
    # -----------------------------------------------------------------------
    while IFS= read -r MP; do
        [[ -z "${MP}" ]] && continue

        log "--- EXP2: ${DS}  metapath=${MP}  depth=${DEPTHS} ---"
        python scripts/exp2_train.py "${DS}" \
            --metapath "${MP}" \
            --depth ${DEPTHS} \
            --epochs "${EPOCHS}" \
            --partition-json "${PART_JSON}" \
            --seed "${SEED}" \
            ${EXTRA_TRAIN_ARGS} \
        || { log "WARNING: exp2 failed for ${DS} / ${MP} (exit $?) — continuing"; continue; }

        log "--- EXP3: ${DS}  metapath=${MP}  k=${K_VALUES} ---"
        python scripts/exp3_inference.py "${DS}" \
            --metapath "${MP}" \
            --depth ${DEPTHS} \
            --k-values ${K_VALUES} \
            --partition-json "${PART_JSON}" \
            --weights-dir "results/${DS}/weights" \
            --max-rss-gb ${MAX_RSS_GB} \
            ${EXTRA_INFER_ARGS} \
        || log "WARNING: exp3 failed for ${DS} / ${MP} (exit $?) — continuing"

    done <<< "${METAPATHS}"

    log "=== DONE: ${DS} ==="
    echo ""
done

# ---------------------------------------------------------------------------
# Exp 4 — Visualize
# ---------------------------------------------------------------------------
log "=== EXP4: Visualize ==="
# Merge per-dataset CSVs into one (simple cat — headers only from first file)
MERGED="results/master_results.csv"
FIRST=1
for DS in "${DATASETS[@]}"; do
    DS_CSV="results/${DS}/master_results.csv"
    [[ -f "${DS_CSV}" ]] || continue
    if [[ ${FIRST} -eq 1 ]]; then
        cp "${DS_CSV}" "${MERGED}"
        FIRST=0
    else
        tail -n +2 "${DS_CSV}" >> "${MERGED}"
    fi
done

[[ -f "${MERGED}" ]] && python scripts/exp4_visualize.py \
    --results "${MERGED}" \
    --output-dir figures/

log "All done."
log "  Weights  → results/*/weights/"
log "  Results  → results/*/master_results.csv"
log "  Merged   → results/master_results.csv"
log "  Figures  → figures/"
