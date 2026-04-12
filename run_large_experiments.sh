#!/usr/bin/env bash
# run_large_experiments.sh
# Extension pipeline for OGB_MAG and OAG_CS.
#
# EXACT behaviour (--skip-exact-inference):
#   Exact: C++ ExactD materialization only — records time/edge count/RAM.
#          No GNN forward pass, no tensor files written to disk.
#   KMV:   Full pipeline (mat + GNN inference) across all k values.
#   MPRW:  Full pipeline (density-matched walks + GNN inference).
#
# OGB_MAG: only metapath "cites,rev_cites" (densest, ~13B edges).
# OAG_CS:  all config metapaths — Exact mat-only acts as a density scan.
#           Check results/OAG_CS/master_results.csv Edge_Count column
#           to pick the densest, then run exp2+exp3 for that one alone.
#
# Usage:
#   bash run_large_experiments.sh               # both datasets
#   bash run_large_experiments.sh OGB_MAG       # single dataset
#   bash run_large_experiments.sh OAG_CS        # density scan only
#
# After the OAG_CS density scan you can train + infer for the winner with:
#   python scripts/exp2_train.py OAG_CS \
#       --metapath <densest_mp> --depth 2 3 4 --epochs 100 \
#       --partition-json results/OAG_CS/partition.json
#   python scripts/exp3_inference.py OAG_CS \
#       --metapath <densest_mp> --depth 2 3 4 \
#       --k-values 2 4 8 16 32 \
#       --partition-json results/OAG_CS/partition.json \
#       --skip-exact-inference

set -uo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_FRAC=0.4
DEPTHS="2 3 4"
K_VALUES="2 4 8 16 32"
EPOCHS=100
SEED=42
MAX_RSS_GB=32
TIMEOUT=3600        # 1 h per materialization subprocess

# Which datasets to run. Override with positional args.
if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=("OGB_MAG" "OAG_CS")
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Per-dataset metapath override
# ---------------------------------------------------------------------------
# Returns newline-separated metapaths for a given dataset.
metapaths_for() {
    local DS="$1"
    case "$DS" in
        OGB_MAG)
            # Only the bibliographic-coupling path — user-confirmed densest.
            echo "cites,rev_cites"
            ;;
        OAG_CS)
            # All config metapaths — Exact mat-only acts as density scan.
            python -c "
from src.config import config
cfg = config.get_dataset_config('OAG_CS')
for p in cfg.suggested_paths:
    print(p)
"
            ;;
        *)
            # Fallback: read from config.
            python -c "
from src.config import config
cfg = config.get_dataset_config('${DS}')
for p in cfg.suggested_paths:
    print(p)
"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for DS in "${DATASETS[@]}"; do
    PART_JSON="results/${DS}/partition.json"
    mkdir -p "results/${DS}"

    log "========================================"
    log "Dataset: ${DS}"
    log "========================================"

    TARGET_TYPE=$(python -c "from src.config import config; print(config.get_dataset_config('${DS}').target_node)")

    # --- Exp 1: Partition ---
    log "--- EXP1: partition ---"
    python scripts/exp1_partition.py \
        --dataset "${DS}" \
        --target-type "${TARGET_TYPE}" \
        --train-frac "${TRAIN_FRAC}" \
        --seed "${SEED}"

    [[ -f "${PART_JSON}" ]] || { log "ERROR: partition.json missing for ${DS}"; exit 1; }
    log "partition.json ready."

    # --- Baselines (dummy + MLP) ---
    log "--- BASELINES ---"
    python scripts/dummy_baseline.py "${DS}" \
        --strategies most_frequent prior mlp \
        --epochs 200 \
        --seed "${SEED}" \
        --master-csv "results/${DS}/master_results.csv" \
    || log "WARNING: dummy_baseline failed (exit $?) — continuing"

    METAPATHS=$(metapaths_for "${DS}")

    # --- Exp 2: Train (skip for OAG_CS density scan) ---
    if [[ "${DS}" == "OAG_CS" ]]; then
        log "--- EXP2 SKIPPED for OAG_CS (density scan mode) ---"
        log "    Exact materialization in EXP3 will record edge counts."
        log "    After this run, inspect results/OAG_CS/master_results.csv"
        log "    to find the densest metapath, then run exp2 manually."
    else
        while IFS= read -r MP; do
            [[ -z "${MP}" ]] && continue
            log "--- EXP2: ${DS}  metapath=${MP} ---"
            python scripts/exp2_train.py "${DS}" \
                --metapath "${MP}" \
                --depth ${DEPTHS} \
                --epochs "${EPOCHS}" \
                --partition-json "${PART_JSON}" \
                --seed "${SEED}" \
            || log "WARNING: exp2 failed for ${DS} / ${MP} (exit $?) — continuing"
        done <<< "${METAPATHS}"
    fi

    # --- Exp 3: Inference (Exact mat-only + full KMV + full MPRW) ---
    while IFS= read -r MP; do
        [[ -z "${MP}" ]] && continue
        log "--- EXP3: ${DS}  metapath=${MP}  k=${K_VALUES} ---"
        python scripts/exp3_inference.py "${DS}" \
            --metapath "${MP}" \
            --depth ${DEPTHS} \
            --k-values ${K_VALUES} \
            --partition-json "${PART_JSON}" \
            --weights-dir "results/${DS}/weights" \
            --timeout "${TIMEOUT}" \
            --max-rss-gb "${MAX_RSS_GB}" \
            --skip-exact-inference \
        || log "WARNING: exp3 failed for ${DS} / ${MP} (exit $?) — continuing"
    done <<< "${METAPATHS}"

    log "=== DONE: ${DS} ==="
    echo ""
done

# ---------------------------------------------------------------------------
# Exp 4: Visualize
# ---------------------------------------------------------------------------
log "=== EXP4: Visualize ==="
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
    --output-dir figures/ \
|| log "WARNING: exp4_visualize failed"

log ""
log "All done."
log "  Results  → results/OGB_MAG/master_results.csv"
log "  Results  → results/OAG_CS/master_results.csv   (Edge_Count = density scan)"
log "  Merged   → results/master_results.csv"
log "  Figures  → figures/"
log ""
log "OAG_CS next step: pick densest metapath from Edge_Count column, then:"
log "  python scripts/exp2_train.py OAG_CS --metapath <mp> --depth 2 3 4 \\"
log "      --epochs ${EPOCHS} --partition-json results/OAG_CS/partition.json"
log "  python scripts/exp3_inference.py OAG_CS --metapath <mp> --depth 2 3 4 \\"
log "      --k-values ${K_VALUES} --partition-json results/OAG_CS/partition.json \\"
log "      --skip-exact-inference"
