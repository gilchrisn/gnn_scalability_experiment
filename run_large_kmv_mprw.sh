#!/usr/bin/env bash
# run_large_kmv_mprw.sh
# KMV + MPRW inference sweep for OGB_MAG and OAG_CS across 5 seeds.
#
# Assumes:
#   - Exact rows already in CSV  (exact results already recorded)
#
# Step 1 (base seed 42): exp1 partition → exp2 weight init (0 epochs) → exp3 KMV+MPRW.
# Step 2 (×4):           exp3 with --hash-seed only (partition + weights reused).
#
# --skip-exact is used on every run — ExactD is never re-materialized.
#
# Usage:
#   bash run_large_kmv_mprw.sh               # both datasets
#   bash run_large_kmv_mprw.sh OGB_MAG       # single dataset
#   bash run_large_kmv_mprw.sh OAG_CS        # single dataset

set -uo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_SEED=42
HASH_SEEDS=(43 44 45 46)   # 4 extra seeds → 5 total replicates

TRAIN_FRAC=0.4
EPOCHS=0           # 0 = init weights without training (saves time on large graphs)
DEPTHS="1 2 3 4"
K_VALUES="2 4 8 16 32"
TIMEOUT=3600
MAX_RSS_GB=100

if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=("OGB_MAG" "OAG_CS")
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Fixed metapaths
# ---------------------------------------------------------------------------
metapath_for() {
    local DS="$1"
    case "$DS" in
        OGB_MAG) echo "cites,rev_cites" ;;
        OAG_CS)  echo "PP_cite,rev_PP_cite" ;;
        *) log "ERROR: unknown dataset '${DS}'"; exit 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Helper: run exp3 for all datasets with a given hash seed (or none for base)
# ---------------------------------------------------------------------------
run_exp3_all() {
    local HASH_SEED="${1:-}"   # empty = base run (uses seed from partition.json)

    for DS in "${DATASETS[@]}"; do
        PART_JSON="results/${DS}/partition.json"
        MP=$(metapath_for "${DS}")
        local SEED_LABEL="${HASH_SEED:-base}"
        log "--- DS=${DS}  metapath=${MP}  seed=${SEED_LABEL} ---"

        HASH_ARG=""
        [[ -n "${HASH_SEED}" ]] && HASH_ARG="--hash-seed ${HASH_SEED}"

        python scripts/exp3_inference.py "${DS}" \
            --metapath "${MP}" \
            --depth ${DEPTHS} \
            --k-values ${K_VALUES} \
            --partition-json "${PART_JSON}" \
            --weights-dir "results/${DS}/weights" \
            --timeout "${TIMEOUT}" \
            --max-rss-gb "${MAX_RSS_GB}" \
            --skip-exact \
            ${HASH_ARG} \
        || log "WARNING: exp3 failed for ${DS} / ${MP} seed=${SEED_LABEL} (exit $?) — continuing"
    done
}

# ---------------------------------------------------------------------------
# Helper: merge per-dataset CSVs into results/master_results.csv
# ---------------------------------------------------------------------------
merge_csvs() {
    local MERGED="results/master_results.csv"
    local FIRST=1
    for DS in "${DATASETS[@]}"; do
        local DS_CSV="results/${DS}/master_results.csv"
        [[ -f "${DS_CSV}" ]] || continue
        if [[ ${FIRST} -eq 1 ]]; then
            cp "${DS_CSV}" "${MERGED}"
            FIRST=0
        else
            tail -n +2 "${DS_CSV}" >> "${MERGED}"
        fi
    done
}

# ---------------------------------------------------------------------------
# Helper: archive results for a seed (excludes weights/ and inf_scratch/)
# ---------------------------------------------------------------------------
archive_results() {
    local SEED="$1"
    local ARCHIVE="results_${SEED}"
    mkdir -p "${ARCHIVE}"
    for DS in "${DATASETS[@]}"; do
        [[ -d "results/${DS}" ]] || continue
        mkdir -p "${ARCHIVE}/${DS}"
        rsync -a --exclude="weights/" --exclude="inf_scratch/" \
            "results/${DS}/" "${ARCHIVE}/${DS}/"
    done
    [[ -f "results/master_results.csv" ]] && cp "results/master_results.csv" "${ARCHIVE}/"
    log "Results archived → ${ARCHIVE}/"
}

# ---------------------------------------------------------------------------
# Helper: clear per-run state (CSV, logs, MPRW work dirs) before each replicate
# ---------------------------------------------------------------------------
clear_run_state() {
    for DS in "${DATASETS[@]}"; do
        rm -f "results/${DS}/master_results.csv"
        rm -f "results/${DS}/run_exp3_"*.log
        # Remove density-matched walk files so MPRW re-runs with the new hash seed
        FOLDER=$(python -c "from src.config import config; print(config.get_folder_name('${DS}'))")
        rm -rf "${FOLDER}/mprw_work"
    done
    rm -f "results/master_results.csv"
}

# ---------------------------------------------------------------------------
# Step 1 — Base seed: train (0 epochs = init weights only), then KMV+MPRW
# ---------------------------------------------------------------------------
log "========================================"
log "STEP 1: Base run (seed=${BASE_SEED})"
log "========================================"

# Exp 1: partition (idempotent — overwrites partition.json with same seed).
for DS in "${DATASETS[@]}"; do
    TARGET_TYPE=$(python -c "from src.config import config; print(config.get_dataset_config('${DS}').target_node)")
    log "--- EXP1: ${DS}  target=${TARGET_TYPE}  train_frac=${TRAIN_FRAC} ---"
    python scripts/exp1_partition.py \
        --dataset "${DS}" \
        --target-type "${TARGET_TYPE}" \
        --train-frac "${TRAIN_FRAC}" \
        --seed "${BASE_SEED}" \
    || { log "ERROR: exp1 failed for ${DS} (exit $?)"; exit 1; }
done

# Exp 2: initialize weights (--epochs 0 skips training loop, just saves θ_init).
# exp2 is idempotent — it checks training_log.csv and skips already-done (mp, L) pairs.
for DS in "${DATASETS[@]}"; do
    PART_JSON="results/${DS}/partition.json"
    MP=$(metapath_for "${DS}")

    # Skip if weights already exist for all depths.
    ALL_EXIST=1
    for L in ${DEPTHS}; do
        MP_SAFE="${MP//,/_}"
        [[ -f "results/${DS}/weights/${MP_SAFE}_L${L}.pt" ]] || { ALL_EXIST=0; break; }
    done
    if [[ ${ALL_EXIST} -eq 1 ]]; then
        log "--- EXP2 ${DS}: weights already present — skipping"
        continue
    fi

    log "--- EXP2: ${DS}  metapath=${MP}  epochs=${EPOCHS} ---"
    python scripts/exp2_train.py "${DS}" \
        --metapath "${MP}" \
        --depth ${DEPTHS} \
        --epochs "${EPOCHS}" \
        --partition-json "${PART_JSON}" \
        --seed "${BASE_SEED}" \
    || log "WARNING: exp2 failed for ${DS} / ${MP} (exit $?) — continuing"
done

run_exp3_all ""
merge_csvs
archive_results "${BASE_SEED}"

# ---------------------------------------------------------------------------
# Step 2 — Extra hash seeds (inference-only replicates)
# ---------------------------------------------------------------------------
for HASH_SEED in "${HASH_SEEDS[@]}"; do
    log ""
    log "========================================"
    log "STEP 2: Inference replicate (hash_seed=${HASH_SEED})"
    log "========================================"
    clear_run_state
    run_exp3_all "${HASH_SEED}"
    merge_csvs
    archive_results "${HASH_SEED}"
done

log ""
log "All done."
log "  Replicates → results_${BASE_SEED}/  $(for s in "${HASH_SEEDS[@]}"; do printf "results_%s/  " "$s"; done)"
log "  Weights (shared) → results/*/weights/"
