#!/usr/bin/env bash
# run_large_kmv_mprw.sh
# KMV + MPRW inference sweep for OGB_MAG and OAG_CS across 5 seeds.
#
# Step 1 (seed=42):  exp1 → exp2 (0 epochs, weight init only) → exp3 KMV+MPRW.
#                    Archives to results_42/.
# Step 2 (×4):       exp3-only with --hash-seed (same partition + weights).
#                    Archives to results_43/ … results_46/.
#
# --skip-exact on every exp3 call — ExactD results already in CSV.
#
# Usage:
#   bash run_large_kmv_mprw.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_SEED=42
HASH_SEEDS=(43 44 45 46)

TRAIN_FRAC=0.4
EPOCHS=0           # 0 = init weights without training
DEPTHS="1 2 3 4"
K_VALUES="2 4 8 16 32"
TIMEOUT=3600
MAX_RSS_GB=100

DATASETS=("OGB_MAG" "OAG_CS")

# Fixed metapaths and target types — hardcoded to avoid importing torch_geometric
MP_OGB_MAG="cites,rev_cites"
MP_OAG_CS="PP_cite,rev_PP_cite"
TARGET_OGB_MAG="paper"
TARGET_OAG_CS="paper"
FOLDER_OGB_MAG="OGB_MAG"
FOLDER_OAG_CS="OAG_CS"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Step 1 — Full pipeline with BASE_SEED
# ---------------------------------------------------------------------------
log "=== STEP 1: Full pipeline (seed=${BASE_SEED}) ==="

for DS in "${DATASETS[@]}"; do
    PART_JSON="results/${DS}/partition.json"
    mkdir -p "results/${DS}"

    if   [[ "${DS}" == "OGB_MAG" ]]; then MP="${MP_OGB_MAG}"; TARGET_TYPE="${TARGET_OGB_MAG}"
    elif [[ "${DS}" == "OAG_CS"  ]]; then MP="${MP_OAG_CS}";  TARGET_TYPE="${TARGET_OAG_CS}"
    fi

    log "=== EXP1: ${DS} ==="
    python scripts/exp1_partition.py \
        --dataset "${DS}" \
        --target-type "${TARGET_TYPE}" \
        --train-frac "${TRAIN_FRAC}" \
        --seed "${BASE_SEED}"

    [[ -f "${PART_JSON}" ]] || { log "ERROR: partition.json missing for ${DS}"; exit 1; }

    log "--- EXP2: ${DS}  metapath=${MP}  epochs=${EPOCHS} ---"
    python scripts/exp2_train.py "${DS}" \
        --metapath "${MP}" \
        --depth ${DEPTHS} \
        --epochs "${EPOCHS}" \
        --partition-json "${PART_JSON}" \
        --seed "${BASE_SEED}" \
    || { log "WARNING: exp2 failed for ${DS} / ${MP} (exit $?) — continuing"; }

    log "--- EXP3: ${DS}  metapath=${MP}  k=${K_VALUES} ---"
    python scripts/exp3_inference.py "${DS}" \
        --metapath "${MP}" \
        --depth ${DEPTHS} \
        --k-values ${K_VALUES} \
        --partition-json "${PART_JSON}" \
        --weights-dir "results/${DS}/weights" \
        --timeout "${TIMEOUT}" \
        --max-rss-gb "${MAX_RSS_GB}" \
        --skip-exact \
    || log "WARNING: exp3 failed for ${DS} / ${MP} (exit $?) — continuing"

    log "=== DONE: ${DS} ==="
    echo ""
done

# Archive base results (keep weights/ and partition.json in place for step 2)
mkdir -p "results_${BASE_SEED}"
for DS in "${DATASETS[@]}"; do
    [[ -d "results/${DS}" ]] || continue
    mkdir -p "results_${BASE_SEED}/${DS}"
    rsync -a --exclude="weights/" --exclude="inf_scratch/" "results/${DS}/" "results_${BASE_SEED}/${DS}/"
done
[[ -f "results/master_results.csv" ]] && cp "results/master_results.csv" "results_${BASE_SEED}/"
log "Base results archived → results_${BASE_SEED}/"

# ---------------------------------------------------------------------------
# Step 2 — Inference-only replicates with different hash seeds
# ---------------------------------------------------------------------------
for HASH_SEED in "${HASH_SEEDS[@]}"; do
    log ""
    log "=== STEP 2: Inference replicate (hash_seed=${HASH_SEED}) ==="

    for DS in "${DATASETS[@]}"; do
        rm -f "results/${DS}/master_results.csv"
        rm -f "results/${DS}/run_exp3_"*.log
        if   [[ "${DS}" == "OGB_MAG" ]]; then FOLDER="${FOLDER_OGB_MAG}"
        elif [[ "${DS}" == "OAG_CS"  ]]; then FOLDER="${FOLDER_OAG_CS}"
        fi
        rm -rf "${FOLDER}/mprw_work"
    done
    rm -f "results/master_results.csv"

    for DS in "${DATASETS[@]}"; do
        PART_JSON="results/${DS}/partition.json"

        if   [[ "${DS}" == "OGB_MAG" ]]; then MP="${MP_OGB_MAG}"
        elif [[ "${DS}" == "OAG_CS"  ]]; then MP="${MP_OAG_CS}"
        fi

        log "--- EXP3: ${DS}  metapath=${MP}  hash_seed=${HASH_SEED} ---"

        python scripts/exp3_inference.py "${DS}" \
            --metapath "${MP}" \
            --depth ${DEPTHS} \
            --k-values ${K_VALUES} \
            --partition-json "${PART_JSON}" \
            --weights-dir "results/${DS}/weights" \
            --timeout "${TIMEOUT}" \
            --max-rss-gb "${MAX_RSS_GB}" \
            --skip-exact \
            --hash-seed "${HASH_SEED}" \
        || log "WARNING: exp3 failed for ${DS} / ${MP} (exit $?) — continuing"
    done

    # Merge per-dataset CSVs
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

    mkdir -p "results_${HASH_SEED}"
    for DS in "${DATASETS[@]}"; do
        [[ -d "results/${DS}" ]] || continue
        mkdir -p "results_${HASH_SEED}/${DS}"
        rsync -a --exclude="weights/" --exclude="inf_scratch/" "results/${DS}/" "results_${HASH_SEED}/${DS}/"
    done
    [[ -f "${MERGED}" ]] && cp "${MERGED}" "results_${HASH_SEED}/"
    log "Replicate archived → results_${HASH_SEED}/"

done

log ""
log "All done."
log "  Weights (shared) → results/*/weights/"
log "  Replicates       → results_${BASE_SEED}/  results_43/  results_44/  results_45/  results_46/"
