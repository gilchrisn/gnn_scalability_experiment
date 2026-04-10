#!/usr/bin/env bash
# run_seeds.sh — Gemini replication protocol.
#
# Step 1 (once): Full pipeline with BASE_SEED — trains θ* and produces
#                the base results.  Weights stay in results/ permanently.
#
# Step 2 (×N):   For each extra hash seed, run exp3 ONLY (no retraining,
#                no repartition) using --hash-seed to vary the KMV sketch
#                and MPRW walk seeds.  The same frozen θ* and the same
#                train/test partition are reused every time.
#
# Result layout:
#   results_<BASE_SEED>/   — full run (exp1+exp2+exp3), base inference
#   results_<SEED>/        — exp3-only replicate for each extra hash seed
#   results/               — weights only (kept for all runs)
#
# Usage:
#   bash run_seeds.sh
#   bash run_seeds.sh --epochs 50 --k-values 8 16 32

set -euo pipefail

BASE_SEED=42
HASH_SEEDS=(1001 1002 1003 1004)   # 4 extra seeds → 5 total replicates

# Forward any extra args to run_experiments.sh (exp2/exp3 flags)
EXTRA_ARGS="$@"

DATASETS=("HGB_ACM" "HGB_DBLP" "HGB_IMDB")
DEPTHS="2 3 4"
K_VALUES="2 4 8 16 32"
MAX_RSS_GB=32

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Step 1 — Full pipeline with BASE_SEED (trains θ*, writes partition.json)
# ---------------------------------------------------------------------------
log "=== STEP 1: Full pipeline (seed=${BASE_SEED}) ==="
bash run_experiments.sh --seed "${BASE_SEED}" ${EXTRA_ARGS}

# Archive base results; keep weights and partition.json in place for step 2.
mkdir -p "results_${BASE_SEED}"
for DS in "${DATASETS[@]}"; do
    [[ -d "results/${DS}" ]] || continue
    mkdir -p "results_${BASE_SEED}/${DS}"
    # Copy everything except weights/ into the archive
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

    # Clear previous inference outputs but leave weights/, partition.json, and
    # inf_scratch/ intact.  inf_scratch holds z_exact_L*.pt — reused via done_runs
    # or overwritten deterministically; no need to delete them.
    for DS in "${DATASETS[@]}"; do
        rm -f "results/${DS}/master_results.csv"
        rm -f "results/${DS}/run_exp3_"*.log
        # Clean actual MPRW work directory (mat_mprw_<k>.pt files)
        FOLDER=$(python -c "from src.config import config; print(config.get_folder_name('${DS}'))")
        rm -rf "${FOLDER}/mprw_work"
    done
    rm -f "results/master_results.csv"

    for DS in "${DATASETS[@]}"; do
        PART_JSON="results/${DS}/partition.json"
        [[ -f "${PART_JSON}" ]] || { log "WARNING: ${PART_JSON} missing — skipping ${DS}"; continue; }

        METAPATHS=$(python -c "
from src.config import config
cfg = config.get_dataset_config('${DS}')
for p in cfg.suggested_paths:
    print(p)
")

        while IFS= read -r MP; do
            [[ -z "${MP}" ]] && continue
            log "--- DS=${DS}  metapath=${MP}  hash_seed=${HASH_SEED} ---"
            python scripts/exp3_inference.py "${DS}" \
                --metapath "${MP}" \
                --depth ${DEPTHS} \
                --k-values ${K_VALUES} \
                --partition-json "${PART_JSON}" \
                --weights-dir "results/${DS}/weights" \
                --hash-seed "${HASH_SEED}" \
                --max-rss-gb ${MAX_RSS_GB} \
            || log "WARNING: exp3 failed for ${DS} / ${MP} (exit $?) — continuing"
        done <<< "${METAPATHS}"
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

    # Archive this replicate's results (no weights, no inf_scratch .pt blobs)
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
log "  Replicates       → results_${BASE_SEED}/  results_1001/  results_1002/  ..."
