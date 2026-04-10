#!/usr/bin/env bash
# rerun_mprw.sh — Regenerate only MPRW rows for all seed archives using the
# new over-sample-and-truncate code.  Exact and KMV rows are preserved.
#
# For each results_{SEED}/:
#   1. Strip MPRW rows from its master_results.csv
#   2. Copy stripped CSV into results/ (so exp3 skips already-done rows)
#   3. Run exp3 with the correct hash seed
#   4. Copy updated results back into results_{SEED}/
#
# Weights and partition.json stay in results/ throughout — no retraining.
#
# Usage:
#   bash rerun_mprw.sh

set -euo pipefail

# Seed 42 = base run (no --hash-seed flag; uses partition.json's hash_seed)
# Seeds 43-46 = Step 2 replicates (--hash-seed passed directly)
BASE_SEED=42
HASH_SEEDS=(43 44 45 46)

DATASETS=("HGB_ACM" "HGB_DBLP" "HGB_IMDB")
DEPTHS="2 3 4"
K_VALUES="2 4 8 16 32"
MAX_RSS_GB=32

log() { echo "[$(date '+%H:%M:%S')] $*"; }

strip_mprw() {
    local csv="$1"
    [[ -f "${csv}" ]] || return 0
    grep -v ",MPRW," "${csv}" > "${csv}.tmp" && mv "${csv}.tmp" "${csv}"
}

run_exp3_for_seed() {
    local archive_dir="$1"   # e.g. results_42
    local hash_seed="$2"     # empty string = base run, else the override value

    log "--- Processing ${archive_dir} (hash_seed=${hash_seed:-from_partition}) ---"

    for DS in "${DATASETS[@]}"; do
        local archive_csv="${archive_dir}/${DS}/master_results.csv"
        local working_csv="results/${DS}/master_results.csv"
        local part_json="results/${DS}/partition.json"

        [[ -f "${part_json}" ]] || { log "WARNING: ${part_json} missing — skipping ${DS}"; continue; }

        # Strip MPRW rows from the archive copy
        strip_mprw "${archive_csv}"

        # Put the stripped CSV into the working dir so exp3 skips KMV/Exact
        cp "${archive_csv}" "${working_csv}"

        METAPATHS=$(python -c "
from src.config import config
cfg = config.get_dataset_config('${DS}')
for p in cfg.suggested_paths:
    print(p)
")

        while IFS= read -r MP; do
            [[ -z "${MP}" ]] && continue
            log "  DS=${DS}  metapath=${MP}"

            local hash_arg=""
            [[ -n "${hash_seed}" ]] && hash_arg="--hash-seed ${hash_seed}"

            python scripts/exp3_inference.py "${DS}" \
                --metapath "${MP}" \
                --depth ${DEPTHS} \
                --k-values ${K_VALUES} \
                --partition-json "${part_json}" \
                --weights-dir "results/${DS}/weights" \
                --max-rss-gb ${MAX_RSS_GB} \
                ${hash_arg} \
            || log "WARNING: exp3 failed for ${DS}/${MP} (exit $?) — continuing"

        done <<< "${METAPATHS}"

        # Copy updated CSV back into the archive
        cp "${working_csv}" "${archive_csv}"
    done

    # Rebuild merged master_results.csv in archive
    local merged="${archive_dir}/master_results.csv"
    local first=1
    for DS in "${DATASETS[@]}"; do
        local ds_csv="${archive_dir}/${DS}/master_results.csv"
        [[ -f "${ds_csv}" ]] || continue
        if [[ ${first} -eq 1 ]]; then
            cp "${ds_csv}" "${merged}"
            first=0
        else
            tail -n +2 "${ds_csv}" >> "${merged}"
        fi
    done

    log "Done: ${archive_dir}/"
}

# Base run (seed=42): no --hash-seed, uses partition.json's hash_seed
run_exp3_for_seed "results_${BASE_SEED}" ""

# Replicate runs
for SEED in "${HASH_SEEDS[@]}"; do
    run_exp3_for_seed "results_${SEED}" "${SEED}"
done

log "All MPRW rows regenerated."
