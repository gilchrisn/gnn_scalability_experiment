#!/usr/bin/env bash
# rerun_mlp.sh — Regenerate only MLP/most_frequent/prior baseline rows for all
# seed archives now that dummy_baseline.py supports single-label datasets.
# Exact, KMV, and MPRW rows are preserved.
#
# For each results_{SEED}/:
#   1. Strip baseline rows (most_frequent, prior, mlp) from its master_results.csv
#   2. Copy stripped CSV into results/ (so exp3 skips already-done rows)
#   3. Run dummy_baseline.py with the correct seed
#   4. Copy updated results back into results_{SEED}/
#
# Usage:
#   bash rerun_mlp.sh

set -euo pipefail

BASE_SEED=42
HASH_SEEDS=(43 44 45 46)

DATASETS=("HGB_ACM" "HGB_DBLP" "HGB_IMDB")

log() { echo "[$(date '+%H:%M:%S')] $*"; }

strip_baselines() {
    local csv="$1"
    [[ -f "${csv}" ]] || return 0
    grep -vE ",(most_frequent|prior|mlp)," "${csv}" > "${csv}.tmp" && mv "${csv}.tmp" "${csv}"
}

run_baselines_for_seed() {
    local archive_dir="$1"   # e.g. results_42
    local seed="$2"          # numeric seed passed to --seed

    log "--- Processing ${archive_dir} (seed=${seed}) ---"

    for DS in "${DATASETS[@]}"; do
        local archive_csv="${archive_dir}/${DS}/master_results.csv"
        local working_csv="results/${DS}/master_results.csv"

        # Strip old baseline rows from the archive copy
        strip_baselines "${archive_csv}"

        # Put the stripped CSV into the working dir
        cp "${archive_csv}" "${working_csv}"

        log "  DS=${DS}  seed=${seed}"

        python scripts/dummy_baseline.py "${DS}" \
            --strategies most_frequent prior mlp \
            --epochs 200 \
            --seed "${seed}" \
            --master-csv "${working_csv}" \
        || log "WARNING: dummy_baseline failed for ${DS} (exit $?) — continuing"

        # Copy updated CSV back into the archive
        cp "${working_csv}" "${archive_csv}"
    done

    # Rebuild merged master_results.csv in archive root
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

# Base run (seed=42)
run_baselines_for_seed "results_${BASE_SEED}" "${BASE_SEED}"

# Replicate runs
for SEED in "${HASH_SEEDS[@]}"; do
    run_baselines_for_seed "results_${SEED}" "${SEED}"
done

log "All baseline rows regenerated."
