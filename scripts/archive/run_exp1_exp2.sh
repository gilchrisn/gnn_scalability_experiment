#!/usr/bin/env bash
# run_exp1_exp2.sh
# Runs Experiment 1 (partition) then Experiment 2 (train) for ACM, DBLP, IMDB.
#
# Usage:
#   bash run_exp1_exp2.sh              # all datasets, defaults
#   bash run_exp1_exp2.sh --epochs 50  # pass extra args to exp2
#
# Exp1 args  (fixed): --train-frac 0.1  --seed 42
# Exp2 args  (fixed): --depth 2 3 4     --seed 42
# Any extra args passed to this script are forwarded to exp2.

set -euo pipefail

SEED=42
TRAIN_FRAC=0.1
DEPTHS="2 3 4"
EXTRA_EXP2_ARGS="$*"   # e.g. --epochs 50

# ---------------------------------------------------------------------------
# Dataset → metapath map
# ---------------------------------------------------------------------------

declare -A METAPATHS
METAPATHS["HGB_DBLP"]="author_to_paper,paper_to_author
author_to_paper,paper_to_term,term_to_paper,paper_to_author
author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"

METAPATHS["HGB_ACM"]="paper_to_author,author_to_paper
paper_to_subject,subject_to_paper
paper_to_term,term_to_paper"

METAPATHS["HGB_IMDB"]="movie_to_actor,actor_to_movie
movie_to_director,director_to_movie
movie_to_keyword,keyword_to_movie"

DATASETS=("HGB_ACM" "HGB_DBLP" "HGB_IMDB")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for DATASET in "${DATASETS[@]}"; do
    PART_JSON="results/${DATASET}/partition.json"

    # -----------------------------------------------------------------------
    # Exp 1 — Partition
    # -----------------------------------------------------------------------
    log "=== EXP1: ${DATASET} ==="
    python scripts/exp1_inductive_split.py "${DATASET}" \
        --train-frac "${TRAIN_FRAC}" \
        --seed "${SEED}"

    if [[ ! -f "${PART_JSON}" ]]; then
        echo "ERROR: partition.json not created for ${DATASET}, aborting."
        exit 1
    fi
    log "partition.json ready: ${PART_JSON}"

    # -----------------------------------------------------------------------
    # Exp 2 — Train, one call per metapath
    # -----------------------------------------------------------------------
    while IFS= read -r METAPATH; do
        [[ -z "${METAPATH}" ]] && continue
        log "--- EXP2: ${DATASET}  metapath=${METAPATH}  depth=${DEPTHS} ---"
        python scripts/exp2_train.py "${DATASET}" \
            --metapath "${METAPATH}" \
            --depth ${DEPTHS} \
            --partition-json "${PART_JSON}" \
            --seed "${SEED}" \
            ${EXTRA_EXP2_ARGS}
    done <<< "${METAPATHS[${DATASET}]}"

    log "=== DONE: ${DATASET} ==="
    echo ""
done

log "All done. Weights in results/*/weights/  |  Logs in results/*/training_log.csv"
