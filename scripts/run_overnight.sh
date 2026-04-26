#!/bin/bash
# run_overnight.sh — overnight scalability + depth-effect validation.
# Just `bash scripts/run_overnight.sh` and walk away.
#
# All phases:
#   - Skip Exact entirely (it OOMs at OGB scale anyway)
#   - Run KMV + MPRW + KGRW (reduced grid)
#   - Measure edges, time, peak RSS  (no SAGE inference, no quality metrics)
#   - Resume-safe via results/scale_validation.csv
#   - One failed metapath does NOT kill the rest of the run
#
# Output:
#   results/scale_validation.csv   — append-only, one row per (dataset, mp, method, params, seed)
#   results/scale_validation_logs/ — per-phase tee'd logs

set +e   # do NOT abort on errors — keep going to next metapath
cd "$(dirname "$0")/.."
# Try Linux venv first, then Windows-style venv
source .venv/bin/activate     2>/dev/null || \
source .venv/Scripts/activate 2>/dev/null || true
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1     # so tee'd output appears in real time

LOG_DIR="results/scale_validation_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
SEEDS=3
KGRW_K="4 16 32 64"
KGRW_W="4 16 64"

run_phase() {
    # $1: phase label  |  $2: dataset  |  $3: kvalues  |  $4: wvalues
    # $5: method-timeout  |  $@: remaining = "--metapaths" arg list
    local label=$1 dataset=$2 kvals=$3 wvals=$4 mtimeout=$5
    shift 5
    echo
    echo "===== $label  ($(date +%H:%M:%S)) ====="
    # KMV+MPRW
    python -u scripts/bench_scale_validation.py \
        --datasets "$dataset" \
        --metapaths "$@" \
        --k-values $kvals \
        --w-values $wvals \
        --seeds $SEEDS \
        --skip-exact \
        --method-timeout "$mtimeout" \
        2>&1 | tee -a "$LOG_DIR/${label// /_}_${TS}.log"
    # KGRW (reduced grid)
    python -u scripts/bench_scale_validation.py \
        --datasets "$dataset" \
        --metapaths "$@" \
        --k-values $KGRW_K \
        --w-values $KGRW_W \
        --seeds $SEEDS \
        --skip-exact --skip-kmv --skip-mprw --no-skip-kgrw \
        --method-timeout "$mtimeout" \
        2>&1 | tee -a "$LOG_DIR/${label// /_}_KGRW_${TS}.log"
}

# ─── PHASE 1: OGB_MAG depth × topology  (L=2 fast) ─────────────────────────
run_phase "OGB_L2" OGB_MAG \
    "4 8 16 32 64 128 256 512 1024" \
    "2 4 8 16 32 64 128 256 512 1024" \
    1800 \
    "rev_writes,writes" \
    "cites,rev_cites" \
    "has_topic,rev_has_topic"

# ─── PHASE 2: OGB_MAG L=4 (params trimmed since edges grow ~exponentially) ─
run_phase "OGB_L4" OGB_MAG \
    "4 8 16 32 64 128 256" \
    "2 4 8 16 32 64 128 256" \
    3600 \
    "rev_writes,writes,rev_writes,writes" \
    "cites,rev_cites,cites,rev_cites" \
    "has_topic,rev_has_topic,has_topic,rev_has_topic"

# ─── PHASE 3: OGB_MAG L=6 (PAP-type only — slow at this scale) ─────────────
run_phase "OGB_L6" OGB_MAG \
    "4 8 16 32 64 128" \
    "2 4 8 16 32 64 128" \
    7200 \
    "rev_writes,writes,rev_writes,writes,rev_writes,writes"

# ─── PHASE 4: DBLP depth × topology  (parallel sanity at small scale) ──────
run_phase "DBLP_L2" HGB_DBLP \
    "4 8 16 32 64 128 256 512 1024" \
    "2 4 8 16 32 64 128 256 512 1024" \
    600 \
    "author_to_paper,paper_to_author"

run_phase "DBLP_L4" HGB_DBLP \
    "4 8 16 32 64 128 256 512" \
    "2 4 8 16 32 64 128 256 512" \
    900 \
    "author_to_paper,paper_to_author,author_to_paper,paper_to_author" \
    "author_to_paper,paper_to_term,term_to_paper,paper_to_author" \
    "author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"

run_phase "DBLP_L6" HGB_DBLP \
    "4 8 16 32 64 128 256" \
    "2 4 8 16 32 64 128 256" \
    1200 \
    "author_to_paper,paper_to_author,author_to_paper,paper_to_author,author_to_paper,paper_to_author" \
    "author_to_paper,paper_to_term,term_to_paper,paper_to_term,term_to_paper,paper_to_author"

echo
echo "===== ALL PHASES DONE  ($(date +%H:%M:%S)) ====="
wc -l results/scale_validation.csv
echo "Logs: $LOG_DIR/"
