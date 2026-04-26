#!/bin/bash
# run_overnight.sh — overnight scalability validation on OGB_MAG.
# Just `bash scripts/run_overnight.sh` and walk away.
#
# Phases (each phase resumes from CSV if interrupted):
#   1. PAP   (2-hop, paper-author-paper):       Exact + KMV + MPRW
#   2. PAPAP (4-hop):                            Exact + KMV + MPRW
#   3. KGRW  (PAP + PAPAP, reduced grid):        only if phases 1+2 finished cleanly
#
# Output: results/scale_validation.csv  (resume-safe, append-only)

set -e
cd "$(dirname "$0")/.."
source .venv/Scripts/activate 2>/dev/null || true
export PYTHONUTF8=1

LOG_DIR="results/scale_validation_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "===== Phase 1: OGB_MAG PAP (2-hop)  Exact + KMV + MPRW ====="
python scripts/bench_scale_validation.py \
    --datasets OGB_MAG \
    --metapaths "rev_writes,writes" \
    --k-values 4 8 16 32 64 128 256 512 1024 \
    --w-values 2 4 8 16 32 64 128 256 512 1024 \
    --seeds 3 \
    --exact-timeout 3600 \
    --method-timeout 1800 \
    2>&1 | tee "$LOG_DIR/phase1_pap_${TS}.log"

echo
echo "===== Phase 2: OGB_MAG PAPAP (4-hop)  Exact + KMV + MPRW ====="
python scripts/bench_scale_validation.py \
    --datasets OGB_MAG \
    --metapaths "rev_writes,writes,rev_writes,writes" \
    --k-values 4 8 16 32 64 128 256 512 1024 \
    --w-values 2 4 8 16 32 64 128 256 512 1024 \
    --seeds 3 \
    --exact-timeout 7200 \
    --method-timeout 3600 \
    2>&1 | tee "$LOG_DIR/phase2_papap_${TS}.log"

echo
echo "===== Phase 3: KGRW on both metapaths (reduced grid k×w = 12 cells) ====="
python scripts/bench_scale_validation.py \
    --datasets OGB_MAG \
    --metapaths "rev_writes,writes" "rev_writes,writes,rev_writes,writes" \
    --k-values 4 16 32 64 \
    --w-values 4 16 64 \
    --seeds 3 \
    --skip-exact --skip-kmv --skip-mprw --no-skip-kgrw \
    --method-timeout 3600 \
    2>&1 | tee "$LOG_DIR/phase3_kgrw_${TS}.log"

echo
echo "===== ALL PHASES COMPLETE ====="
echo "CSV:  results/scale_validation.csv"
echo "Logs: $LOG_DIR/"
wc -l results/scale_validation.csv
