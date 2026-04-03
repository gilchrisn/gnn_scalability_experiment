#!/usr/bin/env bash
# Large datasets: OGB_MAG, OAG_CS
#   Part 1: Base paper reproduction with CONFIG metapaths (mining produces garbage)
#   Part 2: Extension experiments with temporal subgraphs
#
# Resume-safe. Run in tmux.

set -uo pipefail
source .venv/bin/activate
export PYTHONUTF8=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
if [ -d "$HOME/jdk-25" ]; then
    export PATH="$HOME/jdk-25/bin:$PATH"
fi

# Recompile C++ binary (picks up PerD/PerH instance rule fix)
echo "Recompiling C++ binary..."
cd HUB && g++ -O2 -o ../bin/graph_prep main.cpp param.cpp -std=c++17 && cd ..
echo "Compile OK"

TIMEOUT=1800
EPOCHS=50
K=8
MAX_ADJ_MB=5000
BOOLAP_D="parallel-k-P-core-decomposition-code/BoolAPCoreD"
BOOLAP_G="parallel-k-P-core-decomposition-code/BoolAPCoreG"

BOOLAP_ARGS=""
[ -f "$BOOLAP_D" ] && BOOLAP_ARGS="$BOOLAP_ARGS --boolap-binary $BOOLAP_D"
[ -f "$BOOLAP_G" ] && BOOLAP_ARGS="$BOOLAP_ARGS --boolap-plus-binary $BOOLAP_G"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="results/large_run_${TIMESTAMP}.log"
mkdir -p results

echo "==================================================" | tee -a "$LOG"
echo "  Large Dataset Run — $TIMESTAMP"                    | tee -a "$LOG"
echo "  timeout=${TIMEOUT}s  k=${K}  epochs=${EPOCHS}"     | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

# =====================================================
# OGB_MAG
# =====================================================
echo "" | tee -a "$LOG"
echo "=== OGB_MAG ===" | tee -a "$LOG"

# Part 1: AnyBURL instance rules (variable rules timeout on large datasets)
# Clear old results so resume-safe doesn't skip broken data
rm -f results/OGB_MAG/table4.csv results/OGB_MAG/figure4.csv results/OGB_MAG/figure5.csv results/OGB_MAG/figure6.csv
echo "--- Part 1: Base Paper (AnyBURL instance rules) ---" | tee -a "$LOG"
python scripts/run_paper_experiments.py OGB_MAG \
    --instance-rules \
    --max-metapaths 50 \
    --timeout "$TIMEOUT" \
    $BOOLAP_ARGS \
    2>&1 | tee -a "$LOG" || echo "  [WARN] OGB_MAG Part 1 failed" | tee -a "$LOG"

# Part 2: skipped for now (rerun with extension later)
# python scripts/run_extension_experiments.py OGB_MAG \
#     --fractions 0.08 0.16 0.24 0.32 0.40 \
#     --epochs "$EPOCHS" --k "$K" --timeout "$TIMEOUT" \
#     --max-adj-mb "$MAX_ADJ_MB" --num-cpu-threads 2 \
#     --cpu \
#     2>&1 | tee -a "$LOG" || echo "  [WARN] OGB_MAG Part 2 failed" | tee -a "$LOG"

# =====================================================
# OAG_CS
# =====================================================
echo "" | tee -a "$LOG"
echo "=== OAG_CS ===" | tee -a "$LOG"

# Part 1: AnyBURL instance rules (variable rules timeout on large datasets)
rm -f results/OAG_CS/table4.csv results/OAG_CS/figure4.csv results/OAG_CS/figure5.csv results/OAG_CS/figure6.csv
echo "--- Part 1: Base Paper (AnyBURL instance rules) ---" | tee -a "$LOG"
python scripts/run_paper_experiments.py OAG_CS \
    --instance-rules \
    --max-metapaths 50 \
    --timeout "$TIMEOUT" \
    $BOOLAP_ARGS \
    2>&1 | tee -a "$LOG" || echo "  [WARN] OAG_CS Part 1 failed" | tee -a "$LOG"

# Part 2: skipped for now (rerun with extension later)
# python scripts/run_extension_experiments.py OAG_CS \
#     --fractions 0.04 0.08 0.12 0.16 0.20 \
#     --epochs "$EPOCHS" --k "$K" --timeout "$TIMEOUT" \
#     --max-adj-mb "$MAX_ADJ_MB" --num-cpu-threads 2 \
#     --cpu \
#     2>&1 | tee -a "$LOG" || echo "  [WARN] OAG_CS Part 2 failed" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
echo "  Large datasets done. $(date)" | tee -a "$LOG"
echo "  Part 1: results/<DS>/table4.csv, figure4-6.csv" | tee -a "$LOG"
echo "  Part 2: results/<DS>/extension.csv" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
