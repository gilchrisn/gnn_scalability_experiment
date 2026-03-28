#!/usr/bin/env bash
# Overnight run: ALL datasets, Part 1 + Part 2
#
# Uses config metapaths for all datasets (consistent, no mining dependency).
# ACM also uses --force-mine for 20 mined metapaths in Part 1.
#
# Resume-safe. Run in tmux.

set -uo pipefail
source .venv/bin/activate
export PYTHONUTF8=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
if [ -d "$HOME/jdk-25" ]; then
    export PATH="$HOME/jdk-25/bin:$PATH"
fi

TIMEOUT=1800
EPOCHS=50
K=8
MAX_ADJ_MB=5000
BOOLAP_D="parallel-k-P-core-decomposition-code/BoolAPCoreD"
BOOLAP_G="parallel-k-P-core-decomposition-code/BoolAPCoreG"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="results/overnight_${TIMESTAMP}.log"
mkdir -p results

# Recompile C++ (L=8, RAND_MAX fix, NTypes check)
echo "Compiling C++ binary..." | tee -a "$LOG"
g++ -std=c++17 -O3 -o bin/graph_prep HUB/main.cpp HUB/param.cpp 2>&1 | tee -a "$LOG"

# Compile BoolAP if needed
if [ ! -f "$BOOLAP_D" ]; then
    echo "Compiling BoolAP..." | tee -a "$LOG"
    make -C parallel-k-P-core-decomposition-code BoolAPCoreD BoolAPCoreG 2>&1 || echo "[WARN] BoolAP compile failed" | tee -a "$LOG"
fi

BOOLAP_ARGS=""
[ -f "$BOOLAP_D" ] && BOOLAP_ARGS="$BOOLAP_ARGS --boolap-binary $BOOLAP_D"
[ -f "$BOOLAP_G" ] && BOOLAP_ARGS="$BOOLAP_ARGS --boolap-plus-binary $BOOLAP_G"

echo "==================================================" | tee -a "$LOG"
echo "  Overnight Run - $TIMESTAMP"                        | tee -a "$LOG"
echo "  timeout=${TIMEOUT}s  k=${K}  epochs=${EPOCHS}"     | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

# ==========================================================
# Part 1: Base Paper Reproduction
# ==========================================================
echo "" | tee -a "$LOG"
echo "=== Part 1: Base Paper ===" | tee -a "$LOG"

# ACM: use mining (20 metapaths)
echo "--- HGB_ACM (mining, 20 metapaths) ---" | tee -a "$LOG"
python scripts/run_paper_experiments.py HGB_ACM \
    --force-mine --max-metapaths 500 \
    --mining-timeout 10 --timeout "$TIMEOUT" \
    $BOOLAP_ARGS \
    2>&1 | tee -a "$LOG" || echo "  [WARN] HGB_ACM Part 1 failed" | tee -a "$LOG"

# DBLP: config (3 metapaths - mining only finds 1)
echo "--- HGB_DBLP (config, 3 metapaths) ---" | tee -a "$LOG"
python scripts/run_paper_experiments.py HGB_DBLP \
    --timeout "$TIMEOUT" \
    $BOOLAP_ARGS \
    2>&1 | tee -a "$LOG" || echo "  [WARN] HGB_DBLP Part 1 failed" | tee -a "$LOG"

# IMDB: config (9 metapaths)
echo "--- HGB_IMDB (config, 9 metapaths) ---" | tee -a "$LOG"
python scripts/run_paper_experiments.py HGB_IMDB \
    --timeout "$TIMEOUT" \
    $BOOLAP_ARGS \
    2>&1 | tee -a "$LOG" || echo "  [WARN] HGB_IMDB Part 1 failed" | tee -a "$LOG"

# OGB_MAG: config (3 metapaths, h-index may timeout)
echo "--- OGB_MAG (config, 3 metapaths) ---" | tee -a "$LOG"
python scripts/run_paper_experiments.py OGB_MAG \
    --timeout "$TIMEOUT" \
    $BOOLAP_ARGS \
    2>&1 | tee -a "$LOG" || echo "  [WARN] OGB_MAG Part 1 failed" | tee -a "$LOG"

# OAG_CS: config (2 metapaths)
echo "--- OAG_CS (config, 2 metapaths) ---" | tee -a "$LOG"
python scripts/run_paper_experiments.py OAG_CS \
    --timeout "$TIMEOUT" \
    $BOOLAP_ARGS \
    2>&1 | tee -a "$LOG" || echo "  [WARN] OAG_CS Part 1 failed" | tee -a "$LOG"

# ==========================================================
# Part 2: Extension Experiments
# ==========================================================
echo "" | tee -a "$LOG"
echo "=== Part 2: Extension ===" | tee -a "$LOG"

# HGB datasets: fractions 0.2-1.0
for DS in HGB_ACM HGB_DBLP HGB_IMDB; do
    echo "--- $DS (extension) ---" | tee -a "$LOG"
    python scripts/run_extension_experiments.py "$DS" \
        --fractions 0.2 0.4 0.6 0.8 1.0 \
        --epochs "$EPOCHS" --k "$K" --timeout "$TIMEOUT" \
        --max-adj-mb "$MAX_ADJ_MB" --num-cpu-threads 2 \
        --cpu \
        2>&1 | tee -a "$LOG" || echo "  [WARN] $DS extension failed" | tee -a "$LOG"
done

# OGB_MAG: 40% as full, 5 snapshots
echo "--- OGB_MAG (extension, 40% as full) ---" | tee -a "$LOG"
python scripts/run_extension_experiments.py OGB_MAG \
    --fractions 0.08 0.16 0.24 0.32 0.40 \
    --epochs "$EPOCHS" --k "$K" --timeout "$TIMEOUT" \
    --max-adj-mb "$MAX_ADJ_MB" --num-cpu-threads 2 \
    --cpu \
    2>&1 | tee -a "$LOG" || echo "  [WARN] OGB_MAG extension failed" | tee -a "$LOG"

# OAG_CS: 20% as full, 5 snapshots
echo "--- OAG_CS (extension, 20% as full) ---" | tee -a "$LOG"
python scripts/run_extension_experiments.py OAG_CS \
    --fractions 0.04 0.08 0.12 0.16 0.20 \
    --epochs "$EPOCHS" --k "$K" --timeout "$TIMEOUT" \
    --max-adj-mb "$MAX_ADJ_MB" --num-cpu-threads 2 \
    --cpu \
    2>&1 | tee -a "$LOG" || echo "  [WARN] OAG_CS extension failed" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
echo "  Done. $(date)" | tee -a "$LOG"
echo "  Part 1: results/<DS>/table4.csv" | tee -a "$LOG"
echo "  Part 2: results/<DS>/extension.csv" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
