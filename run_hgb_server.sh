#!/usr/bin/env bash
# HGB datasets: ACM, DBLP, IMDB
#   Part 1: Base paper reproduction with MINING (matches original paper's metapath count)
#   Part 2: Extension experiments with config metapaths
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
MAX_METAPATHS=500
EPOCHS=50
K=8
MAX_ADJ_MB=5000
BOOLAP_D="parallel-k-P-core-decomposition-code/BoolAPCoreD"
BOOLAP_G="parallel-k-P-core-decomposition-code/BoolAPCoreG"

# Compile BoolAP if needed
if [ ! -f "$BOOLAP_D" ]; then
    echo "Compiling BoolAP binaries..."
    make -C parallel-k-P-core-decomposition-code BoolAPCoreD BoolAPCoreG 2>&1 || echo "[WARN] BoolAP compile failed"
fi

BOOLAP_ARGS=""
[ -f "$BOOLAP_D" ] && BOOLAP_ARGS="$BOOLAP_ARGS --boolap-binary $BOOLAP_D"
[ -f "$BOOLAP_G" ] && BOOLAP_ARGS="$BOOLAP_ARGS --boolap-plus-binary $BOOLAP_G"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="results/hgb_run_${TIMESTAMP}.log"
mkdir -p results

echo "==================================================" | tee -a "$LOG"
echo "  HGB Server Run — $TIMESTAMP"                      | tee -a "$LOG"
echo "  max_metapaths=${MAX_METAPATHS}  timeout=${TIMEOUT}s" | tee -a "$LOG"
echo "  epochs=${EPOCHS}  k=${K}"                          | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

for DS in HGB_ACM HGB_DBLP HGB_IMDB; do
    echo "" | tee -a "$LOG"
    echo "=== $DS ===" | tee -a "$LOG"

    # Part 1: Base paper with MINING (--force-mine overrides config paths)
    echo "--- Part 1: Base Paper (mining) ---" | tee -a "$LOG"
    python scripts/run_paper_experiments.py "$DS" \
        --max-metapaths "$MAX_METAPATHS" \
        --mining-timeout 10 \
        --timeout "$TIMEOUT" \
        --force-mine \
        $BOOLAP_ARGS \
        2>&1 | tee -a "$LOG" || echo "  [WARN] $DS Part 1 failed" | tee -a "$LOG"

    # Part 2: Extension with config paths
    echo "--- Part 2: Extension ---" | tee -a "$LOG"
    python scripts/run_extension_experiments.py "$DS" \
        --fractions 0.2 0.4 0.6 0.8 1.0 \
        --epochs "$EPOCHS" --k "$K" --timeout "$TIMEOUT" \
        --max-adj-mb "$MAX_ADJ_MB" --num-cpu-threads 2 \
        --cpu \
        2>&1 | tee -a "$LOG" || echo "  [WARN] $DS Part 2 failed" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
echo "  HGB done. $(date)" | tee -a "$LOG"
echo "  Part 1: results/<DS>/table4.csv, figure4-6.csv" | tee -a "$LOG"
echo "  Part 2: results/<DS>/extension.csv" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
