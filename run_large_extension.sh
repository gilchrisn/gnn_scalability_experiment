#!/usr/bin/env bash
# Extension experiments for OGB_MAG and OAG_CS.
#
# Standardized with HGB datasets (ACM, DBLP, IMDB):
#   - Same fractions: 0.2, 0.4, 0.6, 0.8, 1.0
#   - Same k=8, epochs=50, timeout=1800s
#   - Config metapaths (no AnyBURL mining)
#   - --cpu for fair timing
#
# Expected behavior on large datasets:
#   Phase 1: Train SAGE on 20% subgraph (exact materialization).
#            If exact OOM here, metapath is skipped — no model to freeze.
#   Phase 2: For each fraction, attempt exact + always run KMV.
#            Exact will OOM/TLE at larger fractions — reported honestly.
#            KMV completes at all fractions (O(|V|*k) bounded).
#
# Config metapaths:
#   OGB_MAG: rev_writes,writes (PAP), has_topic,rev_has_topic (PFP),
#            rev_writes,affiliated_with,rev_affiliated_with,writes (PAIAP)
#   OAG_CS:  rev_AP_write_first,AP_write_first (PAP),
#            PF_in_L3,rev_PF_in_L3 (PFP)
#
# Resume-safe. Run in tmux.

set -uo pipefail
source .venv/bin/activate
export PYTHONUTF8=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
if [ -d "$HOME/jdk-25" ]; then
    export PATH="$HOME/jdk-25/bin:$PATH"
fi

# Recompile C++ binary
echo "Recompiling graph_prep..."
cd HUB && g++ -O2 -o ../bin/graph_prep main.cpp param.cpp -std=c++17 && cd ..
echo "graph_prep OK"

# ---- Settings (match run_overnight.sh HGB datasets) ----
TIMEOUT=1800            # 30 min per C++ call
EPOCHS=50               # SAGE training epochs
K=8                     # KMV sketch size (same as HGB runs)
MAX_ADJ_MB=0            # No limit — let exact fail naturally on server
MAX_DIRICHLET=50000000  # Skip dirichlet on exact graphs >50M edges (prevents SIGKILL)
NUM_THREADS=2           # PyTorch CPU threads

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="results/large_extension_${TIMESTAMP}.log"
mkdir -p results

echo "==================================================" | tee -a "$LOG"
echo "  Extension: OGB_MAG + OAG_CS — $TIMESTAMP"         | tee -a "$LOG"
echo "  fractions=0.2,0.4,0.6,0.8,1.0"                    | tee -a "$LOG"
echo "  k=${K}  epochs=${EPOCHS}  timeout=${TIMEOUT}s"     | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

# =====================================================
# OGB_MAG  (1.1M papers, 1.2M authors, 7M+ edges)
# =====================================================
echo "" | tee -a "$LOG"
echo "=== OGB_MAG ===" | tee -a "$LOG"

python scripts/run_extension_experiments.py OGB_MAG \
    --fractions 0.2 0.4 0.6 0.8 1.0 \
    --epochs "$EPOCHS" --k "$K" --timeout "$TIMEOUT" \
    --max-adj-mb "$MAX_ADJ_MB" \
    --max-dirichlet-edges "$MAX_DIRICHLET" \
    --num-cpu-threads "$NUM_THREADS" \
    --cpu \
    2>&1 | tee -a "$LOG" || echo "  [WARN] OGB_MAG extension failed" | tee -a "$LOG"

# =====================================================
# OAG_CS   (546K papers, 768-dim XLNet features)
# =====================================================
echo "" | tee -a "$LOG"
echo "=== OAG_CS ===" | tee -a "$LOG"

python scripts/run_extension_experiments.py OAG_CS \
    --fractions 0.2 0.4 0.6 0.8 1.0 \
    --epochs "$EPOCHS" --k "$K" --timeout "$TIMEOUT" \
    --max-adj-mb "$MAX_ADJ_MB" \
    --max-dirichlet-edges "$MAX_DIRICHLET" \
    --num-cpu-threads "$NUM_THREADS" \
    --cpu \
    2>&1 | tee -a "$LOG" || echo "  [WARN] OAG_CS extension failed" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
echo "  Done. $(date)"                                     | tee -a "$LOG"
echo "  Results: results/OGB_MAG/extension.csv"            | tee -a "$LOG"
echo "  Results: results/OAG_CS/extension.csv"             | tee -a "$LOG"
echo "  Log: $LOG"                                         | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"
