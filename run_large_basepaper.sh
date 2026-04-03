#!/bin/bash
set -e

# Use local JDK 25 if available
if [ -d "$HOME/jdk-25" ]; then
    export PATH="$HOME/jdk-25/bin:$PATH"
    echo "Using JDK: $(java -version 2>&1 | head -1)"
fi

echo "=== Recompiling C++ binary ==="
cd HUB && g++ -O2 -o ../bin/graph_prep main.cpp param.cpp -std=c++17 && cd ..

echo "=== Running base paper experiments on OGB_MAG + OAG_CS ==="
echo "  AnyBURL config: 10s snapshot, conf=0.1, MAX_LENGTH_CYCLIC=4"
echo "  Methods: ExactD+, ExactD, ExactH+, ExactH, GloD(k=32), GloH(k=4)"

python scripts/run_large_basepaper.py \
    --datasets OGB_MAG OAG_CS \
    --mining-time 10 \
    --min-conf 0.1 \
    --timeout 1800 \
    2>&1 | tee results/large_basepaper.log

echo "=== Done. Check results/large_basepaper.csv ==="
