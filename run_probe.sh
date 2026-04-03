#!/bin/bash
set -e

# Use local JDK 25 if available (server has Java 17, AnyBURL needs 25)
if [ -d "$HOME/jdk-25" ]; then
    export PATH="$HOME/jdk-25/bin:$PATH"
    echo "Using JDK 25: $(java -version 2>&1 | head -1)"
fi

echo "=== Recompiling C++ binary ==="
cd HUB && g++ -O2 -o ../bin/graph_prep main.cpp param.cpp -std=c++17 && cd ..
echo "Compile OK"

echo "=== Running AnyBURL instance rules (skip variable - already done) ==="
python scripts/probe_metapaths.py --timeout 1800 --skip-variable 2>&1 | tee results/probe_overnight.log

echo "=== Done. Check results/probe_metapaths.csv ==="
