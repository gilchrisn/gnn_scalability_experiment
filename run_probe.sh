#!/bin/bash
set -e

echo "=== Recompiling C++ binary ==="
cd HUB && g++ -O2 -o ../bin/graph_prep main.cpp param.cpp -std=c++17 && cd ..
echo "Compile OK"

echo "=== Phase 1: Variable metapaths + Phase 2: AnyBURL instance rules ==="
python scripts/probe_metapaths.py --timeout 1800 2>&1 | tee results/probe_overnight.log

echo "=== Done. Check results/probe_metapaths.csv ==="
