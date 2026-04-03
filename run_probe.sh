#!/bin/bash
set -e

echo "=== Recompiling C++ binary ==="
cd HUB && g++ -O2 -o ../bin/graph_prep main.cpp param.cpp -std=c++17 && cd ..
echo "Compile OK"

echo "=== Running AnyBURL instance rules (skip variable - already done) ==="
python scripts/probe_metapaths.py --timeout 1800 --skip-variable 2>&1 | tee results/probe_overnight.log

echo "=== Done. Check results/probe_metapaths.csv ==="
