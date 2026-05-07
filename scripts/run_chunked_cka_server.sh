#!/bin/bash
# Launcher for Directive 2 chunked-CKA on the yudong server.
#
# Run inside a tmux session named "chunked_cka":
#   tmux new -s chunked_cka
#   bash scripts/run_chunked_cka_server.sh
#
# Pre-reqs already on server:
#   - results/OGB_MAG/partition.json
#   - results/OGB_MAG/weights/rev_writes_writes_L2.pt
#   - staging/OGB_MAG/mat_exact.adj  (will auto-rerun if missing)
#   - bin/graph_prep
#
# Outputs:
#   - results/OGB_MAG/chunked_cka_subsample_5000.npy
#   - results/OGB_MAG/chunked_cka_subgraph_exact.pt
#   - results/OGB_MAG/chunked_cka_z_exact_layers.pt
#   - results/OGB_MAG/chunked_cka_z_kmv_k32_layers.pt
#   - results/approach_a_2026_05_07/OGB_MAG/SAGE/chunked_cka_seed42_k32.json

set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate 2>/dev/null || source venv/bin/activate

export PYTHONUTF8=1
export PYTHONUNBUFFERED=1

python scripts/chunked_cka_ogb_mag.py \
    --n-subsample 5000 \
    --k 32 \
    --seed 42 \
    --num-layers 2 \
    --max-neighbours-per-node 200 \
    --hard-cap-reach 2000000 \
    --timeout 7200 \
    "$@"
