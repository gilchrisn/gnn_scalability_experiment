#!/usr/bin/env bash
# run_fraction_sweep.sh — Drive the fraction-sweep experiment end-to-end.
#
# Stages:
#   1. PubMed (trained mode, 5 seeds): subsample → train SAGE → Exact +
#      KMV/MPRW/KGRW sweeps with quality metrics vs Exact.
#   2. OGB_MAG (untrained mode, 1 seed): subsample → random-init SAGE →
#      KMV/MPRW/KGRW sweeps with quality metrics vs KMV-k=256 reference.
#      No Exact (won't fit at full size; consistent across fractions).
#   3. Aggregate both into 3way_l2_scatter_fractions.pdf and
#      method_saturation_fractions.pdf.
#
# Resume-safe: each driver appends to its own kgrw_bench_fractions.csv and
# skips already-present (Dataset, Fraction, L, Method, k, w', Seed) cells.
# Re-running this script picks up wherever it left off.
#
# Usage:
#   bash scripts/run_fraction_sweep.sh             # full sweep
#   bash scripts/run_fraction_sweep.sh pubmed      # PubMed only
#   bash scripts/run_fraction_sweep.sh ogb         # OGB only
#   bash scripts/run_fraction_sweep.sh agg         # aggregator only
#
# The script must be launched from the project root.

set -euo pipefail

# Resolve project root (the directory above scripts/)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Pick a Python interpreter — prefer the project venv on Windows, else system python3
if [[ -x ".venv/Scripts/python.exe" ]]; then
    PY=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi
export PYTHONUTF8=1

run_pubmed() {
    echo "============================================================"
    echo "STAGE 1/3 — HNE_PubMed (trained, 5 seeds, 5 fractions)"
    echo "============================================================"
    "$PY" scripts/bench_fraction_sweep.py \
        --dataset HNE_PubMed --mode trained \
        --skip-restage
}

run_ogb() {
    echo "============================================================"
    echo "STAGE 2/3 — OGB_MAG (untrained, 1 seed, 5 fractions)"
    echo "============================================================"
    "$PY" scripts/bench_fraction_sweep.py \
        --dataset OGB_MAG --mode untrained \
        --skip-restage
}

run_agg() {
    echo "============================================================"
    echo "STAGE 3/3 — Aggregate plots + summary"
    echo "============================================================"
    "$PY" scripts/aggregate_fractions.py --datasets HNE_PubMed OGB_MAG
    echo
    echo "Outputs:"
    echo "  results/3way_l2_scatter_fractions.pdf"
    echo "  results/method_saturation_fractions.pdf"
    echo "  results/fractions_summary.csv"
}

stage="${1:-all}"
case "$stage" in
    pubmed) run_pubmed ;;
    ogb)    run_ogb ;;
    agg)    run_agg ;;
    all)    run_pubmed; run_ogb; run_agg ;;
    *)
        echo "Unknown stage: $stage  (expected: pubmed | ogb | agg | all)"
        exit 2
        ;;
esac
