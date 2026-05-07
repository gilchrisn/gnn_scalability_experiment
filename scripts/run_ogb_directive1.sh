#!/usr/bin/env bash
# Directive 1 (Gemini revision): 5-seed OGB-MAG KMV materialisation re-run for
# cost CIs. Exact is deterministic (cached on disk); we vary --hash-seed across
# {42,43,44,45,46} and run all 5 k values per seed via one exp3 invocation each.
# --skip-exact reuses the existing Exact materialise; KMV mat + KMV inference
# both run, capturing mat_time, mat_RAM, edge_count to master_results_v2.csv.
set -uo pipefail
cd ~/gilchris/gnn_scalability_experiment
source .venv/bin/activate

LOG=results/OGB_MAG/ogb_directive1_$(date +%Y%m%d_%H%M%S).log
echo "[$(date)] === Directive 1: OGB-MAG 5-seed KMV mat ===" | tee -a $LOG

for seed in 42 43 44 45 46; do
  echo "[$(date)] [SEED $seed] all k {8,16,32,64,128}" | tee -a $LOG
  timeout 5400 python scripts/exp3_inference.py OGB_MAG \
    --metapath rev_writes,writes \
    --depth 2 \
    --k-values 8 16 32 64 128 \
    --partition-json results/OGB_MAG/partition.json \
    --weights-dir results/OGB_MAG/weights \
    --output results/OGB_MAG/master_results_v2.csv \
    --arch SAGE \
    --skip-mprw \
    --skip-exact \
    --run-id $seed \
    --hash-seed $seed \
    --timeout 3600 2>&1 | tail -30 | tee -a $LOG
done

echo "[$(date)] === Directive 1 DONE ===" | tee -a $LOG
echo "Aggregating cost CIs..." | tee -a $LOG
python -c "
import pandas as pd
df = pd.read_csv('results/OGB_MAG/master_results_v2.csv')
sub = df[(df.Method=='KMV') & (df.Dataset=='OGB_MAG')]
print('=== KMV mat-time + RAM + edges, mean ± std across seeds ===')
agg = sub.groupby('k_value').agg(
    mat_time_mean=('Materialization_Time', 'mean'),
    mat_time_std =('Materialization_Time', 'std'),
    mat_ram_mean =('Mat_RAM_MB', 'mean'),
    mat_ram_std  =('Mat_RAM_MB', 'std'),
    edges_mean   =('Edge_Count', 'mean'),
    edges_std    =('Edge_Count', 'std'),
    n_seeds      =('Seed', 'nunique'),
).round(2)
print(agg.to_string())
" 2>&1 | tee -a $LOG
