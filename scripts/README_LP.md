# Link Prediction Experiment Suite — Reproducibility

**Created**: 2026-04-25
**Purpose**: Reproduce the LP experiments for Frame 1 (Scalable HGNN LP via KMV).

## Pipeline overview

Three scripts:

1. `exp_lp_train.py` — Stage I: train SAGE encoder + dot-product LP decoder on exact `H_train` (edges among V_train nodes). Saves frozen weights.
2. `exp_lp_inference.py` — Stage II: load frozen encoder, run forward pass on Exact / KMV / MPRW sparsified full H, score test edges against 2-hop negatives.
3. `exp_lp_analyze.py` — aggregate `master_lp_results.csv` across seeds + datasets into publication tables + plots.

## Prerequisites

```bash
source .venv/Scripts/activate           # Windows
export PYTHONUTF8=1
# Ensure partitions exist (reuse exp1 output):
ls results/<DATASET>/partition.json
# Ensure C++ binaries are built:
ls bin/graph_prep bin/mprw_exec
```

## Step 1 — Train (one-time per dataset × L × metapath)

```bash
python scripts/exp_lp_train.py HGB_DBLP \
    --metapath author_to_paper,paper_to_author \
    --depth 2 \
    --epochs 50 \
    --partition-json results/HGB_DBLP/partition.json \
    --embedding-dim 64 \
    --max-pos-per-epoch 10000 \
    --seed 42
```

**Key flags**:
- `--max-pos-per-epoch N` — subsample positive edges per epoch (avoid O(|E|²) on dense graphs). Default 10000; reduce to 2000-5000 for dense graphs (ACM PTP, PubMed DCD).
- `--embedding-dim D` — decoder dim. Default 64.

**Output**: `results/<DATASET>/weights_lp/<mp_safe>_L<L>.pt` + `lp_training_log.csv`

## Step 2 — Inference (per seed, sweep k and w)

```bash
python scripts/exp_lp_inference.py HGB_DBLP \
    --metapath author_to_paper,paper_to_author \
    --depth 2 \
    --partition-json results/HGB_DBLP/partition.json \
    --k-values 8 16 32 64 \
    --w-values 8 32 128 \
    --run-id 0 \
    --max-test-edges 2000 \
    --n-negs-per-pos 50
```

**Critical flags**:
- `--force-cpu` — required for large graphs (|E| > 1M or in_dim > 2000); GPU OOMs otherwise.
- `--skip-exact` — skip the Exact row (if already computed or known to OOM).
- `--run-id N` — seed for this run. Stage I weights are shared across seeds; only sparsification + scoring RNG varies.
- `--n-negs-per-pos N` — number of 2-hop negatives per positive for MRR computation. 30-100 typical.

**Output**: appends rows to `results/<DATASET>/master_lp_results.csv`.
Resume-safe: skips runs already present in CSV (keyed by (metapath, L, method, k, w, seed)).

## Step 3 — Multi-seed sweep

```bash
# Seed 0: runs Exact once + KMV k-sweep + MPRW w-sweep
# Seeds 1-4: skip-exact, re-run KMV/MPRW with different --hash-seed
for s in 0 1 2 3 4; do
    python scripts/exp_lp_inference.py HGB_DBLP \
        --metapath author_to_paper,paper_to_author \
        --depth 2 \
        --partition-json results/HGB_DBLP/partition.json \
        --k-values 8 16 32 64 \
        --w-values 8 32 128 \
        --run-id $s \
        --hash-seed $s \
        $([ $s -gt 0 ] && echo "--skip-exact")
done
```

## Step 4 — Analyze

```bash
python scripts/exp_lp_analyze.py --datasets HGB_DBLP HGB_ACM HGB_IMDB HNE_PubMed
```

**Output**:
- `figures/lp/table1_mrr_auc.csv` — main quality table (Exact / KMV / MPRW × datasets)
- `figures/lp/table2_scaling.csv` — time / memory / edges
- `figures/lp/table3_tail_mid_hub.csv` — per-degree-bin MRR
- `figures/lp/plot_mrr_bars.pdf` — bar plot per dataset
- `figures/lp/plot_mrr_vs_density.pdf` — MRR vs edge count (common density axis)
- `figures/lp/plot_perbin.pdf` — per-degree-bin MRR bars

## Known issues

### 1. Dense graphs (ACM PTP)
Full H has millions of edges × thousands of features → GPU OOM on inference.
**Fix**: `--force-cpu` flag.

### 2. High-dim features (IMDB, in_dim=3489)
Message buffer `|E| × in_dim × 4B` exceeds any RAM during SAGE forward.
**Fix**: add a feature projection layer (NOT YET IMPLEMENTED) OR use sparse SpMM forward pass (see `scripts/inference_worker.py` for template).

### 3. 2-hop negative sampling on dense graphs
Original code: O(mean_deg²) per query. On ACM PTP (mean_deg=1208), 10M ops per sample × 10K samples = 1e11 ops/epoch.
**Fix**: `_sample_neg_2hop` in `exp_lp_train.py` auto-detects dense mode (>30% density) and switches to rejection sampling from V \\ N1.

## CSV schema (master_lp_results.csv)

```
Dataset, MetaPath, L, Method, k_value, w_value, Seed,
Materialization_Time, Inference_Time, Mat_RAM_MB, Inf_RAM_MB,
Edge_Count, Graph_Density,
MRR, ROC_AUC, Hits_1, Hits_10, AP, Recall_10,
MRR_tail, MRR_mid, MRR_hub,
ROC_AUC_tail, ROC_AUC_mid, ROC_AUC_hub,
n_test_pos, exact_status
```

- `Method`: "Exact" / "KMV" / "MPRW"
- `k_value`: sketch size (KMV only; empty for Exact/MPRW)
- `w_value`: walk budget (MPRW only; empty for Exact/KMV)
- `Seed`: replicate id (0-indexed). Exact uses Seed=0.
- `MRR_tail/mid/hub`: MRR stratified by source-node degree bin in Exact H (tail=0-50%ile, mid=50-90%, hub=top 10%)
- `exact_status`: "OK" / "INF_OOM(...)" / "MAT_FAIL(...)"

## HGB protocol compliance

- Split: 40% V_train / 60% V_test (existing partition.json). Semi-inductive — train on H_train, test on edges involving V_test.
- Negative sampling: **2-hop neighbor sampling in H_exact** (NOT uniform random — HGB recommendation).
- Metrics: **MRR + ROC-AUC + Hits@10** headline; AP + Recall@10 secondary.
- Decoder: dot product. HGB uses DistMult for KG-style (Amazon, PubMed); we use dot for homophily-style (DBLP, IMDB, ACM).

## Expected wall-clock (seed 0 full sweep, single GPU + CPU fallback)

| Dataset | Train | Inference (k-sweep + w-sweep) |
|---------|-------|-------------------------------|
| HGB_DBLP | ~2s | ~3s |
| HGB_ACM | ~3 min | ~5 min (CPU) |
| HGB_IMDB | ~20 min (small config) | ~10 min |
| HNE_PubMed | ~10 min (CPU) | ~15 min (CPU) |

For 5 seeds, multiply inference by ~5.

## Reference: design doc

See `final_report/research_notes/18_lp_pipeline_design.md` for the full rationale.
