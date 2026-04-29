# Reproducing the Journal Extension Results

This document walks a fresh reader from `git clone` to every numerical result and figure reported in the journal extension. All numbers cited below were measured on the local Windows + NVIDIA-GPU workstation; server-only results (OGB-MAG, OAG-CS) live in their own subsection.

## 1. Environment

```bash
# Windows: activate the venv created during initial setup.
source .venv/Scripts/activate
export PYTHONUTF8=1

# (One-time) install Python deps. torch / torch_geometric / torch_sparse /
# torch_scatter must match your local CUDA + torch versions; see
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install -r requirements.txt
```

If `import torch_sparse` fails, re-install the PyG companion wheels matching your CUDA build. The journal extension's sketch pipeline does not strictly need `torch_sparse`, but the legacy KMV propagation in `src/kernels/kmv.py` does.

## 2. Build the C++ binaries (only required for the legacy LP pipeline and base-paper centrality reproduction)

```bash
make clean && make
```

Produces `bin/graph_prep` (KMV propagation + ExactD/H) and `bin/mprw_exec` (MPRW baseline). The journal extension's new sketch-feature / sketch-sparsifier / sketch-LP scripts use the **Python** `KMVSketchingKernel` and do not require the C++ binaries. Building them is only needed for: (a) reproducing the base-paper centrality numbers and (b) running the legacy `scripts/exp_lp_*.py` pipeline.

## 3. The experiments to reproduce

All reported numbers are aggregated under `results/` after running the relevant scripts. The compile script `scripts/compile_master_results.py` walks every per-(dataset, method, seed) JSON and produces the paper's master tables.

### 3.1 Quality (NC test macro-F1) — `master_table_quality.md`

```bash
# Sketch-as-feature (LoNe-typed MLP) — 4 datasets × 3 seeds.
python scripts/run_sketch_feature_sweep.py --num-seeds 3 --backbone mlp

# Sketch-as-sparsifier (SAGE on union of sketch-decoded edges) — 4 × 3.
python scripts/run_sketch_sparsifier_sweep.py --num-seeds 3

# Simple-HGN baseline (PyG port; reduced config to fit ACM in memory).
for ds in HGB_DBLP HGB_ACM HGB_IMDB HNE_PubMed; do
  python scripts/exp_simple_hgn_baseline.py "$ds" --num-seeds 3 \
      --hidden-dim 64 --n-heads 4 --epochs 100
done

# Compile the master tables.
python scripts/compile_master_results.py
```

After the above, the following will exist:

- `results/<DATASET>/sketch_feature_pilot_k32_mlp_seed{42,43,44}.json`
- `results/<DATASET>/sketch_sparsifier_pilot_k32_seed{42,43,44}.json`
- `results/<DATASET>/simple_hgn_baseline_seed{42,43,44}.json`
- `results/master_results.csv` (49 rows)
- `results/master_table_quality.md` (paper-ready table)

### 3.2 Similarity fidelity — `master_table_similarity.md`

```bash
# Run on DBLP and ACM (HGB-IMDB and HNE-PubMed give degenerate results
# due to multi-label / sparse-label respectively).
for ds in HGB_DBLP HGB_ACM; do
  python scripts/exp_sketch_similarity.py "$ds" --k 32 --seed 42 --sample 200
done

python scripts/compile_master_results.py
```

Produces `results/master_table_similarity.md`.

### 3.3 Multi-query amortisation — `master_table_amortization.md`

```bash
# Per-dataset cost breakdown (Q=2: NC + Sim).
for ds in HGB_DBLP HGB_ACM; do
  python scripts/exp_multi_query_amortization.py "$ds" --k 32 --seed 42
done

python scripts/compile_master_results.py
```

Produces `results/master_table_amortization.md` and `results/<DS>/multi_query_amortization_k32_seed42.json`.

This script:
1. Deletes the cached sketch bundle so precompute is timed cold.
2. Runs sketch-feature MLP for NC.
3. Runs sketch similarity (slot-Jaccard).
4. Runs Simple-HGN baseline if not already cached.
5. Reports KMV total {NC + Sim} vs Baseline total {SHGN-NC + exact-Jaccard}.

### 3.4 Multi-task SHGN baseline — fair multi-query comparison

Per `final_report/research_notes/PLAN_journal_extension_remaining.md` §A4, the per-task SHGN × Q comparison favours us. The fair baseline trains one SHGN encoder with two heads (NC + LP) jointly:

```bash
# Requires a partition.json (from exp1_partition.py).
python scripts/exp1_partition.py --dataset HGB_DBLP \
    --target-type author --train-frac 0.4 --seed 42

python scripts/exp_simple_hgn_multitask.py HGB_DBLP \
    --partition-json results/HGB_DBLP/partition.json \
    --num-seeds 3 --epochs 100 --hidden-dim 64 --n-heads 4
```

Produces `results/HGB_DBLP/simple_hgn_multitask_seed{42,43,44}.json`. Compare its joint training cost against KMV's `precompute + NC consume + LP consume`.

### 3.5 LP via the new sketch pipeline

```bash
# Requires partition.json.
python scripts/exp_sketch_lp_train.py HGB_DBLP \
    --partition-json results/HGB_DBLP/partition.json \
    --k 32 --emb-dim 128 --epochs 100 --seed 42
```

Produces `results/HGB_DBLP/sketch_lp_pilot_k32_seed42.json`.

### 3.6 HGSampling baseline (sampler overhead measurement)

```bash
for ds in HGB_DBLP HGB_ACM HGB_IMDB HNE_PubMed; do
  python scripts/exp_hgsampling_baseline.py "$ds" --num-seeds 3 \
      --hidden-dim 64 --n-heads 4 --epochs 50
done
```

Produces `results/<DATASET>/hgsampling_baseline_seed{42,43,44}.json`. The `sampler_fraction` field measures how much of training wall-clock is sampling vs. forward+backward. On HGB-scale graphs this fraction is small (~5%); on OGB-MAG it dominates.

## 4. Figures

Once all of §3 has been run:

```bash
python scripts/plot_session_results.py
```

Produces five figures under `figures/sketch_session/` (PDF + PNG):

| File | What it shows | Cited in paper § |
|---|---|---|
| `quality_per_dataset` | Grouped bar of test F1 across consumer modes vs Simple-HGN | §7 NC quality |
| `cost_breakdown_q2`   | Stacked bar of KMV vs SHGN+exact total cost at Q=2 | §7 amortisation |
| `amortization_curve`  | Projected total wall-clock vs Q | §7 amortisation |
| `similarity_speedup`  | Sketch-vs-exact Jaccard speedup per meta-path | §7 similarity |
| `quality_vs_time`     | Per-(dataset, method, seed) Pareto scatter | §7 quality–cost trade-off |

## 5. Hardware caveats

All numbers in `master_table_*.md` were measured on:
- Windows 11, NVIDIA GPU, single workstation.
- `train_time_s` is GPU wall-clock around the training loop only (CUDA warm-up dominates the first call inside a process; the amortisation script measures cold-start to be honest about precompute).
- Simple-HGN on ACM uses `hidden_dim=64, n_heads=4` instead of 128/8 because the latter blows past the GPU's memory budget on the 10-edge-type ACM graph. The smaller config matches the paper's published number (~0.94), so this is a hardware portability note, not a fidelity issue.

A reviewer wanting to reproduce on Linux + a different GPU should expect:
- Identical F1 numbers up to seed-level variance (no hardware-dependence).
- Different absolute wall-clock numbers; the **ratios** between methods (e.g. KMV vs SHGN per-task speedup) should be roughly stable.

## 6. Server-only experiments (out of scope of local reproduction)

The journal extension's scalability claim depends on running on a dataset where exact materialisation OOMs. For local reproduction this is **not possible** with HGB or HNE benchmarks; it requires OGB-MAG or OAG-CS on a server with $\geq$ 64 GB RAM.

When the server runs land, they will be documented in a separate appendix with their own reproduction script. Until then, this section is a placeholder.

## 7. Index of all reported numbers

| Where in paper | Source script | Source artefact |
|---|---|---|
| §7 quality table | `run_sketch_*_sweep.py` + `exp_simple_hgn_baseline.py` | `master_table_quality.md` |
| §7 fidelity table | `exp_sketch_similarity.py` | `master_table_similarity.md` |
| §7 amortisation table | `exp_multi_query_amortization.py` | `master_table_amortization.md` |
| §7 multi-task fair-baseline | `exp_simple_hgn_multitask.py` | `simple_hgn_multitask_seed*.json` |
| §7 sampler-overhead row | `exp_hgsampling_baseline.py` | `hgsampling_baseline_seed*.json` |
| §7 figure 1 (quality bars) | `plot_session_results.py` | `figures/sketch_session/quality_per_dataset.{pdf,png}` |
| §7 figure 2 (cost breakdown) | `plot_session_results.py` | `figures/sketch_session/cost_breakdown_q2.{pdf,png}` |
| §7 figure 3 (amortisation curve) | `plot_session_results.py` | `figures/sketch_session/amortization_curve.{pdf,png}` |
| §7 figure 4 (similarity speedup) | `plot_session_results.py` | `figures/sketch_session/similarity_speedup.{pdf,png}` |
| §7 figure 5 (quality vs time) | `plot_session_results.py` | `figures/sketch_session/quality_vs_time.{pdf,png}` |

## 8. Reset / re-run a specific subset

The aggregator `compile_master_results.py` is idempotent — re-running it picks up new JSONs without disturbing existing ones. To force a re-run of a method on a specific dataset, delete the relevant per-(seed) JSON:

```bash
rm results/HGB_DBLP/sketch_feature_pilot_k32_mlp_seed42.json
python scripts/run_sketch_feature_sweep.py --datasets HGB_DBLP --num-seeds 3 --backbone mlp
```

Sweep scripts skip already-cached results unless `--force` is passed.

## 9. Plan for missing pieces

The complete plan for what's still missing (OGB-MAG, theorem write-up, paper §6/§7/§8 LaTeX, etc.) lives in `final_report/research_notes/PLAN_journal_extension_remaining.md`.
