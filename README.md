# KMV Graph Reconstruction for Scalable HGNN Inference

Code accompanying the paper **"KMV Graph Reconstruction: Sketch-Based Approximate Adjacency for Scalable HGNN Inference"** — an extension of the KMV sketch propagation framework of Niu et al. (ICDE 2025) to the heterogeneous-GNN inference setting.

## What's here

```
.
├── src/
│   ├── sketch_feature/  # Path 1 sketch-feature pipeline (LoNe-typed)
│   ├── kernels/         # Python KMV propagation kernel
│   ├── bridge/          # bridge to C++ binaries
│   └── data/            # dataset loaders (HGB / HNE / OGB)
├── scripts/             # experiment driver scripts (see CLAUDE.md for the catalog)
├── csrc/                # C++ source for the MPRW baseline
├── HUB/                 # C++ source for Exact + KMV (from the base paper, used unmodified)
├── tests/               # pytest unit tests
├── main.py              # CLI entry point (legacy, base-paper centrality)
├── Makefile             # builds bin/graph_prep and bin/mprw_exec
├── RESULTS_REPRODUCING.md   # walkthrough from clone to every reported number
└── requirements.txt
```

## Setup

```bash
# 1. Python environment
python -m venv .venv
source .venv/bin/activate         # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt

# 2. C++ binaries (Exact + KMV from HUB/, MPRW from csrc/)
make
```

`torch_geometric`, `torch_sparse`, and `torch_scatter` must match your local CUDA / PyTorch version.

## Quickstart

Single-dataset benchmark (Exact materialisation, GraphSAGE inference):

```bash
python main.py benchmark --dataset HGB_ACM --method exact
```

KMV at sketch budget `k=16` with fidelity check against Exact:

```bash
python main.py benchmark --dataset HGB_ACM --method kmv --k 16 --check-fidelity
```

## Reproducing the paper

**Journal extension (current)**: full reproduction walkthrough in [`RESULTS_REPRODUCING.md`](RESULTS_REPRODUCING.md). The headline scripts:

```bash
# Sketch-as-feature NC across HGB/HNE (3 seeds).
python scripts/run_sketch_feature_sweep.py --num-seeds 3 --backbone mlp

# Sketch-as-sparsifier NC across HGB/HNE (3 seeds).
python scripts/run_sketch_sparsifier_sweep.py --num-seeds 3

# Simple-HGN baseline.
python scripts/exp_simple_hgn_baseline.py HGB_DBLP --num-seeds 3 \
    --hidden-dim 64 --n-heads 4 --epochs 100

# Multi-query amortisation cost breakdown.
python scripts/exp_multi_query_amortization.py HGB_DBLP --k 32 --seed 42

# Compile master tables + figures for the paper.
python scripts/compile_master_results.py
python scripts/plot_session_results.py
```

**Base-paper centrality (legacy)**: the experimental grid (KMV `k`-sweep, MPRW `w`-sweep, depth `L`-sweep) for one dataset is `scripts/run_full_pipeline.py` + `scripts/exp4_visualize.py`. See [`CLAUDE.md`](CLAUDE.md) §"Active Scripts" for the full catalog.

## Citation

If you use this code, please also cite the base paper:

```bibtex
@inproceedings{niu2025sketch,
  author    = {Niu, Yudong and Li, Yuchen and Karras, Panagiotis and Wang, Yanhao},
  title     = {A Sketch Propagation Framework for Hub Queries on Unmaterialized Relational Graphs},
  booktitle = {ICDE},
  year      = {2025},
}
```

## License

GPL-3.0 (see [`LICENSE`](LICENSE)). The `HUB/` subtree originates from the base paper's authors and retains its own licensing terms.
