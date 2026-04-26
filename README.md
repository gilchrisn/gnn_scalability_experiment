# KMV Graph Reconstruction for Scalable HGNN Inference

Code accompanying the paper **"KMV Graph Reconstruction: Sketch-Based Approximate Adjacency for Scalable HGNN Inference"** — an extension of the KMV sketch propagation framework of Niu et al. (ICDE 2025) to the heterogeneous-GNN inference setting.

## What's here

```
.
├── src/                 # Python: bridge to C++ binaries, dataset loaders, GNN models
├── scripts/             # experiment driver scripts (exp1..exp18, bench_*, run_*)
├── csrc/                # C++ source for the MPRW baseline
├── HUB/                 # C++ source for Exact + KMV (from the base paper, used unmodified)
├── tests/               # pytest unit tests
├── main.py              # CLI entry point
├── Makefile             # builds bin/graph_prep and bin/mprw_exec
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

The experimental grid (KMV `k`-sweep, MPRW `w`-sweep, depth `L`-sweep) for one dataset:

```bash
python scripts/run_full_pipeline.py \
  --dataset HGB_DBLP \
  --metapath "author_to_paper,paper_to_term,term_to_paper,paper_to_author" \
  --target-type author \
  --k-values 2 4 8 16 32 64 \
  --w-values 1 2 4 8 16 32 64 128 256 512 \
  --kmv-reps 50 \
  --mprw-reps 10
```

Then regenerate the figures:

```bash
python scripts/exp4_visualize.py --dataset HGB_DBLP
```

Per-experiment scripts (`exp1_partition.py` through `exp18_rigor_check.py`) are individually invokable; each is documented in its own docstring.

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
