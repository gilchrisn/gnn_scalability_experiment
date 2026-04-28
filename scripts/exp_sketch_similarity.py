"""Similarity-from-sketch evaluation — the third query class served from
the same KMV propagation pass.

Mechanism (Cohen 2014, Broder 1997 bottom-k):

For two coordinated bottom-k sketches K_u, K_v over the same hash
universe, the **slot-intersection Jaccard estimate** is:

    Ĵ(u, v) = |slots(u) ∩ slots(v)| / |slots(u) ∪ slots(v)|

with bias O(1/k) for k ≪ |universe| (under the standard min-hash
analysis). This is the third query mode in the multi-query framework:
no training, no GNN — Jaccard is read directly off the sketch.

What this script reports
------------------------

For each meta-path ρ in the dataset's default list:

  1. Time to compute the FULL pairwise Jaccard estimate matrix among
     a sample of S target nodes from the sketch (this is the operational
     cost of similarity from the sketch).
  2. Compare the estimate to the EXACT meta-path-induced Jaccard
     (computed via materialised adjacency) on the same S nodes; report
     mean abs error and Pearson correlation.

The exact computation is O(N·d̄·d̄) which is feasible only on the
HGB-scale graphs and only for the sampled S target nodes — perfectly
adequate for a paper-grade fidelity check.

Usage
-----
    python scripts/exp_sketch_similarity.py HGB_DBLP --k 32
    python scripts/exp_sketch_similarity.py HGB_ACM --sample 500
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.config import config
from src.data import DatasetFactory
from src.sketch_feature import (
    SketchBundle,
    decode_sketches,
    extract_sketches,
)

from scripts.exp_sketch_feature_train import _DEFAULT_META_PATHS  # noqa: E402

INFINITY = torch.iinfo(torch.int64).max


def _exact_meta_path_neighbors(g, mp_str: str, target: str) -> list:
    """Compute the exact meta-path-induced neighbor set for each target
    node — the ground truth against which the sketch's Jaccard estimate
    is checked.

    Walks the adjacency dictionaries directly. O(|E*|) per meta-path,
    which is fine on HGB-scale graphs.
    """
    rels = [r.strip() for r in mp_str.split(",")]
    by_rel = {et[1]: et for et in g.edge_types}
    chain = [by_rel[r] for r in rels]
    if chain[0][0] != target or chain[-1][2] != target:
        raise ValueError(f"Meta-path {mp_str!r} not anchored at {target!r}")

    n = g[target].num_nodes
    # Frontier sets, one per node. Start: {self}.
    frontiers = [{i} for i in range(n)]
    for src_t, _, dst_t in chain:
        ei = g[src_t, _, dst_t].edge_index
        adj_lists = [[] for _ in range(g[src_t].num_nodes)]
        for i in range(ei.size(1)):
            adj_lists[int(ei[0, i])].append(int(ei[1, i]))
        new_frontiers = [set() for _ in range(g[dst_t].num_nodes)]
        # Each frontier is over src_t at this hop, so we propagate:
        for v_idx_dst in range(g[dst_t].num_nodes):
            pass
        # Easier: iterate over frontier nodes of src_t and forward.
        if src_t == target:
            # First hop or symmetric meta-path; expand from the per-target
            # frontiers we built.
            for u in range(g[src_t].num_nodes):
                for x in frontiers[u]:
                    # x is in src_t universe; need to expand x's neighbors
                    # in src_t -> dst_t direction. But x at this point is
                    # actually in some intermediate type. Skip — see below.
                    pass
            # The above doesn't work in general because frontier elements
            # aren't always in target type. Use the fully-explicit approach:
            # propagate along the chain step by step, where frontiers[v]
            # holds the current set of nodes reachable from v.
            pass
        break  # bail and use the explicit version below

    # Explicit step-by-step propagation: frontier[v] is the set of nodes
    # of the *current type* reachable from v after the chain prefix.
    frontiers = [{i} for i in range(n)]
    cur_type = target
    for src_t, _, dst_t in chain:
        if cur_type != src_t:
            raise RuntimeError(f"chain mismatch: cur={cur_type} src={src_t}")
        ei = g[src_t, _, dst_t].edge_index
        adj = [[] for _ in range(g[src_t].num_nodes)]
        for i in range(ei.size(1)):
            adj[int(ei[0, i])].append(int(ei[1, i]))
        new_frontiers = []
        for fr in frontiers:
            new_fr = set()
            for u in fr:
                new_fr.update(adj[u])
            new_frontiers.append(new_fr)
        frontiers = new_frontiers
        cur_type = dst_t
    if cur_type != target:
        raise RuntimeError(f"final frontier type {cur_type} != target {target}")
    return frontiers  # list of sets, one per target node


def _jaccard_set(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    uni = len(a) + len(b) - inter
    return inter / uni if uni else 0.0


def _jaccard_sketch_pair(decoded_u: torch.Tensor, decoded_v: torch.Tensor) -> float:
    """Bottom-k slot-intersection Jaccard estimate for one pair."""
    a = set(int(x) for x in decoded_u.tolist() if x >= 0)
    b = set(int(x) for x in decoded_v.tolist() if x >= 0)
    return _jaccard_set(a, b)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--meta-paths", nargs="+")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample", type=int, default=200,
                   help="Number of target nodes to evaluate exact-vs-sketch "
                        "Jaccard on (the C(S,2) pair count is what dominates).")
    args = p.parse_args()

    cfg = config.get_dataset_config(args.dataset)
    target = cfg.target_node
    out_dir = Path("results") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_paths = args.meta_paths or _DEFAULT_META_PATHS.get(args.dataset)
    if not meta_paths:
        print(f"No default meta-paths for {args.dataset}", file=sys.stderr)
        return 2

    print(f"[load] {args.dataset} target={target}")
    g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, target)
    n_target = g[target].num_nodes

    # Reuse cached sketch bundle if available — same convention as the NC
    # scripts so the propagation cost is paid once across the three query
    # classes (the amortization story).
    cache_path = out_dir / f"sketch_bundle_k{args.k}_seed{args.seed}.pt"
    if cache_path.exists():
        bundle = SketchBundle.load(cache_path)
        if bundle.k != args.k or sorted(bundle.meta_paths) != sorted(meta_paths):
            cache_path.unlink()
            bundle = None
    else:
        bundle = None
    extract_time_s = 0.0
    if bundle is None:
        t0 = time.perf_counter()
        bundle = extract_sketches(
            g, meta_paths, target_type=target, k=args.k, seed=args.seed, device="cpu",
        )
        extract_time_s = time.perf_counter() - t0
        bundle.save(cache_path)
        print(f"[extract] propagated in {extract_time_s:.2f}s")
    else:
        print(f"[extract] reusing cached bundle: {cache_path}")

    rng = torch.Generator().manual_seed(args.seed)
    sample = torch.randperm(n_target, generator=rng)[: args.sample].tolist()
    print(f"[sample] evaluating {len(sample)} target nodes "
          f"({len(sample) * (len(sample) - 1) // 2} pairs)")

    per_mp_results = {}
    for mp in meta_paths:
        print(f"\n[{mp}]")
        decoded = decode_sketches(
            bundle.sketches_by_mp[mp], bundle.sorted_hashes, bundle.sort_indices
        )

        # Sketch Jaccard time (vectorisable but the small-S regime is
        # dominated by Python set ops; this is what the cost equation
        # actually pays for similarity).
        t0 = time.perf_counter()
        sketch_pairs = []
        for i, ui in enumerate(sample):
            for vj in sample[i + 1:]:
                sketch_pairs.append(_jaccard_sketch_pair(decoded[ui], decoded[vj]))
        t_sketch = time.perf_counter() - t0
        print(f"  sketch Jaccard: {len(sketch_pairs)} pairs in {t_sketch:.3f}s")

        # Exact reference: full meta-path neighbor sets, then Jaccard.
        t0 = time.perf_counter()
        exact_neighbors = _exact_meta_path_neighbors(g, mp, target)
        exact_pairs = []
        for i, ui in enumerate(sample):
            ai = exact_neighbors[ui]
            for vj in sample[i + 1:]:
                bj = exact_neighbors[vj]
                exact_pairs.append(_jaccard_set(ai, bj))
        t_exact = time.perf_counter() - t0
        print(f"  exact  Jaccard: {len(exact_pairs)} pairs in {t_exact:.3f}s")

        # Fidelity.
        diffs = [abs(s - e) for s, e in zip(sketch_pairs, exact_pairs)]
        mae = sum(diffs) / len(diffs)
        n = len(sketch_pairs)
        mean_s = sum(sketch_pairs) / n
        mean_e = sum(exact_pairs) / n
        var_s = sum((x - mean_s) ** 2 for x in sketch_pairs) / n
        var_e = sum((x - mean_e) ** 2 for x in exact_pairs) / n
        cov = sum((s - mean_s) * (e - mean_e) for s, e in zip(sketch_pairs, exact_pairs)) / n
        pearson = cov / ((var_s * var_e) ** 0.5) if var_s > 0 and var_e > 0 else float("nan")
        print(f"  fidelity: MAE={mae:.4f}  Pearson={pearson:.4f}  "
              f"speedup_sketch_vs_exact={t_exact / max(t_sketch, 1e-9):.2f}x")

        per_mp_results[mp] = {
            "n_pairs": n,
            "sketch_time_s": t_sketch,
            "exact_time_s": t_exact,
            "speedup_x": t_exact / max(t_sketch, 1e-9),
            "mae": mae,
            "pearson": pearson,
            "mean_sketch": mean_s,
            "mean_exact": mean_e,
        }

    out = {
        "dataset": args.dataset,
        "target_type": target,
        "method": "sketch_similarity_jaccard",
        "meta_paths": meta_paths,
        "k": args.k,
        "seed": args.seed,
        "sample": args.sample,
        "extract_time_s": extract_time_s,
        "per_meta_path": per_mp_results,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    res_path = out_dir / f"sketch_similarity_pilot_k{args.k}_seed{args.seed}.json"
    with open(res_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[save] {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
