"""
Experiment 1 — Deterministic Temporal / Stratified Partition.

Partitions ONLY the target node type into V_train and V_test.
All other node types (authors, subjects, venues …) are structural bridges
and are NOT partitioned.

Temporal path   g[target_type].year exists → sort by year, oldest train_frac.
Fallback path   No year attr → print bold warning, stratified random split
                (stratified by class label if available, otherwise plain random).

Output
------
  <out-dir>/partition.json
    Required keys: dataset, target_type, train_frac, seed, hash_seed,
                   is_temporal, cutoff_year, train_node_ids, test_node_ids

Usage
-----
    python scripts/exp1_partition.py --dataset HGB_ACM --target-type paper
    python scripts/exp1_partition.py --dataset OGB_MAG --target-type paper --train-frac 0.1
    python scripts/exp1_partition.py --dataset HGB_DBLP --target-type author --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.config import config
from src.data import DatasetFactory


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _temporal_split(g, target_type: str, train_frac: float):
    """Sort target nodes by year; oldest train_frac → train."""
    years = g[target_type].year.squeeze().numpy()
    order = np.argsort(years, kind="stable")          # stable: ties keep original order
    n_train  = max(1, int(train_frac * len(order)))
    train_ids = sorted(order[:n_train].tolist())
    test_ids  = sorted(order[n_train:].tolist())
    cutoff_year = int(years[order[n_train - 1]])
    return train_ids, test_ids, cutoff_year


def _stratified_split(g, target_type: str, train_frac: float, seed: int):
    """Stratified random split on target_type by class label (fallback)."""
    n = g[target_type].num_nodes
    rng = np.random.default_rng(seed)

    has_labels = hasattr(g[target_type], "y")
    labels     = g[target_type].y.numpy() if has_labels else None

    if labels is not None and labels.ndim == 1:
        from collections import defaultdict
        buckets = defaultdict(list)
        unlabeled = []
        for i, lbl in enumerate(labels):
            if int(lbl) >= 0:
                buckets[int(lbl)].append(i)
            else:
                unlabeled.append(i)

        train_ids = []
        for cls_ids in buckets.values():
            arr = np.array(cls_ids)
            rng.shuffle(arr)
            n_tr = max(1, int(train_frac * len(arr)))
            train_ids.extend(arr[:n_tr].tolist())

        # proportional share of unlabeled nodes
        unlab_arr = np.array(unlabeled)
        rng.shuffle(unlab_arr)
        n_unlab_tr = int(train_frac * len(unlab_arr))
        train_ids.extend(unlab_arr[:n_unlab_tr].tolist())
    else:
        # plain random
        arr = np.arange(n)
        rng.shuffle(arr)
        n_tr      = max(1, int(train_frac * n))
        train_ids = arr[:n_tr].tolist()

    train_ids = sorted(train_ids)
    test_ids  = sorted(set(range(n)) - set(train_ids))
    return train_ids, test_ids


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",     required=True)
    parser.add_argument("--target-type", required=True,
                        help="Node type to partition, e.g. paper / author / movie")
    parser.add_argument("--train-frac",  type=float, default=0.2,
                        help="Fraction of target nodes for training (default 0.2)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Master random seed (default 42)")
    parser.add_argument("--out-dir",     type=str,   default=None,
                        help="Output directory (default: results/<dataset>)")
    args = parser.parse_args()

    _set_seeds(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / args.dataset
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exp1 Partition  |  dataset={args.dataset}  target={args.target_type}")
    print(f"  train_frac={args.train_frac}  seed={args.seed}")
    print(f"{'='*60}")

    cfg = config.get_dataset_config(args.dataset)
    g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)

    assert args.target_type in g.node_types, (
        f"--target-type '{args.target_type}' not in graph node types: {g.node_types}"
    )

    n_total = g[args.target_type].num_nodes
    print(f"  {args.target_type}: {n_total:,} nodes total")

    # ------------------------------------------------------------------
    # Temporal split or stratified fallback
    # ------------------------------------------------------------------
    has_year = hasattr(g[args.target_type], "year")

    if has_year:
        is_temporal = True
        train_ids, test_ids, cutoff_year = _temporal_split(
            g, args.target_type, args.train_frac
        )
        print(f"  Mode: TEMPORAL  cutoff_year={cutoff_year}")
    else:
        is_temporal = False
        cutoff_year = None
        print(
            "\033[1m\033[33mWARNING: no 'year' attribute on '{}'. "
            "Falling back to stratified random split.\033[0m".format(args.target_type),
            file=sys.stderr,
        )
        train_ids, test_ids = _stratified_split(
            g, args.target_type, args.train_frac, args.seed
        )
        print(f"  Mode: STRATIFIED RANDOM (seed={args.seed})")

    print(f"  train={len(train_ids):,}  test={len(test_ids):,}  "
          f"({len(train_ids)/n_total:.1%} / {len(test_ids)/n_total:.1%})")
    assert len(train_ids) + len(test_ids) == n_total, "partition does not cover all nodes"

    # ------------------------------------------------------------------
    # Generate hash_seed for KMV consistency across all downstream scripts
    # ------------------------------------------------------------------
    hash_seed = random.randint(0, 2**31 - 1)
    print(f"  hash_seed={hash_seed}  (saved to partition.json for KMV)")

    # ------------------------------------------------------------------
    # Write partition.json
    # ------------------------------------------------------------------
    record = {
        "dataset":       args.dataset,
        "target_type":   args.target_type,
        "train_frac":    args.train_frac,
        "seed":          args.seed,
        "hash_seed":     hash_seed,
        "is_temporal":   is_temporal,
        "cutoff_year":   cutoff_year,
        "train_node_ids": train_ids,
        "test_node_ids":  test_ids,
    }

    part_path = out_dir / "partition.json"
    with open(part_path, "w") as f:
        json.dump(record, f)          # no indent — can be large for OGB_MAG

    print(f"\nSaved -> {part_path}  ({part_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
