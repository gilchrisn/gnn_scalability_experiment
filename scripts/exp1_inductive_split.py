"""
Experiment 1 — Inductive Temporal Split.

Determines the V_train / V_infer partition for a dataset and saves it to
results/<dataset>/partition.json.  All downstream scripts (exp2_train.py,
exp3_k_sweep.py) load this file instead of recomputing the split.

Partitioning strategy (auto-detected):
  temporal_year   Any node type carries a .year attr (OGB_MAG, OAG_CS).
                  V_train = nodes with year ≤ year_cutoff(train_frac).
  random_nodes    No year attr (HGB_DBLP, HGB_ACM, HGB_IMDB).
                  V_train = first train_frac of target nodes in a fixed
                  random permutation (seed derived from dataset name).

In both cases the subgraph is a proper induced subgraph — no edge slicing.

Output
------
  results/<dataset>/partition.json

partition.json schema
---------------------
{
  "dataset":        "HGB_DBLP",
  "partition_mode": "random_nodes",
  "pivot_ntype":    "author",
  "train_frac":     0.1,
  "infer_fracs":    [0.28, 0.46, 0.64, 0.82, 1.0],
  "seed":           68076          // random_nodes only
}

Usage
-----
    python scripts/exp1_inductive_split.py HGB_DBLP
    python scripts/exp1_inductive_split.py OGB_MAG --train-frac 0.1 --infer-steps 4
    python scripts/exp1_inductive_split.py HGB_ACM --train-frac 0.2 --infer-fracs 0.4 0.7 1.0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from torch_geometric.data import HeteroData

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.config import config
from src.data import DatasetFactory


DEFAULT_TRAIN_FRAC  = 0.1
DEFAULT_INFER_STEPS = 5


# ---------------------------------------------------------------------------
# Induced-subgraph preview  (no data saved — just count nodes for sanity check)
# ---------------------------------------------------------------------------

def _count_train_nodes(
    g: HeteroData,
    pivot_ntype: str,
    keep_ids: torch.Tensor,
) -> dict:
    """Count surviving nodes per type after induced-subgraph filtering."""
    id_map = torch.full((g[pivot_ntype].num_nodes,), -1, dtype=torch.long)
    id_map[keep_ids] = torch.arange(keep_ids.size(0))

    referenced: dict = {nt: set() for nt in g.node_types if nt != pivot_ntype}
    for src_type, rel, dst_type in g.edge_types:
        ei    = g[src_type, rel, dst_type].edge_index
        valid = torch.ones(ei.size(1), dtype=torch.bool)
        if src_type == pivot_ntype:
            valid &= id_map[ei[0]] >= 0
        if dst_type == pivot_ntype:
            valid &= id_map[ei[1]] >= 0
        if src_type != pivot_ntype and valid.any():
            referenced[src_type].update(ei[0][valid].tolist())
        if dst_type != pivot_ntype and valid.any():
            referenced[dst_type].update(ei[1][valid].tolist())

    counts = {pivot_ntype: keep_ids.size(0)}
    for nt, ids in referenced.items():
        counts[nt] = len(ids)
    return counts


def _keep_ids_for_frac(
    g: HeteroData,
    pivot_ntype: str,
    partition_mode: str,
    train_frac: float,
    seed: Optional[int],
) -> torch.Tensor:
    if partition_mode == "temporal_year":
        years       = g[pivot_ntype].year.squeeze()
        sorted_years, _ = years.sort()
        cutoff_idx  = max(1, int(train_frac * len(sorted_years))) - 1
        year_cutoff = sorted_years[cutoff_idx].item()
        return (years <= year_cutoff).nonzero(as_tuple=False).squeeze(1)
    else:  # random_nodes
        gen  = torch.Generator()
        gen.manual_seed(seed)
        n    = g[pivot_ntype].num_nodes
        perm = torch.randperm(n, generator=gen)
        return perm[:max(1, int(train_frac * n))]


def _count_induced_edges(g: HeteroData, pivot_ntype: str, keep_ids: torch.Tensor) -> int:
    """Count edges in the induced subgraph without allocating the full HeteroData."""
    id_map = torch.full((g[pivot_ntype].num_nodes,), -1, dtype=torch.long)
    id_map[keep_ids] = torch.arange(keep_ids.size(0))

    # Pass 1: find surviving non-pivot nodes
    referenced = {nt: set() for nt in g.node_types if nt != pivot_ntype}
    for src_type, rel, dst_type in g.edge_types:
        ei    = g[src_type, rel, dst_type].edge_index
        valid = torch.ones(ei.size(1), dtype=torch.bool)
        if src_type == pivot_ntype:
            valid &= id_map[ei[0]] >= 0
        if dst_type == pivot_ntype:
            valid &= id_map[ei[1]] >= 0
        if src_type != pivot_ntype and valid.any():
            referenced[src_type].update(ei[0][valid].tolist())
        if dst_type != pivot_ntype and valid.any():
            referenced[dst_type].update(ei[1][valid].tolist())

    # Build id_maps for non-pivot types
    all_maps = {pivot_ntype: id_map}
    for ntype, ids in referenced.items():
        m = torch.full((g[ntype].num_nodes,), -1, dtype=torch.long)
        if ids:
            old = torch.tensor(sorted(ids), dtype=torch.long)
            m[old] = torch.arange(old.size(0))
        all_maps[ntype] = m

    # Pass 2: count edges where both endpoints survive
    total = 0
    for src_type, rel, dst_type in g.edge_types:
        ei    = g[src_type, rel, dst_type].edge_index
        valid = (all_maps[src_type][ei[0]] >= 0) & (all_maps[dst_type][ei[1]] >= 0)
        total += int(valid.sum())
    return total


_LOG_FIELDS = [
    "dataset", "timestamp", "partition_mode", "pivot_ntype", "train_frac",
    "total_nodes", "total_edges", "train_nodes", "train_edges",
    "kmv_seed", "suggested_metapaths",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset",
                        help="Dataset key, e.g. HGB_DBLP, OGB_MAG")
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC,
                        help="Fraction of nodes used for training (default %(default)s)")
    parser.add_argument("--infer-fracs", type=float, nargs="+", default=None,
                        metavar="F",
                        help="Explicit inference fractions (must end at 1.0). "
                             "Default: auto-derive --infer-steps steps.")
    parser.add_argument("--infer-steps", type=int, default=DEFAULT_INFER_STEPS,
                        help="Auto-derive this many inference snapshots (default %(default)s)")
    parser.add_argument("--pivot-ntype", type=str, default=None,
                        help="Node type to fraction. Overrides auto-detection. "
                             "E.g. 'paper' to grow papers, 'author' to grow authors.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for random_nodes mode. "
                             "Default: derived from dataset name (reproducible).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Derive inference fractions
    # ------------------------------------------------------------------
    if args.infer_fracs is not None:
        infer_fracs = sorted(args.infer_fracs)
        assert infer_fracs[-1] == 1.0, "--infer-fracs must end at 1.0"
    else:
        step        = (1.0 - args.train_frac) / args.infer_steps
        infer_fracs = [round(args.train_frac + step * (i + 1), 6)
                       for i in range(args.infer_steps)]
        infer_fracs[-1] = 1.0

    assert all(f > args.train_frac for f in infer_fracs), \
        "All inference fractions must exceed train_frac"

    # ------------------------------------------------------------------
    # Load dataset (read-only — no staging)
    # ------------------------------------------------------------------
    cfg = config.get_dataset_config(args.dataset)
    print(f"\n{'='*60}")
    print(f"Exp1  |  dataset={args.dataset}  train_frac={args.train_frac}")
    print(f"{'='*60}")

    g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    target_ntype = cfg.target_node

    # ------------------------------------------------------------------
    # Detect partition mode
    # ------------------------------------------------------------------
    time_ntype = next(
        (nt for nt in g.node_types if hasattr(g[nt], "year")), None
    )

    # --pivot-ntype overrides auto-detection
    if args.pivot_ntype is not None:
        assert args.pivot_ntype in g.node_types, \
            f"--pivot-ntype '{args.pivot_ntype}' not in {g.node_types}"
        pivot_ntype = args.pivot_ntype
        # Use temporal if the chosen type has year, else random
        if hasattr(g[pivot_ntype], "year"):
            partition_mode = "temporal_year"
            seed           = None
        else:
            partition_mode = "random_nodes"
            seed           = args.seed if args.seed is not None \
                             else sum(ord(c) for c in args.dataset) % (2**31)
    elif time_ntype is not None:
        partition_mode = "temporal_year"
        pivot_ntype    = time_ntype
        seed           = None
    else:
        partition_mode = "random_nodes"
        pivot_ntype    = target_ntype
        seed           = args.seed if args.seed is not None \
                         else sum(ord(c) for c in args.dataset) % (2**31)

    print(f"Mode: {partition_mode}  pivot='{pivot_ntype}'"
          + (f"  seed={seed}" if seed is not None else ""))

    # ------------------------------------------------------------------
    # Preview node counts at each fraction
    # ------------------------------------------------------------------
    print(f"\nNode counts per snapshot:")
    print(f"  {'fraction':>10}  {pivot_ntype:>20}  (induced)")
    for frac in [args.train_frac] + infer_fracs:
        keep_ids = _keep_ids_for_frac(g, pivot_ntype, partition_mode, frac, seed)
        counts   = _count_train_nodes(g, pivot_ntype, keep_ids)
        tag      = " <- train" if frac == args.train_frac else ""
        pivot_n  = counts[pivot_ntype]
        other    = {k: v for k, v in counts.items() if k != pivot_ntype}
        print(f"  {frac:>10.0%}  {pivot_n:>20,}  {other}{tag}")

    # ------------------------------------------------------------------
    # Compute partition stats + generate kmv_seed
    # ------------------------------------------------------------------
    print("\nComputing partition stats...")
    total_nodes = sum(g[nt].num_nodes for nt in g.node_types)
    total_edges = sum(g[et].edge_index.size(1) for et in g.edge_types)

    keep_ids_train = _keep_ids_for_frac(g, pivot_ntype, partition_mode, args.train_frac, seed)
    train_nodes    = sum(_count_train_nodes(g, pivot_ntype, keep_ids_train).values())
    train_edges    = _count_induced_edges(g, pivot_ntype, keep_ids_train)
    kmv_seed       = random.randint(0, 2**31 - 1)

    cfg_tmp        = config.get_dataset_config(args.dataset)
    suggested_mps  = "|".join(cfg_tmp.suggested_paths)

    print(f"  total_nodes={total_nodes:,}  total_edges={total_edges:,}")
    print(f"  train_nodes={train_nodes:,}  train_edges={train_edges:,}")
    print(f"  kmv_seed={kmv_seed}")

    # ------------------------------------------------------------------
    # Write partition.json
    # ------------------------------------------------------------------
    out_dir = Path("results") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    part_path = out_dir / "partition.json"

    record: dict = {
        "dataset":        args.dataset,
        "partition_mode": partition_mode,
        "pivot_ntype":    pivot_ntype,
        "train_frac":     args.train_frac,
        "infer_fracs":    infer_fracs,
        "kmv_seed":       kmv_seed,
    }
    if seed is not None:
        record["seed"] = int(seed)

    with open(part_path, "w") as f:
        json.dump(record, f, indent=2)

    # ------------------------------------------------------------------
    # Write / append partition_log.csv
    # ------------------------------------------------------------------
    log_path = out_dir / "partition_log.csv"
    is_new   = not log_path.exists() or log_path.stat().st_size == 0
    with open(log_path, "a", newline="", encoding="utf-8") as lf:
        w = csv.DictWriter(lf, fieldnames=_LOG_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({
            "dataset":            args.dataset,
            "timestamp":          datetime.now().isoformat(timespec="seconds"),
            "partition_mode":     partition_mode,
            "pivot_ntype":        pivot_ntype,
            "train_frac":         args.train_frac,
            "total_nodes":        total_nodes,
            "total_edges":        total_edges,
            "train_nodes":        train_nodes,
            "train_edges":        train_edges,
            "kmv_seed":           kmv_seed,
            "suggested_metapaths": suggested_mps,
        })

    print(f"\nSaved → {part_path}")
    print(f"Logged → {log_path}")


if __name__ == "__main__":
    main()
