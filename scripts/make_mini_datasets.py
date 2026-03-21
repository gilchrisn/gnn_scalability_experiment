"""
Create small sampled versions of large datasets for local pipeline testing.
Saves to datasets/<KEY>_mini/ so the full data is untouched.

Usage:
    python scripts/make_mini_datasets.py --datasets OGB_MAG OAG_CS RCDD_AliRCD
    python scripts/make_mini_datasets.py --datasets OGB_MAG  --n-target 500
"""
import argparse
import os
import sys
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.factory import DatasetFactory
from src.config import config


def sample_hetero(g, target_ntype: str, n_target: int, n_other: int = 2000, seed: int = 42):
    """
    Sample a small subgraph with bounded size:
      1. Pick n_target nodes of target_ntype at random.
      2. For every other node type, keep at most n_other nodes (those that appear
         as neighbours of the sampled targets, capped at n_other).
      3. Retain only edges where both endpoints are in the kept sets.
    This avoids the blow-up caused by dense citation / co-authorship graphs.
    """
    from torch_geometric.data import HeteroData

    torch.manual_seed(seed)
    num_target = g[target_ntype].num_nodes
    target_idx = torch.randperm(num_target)[:min(n_target, num_target)]
    keep = {target_ntype: target_idx}

    # Pass 1: direct (1-hop) neighbours of target nodes
    for src, rel, dst in g.edge_types:
        ei = g[src, rel, dst].edge_index
        if src == target_ntype and dst != target_ntype:
            mask = torch.isin(ei[0], keep[target_ntype])
            nb = ei[1][mask].unique()[:n_other]
            keep[dst] = nb if dst not in keep else torch.cat([keep[dst], nb]).unique()[:n_other]
        elif dst == target_ntype and src != target_ntype:
            mask = torch.isin(ei[1], keep[target_ntype])
            nb = ei[0][mask].unique()[:n_other]
            keep[src] = nb if src not in keep else torch.cat([keep[src], nb]).unique()[:n_other]

    # Pass 2: 2-hop neighbours (e.g. author->institution for PAIAP)
    hop1_types = set(keep.keys()) - {target_ntype}
    for src, rel, dst in g.edge_types:
        ei = g[src, rel, dst].edge_index
        if src in hop1_types and dst not in keep:
            mask = torch.isin(ei[0], keep[src])
            nb = ei[1][mask].unique()[:n_other]
            if len(nb) > 0:
                keep[dst] = nb
        elif dst in hop1_types and src not in keep:
            mask = torch.isin(ei[1], keep[dst])
            nb = ei[0][mask].unique()[:n_other]
            if len(nb) > 0:
                keep[src] = nb

    # Build remapping and mini graph
    remap = {}
    mini = HeteroData()
    for ntype, nodes in keep.items():
        nodes, _ = nodes.sort()
        keep[ntype] = nodes
        remap[ntype] = torch.full((g[ntype].num_nodes,), -1, dtype=torch.long)
        remap[ntype][nodes] = torch.arange(len(nodes))
        mini[ntype].num_nodes = len(nodes)
        for attr in g[ntype].keys():
            val = g[ntype][attr]
            if isinstance(val, torch.Tensor) and val.size(0) == g[ntype].num_nodes:
                mini[ntype][attr] = val[nodes]

    for src, rel, dst in g.edge_types:
        if src not in keep or dst not in keep:
            continue
        ei = g[src, rel, dst].edge_index
        mask = torch.isin(ei[0], keep[src]) & torch.isin(ei[1], keep[dst])
        if mask.sum() == 0:
            continue
        new_src = remap[src][ei[0][mask]]
        new_dst = remap[dst][ei[1][mask]]
        mini[src, rel, dst].edge_index = torch.stack([new_src, new_dst])

    return mini


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['OGB_MAG'],
                        help='Dataset keys from config (e.g. OGB_MAG OAG_CS)')
    parser.add_argument('--n-target', type=int, default=1000,
                        help='Number of target nodes to keep (default 1000)')
    args = parser.parse_args()

    for key in args.datasets:
        cfg = config.get_dataset_config(key)
        print(f"\n[{key}] Loading full dataset...")
        try:
            g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        except Exception as e:
            print(f"  FAILED to load {key}: {e}")
            continue

        print(f"  Full size: {g.node_types}, target={cfg.target_node} "
              f"({g[cfg.target_node].num_nodes} nodes)")

        print(f"  Sampling {args.n_target} target nodes...")
        mini = sample_hetero(g, cfg.target_node, args.n_target)

        out_dir = os.path.join(config.DATA_DIR, f"{key}_mini")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "data.pt")
        torch.save(mini, out_path)

        print(f"  Saved mini graph -> {out_path}")
        print(f"  Mini node counts:")
        for nt in mini.node_types:
            print(f"    {nt}: {mini[nt].num_nodes}")
        print(f"  Mini edge counts:")
        for et in mini.edge_types:
            print(f"    {et}: {mini[et].edge_index.shape[1]}")


if __name__ == '__main__':
    main()
