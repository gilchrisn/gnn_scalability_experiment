"""
Probe HGB_DBLP, HGB_ACM, HGB_IMDB for temporal (year) metadata on node types.
Determines whether node-level temporal partitioning is viable for each dataset.

Usage:
    python scripts/probe_year_attrs.py
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from src.config import config
from src.data.factory import DatasetFactory

DATASETS_TO_PROBE = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB"]


def probe(key: str):
    cfg = config.get_dataset_config(key)
    print(f"\n{'='*60}")
    print(f"Dataset: {key}  (target={cfg.target_node})")
    print(f"{'='*60}")
    try:
        g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return

    found_year = False
    for ntype in g.node_types:
        store = g[ntype]
        attrs = list(store.keys())
        print(f"  [{ntype}]  num_nodes={store.num_nodes}  attrs={attrs}")
        for attr in attrs:
            val = getattr(store, attr, None)
            if not isinstance(val, torch.Tensor):
                continue
            if attr == "year" or "year" in attr.lower() or "time" in attr.lower():
                year_min = val.min().item()
                year_max = val.max().item()
                n_unique = val.unique().numel()
                print(f"    --> TEMPORAL attr '{attr}': "
                      f"min={year_min}  max={year_max}  n_unique={n_unique}  shape={tuple(val.shape)}")
                found_year = True

    if not found_year:
        print(f"  !! No year/time attribute found on any node type.")


if __name__ == "__main__":
    for key in DATASETS_TO_PROBE:
        probe(key)

    print("\n\nSUMMARY")
    print("-" * 40)
    print("If a dataset has a temporal attr, temporal node partitioning is viable.")
    print("Otherwise it must fall back to random stratified edge splitting.")
