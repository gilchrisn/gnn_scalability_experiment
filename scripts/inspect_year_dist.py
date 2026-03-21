"""
Print year distribution for OGB_MAG (and OAG_CS if available).
Run on server: python scripts/inspect_year_dist.py

Shows cumulative paper counts at each year to help pick a mid-scale cutoff.
"""
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from collections import Counter


def print_year_dist(name, years_tensor):
    years = years_tensor.squeeze().tolist()
    counts = Counter(years)
    total = len(years)
    cumulative = 0

    print(f"\n{'='*60}")
    print(f"  {name}: {total} papers")
    print(f"{'='*60}")
    print(f"  {'Year':>6}  {'Count':>8}  {'Cumul':>8}  {'%':>6}")
    print(f"  {'-'*36}")

    for year in sorted(counts.keys()):
        cumulative += counts[year]
        pct = 100.0 * cumulative / total
        marker = ""
        if 100_000 <= cumulative <= 350_000:
            marker = "  <-- mid-range"
        print(f"  {year:>6}  {counts[year]:>8}  {cumulative:>8}  {pct:>5.1f}%{marker}")


# --- OGB_MAG ---
print("Loading OGB_MAG...")
from ogb.nodeproppred import PygNodePropPredDataset
ds = PygNodePropPredDataset(name="ogbn-mag", root=os.path.join(project_root, "datasets", "OGB"))
g = ds[0]
print_year_dist("OGB_MAG (paper)", g["paper"].year)


# --- OAG_CS ---
try:
    print("\nLoading OAG_CS...")
    from H2GB.datasets import OAGDataset
    ds2 = OAGDataset(root=os.path.join(project_root, "datasets", "OAG"), name="cs")
    g2 = ds2[0]
    if hasattr(g2["paper"], "year"):
        print_year_dist("OAG_CS (paper)", g2["paper"].year)
    else:
        # Check for other time attributes
        print("OAG_CS paper attributes:", list(g2["paper"].keys()))
        for key in g2["paper"].keys():
            val = g2["paper"][key]
            if isinstance(val, torch.Tensor) and val.dim() <= 1:
                print(f"  {key}: min={val.min().item()}, max={val.max().item()}, shape={val.shape}")
except ImportError:
    print("H2GB not installed — skipping OAG_CS. Run: pip install H2GB")
except Exception as e:
    print(f"OAG_CS failed: {e}")
