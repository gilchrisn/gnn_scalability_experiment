"""
Inspect OAG-CS schema and print confirmed metapaths.
Must be run on a machine with enough RAM (~12GB) to process the raw file.
The raw file (graph_CS_20190919.pt) is already at datasets/OAG/cs/raw/.

Usage:
    python scripts/inspect_oag.py
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from H2GB.datasets import OAGDataset
except ImportError:
    print("H2GB not installed. Run: pip install H2GB")
    sys.exit(1)

print("Loading OAG-CS (requires ~12GB RAM)...")
ds = OAGDataset(root=os.path.join(project_root, "datasets", "OAG"), name="cs")
g = ds[0]

print("\nNode types:")
for nt in g.node_types:
    x = g[nt].get('x', None)
    y = g[nt].get('y', None)
    print(f"  {nt}: num_nodes={g[nt].num_nodes}, "
          f"x={tuple(x.shape) if x is not None else None}, "
          f"y={tuple(y.shape) if y is not None else None}")

print("\nEdge types:")
for src, rel, dst in g.edge_types:
    n = g[src, rel, dst].edge_index.shape[1]
    print(f"  ({src}, {rel}, {dst}): {n} edges")

print("\nSuggested symmetric metapaths through 'paper':")
target = 'paper'
seen = set()
for src, rel, dst in g.edge_types:
    if rel.startswith('rev_'):
        continue
    fwd_rel = rel
    rev_rel = f"rev_{rel}"
    if src == target and (target, fwd_rel, dst) in [(s,r,d) for s,r,d in g.edge_types]:
        mp = f"{fwd_rel},{rev_rel}"
        if mp not in seen:
            seen.add(mp)
            print(f"  paper-{dst}-paper: \"{mp}\"")
    if dst == target and (src, fwd_rel, target) in [(s,r,d) for s,r,d in g.edge_types]:
        mp = f"{rev_rel},{fwd_rel}"
        if mp not in seen:
            seen.add(mp)
            print(f"  paper-{src}-paper: \"{mp}\"")

print("\nUpdate src/config.py OAG_CS.suggested_paths with the metapaths above.")
