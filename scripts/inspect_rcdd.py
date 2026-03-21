"""
Inspect RCDD schema after download and print suggested metapaths.
Run on the server after: python -c "from torch_geometric.datasets import RCDD; RCDD(root='datasets/RCDD')"

Usage:
    python scripts/inspect_rcdd.py
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from torch_geometric.datasets import RCDD

print("Loading RCDD...")
ds = RCDD(root=os.path.join(project_root, "datasets", "RCDD"))
g = ds[0]

print("\nNode types:")
for nt in g.node_types:
    x = g[nt].get('x', None)
    y = g[nt].get('y', None)
    print(f"  {nt}: num_nodes={g[nt].num_nodes}, "
          f"x={'x' + str(tuple(x.shape)) if x is not None else 'None'}, "
          f"y={'y' + str(tuple(y.shape)) if y is not None else 'None'}")

print("\nEdge types:")
for src, rel, dst in g.edge_types:
    ei = g[src, rel, dst].edge_index
    print(f"  ({src}, {rel}, {dst}): {ei.shape[1]} edges")

print("\nPossible symmetric metapaths through 'item':")
target = 'item'
for src, rel, dst in g.edge_types:
    if dst == target:
        rev_rel = f"rev_{rel}"
        if (target, rev_rel, src) in [(s, r, d) for s, r, d in g.edge_types]:
            mp = f"{rev_rel},{rel}"
        else:
            mp = f"rev_{rel},{rel}  (check reverse edge name)"
        print(f"  item-{src}-item via ({rel}): \"{mp}\"")
    elif src == target:
        rev_rel = f"rev_{rel}"
        print(f"  item-{dst}-item via ({rel}): \"{rel},{rev_rel}\"")

print("\nAdd the metapaths above to src/config.py under 'RCDD_AliRCD'.")
