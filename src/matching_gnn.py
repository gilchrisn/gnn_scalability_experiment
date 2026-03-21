"""
MatchingGraphGNN: GNN inference on the matching graph G* without materializing H.

For a meta-path P = (x0, e0, x1, e1, ..., eL-1, xL), the matching graph G*
already exists inside HeteroData as the raw KG edges along the meta-path.
Instead of materializing the relational graph H (expensive), we run one
SAGEConv layer per hop through G* (cheap).

Theoretical equivalence (symmetric meta-paths, mean aggregation):
    MatchingGraphGNN(G*) ≡ 1-layer SAGE on H
    i.e. the two computations produce equivalent representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import HeteroData
from typing import List, Tuple


MetaPathEdge = Tuple[str, str, str]  # (src_type, rel, dst_type)


class MatchingGraphGNN(nn.Module):
    """
    GNN that propagates through the matching graph G* level by level.

    For a meta-path with L hops, this module applies L SAGEConv layers,
    one per hop. Each layer propagates from level i to level i+1, using
    the edge type at that hop.

    Because node types differ across levels (e.g. author->paper->author),
    each node type gets a linear input projection to a common hidden_dim
    before any propagation begins.

    Args:
        meta_path: List of (src_type, rel, dst_type) triples, one per hop.
        node_feat_dims: Dict mapping node_type -> its raw feature dimension.
        hidden_dim: Common hidden dimension used throughout propagation.
        out_dim: Output dimension (number of classes).
    """

    def __init__(
        self,
        meta_path: List[MetaPathEdge],
        node_feat_dims: dict,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()

        self.meta_path = meta_path
        self.num_hops  = len(meta_path)

        # One input projection per unique node type in the meta-path.
        seen_types = set()
        for src_t, _, dst_t in meta_path:
            seen_types.add(src_t)
            seen_types.add(dst_t)

        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(node_feat_dims[ntype], hidden_dim)
            for ntype in seen_types
            if ntype in node_feat_dims
        })

        # One SAGEConv per hop.
        # All hops share (hidden_dim, hidden_dim) because input projections
        # normalise all node types to hidden_dim first.
        # The last hop outputs out_dim.
        self.convs = nn.ModuleList()
        for i in range(self.num_hops):
            out = out_dim if i == self.num_hops - 1 else hidden_dim
            self.convs.append(SAGEConv((hidden_dim, hidden_dim), out))

    def forward(self, g: HeteroData) -> torch.Tensor:
        """
        Propagate through the matching graph level by level.

        Args:
            g: HeteroData containing the raw KG edges and node features.

        Returns:
            Tensor of shape [N_target, out_dim] — embeddings for the
            target node type (xL = x0 for symmetric meta-paths).
        """
        # Project every node type's raw features to hidden_dim.
        h = {}
        for ntype, proj in self.input_proj.items():
            h[ntype] = F.relu(proj(g[ntype].x))

        # Propagate hop by hop.
        for i, (src_type, rel, dst_type) in enumerate(self.meta_path):
            edge_index = g[src_type, rel, dst_type].edge_index
            out = self.convs[i]((h[src_type], h[dst_type]), edge_index)

            # Apply ReLU on all but the final hop.
            if i < self.num_hops - 1:
                out = F.relu(out)

            # Update the destination type's representation for the next hop.
            h[dst_type] = out

        target_type = self.meta_path[-1][2]
        return h[target_type]
