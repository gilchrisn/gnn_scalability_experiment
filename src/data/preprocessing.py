"""
Data preprocessing utilities.
"""
import torch
from torch_geometric.data import HeteroData

class FeatureGuard:
    """
    Handles missing node features in HeteroData objects by generating 
    deterministic placeholders.
    """
    
    @staticmethod
    def ensure_features(g: HeteroData, seed: int = 42, default_dim: int = 64) -> None:
        """
        Populates empty 'x' attributes across all node types. 
        Requires consistent seeds across train/inference to maintain 
        embedding alignment.
        """
        # Use local generator to prevent side effects on global torch state
        rng = torch.Generator()
        rng.manual_seed(seed)
        
        # Sort ntypes to ensure deterministic RNG consumption order
        for ntype in sorted(g.node_types):
            if not hasattr(g[ntype], 'x') or g[ntype].x is None:
                num_nodes = g[ntype].num_nodes
                
                # In-place assignment of random features
                g[ntype].x = torch.randn(
                    (num_nodes, default_dim), 
                    generator=rng
                )