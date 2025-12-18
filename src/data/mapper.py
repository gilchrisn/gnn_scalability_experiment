"""
Graph transformation and homogenization logic.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
import torch
from torch_geometric.data import Data, HeteroData

class GraphTransformer(ABC):
    @abstractmethod
    def transform(self, batch: HeteroData) -> Data:
        pass

class GlobalUniverseMapper(GraphTransformer):
    """
    Homogenizes HeteroData into a single Data object with padded 
    feature and label tensors for unified processing.
    """
    
    def __init__(self, node_types: List[str], dims: Dict[str, int]):
        self.node_types = sorted(node_types)
        self.dims = dims
        self.global_max_dim = max(dims.values()) if dims else 0

    def transform(self, batch: HeteroData) -> Data:
        all_features = []
        all_labels = []
        
        for ntype in self.node_types:
            if ntype not in batch.node_types:
                continue
            
            num_nodes = batch[ntype].num_nodes
            
            # Pad features to global_max_dim
            if ntype in self.dims:
                x = batch[ntype].x
                x_padded = self._pad_features(x, self.dims[ntype])
                all_features.append(x_padded)
            else:
                # Fallback for unexpected nodes missing from dims registry
                device = batch[ntype].x.device if hasattr(batch[ntype], 'x') else 'cpu'
                all_features.append(
                    torch.zeros(num_nodes, self.global_max_dim, device=device)
                )

            # Align label vectors; non-target nodes are assigned -1
            if hasattr(batch[ntype], 'y') and batch[ntype].y is not None:
                all_labels.append(batch[ntype].y)
            else:
                dummy_y = torch.full((num_nodes,), -1, dtype=torch.long)
                if hasattr(batch[ntype], 'x'):
                     dummy_y = dummy_y.to(batch[ntype].x.device)
                
                all_labels.append(dummy_y)

        if not all_features:
             raise ValueError("Input graph has no known node types.")
             
        # Extract adjacency structure
        data = batch.to_homogeneous()
        
        # Override with aligned feature/label blocks
        data.x = torch.cat(all_features, dim=0)
        data.y = torch.cat(all_labels, dim=0)
            
        return data

    def _pad_features(self, x: torch.Tensor, original_dim: int) -> torch.Tensor:
        if x.size(1) != original_dim: return x
        if original_dim == self.global_max_dim: return x
        
        pad_size = self.global_max_dim - original_dim
        return torch.nn.functional.pad(x, (0, pad_size), "constant", 0)