"""
Graph transformation and homogenization logic.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData

class GraphTransformer(ABC):
    @abstractmethod
    def transform(self, batch: HeteroData) -> Data:
        pass

class GlobalUniverseMapper(GraphTransformer):
    """ 
    Homogenizes HeteroData using Diagonal Padding (Block-Diagonal Alignment). 
    Assigns disjoint feature subspaces to prevent feature cancellation.
    """
    
    def __init__(self, node_types: List[str], dims: Dict[str, int]):
        self.node_types = sorted(node_types)
        self.dims = dims
        
        # Pre-calc offsets for diagonal alignment
        self.offsets = {}
        self.total_dim = 0
        
        for nt in self.node_types:
            d = dims.get(nt, 0)
            self.offsets[nt] = self.total_dim
            self.total_dim += d
        
        self.global_max_dim = self.total_dim

        print(f"[Mapper] Diagonal Padding enabled. Universe D = {self.total_dim}")

    def transform(self, batch: HeteroData) -> Data:
        all_features = []
        all_labels = []
        masks = {'train': [], 'val': [], 'test': []}
        
        # Iterate sorted types for consistent ID mapping
        for ntype in self.node_types:
            if ntype not in batch.node_types:
                continue
            
            num_nodes = batch[ntype].num_nodes
            device = self._get_device(batch, ntype)
            
            # 1. Feature Diagonalization
            if ntype in self.dims:
                x = getattr(batch[ntype], 'x', None)
                if x is None:
                    x = torch.zeros((num_nodes, self.dims[ntype]), device=device)
                
                # Pad: (pad_left, pad_right)
                pad_l = self.offsets[ntype]
                pad_r = self.total_dim - (pad_l + x.size(1))
                x_padded = F.pad(x, (pad_l, pad_r), "constant", 0)
                all_features.append(x_padded)
            else:
                # No features config -> pure zero row
                all_features.append(torch.zeros(num_nodes, self.total_dim, device=device))

            # 2. Label Processing
            y = getattr(batch[ntype], 'y', None)
            if y is not None:
                if y.dim() > 1: y = y.view(-1)
                # Handle size mismatch (random splits vs full graph)
                if y.size(0) < num_nodes:
                    pad = torch.full((num_nodes - y.size(0),), -100, dtype=torch.long, device=device)
                    y = torch.cat([y, pad])
                elif y.size(0) > num_nodes:
                    y = y[:num_nodes]
            else:
                y = torch.full((num_nodes,), -100, dtype=torch.long, device=device)
            all_labels.append(y)

            # 3. Mask Collection
            for k in masks:
                m = getattr(batch[ntype], f"{k}_mask", None)
                if m is None:
                    m = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                masks[k].append(m)

        if not all_features:
             raise ValueError("Input graph has no known node types.")
             
        # Consolidate
        data = batch.to_homogeneous()
        
        data.x = torch.cat(all_features, dim=0)
        data.y = torch.cat(all_labels, dim=0)
        # Essential for NeighborLoader slicing
        data.num_nodes = data.x.size(0)

        data.train_mask = torch.cat(masks['train'], dim=0)
        data.val_mask = torch.cat(masks['val'], dim=0)
        data.test_mask = torch.cat(masks['test'], dim=0)
            
        return data

    def _get_device(self, batch, ntype):
        return batch[ntype].x.device if hasattr(batch[ntype], 'x') and batch[ntype].x is not None else 'cpu'