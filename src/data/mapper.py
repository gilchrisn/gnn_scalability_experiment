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
    feature and label tensors.
    """
    
    def __init__(self, node_types: List[str], dims: Dict[str, int]):
        self.node_types = sorted(node_types)
        self.dims = dims
        self.global_max_dim = max(dims.values()) if dims else 0

    def transform(self, batch: HeteroData) -> Data:
        all_features = []
        all_labels = []

        # Masks storage
        masks = {'train': [], 'val': [], 'test': []}
        
        for ntype in self.node_types:
            if ntype not in batch.node_types:
                continue
            
            num_nodes = batch[ntype].num_nodes
            device = self._get_device(batch, ntype)
            
            # Features
            if ntype in self.dims:
                x_padded = self._pad_features(batch[ntype].x, self.dims[ntype])
                all_features.append(x_padded)
            else:
                all_features.append(torch.zeros(num_nodes, self.global_max_dim, device=device))

            # Labels - Strict alignment
            if hasattr(batch[ntype], 'y') and batch[ntype].y is not None:
                y = batch[ntype].y
                if y.dim() > 1: y = y.view(-1)
                
                if y.size(0) < num_nodes:
                    pad = torch.full((num_nodes - y.size(0),), -100, dtype=torch.long, device=device)
                    y = torch.cat([y, pad])
                elif y.size(0) > num_nodes:
                    y = y[:num_nodes]
                
                all_labels.append(y)
            else:
                all_labels.append(torch.full((num_nodes,), -100, dtype=torch.long, device=device))

            # Masks
            for k in masks:
                mask_attr = f"{k}_mask"
                if hasattr(batch[ntype], mask_attr) and getattr(batch[ntype], mask_attr) is not None:
                    masks[k].append(getattr(batch[ntype], mask_attr))
                else:
                    masks[k].append(torch.zeros(num_nodes, dtype=torch.bool, device=device))

        if not all_features:
             raise ValueError("Input graph has no known node types.")
             
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

    def _pad_features(self, x: torch.Tensor, original_dim: int) -> torch.Tensor:
        if x.size(1) != original_dim: return x
        if original_dim == self.global_max_dim: return x
        return torch.nn.functional.pad(x, (0, self.global_max_dim - original_dim), "constant", 0)