"""
Base classes for graph data ingestion.
"""
from abc import ABC, abstractmethod
import torch
from typing import Tuple, Dict, Any
import torch_geometric.data as tg_data


class BaseGraphLoader(ABC):
    """
    Interface for graph loaders. 
    Provides shared utilities for mask generation, feature verification, and metadata formatting.
    """
    
    @abstractmethod
    def load(self, dataset_name: str, target_ntype: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        """
        Implementation-specific loading logic. 
        Returns (graph, metadata_dict).
        """
        pass

    def _create_random_masks(self, num_nodes: int, train_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates data splits. If no train_mask is provided, performs a 60/20/20 shuffle.
        Otherwise, splits remaining nodes 50/50 between validation and testing.
        """
        if train_mask is None:
            # Standard 60/20/20 split
            indices = torch.randperm(num_nodes)
            num_train = int(num_nodes * 0.6)
            num_val = int(num_nodes * 0.2)
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[indices[:num_train]] = True
            
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask[indices[num_train:num_train+num_val]] = True
            
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask[indices[num_train+num_val:]] = True
        else:
            # Handle cases where training set is fixed; partition the complement
            non_train_indices = (~train_mask).nonzero(as_tuple=False).view(-1)
            non_train_indices = non_train_indices[torch.randperm(non_train_indices.size(0))]
            
            num_val = non_train_indices.size(0) // 2
            val_indices = non_train_indices[:num_val]
            test_indices = non_train_indices[num_val:]
            
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask[val_indices] = True
            
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask[test_indices] = True
            
        return train_mask, val_mask, test_mask

    def _ensure_features(self, g: tg_data.HeteroData, target_ntype: str, feature_dim: int = 64) -> torch.Tensor:
        """
        Returns existing node features or generates random placeholders if 'x' is missing.
        """
        if hasattr(g[target_ntype], 'x') and g[target_ntype].x is not None:
            return g[target_ntype].x
        
        print(f"[Loader] Missing features for '{target_ntype}'; initializing random {feature_dim}-dim tensor.")
        num_nodes = g[target_ntype].num_nodes
        return torch.randn((num_nodes, feature_dim))

    def _extract_labels(self, g: tg_data.HeteroData, target_ntype: str, default_classes: int = 2) -> Tuple[torch.Tensor, int]:
        """
        Extracts ground truth from 'y'. Flattens multi-dim labels and infers class count.
        """
        if hasattr(g[target_ntype], 'y') and g[target_ntype].y is not None:
            labels = g[target_ntype].y
            if labels.dim() > 1:
                labels = labels.view(-1)
            num_classes = int(labels.max()) + 1
        else:
            print(f"[Loader] No labels found for '{target_ntype}'; using random targets.")
            num_classes = default_classes
            labels = torch.randint(0, num_classes, (g[target_ntype].num_nodes,))
            
        return labels, num_classes

    def create_info_dict(self, 
                        features: torch.Tensor,
                        labels: torch.Tensor,
                        train_mask: torch.Tensor,
                        val_mask: torch.Tensor,
                        test_mask: torch.Tensor,
                        num_classes: int) -> Dict[str, Any]:
        """
        Collate training tensors and dimensions into a standard metadata package.
        """
        return {
            "features": features,
            "labels": labels,
            "masks": {
                'train': train_mask,
                'val': val_mask,
                'test': test_mask
            },
            "num_classes": num_classes,
            "in_dim": features.shape[1],
            "out_dim": num_classes
        }