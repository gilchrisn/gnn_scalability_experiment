"""
Abstract base class for graph data loaders.
Implements the Template Method pattern for consistent data loading.
"""
from abc import ABC, abstractmethod
import torch
from typing import Tuple, Dict, Any
import torch_geometric.data as tg_data


class BaseGraphLoader(ABC):
    """
    Abstract base class for all dataset loaders.
    Implements common data preparation logic using Template Method pattern.
    """
    
    @abstractmethod
    def load(self, dataset_name: str, target_ntype: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        """
        Load the dataset and return graph + metadata.
        Must be implemented by all subclasses.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'DBLP', 'IMDB')
            target_ntype: Target node type for prediction
            
        Returns:
            Tuple of (HeteroData graph, info dictionary)
        """
        pass

    def _create_random_masks(self, num_nodes: int, train_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Creates train/val/test split masks.
        Uses 60/20/20 split by default.
        
        Args:
            num_nodes: Total number of nodes
            train_mask: Optional pre-defined training mask
            
        Returns:
            Tuple of (train_mask, val_mask, test_mask)
        """
        if train_mask is None:
            # Full random 60/20/20 split
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
            # Split remaining nodes 50/50 for val/test
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
        Ensures target node type has features, generates dummy features if missing.
        
        Args:
            g: Heterogeneous graph
            target_ntype: Target node type
            feature_dim: Dimensionality of generated features
            
        Returns:
            Feature tensor for target nodes
        """
        if hasattr(g[target_ntype], 'x') and g[target_ntype].x is not None:
            return g[target_ntype].x
        
        print(f"[Loader] Warning: '{target_ntype}' has no features. Generating {feature_dim}-dim embeddings...")
        num_nodes = g[target_ntype].num_nodes
        return torch.randn((num_nodes, feature_dim))

    def _extract_labels(self, g: tg_data.HeteroData, target_ntype: str, default_classes: int = 2) -> Tuple[torch.Tensor, int]:
        """
        Extracts or generates labels for the target node type.
        
        Args:
            g: Heterogeneous graph
            target_ntype: Target node type
            default_classes: Number of classes to use if labels are missing
            
        Returns:
            Tuple of (labels tensor, num_classes)
        """
        if hasattr(g[target_ntype], 'y') and g[target_ntype].y is not None:
            labels = g[target_ntype].y
            if labels.dim() > 1:
                labels = labels.view(-1)
            num_classes = int(labels.max()) + 1
        else:
            print(f"[Loader] Warning: No labels found for '{target_ntype}'. Creating dummy labels.")
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
        Creates standardized info dictionary for downstream tasks.
        
        Args:
            features: Node features
            labels: Node labels
            train_mask: Training mask
            val_mask: Validation mask
            test_mask: Test mask
            num_classes: Number of classes
            
        Returns:
            Dictionary containing all required metadata
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