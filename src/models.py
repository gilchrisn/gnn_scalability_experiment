"""
GNN model architectures (GCN, GAT, SAGE).
Provides a factory function for model instantiation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN).
    Two-layer GCN with ReLU activation.
    """
    
    def __init__(self, in_feats: int, h_feats: int, num_classes: int):
        """
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden dimension
            num_classes: Number of output classes
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_feats]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Node logits [num_nodes, num_classes]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class SAGE(nn.Module):
    """
    GraphSAGE model.
    Two-layer GraphSAGE with mean aggregation.
    """
    
    def __init__(self, in_feats: int, h_feats: int, num_classes: int):
        """
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden dimension
            num_classes: Number of output classes
        """
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_feats]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Node logits [num_nodes, num_classes]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GAT(nn.Module):
    """
    Graph Attention Network (GAT).
    Two-layer GAT with multi-head attention.
    """
    
    def __init__(self, in_feats: int, h_feats: int, num_classes: int, num_heads: int = 8):
        """
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden dimension (per head)
            num_classes: Number of output classes
            num_heads: Number of attention heads
        """
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            in_feats, 
            h_feats,
            heads=num_heads,
            concat=True
        )
        
        self.conv2 = GATConv(
            h_feats * num_heads, 
            num_classes,
            heads=1,
            concat=False
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_feats]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Node logits [num_nodes, num_classes]
        """
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x


def get_model(model_name: str, 
              in_dim: int, 
              out_dim: int, 
              h_dim: int, 
              gat_heads: int = 8) -> nn.Module:
    """
    Factory function to instantiate GNN models.
    
    Args:
        model_name: Model type ('GCN', 'SAGE', 'GAT')
        in_dim: Input feature dimension
        out_dim: Number of output classes
        h_dim: Hidden dimension
        gat_heads: Number of attention heads (only for GAT)
        
    Returns:
        Initialized model
        
    Raises:
        ValueError: If model_name is not recognized
        
    Example:
        >>> model = get_model('GCN', 128, 10, 64)
        >>> model = get_model('GAT', 128, 10, 64, gat_heads=4)
    """
    model_name = model_name.upper()
    
    if model_name == 'GCN':
        return GCN(in_dim, h_dim, out_dim)
    elif model_name == 'SAGE':
        return SAGE(in_dim, h_dim, out_dim)
    elif model_name == 'GAT':
        return GAT(in_dim, h_dim, out_dim, gat_heads)
    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: GCN, SAGE, GAT"
        )