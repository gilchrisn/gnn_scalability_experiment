"""
GNN model architectures (GCN, GAT, SAGE).
Provides a factory function for model instantiation with variable depth support.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN).
    Supports variable number of layers.
    """
    
    def __init__(self, in_feats: int, h_feats: int, num_classes: int, num_layers: int = 2):
        """
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden dimension
            num_classes: Number of output classes
            num_layers: Number of GNN layers (depth)
        """
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        if num_layers == 1:
            self.layers.append(GCNConv(in_feats, num_classes))
        else:
            # Input Layer
            self.layers.append(GCNConv(in_feats, h_feats))
            
            # Hidden Layers
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(h_feats, h_feats))
            
            # Output Layer
            self.layers.append(GCNConv(h_feats, num_classes))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Applies ReLU and Dropout to all but the last layer.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class SAGE(nn.Module):
    """
    GraphSAGE model.
    Supports variable number of layers.
    """
    
    def __init__(self, in_feats: int, h_feats: int, num_classes: int, num_layers: int = 2):
        """
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden dimension
            num_classes: Number of output classes
            num_layers: Number of GNN layers (depth)
        """
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        if num_layers == 1:
            self.layers.append(SAGEConv(in_feats, num_classes))
        else:
            # Input Layer
            self.layers.append(SAGEConv(in_feats, h_feats))
            
            # Hidden Layers
            for _ in range(num_layers - 2):
                self.layers.append(SAGEConv(h_feats, h_feats))
            
            # Output Layer
            self.layers.append(SAGEConv(h_feats, num_classes))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Applies ReLU and Dropout to all but the last layer.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class GAT(nn.Module):
    """
    Graph Attention Network (GAT).
    Supports variable number of layers and handles multi-head dimension expansion.
    """
    
    def __init__(self, in_feats: int, h_feats: int, num_classes: int, num_layers: int = 2, heads: int = 8):
        """
        Args:
            in_feats: Input feature dimension
            h_feats: Hidden dimension (per head)
            num_classes: Number of output classes
            num_layers: Number of GNN layers (depth)
            heads: Number of attention heads
        """
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        
        if num_layers == 1:
            self.layers.append(GATConv(in_feats, num_classes, heads=1, concat=False))
        else:
            # Input Layer (Multi-head, concat=True -> output dim is h_feats * heads)
            self.layers.append(GATConv(in_feats, h_feats, heads=heads, concat=True))
            
            # Hidden Layers
            # Input to hidden layers is h_feats * heads (from previous concat)
            hidden_in_dim = h_feats * heads
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hidden_in_dim, h_feats, heads=heads, concat=True))
            
            # Output Layer (Single-head, concat=False -> output dim is num_classes)
            self.layers.append(GATConv(hidden_in_dim, num_classes, heads=1, concat=False))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Applies ELU and Dropout to all but the last layer.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x


def get_model(model_name: str, 
              in_dim: int, 
              out_dim: int, 
              h_dim: int, 
              gat_heads: int = 8,
              num_layers: int = 2) -> nn.Module:
    """
    Factory function to instantiate GNN models with variable depth.
    
    Args:
        model_name: Model type ('GCN', 'SAGE', 'GAT')
        in_dim: Input feature dimension
        out_dim: Number of output classes
        h_dim: Hidden dimension
        gat_heads: Number of attention heads (only for GAT)
        num_layers: Number of GNN layers (default: 2)
        
    Returns:
        Initialized model
        
    Raises:
        ValueError: If model_name is not recognized
        
    Example:
        >>> model = get_model('GCN', 128, 10, 64, num_layers=3)
    """
    model_name = model_name.upper()
    
    if model_name == 'GCN':
        return GCN(in_dim, h_dim, out_dim, num_layers=num_layers)
    elif model_name == 'SAGE':
        return SAGE(in_dim, h_dim, out_dim, num_layers=num_layers)
    elif model_name == 'GAT':
        return GAT(in_dim, h_dim, out_dim, num_layers=num_layers, heads=gat_heads)
    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: GCN, SAGE, GAT"
        )