"""
GNN model architectures (GCN, GAT, SAGE).
Provides a factory function for model instantiation with variable depth support.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv as GATConv, GINConv


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
    GraphSAGE model with residual skip connections on hidden layers.
    Supports variable number of layers.
    """

    def __init__(self, in_feats: int, h_feats: int, num_classes: int, num_layers: int = 2):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        self.num_layers = num_layers

        if num_layers == 1:
            self.layers.append(SAGEConv(in_feats, num_classes))
        else:
            self.layers.append(SAGEConv(in_feats, h_feats))
            self.skip_projs.append(nn.Linear(in_feats, h_feats, bias=False))
            for _ in range(num_layers - 2):
                self.layers.append(SAGEConv(h_feats, h_feats))
                self.skip_projs.append(nn.Linear(h_feats, h_feats, bias=False))
            self.layers.append(SAGEConv(h_feats, num_classes))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            h = layer(x, edge_index)
            if i < self.num_layers - 1:
                h = h + self.skip_projs[i](x)
                h = F.relu(h)
                h = F.dropout(h, p=0.5, training=self.training)
            x = h
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


class GIN(nn.Module):
    """Graph Isomorphism Network (Xu et al. ICLR 2019).

    Sum aggregation + MLP per layer. No normalization → robust on dense
    saturated meta-paths where GCN's Laplacian collapses.
    """

    def __init__(self, in_feats: int, h_feats: int, num_classes: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        def _mlp(d_in: int, d_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.ReLU(),
                nn.Linear(d_out, d_out),
            )

        if num_layers == 1:
            self.layers.append(GINConv(_mlp(in_feats, num_classes), train_eps=True))
        else:
            self.layers.append(GINConv(_mlp(in_feats, h_feats), train_eps=True))
            for _ in range(num_layers - 2):
                self.layers.append(GINConv(_mlp(h_feats, h_feats), train_eps=True))
            self.layers.append(GINConv(_mlp(h_feats, num_classes), train_eps=True))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
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
        model_name: Model type ('GCN', 'SAGE', 'GAT', 'GIN')
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
    elif model_name == 'GIN':
        return GIN(in_dim, h_dim, out_dim, num_layers=num_layers)
    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: GCN, SAGE, GAT, GIN"
        )