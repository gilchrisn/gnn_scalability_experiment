# ---------- File: ./src/models.py ----------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
# from ... import config # --- DELETED

class GCN(nn.Module):
    # (No changes here)
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class SAGE(nn.Module):
    # (No changes here)
    def __init__(self, in_feats, h_feats, num_classes):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats) # 'mean' is the default
        self.conv2 = SAGEConv(h_feats, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GAT(nn.Module):
    # (No changes here)
    def __init__(self, in_feats, h_feats, num_classes, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats,
                             heads=num_heads,
                             concat=True)
        
        self.conv2 = GATConv(h_feats * num_heads, num_classes,
                             heads=1,
                             concat=False) 

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

# --- MODIFIED FUNCTION SIGNATURE ---
def get_model(model_name, in_dim, out_dim, h_dim, gat_heads):
    """
    Factory function to get a model by name.
    """
    if model_name.lower() == 'gcn':
        return GCN(in_dim, h_dim, out_dim)
    elif model_name.lower() == 'sage':
        return SAGE(in_dim, h_dim, out_dim)
    elif model_name.lower() == 'gat':
        # Pass gat_heads to the GAT constructor
        return GAT(in_dim, h_dim, out_dim, gat_heads)
    else:
        raise ValueError(f"Unknown model: {model_name}")