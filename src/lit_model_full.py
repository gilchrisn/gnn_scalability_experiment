import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from typing import Any
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset

class LitFullBatchGNN(pl.LightningModule):
    """
    Lightning Module for Full-Batch Training (GCN style).
    
    Design Pattern: Bridge Pattern
    Separates the 'Abstraction' (Training Loop) from 'Implementation' (Graph Structure).
    The Graph Structure is injectable via `update_graph`.
    """
    
    def __init__(self, 
                 encoder: torch.nn.Module, 
                 initial_graph: Data,
                 lr: float = 0.01, 
                 weight_decay: float = 5e-4):
        super().__init__()
        self.encoder = encoder
        self.lr = lr
        self.weight_decay = weight_decay
        
        # State: The current realization of the graph A_t
        # Registered as buffer so it moves to GPU automatically but isn't a parameter
        self.register_buffer('node_features', initial_graph.x)
        self.register_buffer('edge_index', initial_graph.edge_index)
        self.register_buffer('labels', initial_graph.y)
        self.register_buffer('train_mask', initial_graph.train_mask)
        self.register_buffer('val_mask', initial_graph.val_mask)
        self.register_buffer('test_mask', initial_graph.test_mask)

    def update_graph(self, new_graph: Data):
        """
        Interface for DynamicGraphCallback to inject A_t.
        """
        self.edge_index = new_graph.edge_index
        # Note: Features/Labels usually stay static, but if doing data augmentation
        # on features, update them here too.

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    def _common_step(self, batch_idx, mask_name):
        """Full-batch forward pass."""
        # Note: 'batch_idx' is ignored in full-batch training
        
        out = self(self.node_features, self.edge_index)
        
        mask = getattr(self, mask_name)
        labels = self.labels
        
        loss = F.cross_entropy(out[mask], labels[mask])
        
        num_classes = out.size(-1)
        acc = accuracy(out[mask], labels[mask], task="multiclass", num_classes=num_classes)
        
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch_idx, 'train_mask')
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch_idx, 'val_mask')
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch_idx, 'test_mask')
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    # --- FIX START ---
    def _dummy_loader(self):
        """
        Returns a dummy DataLoader with 1 step.
        This satisfies Lightning's requirement for an iterable without 
        moving the massive graph data through the CPU-GPU bus unnecessarily.
        """
        dummy_ds = TensorDataset(torch.tensor([0.0]))
        return DataLoader(dummy_ds, batch_size=1)

    def train_dataloader(self): return self._dummy_loader()
    def val_dataloader(self): return self._dummy_loader()
    def test_dataloader(self): return self._dummy_loader()
    # --- FIX END ---