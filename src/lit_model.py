"""
PyTorch Lightning wrapper for GNN models.
Handles training, validation, and testing loops.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from typing import Any


class LitGNN(pl.LightningModule):
    """
    PyTorch Lightning wrapper for GNN encoders.
    Implements training, validation, and testing logic.
    """
    
    def __init__(self, 
                 encoder: torch.nn.Module, 
                 lr: float = 0.01, 
                 weight_decay: float = 5e-4):
        """
        Args:
            encoder: GNN encoder (e.g., GCN, GAT, SAGE)
            lr: Learning rate
            weight_decay: L2 regularization coefficient
        """
        super().__init__()
        self.encoder = encoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=['encoder'])

    def forward(self, x: Any, edge_index: Any) -> torch.Tensor:
        """
        Forward pass through the encoder.
        Supports both Dict (Hetero) and Tensor (Homo) inputs.
        """
        return self.encoder(x, edge_index)

    def _calculate_loss(self, batch: Any, mask: torch.Tensor) -> tuple:
        """
        Calculate loss and accuracy for a given split.
        Supports both HeteroData (dict-based) and Data (tensor-based).
        """
        # Handle both single graph and list formats
        data = batch[0] if isinstance(batch, list) else batch
        
        # 1. Forward Pass logic for Hetero vs Homo
        if hasattr(data, 'x_dict'):
            # Heterogeneous training path
            out_dict = self.encoder(data.x_dict, data.edge_index_dict)
            # Slice logits for target node type (assumes labels are on target)
            target_node = next(iter(data.y_dict.keys()))
            logits = out_dict[target_node]
            labels = data.y_dict[target_node]
        else:
            # Homogeneous/Materialized inference path
            logits = self.encoder(data.x, data.edge_index)
            labels = data.y
        
        # 2. Apply mask and calculate loss
        out = logits[mask]
        target_y = labels[mask]
        
        loss = F.cross_entropy(out, target_y)
        
        # 3. Calculate accuracy
        # Infer num_classes from output dimension if not explicitly set
        num_classes = logits.size(-1)
        acc = accuracy(out, target_y, task="multiclass", num_classes=num_classes)
        
        return loss, acc
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        data = batch[0] if isinstance(batch, list) else batch
        # We assume the mask is stored on the target node type or the root
        mask = data.train_mask if not hasattr(data, 'train_mask_dict') else next(iter(data.train_mask_dict.values()))
        loss, acc = self._calculate_loss(batch, mask)
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        data = batch[0] if isinstance(batch, list) else batch
        mask = data.val_mask if not hasattr(data, 'val_mask_dict') else next(iter(data.val_mask_dict.values()))
        loss, acc = self._calculate_loss(batch, mask)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        data = batch[0] if isinstance(batch, list) else batch
        mask = data.test_mask if not hasattr(data, 'test_mask_dict') else next(iter(data.test_mask_dict.values()))
        loss, acc = self._calculate_loss(batch, mask)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )