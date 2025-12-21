"""
LightningModule wrapper for GNN training and evaluation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from typing import Any


class LitGNN(pl.LightningModule):
    """
    Standardizes training and test loops for arbitrary GNN encoders.
    """
    
    def __init__(self, 
                 encoder: torch.nn.Module, 
                 lr: float = 0.01, 
                 weight_decay: float = 5e-4):
        super().__init__()
        self.encoder = encoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=['encoder'])

    def forward(self, x: Any, edge_index: Any) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def _calculate_loss(self, batch: Any, mask: torch.Tensor) -> tuple:
        """
        Shared loss calculation. Handles extraction from single graph or batch list,
        and manages attribute dispatch for HeteroData vs standard Data objects.
        """
        # Unwrap data if passed as a single-item list from certain loaders
        data = batch[0] if isinstance(batch, list) else batch
        
        if hasattr(data, 'x_dict'):
            # Heterogeneous path: assumes target labels are on the first available node type
            out_dict = self.encoder(data.x_dict, data.edge_index_dict)
            target_node = next(iter(data.y_dict.keys()))
            logits = out_dict[target_node]
            labels = data.y_dict[target_node]
        else:
            # Homogeneous path
            logits = self.encoder(data.x, data.edge_index)
            labels = data.y
        
        # Isolate masked indices for metric calculation
        out = logits[mask]
        target_y = labels[mask]
        
        loss = F.cross_entropy(out, target_y)
        
        # Dynamic class inference for multiclass accuracy
        num_classes = logits.size(-1)
        acc = accuracy(out, target_y, task="multiclass", num_classes=num_classes, ignore_index=-100)
        
        return loss, acc
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        data = batch[0] if isinstance(batch, list) else batch
        
        # Resolve training mask location
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