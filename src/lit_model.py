"""
PyTorch Lightning wrapper for GNN models.
Handles training, validation, and testing loops.
"""
import torch
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Node logits [num_nodes, num_classes]
        """
        return self.encoder(x, edge_index)

    def _calculate_loss(self, batch: Any, mask: torch.Tensor) -> tuple:
        """
        Calculate loss and accuracy for a given split.
        
        Args:
            batch: Batch data (can be single graph or list)
            mask: Boolean mask indicating which nodes to evaluate
            
        Returns:
            Tuple of (loss, accuracy)
        """
        # Handle both single graph and list formats
        if isinstance(batch, list):
            data = batch[0]
        else:
            data = batch
        
        # Forward pass
        logits = self.encoder(data.x, data.edge_index)
        
        # Apply mask
        out = logits[mask]
        labels = data.y[mask]
        
        # Calculate loss
        loss = F.cross_entropy(out, labels)
        
        # Calculate accuracy
        num_classes = self.encoder.conv2.out_channels
        acc = accuracy(out, labels, task="multiclass", num_classes=num_classes)
        
        return loss, acc

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Training batch
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        data = batch[0] if isinstance(batch, list) else batch
        loss, acc = self._calculate_loss(batch, data.train_mask)
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        Validation step.
        
        Args:
            batch: Validation batch
            batch_idx: Batch index
        """
        data = batch[0] if isinstance(batch, list) else batch
        loss, acc = self._calculate_loss(batch, data.val_mask)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """
        Test step.
        
        Args:
            batch: Test batch
            batch_idx: Batch index
        """
        data = batch[0] if isinstance(batch, list) else batch
        loss, acc = self._calculate_loss(batch, data.test_mask)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer.
        
        Returns:
            AdamW optimizer
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        return optimizer

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Prediction step.
        
        Args:
            batch: Prediction batch
            batch_idx: Batch index
            
        Returns:
            Predicted class labels
        """
        data = batch[0] if isinstance(batch, list) else batch
        logits = self.encoder(data.x, data.edge_index)
        return logits.argmax(dim=1)