# ---------- File: ./src/lit_model.py ----------
# (This is the CORRECT version for this project)

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

class LitGNN(pl.LightningModule):
    """
    Generic PyTorch Lightning wrapper for a GNN encoder.
    This class handles the training, validation, testing, and optimization logic.
    """
    def __init__(self, encoder, lr, weight_decay):
        super().__init__()
        self.encoder = encoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=['encoder'])

    def forward(self, x, edge_index):
        """
        Forward pass simply calls the underlying encoder.
        NOTE: This encoder does NOT support 'return_intermediate'.
        """
        return self.encoder(x, edge_index)

    def _calculate_loss(self, batch, mask):
        """
        A private helper function to calculate loss and accuracy for a given split.
        """
        # In our setup (full-graph training), the "batch" is the single graph object.
        if isinstance(batch, list):
            data = batch[0]
        else:
            data = batch
        
        # Get final logits
        # NOTE: We just call the encoder directly. It returns one value.
        logits = self.encoder(data.x, data.edge_index)
        
        out = logits[mask]
        labels = data.y[mask]
        
        loss = F.cross_entropy(out, labels)
        
        # Get num_classes from the encoder's final layer
        num_classes = self.encoder.conv2.out_channels
        
        acc = accuracy(out, labels, task="multiclass", num_classes=num_classes)
        
        return loss, acc

    def training_step(self, batch, batch_idx):
        data = batch[0]
        loss, acc = self._calculate_loss(batch, data.train_mask)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0]
        loss, acc = self._calculate_loss(batch, data.val_mask)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        data = batch[0]
        loss, acc = self._calculate_loss(batch, data.test_mask)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        return optimizer