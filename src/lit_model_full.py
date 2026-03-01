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

    Assumptions:
        - Node features (x) and labels (y) are STATIC across epochs.
          Only the graph topology (edge_index) changes between epochs.
        - If you need dynamic node features (e.g., feature augmentation),
          extend update_graph() to also update self.node_features.
        - Masks (train/val/test) are fixed at construction time.
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

    def update_graph(self, new_graph: Data) -> None:
        """
        Interface for DynamicGraphCallback to inject a new graph topology A_t.

        IMPORTANT — Only updates edge_index, not node_features or labels.
        This is intentional: in the KMV-Dynamic training paradigm, features
        and labels are fixed properties of the nodes. Only the sampled
        neighbourhood structure changes each epoch.

        If your use case requires dynamic features, update self.node_features
        here as well. Be aware this will trigger a full host-to-device
        transfer every epoch.

        Args:
            new_graph: The newly sampled homogeneous graph for this epoch.
                       Must have the same num_nodes as the initial graph.
        """
        self.edge_index = new_graph.edge_index

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def _common_step(self, batch_idx: int, mask_name: str) -> tuple:
        """Full-batch forward pass. batch_idx is unused in full-batch training."""
        out = self(self.node_features, self.edge_index)

        mask = getattr(self, mask_name)
        labels = self.labels

        loss = F.cross_entropy(out[mask], labels[mask])

        num_classes = out.size(-1)
        acc = accuracy(out[mask], labels[mask], task="multiclass", num_classes=num_classes)

        return loss, acc

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, acc = self._common_step(batch_idx, 'train_mask')
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss, acc = self._common_step(batch_idx, 'val_mask')
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss, acc = self._common_step(batch_idx, 'test_mask')
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _dummy_loader(self) -> DataLoader:
        """
        Returns a dummy DataLoader with 1 step.
        Satisfies Lightning's requirement for an iterable without moving
        the full graph through the CPU-GPU bus unnecessarily.
        num_workers=0 avoids subprocess overhead for this trivial task.
        """
        dummy_ds = TensorDataset(torch.tensor([0.0]))
        return DataLoader(dummy_ds, batch_size=1, num_workers=0)

    def train_dataloader(self) -> DataLoader: return self._dummy_loader()
    def val_dataloader(self) -> DataLoader: return self._dummy_loader()
    def test_dataloader(self) -> DataLoader: return self._dummy_loader()

    