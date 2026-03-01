"""
MLP Baseline Command — Metapath Informativeness Gate.

Trains a feature-only MLP (no graph edges) on target node features.
Acts as a null hypothesis test before running materialization experiments.

Interpretation:
    MLP accuracy << GNN accuracy → metapath carries structural signal ✅
    MLP accuracy ≈  GNN accuracy → metapath is uninformative          ❌

If the gate fails, revisit AnyBURL rule selection with a discriminability
criterion (mutual information) before investing in materialization experiments.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
from argparse import Namespace
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import EarlyStopping

from .base import BaseCommand
from ..config import config
from ..data import DatasetFactory


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Feature-only MLP with no graph structure.

    Null hypothesis: GNN accuracy minus MLP accuracy measures how much
    the graph topology contributes beyond raw node features.
    """

    def __init__(self, in_dim: int, h_dim: int, out_dim: int, num_layers: int = 3):
        """
        Args:
            in_dim:     Input feature dimension.
            h_dim:      Hidden layer dimension.
            out_dim:    Number of output classes.
            num_layers: Total depth including input and output layers.
        """
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers: list[nn.Module] = []

        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Dropout(0.5)]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(0.5)]
            layers.append(nn.Linear(h_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Lightning Wrapper
# ---------------------------------------------------------------------------

class LitMLP(pl.LightningModule):
    """
    Full-batch Lightning wrapper for the MLP baseline.

    Stores features, labels, and masks as buffers so they move to GPU
    automatically alongside the model parameters.
    """

    def __init__(self,
                 model: MLP,
                 features: torch.Tensor,
                 labels: torch.Tensor,
                 train_mask: torch.Tensor,
                 val_mask: torch.Tensor,
                 test_mask: torch.Tensor,
                 lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

        self.register_buffer('features', features)
        self.register_buffer('labels', labels)
        self.register_buffer('train_mask', train_mask)
        self.register_buffer('val_mask', val_mask)
        self.register_buffer('test_mask', test_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shared forward pass for train / val / test."""
        out = self(self.features)
        loss = F.cross_entropy(out[mask], self.labels[mask])
        acc = accuracy(
            out[mask], self.labels[mask],
            task="multiclass", num_classes=out.size(-1)
        )
        return loss, acc

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, acc = self._step(self.train_mask)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc',  acc,  prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        loss, acc = self._step(self.val_mask)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc',  acc,  prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        loss, acc = self._step(self.test_mask)
        self.log('test_loss', loss)
        self.log('test_acc',  acc)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _dummy_loader(self) -> DataLoader:
        """Satisfies Lightning's loader requirement without moving data."""
        return DataLoader(TensorDataset(torch.tensor([0.0])), batch_size=1, num_workers=0)

    def train_dataloader(self) -> DataLoader: return self._dummy_loader()
    def val_dataloader(self)   -> DataLoader: return self._dummy_loader()
    def test_dataloader(self)  -> DataLoader: return self._dummy_loader()


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

class MLPBaselineCommand(BaseCommand):
    """
    Runs the MLP null-hypothesis baseline.

    Use this BEFORE running any materialization experiment to confirm
    the selected metapath is actually carrying structural signal.
    """

    # Gap threshold below which metapath is flagged as uninformative
    _DEFAULT_TOLERANCE: float = 0.02

    def execute(self, args: Namespace) -> None:
        print(f"\n{'='*60}")
        print(f"MLP BASELINE (Null Hypothesis Test): {args.dataset}")
        print(f"Question: does the metapath add anything beyond features?")
        print(f"{'='*60}\n")

        # 1. Load data -------------------------------------------------------
        dataset_cfg = config.get_dataset_config(args.dataset)
        _, info = DatasetFactory.get_data(
            dataset_cfg.source, dataset_cfg.dataset_name, dataset_cfg.target_node
        )

        features   = info['features']
        labels     = info['labels']
        masks      = info['masks']
        num_classes = info['num_classes']

        print(f"Node features : {features.shape}")
        print(f"Classes       : {num_classes}")
        print(f"Train / Val / Test : "
              f"{masks['train'].sum().item()} / "
              f"{masks['val'].sum().item()} / "
              f"{masks['test'].sum().item()}")

        # 2. Build model -----------------------------------------------------
        mlp = MLP(
            in_dim=features.shape[1],
            h_dim=config.HIDDEN_DIM,
            out_dim=num_classes,
            num_layers=args.layers
        )
        lit_model = LitMLP(
            model=mlp,
            features=features,
            labels=labels,
            train_mask=masks['train'],
            val_mask=masks['val'],
            test_mask=masks['test'],
            lr=config.LEARNING_RATE
        )

        # 3. Train -----------------------------------------------------------
        print(f"\nTraining MLP ({args.layers} layers, up to {args.epochs} epochs)...")
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            log_every_n_steps=1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=15, mode='min')]
        )
        trainer.fit(lit_model)

        # 4. Evaluate --------------------------------------------------------
        results  = trainer.test(lit_model, verbose=False)[0]
        mlp_acc  = results['test_acc']
        gnn_acc  = args.gnn_acc
        gap      = gnn_acc - mlp_acc
        informative = gap > args.tolerance

        # 5. Report ----------------------------------------------------------
        print(f"\n{'='*40}")
        print(f"MLP  accuracy : {mlp_acc:.4f}")
        if gnn_acc > 0:
            print(f"GNN  accuracy : {gnn_acc:.4f}  (provided via --gnn-acc)")
            print(f"Gap           : {gap:+.4f}")
        print(f"{'='*40}")

        if gnn_acc > 0:
            if informative:
                print(f"✅  PASS — metapath contributes structural signal (gap > {args.tolerance})")
                print(f"    Safe to proceed with materialization experiments.")
            else:
                print(f"⚠️  FAIL — MLP ≈ GNN (gap ≤ {args.tolerance})")
                print(f"    The metapath is not adding useful structural information.")
                print(f"    → Revisit AnyBURL rule selection before proceeding.")
        else:
            print("    (Provide --gnn-acc <float> to enable pass/fail comparison)")

        # 6. Save ------------------------------------------------------------
        if args.csv_output:
            row = {
                'Dataset':     args.dataset,
                'MLP_Acc':     mlp_acc,
                'GNN_Acc':     gnn_acc,
                'Gap':         gap,
                'Informative': informative,
                'Tolerance':   args.tolerance,
            }
            df = pd.DataFrame([row])
            header = not os.path.exists(args.csv_output)
            df.to_csv(args.csv_output, mode='a', header=header, index=False)
            print(f"\n[IO] Results appended to {args.csv_output}")

    @staticmethod
    def register_subparser(subparsers) -> None:
        parser = subparsers.add_parser(
            'mlp_baseline',
            help='Run MLP null-hypothesis baseline (metapath informativeness gate)'
        )
        parser.add_argument('--dataset',   type=str,   required=True)
        parser.add_argument('--epochs',    type=int,   default=200)
        parser.add_argument('--layers',    type=int,   default=3,
                            help='Number of MLP layers (default: 3)')
        parser.add_argument('--gnn-acc',   type=float, default=0.0,
                            help='GNN exact test accuracy to compare against')
        parser.add_argument('--tolerance', type=float,
                            default=MLPBaselineCommand._DEFAULT_TOLERANCE,
                            help='Min gap for metapath to be considered informative (default: 0.02)')
        parser.add_argument('--csv-output', type=str,
                            help='Path to append results CSV')