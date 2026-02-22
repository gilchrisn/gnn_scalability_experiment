# ---------- File: ./src/experiments/training.py ----------
import os
import json
import torch
import pytorch_lightning as pl
from torch_geometric.loader import NeighborLoader
from pytorch_lightning.callbacks import EarlyStopping

from ..config import config
from ..data import DatasetFactory, FeatureGuard, GlobalUniverseMapper
from ..models import get_model
from ..lit_model import LitGNN
from .base import AbstractExperimentPhase
from .config import ExperimentConfig

class Phase2FoundationTraining(AbstractExperimentPhase):
    """
    Phase 2: Foundation Model Training.
    Creates the 'Frozen Model' used for fidelity checks.
    Outputs: models/{dataset}_{model_name}.pt
    """

    def __init__(self, cfg: ExperimentConfig, num_layers: int = 2):
        super().__init__(cfg)
        self.num_layers = num_layers  # Store depth for this run

    def execute(self) -> None:
        model_name = self.cfg.current_model_name
        print(f"\n>>> [Phase 2] Training: {model_name} (Layers={self.num_layers})")
        
        model_path = config.get_model_path(self.cfg.dataset, model_name)
        
        if os.path.exists(model_path) and not self.cfg.force_retrain:
            print(f"    [Cache] Frozen model exists: {model_path}")
            return

        # 1. Load Data
        g_hetero, info = DatasetFactory.get_data(
            self.dataset_cfg.source, 
            self.dataset_cfg.dataset_name, 
            self.dataset_cfg.target_node
        )
        FeatureGuard.ensure_features(g_hetero, seed=42)

        # 2. Attach Masks BEFORE Mapping
        # This ensures the Mapper propagates masks to the homogeneous graph correctly.
        target = self.dataset_cfg.target_node
        g_hetero[target].train_mask = info['masks']['train']
        g_hetero[target].val_mask = info['masks']['val']
        g_hetero[target].test_mask = info['masks']['test']
        
        if 'labels' in info:
            g_hetero[target].y = info['labels']

        # 3. Homogenization (Diagonal Padding)
        dims = {nt: g_hetero[nt].x.shape[1] for nt in g_hetero.node_types}
        mapper = GlobalUniverseMapper(g_hetero.node_types, dims)
        g_homo = mapper.transform(g_hetero)

        # [FIX] REMOVED: g_homo.y = info['labels'] ...
        # Justification: The Mapper has already correctly padded and concatenated 
        # the labels into g_homo.y (size N_total). Overwriting it with info['labels'] 
        # (size N_target) breaks structural alignment.

        # 4. Setup Training
        model = get_model(
            self.cfg.model_arch, 
            in_dim=mapper.global_max_dim, 
            out_dim=info['num_classes'], 
            h_dim=config.HIDDEN_DIM,
            num_layers=self.num_layers
        )
        lit_model = LitGNN(model, lr=config.LEARNING_RATE)

        # 5. Data Loaders
        loader_kwargs = {'batch_size': 1024, 'num_workers': 0}
        
        # Note: We must ensure the input_nodes are valid indices in the homogenized space.
        # However, since NeighborLoader handles masking internally if passed correctly,
        # we rely on the masks attached to g_homo by the Mapper.
        
        train_loader = NeighborLoader(
            g_homo, 
            num_neighbors=[10] * 2, 
            shuffle=True, 
            input_nodes=g_homo.train_mask, # Use the mapped mask
            **loader_kwargs
        )
        
        val_loader = NeighborLoader(
            g_homo, 
            num_neighbors=[10] * 2, 
            shuffle=False, 
            input_nodes=g_homo.val_mask, # Use the mapped mask
            **loader_kwargs
        )

        # 6. Train
        trainer = pl.Trainer(
            max_epochs=self.cfg.epochs,
            accelerator="auto",
            devices=1,
            enable_checkpointing=False,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")]
        )
        trainer.fit(lit_model, train_loader, val_loader)

        # 7. Save Artifacts
        self._save_artifacts(lit_model.encoder, mapper, self.cfg.dataset, model_name)
        
        print(f"    [Success] Model frozen at {model_path}")

    def _save_artifacts(self, model, mapper, dataset_name, model_name):
        """Helper to save model weights and mapper config consistently."""
        model_path = config.get_model_path(dataset_name, model_name)
        torch.save(model.state_dict(), model_path)
        
        config_path = model_path.replace('.pt', '_mapper.json')
        mapper_data = {
            'node_types': mapper.node_types,
            'dims': mapper.dims,
            'global_max_dim': mapper.global_max_dim
        }
        
        with open(config_path, 'w') as f:
            json.dump(mapper_data, f, indent=2)
            
        print(f"      Weights: {model_path}")
        print(f"      Mapper:  {config_path}")