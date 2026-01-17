import os
import json
import torch
import pytorch_lightning as pl
from argparse import Namespace
from torch_geometric.loader import NeighborLoader
from pytorch_lightning.callbacks import EarlyStopping

from .base import BaseCommand
from ..config import config
from ..data import DatasetFactory, FeatureGuard, GlobalUniverseMapper
from ..models import get_model
from ..lit_model import LitGNN

class TrainCommand(BaseCommand):
    """
    Trains a GNN Foundation Model on the homogenized 'Universe' space.
    Saves both the model weights and the Mapper configuration.
    """

    def execute(self, args: Namespace) -> None:
        model_path = config.get_model_path(args.dataset, args.model)
        
        if os.path.exists(model_path) and not args.force:
            print(f"\n[Info] Model already exists at {model_path}.")
            print("Skipping training. Use --force to retrain.")
            return
        
        print(f"\n{'='*60}")
        print(f"FOUNDATION TRAINING: {args.dataset}")
        print(f"{'='*60}\n")

        # 1. Load Data
        dataset_cfg = config.get_dataset_config(args.dataset)
        g_hetero, info = DatasetFactory.get_data(
            dataset_cfg.source, dataset_cfg.dataset_name, dataset_cfg.target_node
        )
        
        # Ensure features exist (handle missing attributes)
        FeatureGuard.ensure_features(g_hetero, seed=42, default_dim=64)
        
        # Attach labels/masks to graph object for Mapper to process
        target = dataset_cfg.target_node
        g_hetero[target].train_mask = info['masks']['train']
        g_hetero[target].val_mask = info['masks']['val']
        g_hetero[target].test_mask = info['masks']['test']
        
        if 'labels' in info:
            g_hetero[target].y = info['labels']
        
        # 2. Map to Universe Space (Diagonal Padding)
        print("Mapping to Universe space...")
        dims = {}
        for nt in g_hetero.node_types:
            if hasattr(g_hetero[nt], 'x') and g_hetero[nt].x is not None:
                dims[nt] = g_hetero[nt].x.shape[1]

        mapper = GlobalUniverseMapper(g_hetero.node_types, dims)
        g_homo = mapper.transform(g_hetero)
        
        print(f"      Dimension D = {mapper.global_max_dim}")

        # 3. Setup Model
        model = get_model(
            args.model, 
            in_dim=mapper.global_max_dim, 
            out_dim=info['num_classes'], 
            h_dim=config.HIDDEN_DIM
        )
        lit_model = LitGNN(model, lr=config.LEARNING_RATE)

        # 4. Setup Loaders
        print(f"\nTraining {args.model} ({args.epochs} epochs)...")
        
        train_loader = NeighborLoader(
            g_homo,
            num_neighbors=[10] * 2,  
            batch_size=1024,        
            shuffle=True,
            num_workers=0
        )

        val_loader = NeighborLoader(
            g_homo,
            num_neighbors=[10] * 2,
            batch_size=1024,
            shuffle=False,
            num_workers=0
        )
        
        # 5. Training Loop
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=5,
            verbose=True,
            mode="min"
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            gradient_clip_val=1.0,
            callbacks=[early_stop_callback]
        )
        
        trainer.fit(lit_model, train_loader, val_loader)

        # 6. Save Artifacts
        print("\nSaving artifacts...")
        self._save_artifacts(lit_model.encoder, mapper, args.dataset, args.model)
        print("Done.")

    def _save_artifacts(self, model, mapper, dataset_name, model_name):
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

    @staticmethod
    def register_subparser(subparsers) -> None:
        parser = subparsers.add_parser('train', help='Train foundation model')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--model', type=str, default='SAGE')
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--force', action='store_true', help="Overwrite existing model")