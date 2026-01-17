import time
import torch
import pytorch_lightning as pl
from argparse import Namespace
import pandas as pd
import os

from .base import BaseCommand
from ..config import config
from ..data import DatasetFactory, GlobalUniverseMapper
from ..backend import BackendFactory
from ..models import get_model
from ..utils import SchemaMatcher
from ..lit_model_full import LitFullBatchGNN
from ..callbacks import DynamicGraphCallback

class TrainFidelityCommand(BaseCommand):
    """
    Executes Journal Experiments A & B: Fidelity and Scalability.
    
    Modes:
    1. EXACT: Baseline (Control). SpGEMM materialization.
    2. KMV-STATIC: Pre-compute sketch once. (Scalability Claim).
    3. KMV-DYNAMIC: Re-sketch every epoch. (Regularization Claim).
    """

    def execute(self, args: Namespace) -> None:
        print(f"\n{'='*60}")
        print(f"FIDELITY EXPERIMENT: {args.dataset}")
        print(f"Mode: {args.mode.upper()} | Model: {args.model}")
        print(f"{'='*60}\n")

        # 1. Load Heterogeneous Data
        cfg = config.get_dataset_config(args.dataset)
        g_hetero, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        
        # 2. Parse Metapath
        path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in args.metapath.split(',')]
        
        # 3. Initial Graph Materialization (Pre-processing)
        print(f"[Phase 1] Initial Materialization ({args.mode})...")
        start_t = time.perf_counter()
        
        backend = BackendFactory.create('python', device=config.DEVICE)
        backend.initialize(g_hetero, path_list, info)
        
        if args.mode == 'exact':
            g_initial = backend.materialize_exact()
        elif args.mode in ['kmv-static', 'kmv-dynamic']:
            g_initial = backend.materialize_kmv(k=args.k)
        elif args.mode == 'random':
            g_initial = backend.materialize_random(k=args.k)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
            
        prep_time = time.perf_counter() - start_t
        print(f"  -> Graph built in {prep_time:.4f}s | Edges: {g_initial.num_edges}")

        # 4. Feature Homogenization (Global Mapper)
        # Note: For full-batch, we ensure features align with the materialized graph
        # Currently assuming features are already on the nodes from the backend
        
        # 5. Model Setup
        # We need the input dimension D after materialization
        in_dim = g_initial.x.shape[1]
        model = get_model(args.model, in_dim, info['num_classes'], config.HIDDEN_DIM)
        
        lit_model = LitFullBatchGNN(
            encoder=model,
            initial_graph=g_initial,
            lr=config.LEARNING_RATE
        )

        # 6. Callback Setup (The Dynamic Defense)
        callbacks = []
        if args.mode == 'kmv-dynamic':
            print("[System] Engaging Dynamic Graph Callback (Resampling per epoch)")
            dg_callback = DynamicGraphCallback(
                g_hetero=g_hetero, # Pass raw hetero graph for re-sketching
                metapath=path_list,
                target_ntype=cfg.target_node,
                method='kmv',
                k=args.k,
                features=info['features'],
                labels=info['labels'],
                masks=info['masks'],
                device=config.DEVICE
            )
            callbacks.append(dg_callback)

        # 7. Training
        print(f"\n[Phase 2] Training ({args.epochs} epochs)...")
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=callbacks,
            enable_checkpointing=False,
            logger=False, # Disable for speed in benchmarks
            log_every_n_steps=1
        )
        
        start_train = time.perf_counter()
        trainer.fit(lit_model)
        train_time = time.perf_counter() - start_train
        
        # 8. Testing (Final Fidelity Check)
        res = trainer.test(lit_model, verbose=False)[0]
        
        # 9. Report Generation
        stats = {
            "Dataset": args.dataset,
            "Mode": args.mode,
            "K": args.k if 'kmv' in args.mode else 'N/A',
            "PrepTime": prep_time,
            "TrainTime": train_time,
            "TestAcc": res['test_acc'],
            "TestLoss": res['test_loss']
        }
        
        print("\n" + "-"*30)
        print("RESULTS SUMMARY")
        print("-"*30)
        print(pd.DataFrame([stats]).T)
        
        if args.csv_output:
            df = pd.DataFrame([stats])
            header = not os.path.exists(args.csv_output)
            df.to_csv(args.csv_output, mode='a', header=header, index=False)

    @staticmethod
    def register_subparser(subparsers) -> None:
        parser = subparsers.add_parser('train_fidelity', help='Run Journal Fidelity Experiments')
        parser.add_argument('--dataset', required=True)
        parser.add_argument('--metapath', required=True)
        parser.add_argument('--mode', choices=['exact', 'kmv-static', 'kmv-dynamic', 'random'], required=True)
        parser.add_argument('--model', default='GCN')
        parser.add_argument('--k', type=int, default=32)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--csv-output', type=str)
        parser.add_argument('--backend', type=str, default='python', choices=['python', 'cpp'], help='Backend to use for graph materialization')