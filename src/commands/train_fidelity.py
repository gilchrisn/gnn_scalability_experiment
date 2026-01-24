import time
import torch
import pytorch_lightning as pl
from argparse import Namespace
import pandas as pd
import os
from pytorch_lightning.callbacks import EarlyStopping  # <--- Added Import

from .base import BaseCommand
from ..config import config
from ..data import DatasetFactory
from ..backend import BackendFactory
from ..models import get_model
from ..utils import SchemaMatcher
from ..lit_model_full import LitFullBatchGNN
from ..callbacks import DynamicGraphCallback, GraphCyclingCallback

class TrainFidelityCommand(BaseCommand):
    """
    Executes Journal Experiments A & B: Fidelity and Scalability.
    """

    def execute(self, args: Namespace) -> None:
        print(f"\n{'='*60}")
        print(f"FIDELITY EXPERIMENT: {args.dataset}")
        print(f"Mode: {args.mode.upper()} | Backend: {args.backend.upper()} | L={args.l}")
        print(f"{'='*60}\n")

        # 1. Load Heterogeneous Data
        cfg = config.get_dataset_config(args.dataset)
        g_hetero, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        
        # 2. Parse Metapath
        path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in args.metapath.split(',')]
        
        # 3. Initialize Backend
        # Note: We pass num_sketches=args.l here so C++ backend knows how many to generate
        backend = BackendFactory.create(
            args.backend, 
            executable_path=config.CPP_EXECUTABLE, 
            temp_dir=config.TEMP_DIR, 
            device=config.DEVICE,
            num_sketches=args.l
        )
        backend.initialize(g_hetero, path_list, info)
        
        # 4. Materialization Strategy (Factory Logic)
        print(f"[Phase 1] Materialization...")
        start_t = time.perf_counter()
        
        g_initial = None
        callbacks = []

        # Case A: C++ Ensemble Dynamic (Pre-compute L graphs & Cycle)
        if args.mode == 'kmv-dynamic' and args.backend == 'cpp':
            print(f"   -> Mode: Pre-computed Ensemble (L={args.l})")
            graphs = backend.materialize_kmv_ensemble(k=args.k)
            
            if not graphs:
                raise RuntimeError("Backend returned 0 graphs for ensemble.")
            
            g_initial = graphs[0] # Initialize model with first graph
            callbacks.append(GraphCyclingCallback(graphs))

        # Case B: Python On-the-Fly Dynamic
        elif args.mode == 'kmv-dynamic' and args.backend == 'python':
            print(f"   -> Mode: On-the-Fly Re-sampling")
            g_initial = backend.materialize_kmv(k=args.k) # Initial
            
            dg_callback = DynamicGraphCallback(
                g_hetero=g_hetero,
                metapath=path_list,
                target_ntype=cfg.target_node,
                method='kmv',
                k=args.k,
                data_info=info,
                device=config.DEVICE
            )
            callbacks.append(dg_callback)

        # Case C: Random Baseline
        elif args.mode == 'random':
            print(f"   -> Mode: Random Sampling")
            if hasattr(backend, 'materialize_random'):
                g_initial = backend.materialize_random(k=args.k)
            else:
                # Fallback for C++ if random not impl yet (defaults to exact usually or error)
                raise NotImplementedError("Random sampling not supported in C++ backend yet.")

        # Case D: Static (Exact or KMV-Static)
        else:
            print(f"   -> Mode: Static ({args.mode})")
            if args.mode == 'exact':
                g_initial = backend.materialize_exact()
            else:
                g_initial = backend.materialize_kmv(k=args.k)

        prep_time = time.perf_counter() - start_t
        print(f"   -> Prep Time: {prep_time:.4f}s")

        # --- NEW: Inject Early Stopping & Tracking ---
        early_stop_ref = None # Reference to callback for stat extraction
        if args.epochs > 0:
            early_stop_ref = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=10,        # Slightly higher patience for noisy dynamic graphs
                verbose=False,
                mode="min"
            )
            callbacks.append(early_stop_ref)
        # ---------------------------------------------

        # 5. Model Setup
        in_dim = g_initial.x.shape[1]
        model = get_model(args.model, in_dim, info['num_classes'], config.HIDDEN_DIM)
        
        lit_model = LitFullBatchGNN(
            encoder=model,
            initial_graph=g_initial,
            lr=config.LEARNING_RATE
        )

        # 6. Training
        print(f"\n[Phase 2] Training ({args.epochs} epochs)...")
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=callbacks, # Now includes EarlyStopping + GraphCycling/Dynamic
            enable_checkpointing=False,
            logger=False, 
            log_every_n_steps=1
        )
        
        start_train = time.perf_counter()
        trainer.fit(lit_model)
        train_time = time.perf_counter() - start_train
        
        # 7. Testing
        res = trainer.test(lit_model, verbose=False)[0]
        
        # --- NEW: Calculate Convergence Metrics ---
        total_epochs = trainer.current_epoch + 1 # 0-indexed
        
        # Heuristic: If stopped early, true convergence was 'patience' epochs ago
        stopped_early = (early_stop_ref and early_stop_ref.stopped_epoch > 0)
        patience_buffer = early_stop_ref.patience if stopped_early else 0
        effective_epochs = max(1, total_epochs - patience_buffer)
        
        avg_time_per_epoch = train_time / total_epochs if total_epochs > 0 else 0
        effective_converge_time = avg_time_per_epoch * effective_epochs
        # ------------------------------------------

        # 8. Report
        stats = {
            "Dataset": args.dataset,
            "Mode": args.mode,
            "Backend": args.backend,
            "K": args.k if 'kmv' in args.mode else 'N/A',
            "L": args.l,
            "PrepTime": prep_time,
            "TrainTime": train_time,
            "TestAcc": res['test_acc'],
            # Convergence Stats
            "TotalEpochs": total_epochs,
            "ConvergeEpoch": effective_epochs,
            "TimePerEpoch": avg_time_per_epoch,
            "TimeToConverge": effective_converge_time
        }
        
        print("\n" + "-"*30)
        print("RESULTS SUMMARY")
        print("-"*30)
        print(pd.DataFrame([stats]).T)
        
        if args.csv_output:
            df = pd.DataFrame([stats])
            header = not os.path.exists(args.csv_output)
            df.to_csv(args.csv_output, mode='a', header=header, index=False)

        backend.cleanup()

    @staticmethod
    def register_subparser(subparsers) -> None:
        parser = subparsers.add_parser('train_fidelity', help='Run Fidelity Experiments')
        parser.add_argument('--dataset', required=True)
        parser.add_argument('--metapath', required=True)
        parser.add_argument('--mode', choices=['exact', 'kmv-static', 'kmv-dynamic', 'random'], required=True)
        parser.add_argument('--model', default='GCN')
        parser.add_argument('--k', type=int, default=32)
        parser.add_argument('--l', type=int, default=1, help='Ensemble size (for kmv-dynamic with C++)')
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--csv-output', type=str)
        parser.add_argument('--backend', type=str, default='python', choices=['python', 'cpp'])