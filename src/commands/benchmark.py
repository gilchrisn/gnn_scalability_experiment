import time
import os
import torch
import gc
import pandas as pd
import torch.nn.functional as F
from argparse import Namespace
from typing import List, Tuple

from .base import BaseCommand
from ..config import config
from ..data import DatasetFactory, BaselineCache  # Removed ArtifactManager
from ..backend import BackendFactory, BenchmarkResult
from ..utils import generate_random_metapath, SchemaMatcher
from ..models import get_model
from ..analysis import CosineFidelityMetric, PredictionAgreementMetric

class BenchmarkCommand(BaseCommand):
    """
    Executes benchmark: Materialization Speed + Model Fidelity.
    
    Features:
    1. Fidelity Checking: Compares Approximate vs. Exact embeddings.
    2. Full Loading: Always loads fresh data from source.
    """

    def execute(self, args: Namespace) -> None:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {args.dataset} | {args.model}")
        print(f"Method: {args.method.upper()} (k={args.k}) | Backend: {args.backend.upper()}")
        print(f"{'='*60}\n")

        # --- 1. Data Loading Strategy (Full Load Only) ---
        dataset_cfg = config.get_dataset_config(args.dataset)
        
        print(f"Loading dataset: {dataset_cfg}...")
        g_hetero, info = DatasetFactory.get_data(
            dataset_cfg.source,
            dataset_cfg.dataset_name,
            dataset_cfg.target_node
        )
        print(f"      Graph size: {g_hetero.num_nodes} nodes.")
        
        # Enrich info with schema for consistency
        info['schema'] = {
            'node_types': sorted(list(g_hetero.node_types)),
            'edge_types': sorted(list(g_hetero.edge_types))
        }

        # --- 2. Setup Model & Metrics (If Fidelity Requested) ---
        model = None
        exact_baseline_out = None
        
        if args.check_fidelity:
            model = self._load_model(args.dataset, args.model, info)
            metric_cos = CosineFidelityMetric()
            metric_agree = PredictionAgreementMetric()

        # --- 3. Path Parsing ---
        if args.manual_path:
            print(f"\nParsing manual path: '{args.manual_path}'")
            metapath_strings = [args.manual_path]
        else:
            suggested = dataset_cfg.suggested_paths
            if suggested:
                print(f"\nUsing config-suggested path: '{suggested[0]}'")
                metapath_strings = [suggested[0]]
            else:
                print(f"\nGenerating random walk path (len={args.metapath_length})...")
                random_path = generate_random_metapath(g_hetero, dataset_cfg.target_node, args.metapath_length)
                metapath_strings = [",".join([m[1] for m in random_path])]

        metapath = self._parse_manual_path(metapath_strings[0], g_hetero)
        path_str = " -> ".join([f"[{m[1]}]" for m in metapath])
        print(f"      Schema: {path_str}")

        # --- 4. Initialize Backend ---
        print(f"\nInitializing {args.backend} backend...")
        backend = BackendFactory.create(args.backend, 
                                        executable_path=config.CPP_EXECUTABLE,
                                        temp_dir=config.TEMP_DIR,
                                        device=config.DEVICE)
        
        backend.initialize(g_hetero, metapath, info)

        # Memory optimization: Free Python graph before C++ execution
        if args.backend == 'cpp' and g_hetero is not None:
            print("[System] Deallocating Python source graph to free RAM for C++...")
            del g_hetero
            gc.collect()

        # Initialize Cache
        baseline_cache = BaselineCache(config.TEMP_DIR)
        exact_baseline_out = None

        # --- 5 & 6. Combined Materialization & Baseline Logic ---
        
        # CASE A: The user is running EXACT. This is the "Producer" run.
        if args.method == 'exact':
            print(f"\n[Target] Running EXACT Materialization...")
            start_t = time.perf_counter()
            g_target = backend.materialize_exact()
            
            # Helper logic to capture prep time (fallback if backend doesn't provide it)
            prep_time = backend.get_prep_time()
            if prep_time <= 0: prep_time = time.perf_counter() - start_t

            # If we need fidelity (or just want to populate cache), we run inference now
            if args.check_fidelity:
                with torch.no_grad():
                    # Using your existing _pad_features
                    x_target = self._pad_features(g_target.x, model.conv1.in_channels)
                    out_target = model(x_target, g_target.edge_index.to(config.DEVICE))
                
                # SAVE TO CACHE
                print("[Cache] Saving Exact results to disk...")
                baseline_cache.save(args.dataset, args.model, path_str, {
                    'logits': out_target.cpu()
                })
                
                # For Exact, the target IS the baseline
                exact_baseline_out = out_target

        # CASE B: The user is running KMV/RANDOM. This is the "Consumer" run.
        else:
            # 1. First, establish the baseline (Try Cache -> Fallback to Compute)
            if args.check_fidelity:
                print("\n[Baseline] Checking cache for Exact Baseline...")
                cached_data = baseline_cache.load(args.dataset, args.model, path_str)
                
                if cached_data is not None:
                    # CACHE HIT: Instant load
                    exact_baseline_out = cached_data['logits'].to(config.DEVICE)
                else:
                    # CACHE MISS: Compute strictly as fallback (The old slow way)
                    print(" -> Cache miss. Computing Exact Baseline on-the-fly...")
                    g_exact = backend.materialize_exact()
                    with torch.no_grad():
                        x_ex = self._pad_features(g_exact.x, model.conv1.in_channels)
                        exact_baseline_out = model(x_ex, g_exact.edge_index.to(config.DEVICE))
                    
                    # Clean up strictly
                    del g_exact
                    gc.collect()

            # 2. Now run the actual Target Materialization (KMV/Random)
            print(f"\n[Target] Running {args.method.upper()} Materialization...")
            start_t = time.perf_counter()
            
            if args.method == 'kmv':
                g_target = backend.materialize_kmv(k=args.k)
            elif args.method == 'random':
                if args.backend != 'python':
                    raise ValueError("Random sampling is only supported with the Python backend.")
                g_target = backend.materialize_random(k=args.k)
            else:
                raise ValueError(f"Unknown method: {args.method}")

            # Capture timing
            prep_time = backend.get_prep_time()
            if prep_time <= 0: prep_time = time.perf_counter() - start_t

        # --- 7. Run Inference & Compute Metrics ---
        stats = {
            "Dataset": args.dataset,
            "Model": args.model,
            "Method": args.method,
            "K": args.k,
            "Metapath": path_str,
            "Backend": args.backend,
            "PrepTime": prep_time,
            "Acc": 0.0,
            "Fidelity": 1.0,
            "Agreement": 1.0
        }

        if args.check_fidelity:
            with torch.no_grad():
                x_target = self._pad_features(g_target.x, model.conv1.in_channels)
                out_target = model(x_target, g_target.edge_index.to(config.DEVICE))
                
                # Accuracy
                if 'masks' in info and 'test' in info['masks']:
                    mask = info['masks']['test'].to(config.DEVICE)
                    labels = info['labels'].to(config.DEVICE)
                    pred = out_target[mask].argmax(dim=1)
                    acc = (pred == labels[mask]).float().mean().item()
                    stats["Acc"] = acc

                # Fidelity Metrics (Compare against Baseline)
                if exact_baseline_out is not None:
                    # Filter only test nodes
                    emb_a = exact_baseline_out[mask]
                    emb_b = out_target[mask]
                    
                    stats["Fidelity"] = metric_cos.calculate(emb_a, emb_b)
                    stats["Agreement"] = metric_agree.calculate(exact_baseline_out[mask], out_target[mask])
                
                print(f"  -> Acc: {stats['Acc']:.4f} | Fid: {stats['Fidelity']:.4f} | Agree: {stats['Agreement']:.4f}")

        # --- 8. Save Results ---
        if args.csv_output:
            self._save_to_csv(stats, args.csv_output)

        backend.cleanup()

    def _parse_manual_path(self, path_str: str, g_hetero) -> List[Tuple[str, str, str]]:
        tuples = []
        rels = path_str.split(',')
        for r in rels:
            r = r.strip()
            # SchemaMatcher is robust enough to handle dummy g_hetero if edge_types are present
            matched_edge = SchemaMatcher.match(r, g_hetero)
            tuples.append(matched_edge)
        return tuples

    def _load_model(self, dataset, model_name, info):
        model_path = config.get_model_path(dataset, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
            
        import json
        with open(model_path.replace('.pt', '_mapper.json'), 'r') as f:
            mapper_cfg = json.load(f)
            
        model = get_model(model_name, mapper_cfg['global_max_dim'], info['num_classes'], config.HIDDEN_DIM)
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.to(config.DEVICE).eval()
        return model

    def _pad_features(self, x, target_dim):
        x = x.to(config.DEVICE)
        if x.size(1) == target_dim: return x
        return F.pad(x, (0, target_dim - x.size(1)), "constant", 0)

    def _save_to_csv(self, stats, filepath):
        df = pd.DataFrame([stats])
        header = not os.path.exists(filepath)
        df.to_csv(filepath, mode='a', header=header, index=False)
        print(f"[IO] Results appended to {filepath}")

    @staticmethod
    def register_subparser(subparsers) -> None:
        parser = subparsers.add_parser('benchmark', help='Run materialization & fidelity benchmark')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--method', type=str, choices=['exact', 'kmv', 'random'], default='exact')
        parser.add_argument('--backend', type=str, choices=['python', 'cpp'], default='python')
        parser.add_argument('--k', type=int, default=32)
        parser.add_argument('--metapath-length', type=int, default=config.METAPATH_LENGTH)
        parser.add_argument('--manual-path', type=str)
        parser.add_argument('--check-fidelity', action='store_true', help="Run inference and calculate fidelity metrics")
        parser.add_argument('--model', type=str, choices=['GCN', 'SAGE', 'GAT'], default='SAGE')
        parser.add_argument('--csv-output', type=str, help="Path to append results CSV")