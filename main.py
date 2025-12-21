"""
Main execution script for GNN scalability benchmarks and AnyBURL rule mining.
"""
import argparse
import sys
import os
import shutil
import time
import json
import gc
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader, NeighborLoader
from pytorch_lightning.callbacks import EarlyStopping
from torch_geometric.nn import to_hetero
from typing import Optional, List, Tuple

# Ensure local source imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import config, DatasetConfig
from src.data import DatasetFactory, FeatureGuard, GlobalUniverseMapper, ArtifactManager
from src.backend import BackendFactory, BenchmarkResult
from src.bridge import AnyBURLRunner
from src.utils import generate_random_metapath
from src.models import get_model
from src.sampling.disk import HNESnowballDiskSampler
from src.lit_model import LitGNN
from src.analysis import CosineFidelityMetric, PredictionAgreementMetric
import torch


class BenchmarkCommand:
    """ Handles performance benchmarking for different materialization backends. """
    
    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        self._backend = None
        
    def execute(self, args) -> None:
        # Console header
        print(f"\n{'='*60}")
        print(f"BENCHMARK MODE: {args.dataset}")
        print(f"Method: {args.method.upper()} | Backend: {self.backend_name.upper()}")
        print(f"{'='*60}\n")

        dataset_cfg = config.get_dataset_config(args.dataset)
        artifact_mgr = ArtifactManager(config.TEMP_DIR)
        
        g_hetero = None
        info = None
        
        # Zero-Copy Logic: Check for cached artifacts to avoid RAM spike
        # Only valid for C++ backend which can read directly from disk
        if self.backend_name == 'cpp' and artifact_mgr.exists():
            try:
                print(f"[System] Found cached artifacts. Attempting Zero-Copy load...")
                info = artifact_mgr.load_metadata()
                if 'schema' in info:
                    print("[System] Zero-Copy successful. Python Graph Memory = 0GB.")
                else:
                    # Cache is stale (missing schema for rule generation), force reload
                    info = None 
            except Exception:
                info = None

        # Fallback: Full load via Factory if cache missed or invalid
        if info is None:
            print(f"Loading dataset: {dataset_cfg}...")
            g_hetero, info = DatasetFactory.get_data(
                dataset_cfg.source,
                dataset_cfg.dataset_name,
                dataset_cfg.target_node
            )
            print(f"      Graph size: {g_hetero.num_nodes} nodes.")
            
            # Enrich info with schema for future zero-copy runs
            # This allows parsing metapaths without loading the full graph object
            info['schema'] = {
                'node_types': sorted(list(g_hetero.node_types)),
                'edge_types': sorted(list(g_hetero.edge_types))
            }
            if self.backend_name == 'cpp':
                artifact_mgr.save_metadata(info)

        # Resolve path
        if args.manual_path:
            print(f"\nParsing manual path: '{args.manual_path}'")
            metapath_strings = [args.manual_path]
        else:
            suggested = dataset_cfg.suggested_paths
            if suggested:
                print(f"\nUsing config-suggested path: '{suggested[0]}'")
                metapath_strings = [suggested[0]]
            else:
                # Random generation requires traversal, so we can't do this in zero-copy mode
                if g_hetero is None:
                    raise RuntimeError("Random path generation requires full graph load. Clear cache or specify --manual-path.")
                print(f"\nGenerating random walk path (len={args.metapath_length})...")
                random_path = generate_random_metapath(g_hetero, dataset_cfg.target_node, args.metapath_length)
                metapath_strings = [",".join([m[1] for m in random_path])]

        # Adapter: If g_hetero is None, create a dummy object so SchemaMatcher can read edge_types
        schema_provider = g_hetero
        if schema_provider is None:
            from types import SimpleNamespace
            schema_provider = SimpleNamespace(edge_types=info['schema']['edge_types'])

        metapath = self._parse_manual_path(metapath_strings[0], schema_provider)
        
        path_str = " -> ".join([f"[{m[1]}]" for m in metapath])
        print(f"      Schema: {path_str}")
        
        # Materialization
        print(f"\nInitializing {self.backend_name} backend...")
        self._backend = self._create_backend(args)
        self._backend.initialize(g_hetero, metapath, info)
        
        # Memory optimization: Free Python graph before C++ execution
        # This is the critical step preventing OOM during subprocess execution
        if self.backend_name == 'cpp' and g_hetero is not None:
            print("[System] Deallocating Python source graph to free RAM for C++...")
            del g_hetero
            import gc
            gc.collect()
        
        print(f"\nRunning {args.method} materialization...")
        try:
            result = self._run_materialization(args, dataset_cfg.target_node)
        except Exception as e:
            print(f"\n[ERROR] Materialization failed: {e}")
            raise e
        
        print(f"\n{'='*60}\nRESULTS\n{'='*60}")
        print(result)
        
        if args.run_inference:
            if self._backend.supports_inference:
                # Note: g_materialized might be None if backend manages memory internally
                self._run_inference_test(args, result, info, getattr(self, 'g_materialized', None), dataset_cfg)
        
        self._backend.cleanup()

        
    def _parse_manual_path(self, path_str: str, g_hetero) -> List[Tuple[str, str, str]]:
        tuples = []
        rels = path_str.split(',')
        
        # Local import to prevent circular dependency
        from src.utils import SchemaMatcher 
        
        for r in rels:
            r = r.strip()
            matched_edge = SchemaMatcher.match(r, g_hetero)
            tuples.append(matched_edge)
                
        return tuples
    
    def _create_backend(self, args):
        if self.backend_name == 'python':
            return BackendFactory.create('python', device=config.DEVICE)
        elif self.backend_name == 'cpp':
            return BackendFactory.create(
                'cpp',
                executable_path=config.CPP_EXECUTABLE,
                temp_dir=config.TEMP_DIR
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")
    
    def _run_materialization(self, args, target_ntype: str) -> BenchmarkResult:
        if args.method == 'exact':
            g_result = self._backend.materialize_exact()
        elif args.method == 'kmv':
            g_result = self._backend.materialize_kmv(k=args.k)
        else:
            raise ValueError(f"Unknown method: {args.method}")
        
        return BenchmarkResult(
            method=args.method,
            backend=self.backend_name,
            num_edges=g_result.num_edges,
            num_nodes=g_result.num_nodes,
            prep_time=self._backend.get_prep_time()
        )

    def _run_inference_test(self, args, result, info, g_materialized, dataset_cfg):
        model_path = config.get_model_path(args.dataset, args.model)
        if not os.path.exists(model_path):
            print(f"  [Warning] Model not found at {model_path}. Skipping.")
            return

        model = get_model(args.model, info['in_dim'], info['out_dim'], config.HIDDEN_DIM).to(config.DEVICE)
        model.eval()
        
        start_t = time.perf_counter()
        with torch.no_grad():
            out = model(g_materialized.x.to(config.DEVICE), g_materialized.edge_index.to(config.DEVICE))
            pred = out[info['masks']['test']].argmax(dim=1)
            correct = (pred == info['labels'][info['masks']['test']].to(config.DEVICE)).sum().item()
            acc = correct / info['masks']['test'].sum().item()
        
        result.infer_time = time.perf_counter() - start_t
        result.accuracy = acc
        print(f"  [Inference] Accuracy: {acc:.4f} | Time: {result.infer_time:.4f}s")


class MiningCommand:
    """ Rule mining pipeline using AnyBURL with optional disk sampling. """
    
    def execute(self, args) -> None:
        print(f"\n{'='*60}")
        print(f"RULE MINING MODE: {args.dataset}")
        print(f"{'='*60}\n")
        
        dataset_cfg = config.get_dataset_config(args.dataset)
        main_working_dir = os.path.join(config.DATA_DIR, f"{dataset_cfg.source}_{dataset_cfg.dataset_name}")
        
        print(f"Working directory: {main_working_dir}")
        
        # Disk-based sampling for large datasets (OOM prevention)
        if args.sample_method == 'snowball':
            print(f"\nRunning snowball disk sampler...")
            
            if dataset_cfg.source != 'HNE':
                print(" Disk sampling currently restricted to 'HNE' format.")
                print(" Skipping sampling...")
            else:
                source_dir = os.path.join(config.DATA_DIR, f"HNE_{dataset_cfg.dataset_name}")
                sampled_name = f"{dataset_cfg.dataset_name}_Sampled"
                target_dir = os.path.join(config.DATA_DIR, f"HNE_{sampled_name}")
                
                expected_file = os.path.join(target_dir, "link.dat")
                if os.path.exists(expected_file):
                    print(f" Using cached sample at {target_dir}")
                else:
                    print(f" Generating new sample (Seeds={args.sample_seeds}, Hops={args.sample_hops})")
                    sampler = HNESnowballDiskSampler()
                    sampler_config = {
                        'seeds': args.sample_seeds, 
                        'hops': args.sample_hops
                    }
                    sampler.sample(source_dir, target_dir, sampler_config)
                
                sampled_key = f"{args.dataset}_SAMPLED"
                new_config = DatasetConfig(
                    source='HNE',
                    dataset_name=sampled_name, 
                    target_node=dataset_cfg.target_node
                )
                config.register_dataset(sampled_key, new_config)
                
                args.dataset = sampled_key
                dataset_cfg = new_config
        else:
            print("Loading full dataset...")

        # Memory load
        print(f"\n      Loading data: {dataset_cfg}...")
        g_hetero, _ = DatasetFactory.get_data(
            dataset_cfg.source,
            dataset_cfg.dataset_name,
            dataset_cfg.target_node
        )
        
        # Rule extraction
        print(f"\nExporting for AnyBURL...")
        runner = AnyBURLRunner(main_working_dir, config.ANYBURL_JAR)
        runner.export_graph(g_hetero)
        
        print(f"\nMining rules (timeout={args.timeout}s)...")
        runner.run_mining(
            timeout=args.timeout,
            max_length=args.max_length,
            num_threads=args.threads
        )
        
        # Result parsing
        print("\nParsing mined metapaths...")
        clean_file = runner.save_clean_list()
        
        best_path = runner.parse_best_metapath(
            dataset_cfg.target_node,
            dataset_cfg.target_node,
            min_confidence=args.min_confidence
        )
        
        if os.path.exists(clean_file):
            print(f"\n List saved: {clean_file}")
            clean_name = dataset_cfg.dataset_name.replace('_Sampled', '').replace('_SAMPLED', '')
            print(f" Run batch benchmarks: python run_batch.py {dataset_cfg.source}_{clean_name}")
        
        if best_path:
            print(f"\n Best Path (Conf={args.min_confidence}):")
            print(f" {' -> '.join(best_path)}")
        else:
            print("\n No valid metapaths found.")

class FoundationTrainCommand:
    """ Train GNN on the homogenized 'Universe' space. """

    def execute(self, args) -> None:
        model_path = config.get_model_path(args.dataset, args.model)
        
        if os.path.exists(model_path):
            print(f"\n[Info] Model already exists at {model_path}.")
            print("Skipping training. (Delete the file to force retrain).")
            return
        
        print(f"\n{'='*60}")
        print(f"FOUNDATION TRAINING: {args.dataset}")
        print(f"{'='*60}\n")

        dataset_cfg = config.get_dataset_config(args.dataset)
        g_hetero, info = DatasetFactory.get_data(
            dataset_cfg.source, dataset_cfg.dataset_name, dataset_cfg.target_node
        )
        
        FeatureGuard.ensure_features(g_hetero, seed=42, default_dim=64)
        
        # Attach labels/masks to graph for Mapper
        target = dataset_cfg.target_node
        g_hetero[target].train_mask = info['masks']['train']
        g_hetero[target].val_mask = info['masks']['val']
        g_hetero[target].test_mask = info['masks']['test']
        
        if 'labels' in info:
            g_hetero[target].y = info['labels']
        
        print("Mapping to Universe space...")
        # Recalculate dims in case FeatureGuard added something
        dims = {}
        for nt in g_hetero.node_types:
            if hasattr(g_hetero[nt], 'x') and g_hetero[nt].x is not None:
                dims[nt] = g_hetero[nt].x.shape[1]

        mapper = GlobalUniverseMapper(g_hetero.node_types, dims)
        g_homo = mapper.transform(g_hetero)
        
        print(f"      Dimension D = {mapper.global_max_dim}")

        model = get_model(
            args.model, 
            in_dim=mapper.global_max_dim, 
            out_dim=info['num_classes'], 
            h_dim=config.HIDDEN_DIM
        )
        lit_model = LitGNN(model, lr=config.LEARNING_RATE)

        print(f"\nTraining {args.model} ({args.epochs} epochs)...")
        
        train_loader = NeighborLoader(
            g_homo,
            num_neighbors=[10] * 2,  
            batch_size=1024,        
            shuffle=True,
            num_workers=4
        )

        val_loader = NeighborLoader(
            g_homo,
            num_neighbors=[10] * 2,
            batch_size=1024,
            shuffle=False,
            num_workers=4
        )
        
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

class FidelityCommand:
    """ Evaluates representation shift between Exact and KMV materialization. """

    def execute(self, args) -> None:
        print(f"\n{'='*60}")
        print(f"FIDELITY CHECK: {args.model} on {args.dataset}")
        print(f"Metapath: {args.metapath} | k={args.k}")
        print(f"{'='*60}\n")

        model_path = config.get_model_path(args.dataset, args.model)
        config_path = model_path.replace('.pt', '_mapper.json')
        
        if not os.path.exists(model_path):
            print("[Error] weights not found.")
            return

        with open(config_path, 'r') as f:
            mapper_cfg = json.load(f)

        dataset_cfg = config.get_dataset_config(args.dataset)
        g_hetero, info = DatasetFactory.get_data(
            dataset_cfg.source, dataset_cfg.dataset_name, dataset_cfg.target_node
        )

        from src.utils import SchemaMatcher
        
        path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in args.metapath.split(',')]
        
        if args.backend == 'cpp':
            backend = BackendFactory.create(
                'cpp',
                executable_path=config.CPP_EXECUTABLE,
                temp_dir=config.TEMP_DIR
            )
        else:
            backend = BackendFactory.create('python', device=config.DEVICE)

        backend.initialize(g_hetero, path_list, info)

        g_exact = backend.materialize_exact()
        g_kmv = backend.materialize_kmv(k=args.k)

        model = get_model(
            args.model, 
            in_dim=mapper_cfg['global_max_dim'], 
            out_dim=info['num_classes'], 
            h_dim=config.HIDDEN_DIM
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(config.DEVICE).eval()

        print("\nComputing metrics...")
        metric_cos = CosineFidelityMetric()
        metric_agree = PredictionAgreementMetric()

        with torch.no_grad():
            x_exact = self._pad_to_universe(g_exact.x, mapper_cfg['global_max_dim'])
            x_kmv = self._pad_to_universe(g_kmv.x, mapper_cfg['global_max_dim'])
            
            out_exact = model(x_exact, g_exact.edge_index.to(config.DEVICE))
            out_kmv = model(x_kmv, g_kmv.edge_index.to(config.DEVICE))

            test_mask = info['masks']['test'].to(config.DEVICE)
            labels = info['labels'].to(config.DEVICE)
            
            acc_exact = self._calc_accuracy(out_exact, labels, test_mask)
            acc_kmv = self._calc_accuracy(out_kmv, labels, test_mask)
            
            fid_score = metric_cos.calculate(out_exact[test_mask], out_kmv[test_mask])
            agree_score = metric_agree.calculate(out_exact[test_mask], out_kmv[test_mask])

        print(f"\n{'-'*30}")
        print(f"RESULTS (k={args.k})")
        print(f"{'-'*30}")
        print(f"  Exact Accuracy:    {acc_exact:.4f}")
        print(f"  KMV Accuracy:      {acc_kmv:.4f}")
        print(f"  ------------------------------")
        print(f"  Fidelity (Cosine): {fid_score:.4f}")
        print(f"  Prediction Agree:  {agree_score:.4f}")
        print(f"{'-'*30}")
        
        backend.cleanup()

    def _pad_to_universe(self, x, target_dim):
        x = x.to(config.DEVICE)
        if x.size(1) == target_dim: return x
        return torch.nn.functional.pad(x, (0, target_dim - x.size(1)), "constant", 0)

    def _calc_accuracy(self, logits, labels, mask):
        pred = logits[mask].argmax(dim=1)
        correct = (pred == labels[mask]).sum().item()
        return correct / mask.sum().item()
    
class ListCommand:
    """ Lists registered resources. """
    
    def execute(self, args) -> None:
        print("\n" + "="*60)
        print("DATASETS")
        print("="*60 + "\n")
        
        datasets = config.list_datasets()
        for key in sorted(datasets):
            cfg = config.get_dataset_config(key)
            print(f"  {key:20s} | Source: {cfg.source:5s} | Target: {cfg.target_node}")
        
        print(f"\nTotal: {len(datasets)}")
        
        print("\n" + "="*60)
        print("BACKENDS")
        print("="*60 + "\n")
        
        backends = BackendFactory.list_backends()
        for b in backends:
            print(f"  - {b}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GNN Scalability Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Global toggle for hardware override
    parser.add_argument('--cpu', action='store_true', help='Force CPU execution')
    
    subparsers = parser.add_subparsers(dest='mode', help='Task mode')
    
    # Benchmark
    bench_parser = subparsers.add_parser('benchmark')
    bench_parser.add_argument('--dataset', type=str, required=True)
    bench_parser.add_argument('--method', type=str, choices=['exact', 'kmv'], default='exact')
    bench_parser.add_argument('--backend', type=str, choices=['python', 'cpp'], default='python')
    bench_parser.add_argument('--k', type=int, default=32)
    bench_parser.add_argument('--metapath-length', type=int, default=config.METAPATH_LENGTH)
    bench_parser.add_argument('--manual-path', type=str)
    bench_parser.add_argument('--run-inference', action='store_true')
    bench_parser.add_argument('--model', type=str, choices=['GCN', 'SAGE', 'GAT'], default='SAGE') 
    
    # Mine
    mine_parser = subparsers.add_parser('mine')
    mine_parser.add_argument('--dataset', type=str, required=True)
    mine_parser.add_argument('--timeout', type=int, default=60)
    mine_parser.add_argument('--max-length', type=int, default=4)
    mine_parser.add_argument('--threads', type=int, default=4)
    mine_parser.add_argument('--min-confidence', type=float, default=0.001)
    mine_parser.add_argument('--export', action='store_true')
    mine_parser.add_argument('--sample-method', type=str, choices=['snowball'], default=None)
    mine_parser.add_argument('--sample-seeds', type=int, default=1000)
    mine_parser.add_argument('--sample-hops', type=int, default=2)

    # Train
    train_parser = subparsers.add_parser('train_foundation')
    train_parser.add_argument('--dataset', type=str, required=True)
    train_parser.add_argument('--model', type=str, default='SAGE')
    train_parser.add_argument('--epochs', type=int, default=50)

    # Fidelity
    fid_parser = subparsers.add_parser('fidelity')
    fid_parser.add_argument('--dataset', type=str, required=True)
    fid_parser.add_argument('--model', type=str, required=True)
    fid_parser.add_argument('--metapath', type=str, required=True)
    fid_parser.add_argument('--k', type=int, default=32)
    fid_parser.add_argument('--backend', type=str, choices=['python', 'cpp'], default='python')

    # List
    subparsers.add_parser('list')
    
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.cpu:
        print("[System] CPU override requested. Disabling CUDA for this run.")
        config.DEVICE = torch.device('cpu')
    
    if not config.validate():
        print("\n[Warning] Config validation issues.")
    
    try:
        if args.mode == 'benchmark':
            command = BenchmarkCommand(args.backend)
            command.execute(args)
        elif args.mode == 'mine':
            command = MiningCommand()
            command.execute(args)
        elif args.mode == 'list':
            command = ListCommand()
            command.execute(args)
        elif args.mode == 'train_foundation':
            command = FoundationTrainCommand()
            command.execute(args)
        elif args.mode == 'fidelity':
            command = FidelityCommand()
            command.execute(args)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()