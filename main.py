"""
Unified entry point for GNN scalability benchmarking and rule mining.
Implements the Command pattern with SOLID principles.

Usage Examples:
    python main.py benchmark --dataset HGB_DBLP --method exact --backend python
    python main.py mine --dataset HNE_DBLP --sample-method snowball --sample-seeds 1000
"""
import argparse
import sys
import os
import shutil
import time
from typing import Optional, List, Tuple

# Add project root to path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Internal imports
from src.config import config, DatasetConfig
from src.data import DatasetFactory
from src.backend import BackendFactory, BenchmarkResult
from src.bridge import AnyBURLRunner
from src.utils import generate_random_metapath
from src.models import get_model
from src.sampling.disk import HNESnowballDiskSampler
import torch


class BenchmarkCommand:
    """
    Command for running benchmarks.
    Follows Command Pattern and Dependency Injection.
    """
    
    def __init__(self, backend_name: str):
        """
        Initialize the benchmark command.
        
        Args:
            backend_name: Backend identifier ('python' or 'cpp')
        """
        self.backend_name = backend_name
        self._backend = None
    
    def execute(self, args) -> None:
        """
        Execute benchmark with specified configuration.
        """
        # Header Logging
        print(f"\n{'='*60}")
        print(f"BENCHMARK MODE: {args.dataset}")
        print(f"Method: {args.method.upper()} | Backend: {self.backend_name.upper()}")
        print(f"{'='*60}\n")
        
        # ---------------------------------------------------------
        # Step 1: Load Dataset
        # ---------------------------------------------------------
        dataset_cfg = config.get_dataset_config(args.dataset)
        print(f"[1/4] Loading dataset configuration: {dataset_cfg}...")
        
        # Use Factory to get data loader
        g_hetero, info = DatasetFactory.get_data(
            dataset_cfg.source,
            dataset_cfg.dataset_name,
            dataset_cfg.target_node
        )
        print(f"      Successfully loaded graph with {g_hetero.num_nodes} nodes.")
        
        # ---------------------------------------------------------
        # Step 2: Determine Metapath
        # ---------------------------------------------------------
        # Check if user provided a manual path (e.g. from AnyBURL)
        if args.manual_path:
            print(f"\n[2/4] Using manually specified metapath...")
            print(f"      Input string: '{args.manual_path}'")
            metapath = self._parse_manual_path(args.manual_path)
        else:
            print(f"\n[2/4] Generating random metapath (length={args.metapath_length})...")
            metapath = generate_random_metapath(
                g_hetero,
                dataset_cfg.target_node,
                args.metapath_length
            )
        
        # Display the path clearly
        path_str = " -> ".join([m[1] for m in metapath])
        print(f"      Selected Path: {path_str}")
        
        # ---------------------------------------------------------
        # Step 3: Initialize Backend
        # ---------------------------------------------------------
        print(f"\n[3/4] Initializing {self.backend_name} backend...")
        
        # Create backend instance via Factory
        self._backend = self._create_backend(args)
        
        # Pass data to backend for preparation
        # This step might involve converting data formats (e.g. for C++)
        self._backend.initialize(g_hetero, metapath, info)
        
        # ---------------------------------------------------------
        # Step 4: Execute Materialization
        # ---------------------------------------------------------
        print(f"\n[4/4] Running {args.method} materialization...")
        
        try:
            result = self._run_materialization(args, dataset_cfg.target_node)
        except Exception as e:
            print(f"\n[ERROR] Materialization failed: {e}")
            raise e
        
        # ---------------------------------------------------------
        # Results Display
        # ---------------------------------------------------------
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(result)
        
        # Optional: Run inference if model specified
        if args.run_inference:
            if self._backend.supports_inference:
                self._run_inference_test(args, result, info)
            else:
                print("\n[Info] Inference skipped (Backend does not support it).")
        
        # Cleanup backend resources
        self._backend.cleanup()
    
    def _parse_manual_path(self, path_str: str) -> List[Tuple[str, str, str]]:
        """
        Parses a manual path string into the expected tuple format.
        Expected format: "rel1,rel2" or "src_to_dst,dst_to_src"
        
        Args:
            path_str: Comma-separated relations
            
        Returns:
            List of (src, rel, dst) tuples.
        """
        tuples = []
        rels = path_str.split(',')
        
        for r in rels:
            r = r.strip()
            # Try to infer types from HNE naming convention "type1_to_type2"
            if "_to_" in r:
                parts = r.split('_to_')
                src = parts[0]
                dst = parts[1]
                tuples.append((src, r, dst))
            else:
                # Fallback: Assume homogeneous or unknown types
                # This might fail if the backend strictly checks types against the schema
                tuples.append(('node', r, 'node'))
                
        return tuples
    
    def _create_backend(self, args):
        """Factory method to create backend instance."""
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
        """Run materialization using the backend."""
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
    
    def _run_inference_test(self, args, result: BenchmarkResult, info: dict) -> None:
        """Optional: Run GNN inference to measure end-to-end performance."""
        print(f"\n[BONUS] Running inference test...")
        print("  (Inference testing requires full GNN pipeline integration)")
        # This is where the LitModel would be called


class MiningCommand:
    """
    Command for rule mining using AnyBURL.
    Supports Disk-Based Sampling to prevent OOM on large datasets.
    """
    
    def execute(self, args) -> None:
        """Execute rule mining pipeline."""
        print(f"\n{'='*60}")
        print(f"RULE MINING MODE: {args.dataset}")
        print(f"{'='*60}\n")
        
        # ---------------------------------------------------------
        # Step 1: Configuration & Directory Setup
        # ---------------------------------------------------------
        dataset_cfg = config.get_dataset_config(args.dataset)

        main_working_dir = os.path.join(config.DATA_DIR, f"{dataset_cfg.source}_{dataset_cfg.dataset_name}")
        
        print(f"[Setup] Rules will be saved to: {main_working_dir}")
        
        # ---------------------------------------------------------
        # Step 2: Disk Sampling (Optional Prevention of OOM)
        # ---------------------------------------------------------
        if args.sample_method == 'snowball':
            print(f"\n[1/4] 🔴 Preparing DISK-BASED sampling...")
            
            if dataset_cfg.source != 'HNE':
                print("⚠️ Disk sampling currently only supports 'HNE' format datasets.")
                print("   Skipping sampling (might OOM)...")
            else:
                # Define paths for sampling
                source_dir = os.path.join(config.DATA_DIR, f"HNE_{dataset_cfg.dataset_name}")
                sampled_name = f"{dataset_cfg.dataset_name}_Sampled"
                target_dir = os.path.join(config.DATA_DIR, f"HNE_{sampled_name}")
                
                # Check if sample already exists to avoid re-running
                expected_file = os.path.join(target_dir, "link.dat")
                if os.path.exists(expected_file):
                    print(f"      ✅ Found existing sample at {target_dir}")
                    print("      ⏭️  Skipping sampling step (using cached data).")
                else:
                    print(f"      No existing sample found. Running sampler...")
                    print(f"      Configuration: Seeds={args.sample_seeds}, Hops={args.sample_hops}")
                    
                    sampler = HNESnowballDiskSampler()
                    sampler_config = {
                        'seeds': args.sample_seeds, 
                        'hops': args.sample_hops
                    }
                    sampler.sample(source_dir, target_dir, sampler_config)
                
                # We register the sampled dataset so the Loader knows how to read it.
                sampled_key = f"{args.dataset}_SAMPLED"
                new_config = DatasetConfig(
                    source='HNE',
                    dataset_name=sampled_name, 
                    target_node=dataset_cfg.target_node
                )
                config.register_dataset(sampled_key, new_config)
                
                # Update target to point to the new sampled dataset for LOADING
                args.dataset = sampled_key
                dataset_cfg = new_config
                
                
        else:
            print("[1/4] Loading full dataset (No sampling)...")

        # ---------------------------------------------------------
        # Step 3: Load Data
        # ---------------------------------------------------------
        print(f"\n      Loading into memory: {dataset_cfg}...")
        g_hetero, _ = DatasetFactory.get_data(
            dataset_cfg.source,
            dataset_cfg.dataset_name,
            dataset_cfg.target_node
        )
        
        # ---------------------------------------------------------
        # Step 4: Export & Mine
        # ---------------------------------------------------------
        print(f"\n[2/4] Exporting to AnyBURL format...")
        print(f"      Target Directory: {main_working_dir}")
        
        # Initialize Runner pointing to the MAIN WORKING DIR
        runner = AnyBURLRunner(main_working_dir, config.ANYBURL_JAR)
        
        # Export Graph (Runner handles skipping if exists)
        runner.export_graph(g_hetero)
        
        print(f"\n[3/4] Mining rules (timeout={args.timeout}s)...")
        # Run Mining (Runner handles skipping if exists)
        runner.run_mining(
            timeout=args.timeout,
            max_length=args.max_length,
            num_threads=args.threads
        )
        
        # ---------------------------------------------------------
        # Step 5: Parse Results & Generate Lists
        # ---------------------------------------------------------
        print("\n[4/4] Parsing results...")
        
        # 1. Generate clean list for benchmarking (metapaths_clean.txt)
        clean_file = runner.save_clean_list()
        
        # 2. Find best path for immediate feedback
        best_path = runner.parse_best_metapath(
            dataset_cfg.target_node,
            dataset_cfg.target_node,
            min_confidence=args.min_confidence
        )
        
        if os.path.exists(clean_file):
            print(f"\n✅ Clean metapath list generated: {clean_file}")
            print(f"   You can now run batch benchmarks using:")
            # Generate the helpful command string
            clean_name = dataset_cfg.dataset_name.replace('_Sampled', '').replace('_SAMPLED', '')
            print(f"   python run_batch.py {dataset_cfg.source}_{clean_name}")
        
        if best_path:
            print(f"\n✅ Single Best Path found (Conf={args.min_confidence}):")
            print(f"   {' -> '.join(best_path)}")
        else:
            print("\n❌ No valid metapaths found")


class ListCommand:
    """Command for listing available datasets."""
    
    def execute(self, args) -> None:
        """Display available datasets and backends."""
        print("\n" + "="*60)
        print("AVAILABLE DATASETS")
        print("="*60 + "\n")
        
        datasets = config.list_datasets()
        for key in sorted(datasets):
            cfg = config.get_dataset_config(key)
            print(f"  {key:20s} | Source: {cfg.source:5s} | Target: {cfg.target_node}")
        
        print(f"\nTotal: {len(datasets)} datasets")
        
        # List backends
        print("\n" + "="*60)
        print("AVAILABLE BACKENDS")
        print("="*60 + "\n")
        
        backends = BackendFactory.list_backends()
        for backend in backends:
            print(f"  • {backend}")
        
        print(f"\nTotal: {len(backends)} backends")


def create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        description="GNN Scalability Benchmarking & Rule Mining Engine (SOLID Edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark exact method (Python backend)
  python main.py benchmark --dataset HGB_DBLP --method exact --backend python
  
  # Benchmark with MANUAL METAPATH
  python main.py benchmark --dataset HNE_DBLP --manual-path "author_to_paper,paper_to_author"
  
  # Mine metapaths with Disk-Based Sampling (Prevents OOM)
  python main.py mine --dataset HNE_DBLP --sample-method snowball --sample-seeds 1000
  
  # List all datasets and backends
  python main.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operational mode')
    
    # Benchmark subcommand
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    bench_parser.add_argument('--dataset', type=str, required=True,
                            help='Dataset key (e.g., HGB_DBLP)')
    bench_parser.add_argument('--method', type=str, choices=['exact', 'kmv'], 
                            default='exact', help='Materialization method')
    bench_parser.add_argument('--backend', type=str, choices=['python', 'cpp'],
                            default='python', help='Execution backend (EASY TO TOGGLE!)')
    bench_parser.add_argument('--k', type=int, default=32,
                            help='Sketch size for KMV method')
    bench_parser.add_argument('--metapath-length', type=int, default=config.METAPATH_LENGTH,
                            help='Length of random metapath')
    bench_parser.add_argument('--manual-path', type=str, 
                            help='Manually specify metapath (e.g. "rel1,rel2")')
    bench_parser.add_argument('--run-inference', action='store_true',
                            help='Run GNN inference test')
    
    # Mining subcommand
    mine_parser = subparsers.add_parser('mine', help='Mine metapaths using AnyBURL')
    mine_parser.add_argument('--dataset', type=str, required=True,
                           help='Dataset key')
    mine_parser.add_argument('--timeout', type=int, default=60,
                           help='Mining duration (seconds)')
    mine_parser.add_argument('--max-length', type=int, default=4,
                           help='Maximum rule length')
    mine_parser.add_argument('--threads', type=int, default=4,
                           help='Number of worker threads')
    mine_parser.add_argument('--min-confidence', type=float, default=0.001,
                           help='Minimum confidence threshold')
    mine_parser.add_argument('--export', action='store_true',
                           help='Export mined metapath to file')
    
    # Sampling arguments
    mine_parser.add_argument('--sample-method', type=str, choices=['snowball'], default=None, 
                           help='Sampling strategy (use "snowball" to prevent OOM)')
    mine_parser.add_argument('--sample-seeds', type=int, default=1000, 
                           help='Number of seed nodes for sampling')
    mine_parser.add_argument('--sample-hops', type=int, default=2, 
                           help='Number of hops for sampling')
    
    # List subcommand
    list_parser = subparsers.add_parser('list', help='List available datasets and backends')
    
    return parser


def main():
    """Main entry point with command routing."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate configuration
    if not config.validate():
        print("\n[Warning] Some configuration issues detected. Proceeding anyway...")
    
    # Route to appropriate command (Command Pattern)
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
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()