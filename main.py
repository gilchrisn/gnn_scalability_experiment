"""
Unified entry point for GNN scalability benchmarking and rule mining.
Implements the Command pattern for different operational modes.
"""
import argparse
import sys
import os
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import config
from src.data import DatasetFactory
from src.kernels import ExactMaterializationKernel, KMVSketchingKernel
from src.bridge import AnyBURLRunner, PyGToCppAdapter, CppBridge
from src.utils import generate_random_metapath


class CommandExecutor:
    """
    Executor class implementing the Command pattern.
    Encapsulates different operational modes.
    """
    
    @staticmethod
    def run_benchmark(args) -> None:
        """Executes benchmarking pipeline."""
        print(f"\n{'='*60}")
        print(f"BENCHMARK MODE: {args.dataset}")
        print(f"Method: {args.method.upper()} | Backend: {args.backend.upper()}")
        print(f"{'='*60}\n")
        
        # Load dataset
        dataset_cfg = config.get_dataset_config(args.dataset)
        print(f"[1/4] Loading {dataset_cfg}...")
        
        g_hetero, info = DatasetFactory.get_data(
            dataset_cfg.source,
            dataset_cfg.dataset_name,
            dataset_cfg.target_node
        )
        
        # Generate metapath
        print(f"\n[2/4] Generating metapath (length={args.metapath_length})...")
        metapath = generate_random_metapath(
            g_hetero,
            dataset_cfg.target_node,
            args.metapath_length
        )
        path_str = " -> ".join([rel for _, rel, _ in metapath])
        print(f"   Path: {path_str}")
        
        # Move to device
        g_hetero = g_hetero.to(config.DEVICE)
        
        # Execute selected method
        if args.backend == 'python':
            CommandExecutor._benchmark_python(args, g_hetero, metapath, dataset_cfg, info)
        elif args.backend == 'cpp':
            CommandExecutor._benchmark_cpp(args, g_hetero, metapath, dataset_cfg, info)
        else:
            raise ValueError(f"Unknown backend: {args.backend}")
    
    @staticmethod
    def _benchmark_python(args, g_hetero, metapath, dataset_cfg, info) -> None:
        """Python backend benchmarking."""
        print(f"\n[3/4] Running {args.method.upper()} (Python)...")
        
        if args.method == 'exact':
            kernel = ExactMaterializationKernel(device=config.DEVICE)
            g_result, time_taken = kernel.materialize(
                g_hetero, metapath, dataset_cfg.target_node,
                features=info['features'],
                labels=info['labels'],
                masks=info['masks']
            )
            
            print(f"\n[4/4] Results:")
            print(f"   Edges: {g_result.num_edges:,}")
            print(f"   Time: {time_taken:.4f}s")
            
        elif args.method == 'kmv':
            kernel = KMVSketchingKernel(k=args.k, device=config.DEVICE)
            g_result, t_prop, t_build = kernel.sketch_and_sample(
                g_hetero, metapath, dataset_cfg.target_node,
                features=info['features'],
                labels=info['labels'],
                masks=info['masks']
            )
            
            print(f"\n[4/4] Results:")
            print(f"   Edges: {g_result.num_edges:,}")
            print(f"   Propagation: {t_prop:.4f}s")
            print(f"   Building: {t_build:.4f}s")
            print(f"   Total: {t_prop + t_build:.4f}s")
    
    @staticmethod
    def _benchmark_cpp(args, g_hetero, metapath, dataset_cfg, info) -> None:
        """C++ backend benchmarking."""
        print(f"\n[3/4] Converting to C++ format...")
        
        # Setup adapter
        adapter = PyGToCppAdapter(config.TEMP_DIR)
        adapter.convert(g_hetero)
        
        # Generate rule
        rule_str = adapter.generate_cpp_rule(metapath)
        rule_path = adapter.write_rule_file(rule_str)
        
        # Setup bridge
        bridge = CppBridge(config.CPP_EXECUTABLE, config.TEMP_DIR)
        target_offset = adapter.type_offsets[dataset_cfg.target_node]
        num_nodes = info['features'].shape[0]
        
        print(f"\n[4/4] Running {args.method.upper()} (C++)...")
        
        if args.method == 'exact':
            output_file = os.path.join(config.TEMP_DIR, "result.txt")
            t_exec = bridge.run_command("materialize", rule_path, output_file)
            g_result = bridge.load_result_graph(output_file, num_nodes, target_offset)
            
            print(f"\n   Results:")
            print(f"   Edges: {g_result.num_edges:,}")
            print(f"   Time: {t_exec:.4f}s")
            
        elif args.method == 'kmv':
            output_file = os.path.join(config.TEMP_DIR, f"sketch_k{args.k}.txt")
            t_exec = bridge.run_command("sketch", rule_path, output_file, k=args.k)
            g_result = bridge.load_result_graph(output_file, num_nodes, target_offset)
            
            print(f"\n   Results:")
            print(f"   Edges: {g_result.num_edges:,}")
            print(f"   Time: {t_exec:.4f}s")
    
    @staticmethod
    def run_mining(args) -> None:
        """Executes rule mining pipeline."""
        print(f"\n{'='*60}")
        print(f"RULE MINING MODE: {args.dataset}")
        print(f"{'='*60}\n")
        
        # Load dataset
        dataset_cfg = config.get_dataset_config(args.dataset)
        print(f"[1/3] Loading {dataset_cfg}...")
        
        g_hetero, _ = DatasetFactory.get_data(
            dataset_cfg.source,
            dataset_cfg.dataset_name,
            dataset_cfg.target_node
        )
        
        # Setup AnyBURL
        print(f"\n[2/3] Exporting to AnyBURL format...")
        data_dir = os.path.join(config.DATA_DIR, str(dataset_cfg))
        runner = AnyBURLRunner(data_dir, config.ANYBURL_JAR)
        runner.export_graph(g_hetero)
        
        # Run mining
        print(f"\n[3/3] Mining rules (timeout={args.timeout}s)...")
        runner.run_mining(
            timeout=args.timeout,
            max_length=args.max_length,
            num_threads=args.threads
        )
        
        # Parse results
        print("\nParsing best metapath...")
        best_path = runner.parse_best_metapath(
            dataset_cfg.target_node,
            dataset_cfg.target_node,
            min_confidence=args.min_confidence
        )
        
        if best_path:
            print(f"\nRecommended metapath for {args.dataset}:")
            print(f"   {' -> '.join(best_path)}")
            
            # Export to file
            if args.export:
                runner.export_to_config_format(best_path)
    
    @staticmethod
    def list_datasets(args) -> None:
        """Lists available datasets."""
        print("\nAvailable Datasets:")
        print("-" * 60)
        
        datasets = config.list_datasets()
        for key in sorted(datasets):
            cfg = config.get_dataset_config(key)
            print(f"  {key:20s} | Source: {cfg.source:5s} | Target: {cfg.target_node}")
        
        print(f"\nTotal: {len(datasets)} datasets")


def create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        description="GNN Scalability Benchmarking & Rule Mining Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark exact method (Python backend)
  python main.py benchmark --dataset HGB_DBLP --method exact
  
  # Benchmark KMV with k=32 (C++ backend)
  python main.py benchmark --dataset HGB_DBLP --method kmv --k 32 --backend cpp
  
  # Mine metapaths
  python main.py mine --dataset HNE_DBLP --timeout 120
  
  # List all datasets
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
                            default='python', help='Execution backend')
    bench_parser.add_argument('--k', type=int, default=32,
                            help='Sketch size for KMV method')
    bench_parser.add_argument('--metapath-length', type=int, default=config.METAPATH_LENGTH,
                            help='Length of random metapath')
    
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
    mine_parser.add_argument('--min-confidence', type=float, default=0.5,
                           help='Minimum confidence threshold')
    mine_parser.add_argument('--export', action='store_true',
                           help='Export mined metapath to file')
    
    # List subcommand
    list_parser = subparsers.add_parser('list', help='List available datasets')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate configuration
    if not config.validate():
        print("\n[Warning] Some configuration issues detected. Proceeding anyway...")
    
    # Execute command
    executor = CommandExecutor()
    
    if args.mode == 'benchmark':
        executor.run_benchmark(args)
    elif args.mode == 'mine':
        executor.run_mining(args)
    elif args.mode == 'list':
        executor.list_datasets(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()