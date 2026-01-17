import os
from argparse import Namespace
from .base import BaseCommand
from ..config import config, DatasetConfig
from ..data import DatasetFactory
from ..sampling import SamplerFactory
from ..bridge import AnyBURLRunner

class MineCommand(BaseCommand):
    """
    Runs AnyBURL rule mining.
    Supports 'Snowball' sampling for large datasets to prevent OOM.
    """

    def execute(self, args: Namespace) -> None:
        print(f"\n{'='*60}")
        print(f"RULE MINING MODE: {args.dataset}")
        print(f"{'='*60}\n")
        
        dataset_cfg = config.get_dataset_config(args.dataset)
        main_working_dir = os.path.join(config.DATA_DIR, f"{dataset_cfg.source}_{dataset_cfg.dataset_name}")
        
        print(f"Working directory: {main_working_dir}")
        
        # --- 1. Disk Sampling (Optional) ---
        if args.sample_method == 'snowball':
            print(f"\nRunning snowball disk sampler...")
            
            if dataset_cfg.source != 'HNE':
                print(" [Warning] Disk sampling currently restricted to 'HNE' format. Skipping.")
            else:
                from ..sampling.disk import HNESnowballDiskSampler # Import here to avoid early loading
                
                source_dir = os.path.join(config.DATA_DIR, f"HNE_{dataset_cfg.dataset_name}")
                sampled_name = f"{dataset_cfg.dataset_name}_Sampled"
                target_dir = os.path.join(config.DATA_DIR, f"HNE_{sampled_name}")
                
                expected_file = os.path.join(target_dir, "link.dat")
                
                if os.path.exists(expected_file) and not args.force_sample:
                    print(f" Using cached sample at {target_dir}")
                else:
                    print(f" Generating new sample (Seeds={args.sample_seeds}, Hops={args.sample_hops})")
                    sampler = HNESnowballDiskSampler()
                    sampler_config = {
                        'seeds': args.sample_seeds, 
                        'hops': args.sample_hops
                    }
                    sampler.sample(source_dir, target_dir, sampler_config)
                
                # Register the temporary sampled dataset
                sampled_key = f"{args.dataset}_SAMPLED"
                new_config = DatasetConfig(
                    source='HNE',
                    dataset_name=sampled_name, 
                    target_node=dataset_cfg.target_node
                )
                config.register_dataset(sampled_key, new_config)
                
                # Switch context to the sampled dataset
                args.dataset = sampled_key
                dataset_cfg = new_config
        
        # --- 2. Load Data ---
        print(f"\n      Loading data: {dataset_cfg}...")
        g_hetero, _ = DatasetFactory.get_data(
            dataset_cfg.source,
            dataset_cfg.dataset_name,
            dataset_cfg.target_node
        )
        
        # --- 3. AnyBURL Execution ---
        print(f"\nExporting for AnyBURL...")
        runner = AnyBURLRunner(main_working_dir, config.ANYBURL_JAR)
        runner.export_graph(g_hetero)
        
        print(f"\nMining rules (timeout={args.timeout}s)...")
        runner.run_mining(
            timeout=args.timeout,
            max_length=args.max_length,
            num_threads=args.threads
        )
        
        # --- 4. Result Parsing ---
        print("\nParsing mined metapaths...")
        clean_file = runner.save_clean_list()
        
        best_path = runner.parse_best_metapath(
            dataset_cfg.target_node,
            dataset_cfg.target_node,
            min_confidence=args.min_confidence
        )
        
        if os.path.exists(clean_file):
            print(f"\n List saved: {clean_file}")
        
        if best_path:
            print(f"\n Best Path (Conf={args.min_confidence}):")
            print(f" {' -> '.join(best_path)}")
        else:
            print("\n No valid metapaths found.")

    @staticmethod
    def register_subparser(subparsers) -> None:
        parser = subparsers.add_parser('mine', help='Mine metapaths using AnyBURL')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--timeout', type=int, default=60)
        parser.add_argument('--max-length', type=int, default=4)
        parser.add_argument('--threads', type=int, default=4)
        parser.add_argument('--min-confidence', type=float, default=0.001)
        parser.add_argument('--sample-method', type=str, choices=['snowball'], default=None)
        parser.add_argument('--sample-seeds', type=int, default=1000)
        parser.add_argument('--sample-hops', type=int, default=2)
        parser.add_argument('--force-sample', action='store_true', help="Force re-sampling")