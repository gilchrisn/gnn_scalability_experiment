# ---------- File: ./scripts/run_master_protocol.py ----------
"""
Entry point for running the GNN Scalability Master Protocol V2.1.
Executes Phases 1-4: Mining, Model Zoo Training, Robustness, and Depth Bias Study.
"""
import argparse
import sys
import os
import torch

# Add project root to path so we can import 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import config
from src.experiments import ExperimentConfig, MasterExperimentOrchestrator

def main():
    parser = argparse.ArgumentParser(description="Execute GNN Scalability Master Protocol V2.1")
    
    # Core Arguments
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Target dataset key (e.g., HGB_ACM, HGB_DBLP)')
    parser.add_argument('--model', type=str, default='SAGE', choices=['GCN', 'SAGE', 'GAT'],
                       help='Backbone GNN architecture')
    
    # Phase 1 Controls (NEW)
    parser.add_argument('--min-conf', type=float, default=0.05,
                       help='Minimum AnyBURL confidence score (0.0 to 1.0)')
    parser.add_argument('--top-paths', type=int, default=3,
                       help='Maximum number of metapaths to evaluate per dataset')
    
    parser.add_argument('--mining-strategy', type=str, default='stratified', 
                       choices=['stratified', 'top_k'])
    parser.add_argument('--buckets', type=int, nargs='+', default=[2, 4, 6, 8],
                       help='Target path lengths for stratified curriculum')
    
    # Protocol V2.1 Controls
    parser.add_argument('--depths', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                       help='List of GNN depths to train for Model Zoo (Phase 2 & 4)')
    parser.add_argument('--standard-depth', type=int, default=2,
                       help='The standard depth used for the Robustness Study (Phase 3)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[2, 4, 8, 16, 32],
                       help='Sketch sizes for Robustness Study')
    
    # Execution Controls
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs for training')
    parser.add_argument('--force-retrain', action='store_true', 
                       help='Delete cached model weights and retrain')
    parser.add_argument('--force-remine', action='store_true', 
                       help='Delete cached rules and re-run AnyBURL')
    parser.add_argument('--cpu', action='store_true',
                       help='Force execution on CPU')

    args = parser.parse_args()

    # 0. Hardware Setup
    if args.cpu:
        print("[System] Forcing CPU execution.")
        config.DEVICE = torch.device('cpu')
    else:
        print(f"[System] Using device: {config.DEVICE}")

    print(f"Initializing Protocol V2.1 for {args.dataset}...")

    # 1. Configuration
    cfg = ExperimentConfig(
        dataset=args.dataset,
        model_arch=args.model,
        epochs=args.epochs,
        force_retrain=args.force_retrain,
        force_remine=args.force_remine,
        
        mining_strategy=args.mining_strategy,
        stratified_buckets=args.buckets,
        
        # New Phase 1 Settings
        min_confidence=args.min_conf,
        top_n_paths=args.top_paths,
        
        # V2.1 Specifics
        model_depths=args.depths,
        standard_depth=args.standard_depth,
        k_values=args.k_values
    )

    # 2. Orchestration
    orchestrator = MasterExperimentOrchestrator(cfg)

    # 3. Execution
    orchestrator.run_all()


if __name__ == "__main__": 
    main()