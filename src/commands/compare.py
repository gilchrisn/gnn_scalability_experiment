import pandas as pd
import torch
import os
from .base import BaseCommand
from ..config import config
from ..data import DatasetFactory
from ..backend import BackendFactory
from ..kernels import KMVSketchingKernel, RandomSamplingKernel
from ..models import get_model
from ..utils import SchemaMatcher

class CompareCommand(BaseCommand):
    """
    Runs a head-to-head comparison: Exact vs. KMV vs. Random.
    Saves results to CSV for plotting.
    """

    def execute(self, args) -> None:
        print(f"--- COMPARISON MODE: {args.dataset} ---")
        
        # 1. Setup Data
        cfg = config.get_dataset_config(args.dataset)
        g_hetero, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        
        # 2. Setup Model (Pre-trained)
        model_path = config.get_model_path(args.dataset, args.model)
        if not os.path.exists(model_path):
            raise FileNotFoundError("Please run 'train_foundation' first to generate a model.")
        
        # Load Mapper logic (omitted for brevity, assume similar to your existing code)
        import json
        with open(model_path.replace('.pt', '_mapper.json'), 'r') as f:
            mapper_cfg = json.load(f)
            
        model = get_model(args.model, mapper_cfg['global_max_dim'], info['num_classes'], config.HIDDEN_DIM)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # 3. Define Kernels
        kernels = {
            'KMV': KMVSketchingKernel(k=args.k, device=config.DEVICE),
            'Random': RandomSamplingKernel(k=args.k, device=config.DEVICE)
        }

        results = []
        path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in args.metapath.split(',')]

        # 4. Run Comparisons
        for name, kernel in kernels.items():
            print(f"Running {name} Sampling (k={args.k})...")
            
            # Reuse backend for initialization/materialization logic
            # (In a cleaner refactor, kernels might be injected into Backends)
            if name == 'KMV':
                 g_out, _, _ = kernel.sketch_and_sample(g_hetero, path_list, cfg.target_node, 
                                                      features=info['features'], labels=info['labels'], masks=info['masks'])
            else:
                 g_out, _, _ = kernel.sketch_and_sample(g_hetero, path_list, cfg.target_node,
                                                      features=info['features'], labels=info['labels'], masks=info['masks'])

            # Inference
            with torch.no_grad():
                # (Assuming padding logic exists in utils)
                from ..utils import pad_features 
                x_padded = pad_features(g_out.x, mapper_cfg['global_max_dim'])
                out = model(x_padded, g_out.edge_index)
                
                mask = info['masks']['test']
                pred = out[mask].argmax(dim=1)
                acc = (pred == info['labels'][mask]).sum().item() / mask.sum().item()
                
            print(f"  -> Accuracy: {acc:.4f}")
            results.append({'Method': name, 'K': args.k, 'Accuracy': acc})

        # Save
        df = pd.DataFrame(results)
        df.to_csv(f"output/results/comparison_{args.dataset}_k{args.k}.csv", index=False)
        print("Comparison complete.")

    @staticmethod
    def register_subparser(subparsers):
        parser = subparsers.add_parser('compare', help='Run KMV vs Random baseline')
        parser.add_argument('--dataset', required=True)
        parser.add_argument('--metapath', required=True)
        parser.add_argument('--model', default='SAGE')
        parser.add_argument('--k', type=int, default=32)