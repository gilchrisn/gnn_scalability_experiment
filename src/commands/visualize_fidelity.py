import torch
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import Namespace

from .base import BaseCommand
from ..config import config
from ..data import DatasetFactory
from ..backend import BackendFactory
from ..models import get_model
from ..utils import SchemaMatcher
from ..analysis.cka import LinearCKA

class VisualizeFidelityCommand(BaseCommand):
    """
    Executes Experiment D: Semantic Preservation.
    
    1. Loads 'Exact' Model and 'KMV' Model.
    2. Extracts node embeddings Z_exact and Z_kmv for the Test Set.
    3. Computes Linear CKA score.
    4. Generates a heatmap comparison.
    """

    def execute(self, args: Namespace) -> None:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT D: SEMANTIC FIDELITY (CKA)")
        print(f"Comparing: Exact vs. KMV (k={args.k})")
        print(f"{'='*60}\n")

        # 1. Setup Data & Backend
        cfg = config.get_dataset_config(args.dataset)
        g_hetero, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in args.metapath.split(',')]
        
        backend = BackendFactory.create('python', device=config.DEVICE)
        backend.initialize(g_hetero, path_list, info)

        # 2. Extract Z_Exact (Control)
        print("[1/2] Extracting EXACT embeddings...")
        g_exact = backend.materialize_exact()
        # FIX: Pass args.dataset explicitly
        z_exact = self._get_embeddings(g_exact, info, args.model, 'exact', args.dataset)
        
        # 3. Extract Z_KMV (Experimental)
        print(f"[2/2] Extracting KMV (k={args.k}) embeddings...")
        g_kmv = backend.materialize_kmv(k=args.k)
        # FIX: Pass args.dataset explicitly
        z_kmv = self._get_embeddings(g_kmv, info, args.model, f'kmv_k{args.k}', args.dataset) 

        # 4. Filter to Test Set
        mask = info['masks']['test'].cpu()
        z_exact_test = z_exact[mask]
        z_kmv_test = z_kmv[mask]
        
        # 5. Compute CKA
        print("\nComputing Centered Kernel Alignment (CKA)...")
        cka = LinearCKA(device=config.DEVICE)
        score = cka.calculate(z_exact_test.to(config.DEVICE), z_kmv_test.to(config.DEVICE))
        
        print(f"\n[RESULT] Semantic Similarity (CKA): {score:.4f}")
        print("Interpretation: >0.9 is excellent, >0.8 is acceptable for approximation.")

        # 6. Generate Plot
        self._plot_correlation_comparison(z_exact_test, z_kmv_test, score, args.dataset)

    def _get_embeddings(self, graph, info, model_arch, mode_suffix, dataset_name):
        """
        Loads model and runs inference with STRICT safety checks.
        """
        in_dim = graph.x.shape[1]
        num_nodes = graph.x.shape[0] # The limit of valid IDs
        
        # --- SAFETY FIX: Sanitize Graph ---
        # Ensure we don't have edges pointing to nodes >= num_nodes
        if graph.edge_index.numel() > 0:
            max_idx = graph.edge_index.max().item()
            if max_idx >= num_nodes:
                print(f" [WARNING] Found invalid edges! Max ID {max_idx} >= Num Nodes {num_nodes}")
                print(f"           Pruning invalid edges to prevent CUDA crash...")
                
                # Filter edges where both source and dest are valid
                mask = (graph.edge_index[0] < num_nodes) & (graph.edge_index[1] < num_nodes)
                graph.edge_index = graph.edge_index[:, mask]
        # ----------------------------------

        model = get_model(model_arch, in_dim, info['num_classes'], config.HIDDEN_DIM)
        
        # FIX: Use the dataset_name passed explicitly
        foundation_path = config.get_model_path(dataset_name, model_arch)
        
        if os.path.exists(foundation_path):
            try:
                print(f" [Info] Loading model weights from: {foundation_path}")
                model.load_state_dict(torch.load(foundation_path, map_location=config.DEVICE))
            except Exception as e:
                print(f" [Warn] Could not load weights ({e}), using random init for visualization.")
        else:
            print(f" [Warn] Model path not found: {foundation_path}. Using random init.")
        
        model.to(config.DEVICE).eval()
        
        with torch.no_grad():
            # Move data to GPU
            x = graph.x.to(config.DEVICE)
            edge_index = graph.edge_index.to(config.DEVICE)
            
            # Run Model
            out = model(x, edge_index)
            
        return out.cpu()

    def _plot_correlation_comparison(self, z_a, z_b, cka_score, dataset):
        """Plots feature correlation matrices side-by-side."""
        # Take a subset of dimensions for readability
        dim = min(z_a.shape[1], 32)
        z_a_sub = z_a[:, :dim]
        z_b_sub = z_b[:, :dim]
        
        corr_a = torch.corrcoef(z_a_sub.T).numpy()
        corr_b = torch.corrcoef(z_b_sub.T).numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(corr_a, ax=axes[0], cmap='viridis', vmin=-1, vmax=1)
        axes[0].set_title("Exact Latent Correlations")
        
        sns.heatmap(corr_b, ax=axes[1], cmap='viridis', vmin=-1, vmax=1)
        axes[1].set_title(f"KMV Latent Correlations")
        
        plt.suptitle(f"Semantic Topology Preservation (CKA = {cka_score:.3f})")
        
        out_path = os.path.join(config.RESULTS_DIR, f"fidelity_plot_{dataset}.png")
        plt.savefig(out_path)
        print(f"Plot saved to: {out_path}")

    @staticmethod
    def register_subparser(subparsers):
        parser = subparsers.add_parser('visualize_fidelity', help='Run CKA Semantic Analysis')
        parser.add_argument('--dataset', required=True)
        parser.add_argument('--metapath', required=True)
        parser.add_argument('--model', default='GCN')
        parser.add_argument('--k', type=int, default=32)