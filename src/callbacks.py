import time
import torch
import pytorch_lightning as pl
from typing import Optional, List, Tuple
from torch_geometric.data import Data, HeteroData
from src.kernels import KMVSketchingKernel, RandomSamplingKernel

class DynamicGraphCallback(pl.Callback):
    r"""
    Implements Stochastic Structure Generation (The "Dynamic Defense").
    
    Mathematically:
        At epoch t:
        1. Sample random state \xi_t
        2. A_t = Kernel(G, \mathcal{P}, k, \xi_t)
        3. Model.update_graph(A_t)
    
    This turns the Adjacency Matrix from a constant C into a random variable A ~ P(A|G).
    """
    
    def __init__(self, 
                 g_hetero: HeteroData,
                 metapath: List[Tuple[str, str, str]],
                 target_ntype: str,
                 method: str = 'kmv',
                 k: int = 32,
                 features: torch.Tensor = None,
                 labels: torch.Tensor = None,
                 masks: dict = None,
                 device: torch.device = None):
        super().__init__()
        self.g_hetero = g_hetero
        self.metapath = metapath
        self.target_ntype = target_ntype
        self.method = method
        self.k = k
        self.features = features
        self.labels = labels
        self.masks = masks
        self.device = device or torch.device('cpu')
        
        # Initialize Kernel Strategy
        if method == 'kmv':
            self.kernel = KMVSketchingKernel(k=k, device=self.device)
        elif method == 'random':
            self.kernel = RandomSamplingKernel(k=k, device=self.device)
        else:
            raise ValueError(f"Dynamic training not supported for '{method}'")

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Regenerate the adjacency matrix A_t before the forward pass of epoch t.
        """
        start = time.perf_counter()
        
        # 1. Stochastic Materialization
        # Note: We don't need gradients for the structure generation itself
        with torch.no_grad():
            if self.method == 'kmv':
                # KMV is naturally stochastic due to MinHash collisions and random tie-breaking
                g_sampled, _, _ = self.kernel.sketch_and_sample(
                    self.g_hetero, 
                    self.metapath, 
                    self.target_ntype,
                    features=self.features,
                    labels=self.labels,
                    masks=self.masks
                )
            else:
                # Random kernel explicitly samples neighbors
                g_sampled, _, _ = self.kernel.sketch_and_sample(
                    self.g_hetero,
                    self.metapath,
                    self.target_ntype,
                    features=self.features,
                    labels=self.labels,
                    masks=self.masks
                )
        
        # 2. Injection: Hot-swap the graph in the model
        # The model must expose an `update_graph` method (Interface Segregation)
        if hasattr(pl_module, 'update_graph'):
            # Move to correct device strictly
            g_sampled = g_sampled.to(pl_module.device)
            pl_module.update_graph(g_sampled)
            
        duration = time.perf_counter() - start
        
        # Logging for "Phase 1 - Experiment B" (Overhead Analysis)
        # Fix: Check if logger exists (logger=False in Trainer makes this None)
        if trainer.logger is not None:
            trainer.logger.log_metrics({'graph_gen_time': duration}, step=trainer.global_step)