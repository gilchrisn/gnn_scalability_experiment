"""
PyTorch Lightning Callbacks for Dynamic Graph Injection.
Follows the Strategy Pattern for graph updates during training.
"""
import time
import torch
import pytorch_lightning as pl
from typing import List, Tuple, Dict, Any, Optional
from torch_geometric.data import Data, HeteroData
from src.kernels import KMVSketchingKernel, RandomSamplingKernel

class GraphCyclingCallback(pl.Callback):
    """
    Implements the 'Pre-computed Dynamic' strategy.
    
    Cycles through a list of pre-materialized graphs (G_0, G_1, ... G_L)
    at the start of each epoch. This avoids I/O overhead during training.
    
    Principle: Single Responsibility (Only handles graph injection).
    """
    
    def __init__(self, graphs: List[Data], verbose: bool = False):
        """
        Args:
            graphs: List of homogeneous Data objects (A_0, ... A_L).
            verbose: If True, logs graph swapping.
        """
        super().__init__()
        self.graphs = graphs
        self.num_graphs = len(graphs)
        self.verbose = verbose
        
        if self.num_graphs == 0:
            raise ValueError("[GraphCyclingCallback] Received empty graph list.")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Injects the graph G_{epoch % L} into the model.
        """
        # 1. Select Graph
        epoch = trainer.current_epoch
        graph_idx = epoch % self.num_graphs
        g_next = self.graphs[graph_idx]
        
        # 2. Move to Device (Critical for GPU training)
        # We assume pl_module.device is the correct target.
        g_next = g_next.to(pl_module.device)
        
        # 3. Inject (Bridge Pattern Interface)
        if hasattr(pl_module, 'update_graph'):
            pl_module.update_graph(g_next)
            if self.verbose:
                print(f"[Callback] Epoch {epoch}: Swapped to graph sample {graph_idx}")
        else:
            raise RuntimeError("Model does not support dynamic graph updates. Implement 'update_graph()'.")


class DynamicGraphCallback(pl.Callback):
    """
    Implements the 'On-the-Fly Dynamic' strategy.
    
    Regenerates the graph structure A_t ~ P(A|G) using a Python kernel 
    at the start of every epoch. Best for infinite variance but higher CPU cost.
    """

    def __init__(self, 
                 g_hetero: HeteroData,
                 metapath: List[Tuple[str, str, str]],
                 target_ntype: str,
                 method: str = 'kmv',
                 k: int = 32,
                 data_info: Optional[Dict[str, Any]] = None,
                 device: Optional[torch.device] = None):
        
        super().__init__()
        self.g_hetero = g_hetero
        self.metapath = metapath
        self.target_ntype = target_ntype
        self.data_info = data_info or {}
        self.device = device or torch.device('cpu')
        
        # Strategy Factory for Kernels
        if method == 'kmv':
            self.kernel = KMVSketchingKernel(k=k, device=self.device)
        elif method == 'random':
            self.kernel = RandomSamplingKernel(k=k, device=self.device)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        start = time.perf_counter()
        
        # 1. Sampling (Stochastic Materialization)
        with torch.no_grad():
            g_sampled, _, _ = self.kernel.sketch_and_sample(
                self.g_hetero,
                self.metapath,
                self.target_ntype,
                features=self.data_info.get('features'),
                labels=self.data_info.get('labels'),
                masks=self.data_info.get('masks')
            )
        
        # 2. Injection
        if hasattr(pl_module, 'update_graph'):
            g_sampled = g_sampled.to(pl_module.device)
            pl_module.update_graph(g_sampled)
            
        duration = time.perf_counter() - start
        
        # 3. Logging (if available)
        if trainer.logger is not None:
            trainer.logger.log_metrics({'graph_gen_time': duration}, step=trainer.global_step)