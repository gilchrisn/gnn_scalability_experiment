"""
Python-based backend implementation.
Uses PyTorch operations for graph materialization.
"""
from typing import Dict, Any, List, Tuple
import torch
from torch_geometric.data import Data, HeteroData

from .base import GraphBackend
from ..kernels import ExactMaterializationKernel, KMVSketchingKernel, RandomSamplingKernel


class PythonBackend(GraphBackend):
    """
    Pure Python/PyTorch backend.
    Implements graph operations using in-memory tensor operations.
    """
    
    def __init__(self, device: torch.device = None, **kwargs):
        """
        Args:
            device: Computation device (CPU or CUDA)
        """
        self.device = device or torch.device('cpu')
        self._g_hetero = None
        self._metapath = None
        self._info = None
        self._target_ntype = None
        self._last_prep_time = 0.0
        
        # Initialize kernels
        self._exact_kernel = ExactMaterializationKernel(device=self.device)
        self._kmv_kernel = None  # Created on-demand with specific k
    
    @property
    def name(self) -> str:
        return "python"
    
    @property
    def supports_inference(self) -> bool:
        return True
    
    def initialize(self,
                   g_hetero: HeteroData,
                   metapath: List[Tuple[str, str, str]],
                   info: Dict[str, Any]) -> None:
        """
        Store graph data for processing.
        
        Args:
            g_hetero: Input graph
            metapath: Path definition
            info: Metadata
        """
        self._g_hetero = g_hetero.to(self.device)
        self._metapath = metapath
        self._info = info
        self._target_ntype = metapath[-1][2]  # Last destination type
        
        print(f"[PythonBackend] Initialized on {self.device}")
    
    def materialize_exact(self) -> Data:
        """
        Exact materialization using sparse matrix multiplication.
        
        Returns:
            Materialized graph
        """
        if self._g_hetero is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        
        print(f"[PythonBackend] Running exact materialization...")
        
        g_result, prep_time = self._exact_kernel.materialize(
            self._g_hetero,
            self._metapath,
            self._target_ntype,
            features=self._info['features'],
            labels=self._info['labels'],
            masks=self._info['masks']
        )
        
        self._last_prep_time = prep_time
        return g_result
    
    def materialize_kmv(self, k: int) -> Data:
        """
        KMV-based approximate materialization.
        
        Args:
            k: Sketch size
            
        Returns:
            Sampled graph
        """
        if self._g_hetero is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        
        print(f"[PythonBackend] Running KMV (k={k})...")
        
        # Create kernel with specific k value
        kmv_kernel = KMVSketchingKernel(k=k, nk=1, device=self.device)
        
        g_result, t_prop, t_build = kmv_kernel.sketch_and_sample(
            self._g_hetero,
            self._metapath,
            self._target_ntype,
            features=self._info['features'],
            labels=self._info['labels'],
            masks=self._info['masks']
        )
        
        self._last_prep_time = t_prop + t_build
        return g_result
    
    def materialize_random(self, k: int) -> Data:
        """
        Random sampling-based approximate materialization.
        
        Args:
            k: Number of neighbors to sample

        Returns:
            Sampled graph
        """ 

        if self._g_hetero is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        
        print(f"[PythonBackend] Running Random Sampling (k={k})...")
        
        random_kernel = RandomSamplingKernel(k=k, device=self.device)
        
        g_result, t_prep, _ = random_kernel.sketch_and_sample(
            self._g_hetero,
            self._metapath,
            self._target_ntype,
            features=self._info['features'],
            labels=self._info['labels'],
            masks=self._info['masks']
        )
        
        self._last_prep_time = t_prep
        return g_result
    
    def get_prep_time(self) -> float:
        """Return last recorded preprocessing time."""
        return self._last_prep_time
    
    def cleanup(self) -> None:
        """Clean up GPU memory if using CUDA."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        self._g_hetero = None
        self._metapath = None
        self._info = None