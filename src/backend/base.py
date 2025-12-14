"""
Abstract backend interface for graph materialization.
Implements Strategy Pattern for interchangeable backends.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
import torch
from torch_geometric.data import Data, HeteroData


@dataclass
class BenchmarkResult:
    """
    Standardized result object for benchmarking.
    Ensures all backends return consistent data.
    """
    method: str  # 'exact' or 'kmv'
    backend: str  # 'python' or 'cpp'
    num_edges: int
    num_nodes: int
    prep_time: float
    infer_time: float = 0.0
    
    @property
    def total_time(self) -> float:
        """Total execution time."""
        return self.prep_time + self.infer_time
    
    def __str__(self) -> str:
        return (
            f"{self.backend.upper()} Backend | {self.method.upper()}\n"
            f"  Nodes: {self.num_nodes:,} | Edges: {self.num_edges:,}\n"
            f"  Prep: {self.prep_time:.4f}s | Infer: {self.infer_time:.4f}s | "
            f"Total: {self.total_time:.4f}s"
        )


class GraphBackend(ABC):
    """
    Abstract base class for graph materialization backends.
    Defines the contract that all backends must implement.
    
    This follows the Strategy Pattern, allowing runtime backend selection.
    """
    
    @abstractmethod
    def initialize(self, 
                   g_hetero: HeteroData, 
                   metapath: List[Tuple[str, str, str]],
                   info: Dict[str, Any]) -> None:
        """
        Initialize backend with graph data.
        
        Args:
            g_hetero: Heterogeneous input graph
            metapath: Metapath definition
            info: Metadata dictionary (features, labels, masks)
        """
        pass
    
    @abstractmethod
    def materialize_exact(self) -> Data:
        """
        Perform exact materialization.
        
        Returns:
            Materialized homogeneous graph
        """
        pass
    
    @abstractmethod
    def materialize_kmv(self, k: int) -> Data:
        """
        Perform KMV-based approximate materialization.
        
        Args:
            k: Sketch size
            
        Returns:
            Sampled homogeneous graph
        """
        pass
    
    @abstractmethod
    def get_prep_time(self) -> float:
        """
        Get preprocessing time for last operation.
        
        Returns:
            Time in seconds
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up backend resources.
        Called after benchmark completion.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'python', 'cpp')."""
        pass
    
    @property
    @abstractmethod
    def supports_inference(self) -> bool:
        """Whether this backend supports GNN inference."""
        pass


class BackendFactory:
    """
    Factory for creating backend instances.
    Implements Factory Pattern with runtime registration.
    """
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, backend_class: type) -> None:
        """
        Register a new backend type.
        
        Args:
            name: Backend identifier
            backend_class: Class implementing GraphBackend
            
        Example:
            >>> BackendFactory.register('gpu', GPUBackend)
        """
        if not issubclass(backend_class, GraphBackend):
            raise TypeError(f"{backend_class} must inherit from GraphBackend")
        
        cls._registry[name.lower()] = backend_class
        print(f"[BackendFactory] Registered backend: {name}")
    
    @classmethod
    def create(cls, name: str, **kwargs) -> GraphBackend:
        """
        Create a backend instance.
        
        Args:
            name: Backend identifier
            **kwargs: Backend-specific initialization arguments
            
        Returns:
            Initialized backend instance
            
        Raises:
            ValueError: If backend not found
        """
        name = name.lower()
        if name not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(
                f"Unknown backend: '{name}'. "
                f"Available: {available}"
            )
        
        backend_class = cls._registry[name]
        return backend_class(**kwargs)
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """Get list of registered backends."""
        return list(cls._registry.keys())