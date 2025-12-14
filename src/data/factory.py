"""
Dataset Factory implementing the Factory Pattern.
Provides a unified interface for loading different dataset sources.
"""
from typing import Tuple, Dict, Any, Type
import torch_geometric.data as tg_data

from .base import BaseGraphLoader
from .loaders import HGBLoader, OGBLoader, PyGStandardLoader, HNELoader


class DatasetFactory:
    """
    Factory class for creating appropriate data loaders.
    Implements the Factory Pattern for extensibility.
    """
    
    # Registry of available loaders
    _LOADER_REGISTRY: Dict[str, Type[BaseGraphLoader]] = {
        'HGB': HGBLoader,
        'OGB': OGBLoader,
        'PyG': PyGStandardLoader,
        'HNE': HNELoader,
    }
    
    @classmethod
    def register_loader(cls, source_type: str, loader_class: Type[BaseGraphLoader]) -> None:
        """
        Register a new loader type.
        Allows for runtime extension of supported datasets.
        
        Args:
            source_type: Identifier for the loader (e.g., 'CustomDB')
            loader_class: Class implementing BaseGraphLoader
        """
        if not issubclass(loader_class, BaseGraphLoader):
            raise TypeError(f"Loader must inherit from BaseGraphLoader, got {loader_class}")
        
        cls._LOADER_REGISTRY[source_type] = loader_class
        print(f"[Factory] Registered new loader: {source_type}")
    
    @classmethod
    def get_supported_sources(cls) -> list:
        """Returns list of supported data sources."""
        return list(cls._LOADER_REGISTRY.keys())
    
    @classmethod
    def get_data(cls, 
                 source_type: str, 
                 dataset_name: str, 
                 target_ntype: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        """
        Factory method to load datasets.
        
        Args:
            source_type: Data source identifier (e.g., 'HGB', 'OGB', 'PyG', 'HNE')
            dataset_name: Specific dataset name (e.g., 'DBLP', 'IMDB')
            target_ntype: Target node type for prediction task
            
        Returns:
            Tuple of (heterogeneous graph, metadata dictionary)
            
        Raises:
            ValueError: If source_type is not registered
            
        Example:
            >>> g, info = DatasetFactory.get_data('HGB', 'DBLP', 'author')
        """
        if source_type not in cls._LOADER_REGISTRY:
            supported = ', '.join(cls.get_supported_sources())
            raise ValueError(
                f"Unknown data source: '{source_type}'. "
                f"Supported sources: {supported}"
            )
        
        loader_class = cls._LOADER_REGISTRY[source_type]
        loader = loader_class()
        
        try:
            return loader.load(dataset_name, target_ntype)
        except Exception as e:
            print(f"[Factory] Error loading {source_type}/{dataset_name}: {e}")
            raise