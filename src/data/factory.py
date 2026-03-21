"""
Unified entry point for graph data loading.
"""
from typing import Tuple, Dict, Any, Type
import torch_geometric.data as tg_data

from .base import BaseGraphLoader
from .loaders import HGBLoader, OGBLoader, PyGStandardLoader, HNELoader, CustomLoader, RCDDLoader, OAGLoader, MiniLoader
from ..config import config


class DatasetFactory:
    """
    Registry-based factory for instantiating and executing graph loaders.
    """
    
    _LOADER_REGISTRY: Dict[str, Type[BaseGraphLoader]] = {
        'HGB':  HGBLoader,
        'OGB':  OGBLoader,
        'OAG':  OAGLoader,
        'MINI': MiniLoader,
        'PyG':  PyGStandardLoader,
        'HNE':  HNELoader,
        'RCDD': RCDDLoader,
        'CUSTOM': CustomLoader,
    }
    
    @classmethod
    def register_loader(cls, source_type: str, loader_class: Type[BaseGraphLoader]) -> None:
        """
        Extends the factory with a new loader implementation at runtime.
        """
        if not issubclass(loader_class, BaseGraphLoader):
            raise TypeError(f"Loader must inherit from BaseGraphLoader, got {loader_class}")
        
        cls._LOADER_REGISTRY[source_type] = loader_class
        print(f"[Factory] Registered loader: {source_type}")
    
    @classmethod
    def get_supported_sources(cls) -> list:
        return list(cls._LOADER_REGISTRY.keys())
    
    @classmethod
    def get_data(cls, 
                 source_type: str, 
                 dataset_name: str, 
                 target_ntype: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        """
        Main interface for dataset retrieval.
        
        Example:
            g, info = DatasetFactory.get_data('HGB', 'DBLP', 'author')
        """
        if source_type not in cls._LOADER_REGISTRY:
            supported = ', '.join(cls.get_supported_sources())
            raise ValueError(
                f"Unknown data source: '{source_type}'. "
                f"Supported: {supported}"
            )
        
        loader_class = cls._LOADER_REGISTRY[source_type]
        loader = loader_class()
        
        try:
            return loader.load(dataset_name, target_ntype, config.DATA_DIR)
        except Exception as e:
            print(f"[Factory] Failed to load {source_type}/{dataset_name}: {e}")
            raise