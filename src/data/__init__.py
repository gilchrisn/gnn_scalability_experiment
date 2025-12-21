"""
Data module for heterogeneous graph loading.
Provides a clean interface for all dataset operations.
"""

from .factory import DatasetFactory
from .base import BaseGraphLoader
from .loaders import HGBLoader, OGBLoader, PyGStandardLoader, HNELoader
from .preprocessing import FeatureGuard
from .mapper import GlobalUniverseMapper
from .caching import ArtifactManager


__all__ = [
    'DatasetFactory',
    'BaseGraphLoader',
    'HGBLoader',
    'OGBLoader', 
    'PyGStandardLoader',
    'HNELoader',
    'FeatureGuard',
    'GlobalUniverseMapper',
    'ArtifactManager',
]