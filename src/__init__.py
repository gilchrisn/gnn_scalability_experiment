"""
GNN Scalability Experiment Package
Provides tools for benchmarking and analyzing heterogeneous graph neural networks.
"""

__version__ = "2.0.0"
__author__ = "Your Name"

# Import key components for convenient access
from .config import config
from .data import DatasetFactory
from .kernels import ExactMaterializationKernel, KMVSketchingKernel
from .bridge import CppEngine, PyGToCppAdapter, AnyBURLRunner

__all__ = [
    'config',
    'DatasetFactory',
    'ExactMaterializationKernel',
    'KMVSketchingKernel',
    'CppEngine',
    'PyGToCppAdapter',
    'AnyBURLRunner'
]