"""
Interop layer for C++ (graph processing) and Java (AnyBURL) backends.
Exports clean interfaces and concrete implementations.
"""

from .base import GraphConverter, RuleMiner, ExecutionEngine
from .converter import PyGToCppAdapter
from .engine import CppEngine
from .anyburl import AnyBURLRunner

__all__ = [
    'GraphConverter',
    'RuleMiner',
    'ExecutionEngine',
    'PyGToCppAdapter',
    'CppEngine',
    'AnyBURLRunner'
]