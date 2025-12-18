"""
Backend module for pluggable graph materialization engines.
"""

from .base import GraphBackend, BackendFactory, BenchmarkResult
from .python_backend import PythonBackend
from .cpp_backend import CppBackend

# Auto-register available backends
BackendFactory.register('python', PythonBackend)
BackendFactory.register('cpp', CppBackend)

__all__ = [
    'GraphBackend',
    'BackendFactory',
    'BenchmarkResult',
    'PythonBackend',
    'CppBackend'
]