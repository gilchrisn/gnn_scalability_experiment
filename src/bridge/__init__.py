"""
Interop layer for C++ (graph processing) and Java (AnyBURL) backends.
"""

from .cpp_adapter import CppBridge, PyGToCppAdapter
from .anyburl import AnyBURLRunner

__all__ = [
    'CppBridge',
    'PyGToCppAdapter',
    'AnyBURLRunner'
]