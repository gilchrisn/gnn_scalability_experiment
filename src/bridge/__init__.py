"""
Bridge module for external tool interoperability.
Provides interfaces to C++ and Java components.
"""

from .cpp_adapter import CppBridge, PyGToCppAdapter
from .anyburl import AnyBURLRunner

__all__ = [
    'CppBridge',
    'PyGToCppAdapter',
    'AnyBURLRunner'
]