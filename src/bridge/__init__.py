# src/bridge/__init__.py
#
# Public API for the bridge layer.
#
#   CppEngine         → materialize / sketch (training pipeline)
#   GraphPrepRunner   → all experiment-side commands (bench scripts)
#   PyGToCppAdapter   → PyG graph → C++ .dat file conversion
#
# DELETED: cpp_adapter.py
#   CppBridge was superseded by CppEngine.
#   Its PyGToCppAdapter copy was a duplicate of converter.py.
#   Neither class was ever imported. Removed to eliminate confusion.

from src.bridge.engine import CppEngine
from src.bridge.converter import PyGToCppAdapter
from src.bridge.runner import GraphPrepRunner
from src.bridge.anyburl import AnyBURLRunner

__all__ = ["CppEngine", "PyGToCppAdapter", "GraphPrepRunner", "AnyBURLRunner"]