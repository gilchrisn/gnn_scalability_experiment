# src/bridge/__init__.py
#
# Public API for the bridge layer.
#
#   CppEngine         → materialize / sketch (training pipeline)
#   GraphPrepRunner   → all experiment-side commands (bench scripts)
#   PyGToCppAdapter   → PyG graph → C++ .dat file conversion
#   BoolAPConverter   → PyG graph → BoolAP HIN/metapath/vertices format
#   BoolAPRunner      → subprocess wrapper for BoolAPCoreD / BoolAPCoreG
#   BoolAPResult      → parsed timing result from a BoolAP run
#   BoolAPFiles       → file paths written by BoolAPConverter
#
# DELETED: cpp_adapter.py
#   CppBridge was superseded by CppEngine.
#   Its PyGToCppAdapter copy was a duplicate of converter.py.
#   Neither class was ever imported. Removed to eliminate confusion.

from src.bridge.engine import CppEngine
from src.bridge.converter import PyGToCppAdapter
from src.bridge.runner import GraphPrepRunner
from src.bridge.anyburl import (
    AnyBURLRunner,
    build_schema,
    build_canonical,
    normalize_rel,
    normalize_path,
    parse_rules,
    mirror,
    validate_and_trim,
    load_validated_metapaths,
)
from src.bridge.boolap import BoolAPConverter, BoolAPRunner, BoolAPResult, BoolAPFiles

__all__ = [
    "CppEngine", "PyGToCppAdapter", "GraphPrepRunner", "AnyBURLRunner",
    "build_schema", "build_canonical", "normalize_rel", "normalize_path",
    "parse_rules", "mirror", "validate_and_trim", "load_validated_metapaths",
    "BoolAPConverter", "BoolAPRunner", "BoolAPResult", "BoolAPFiles",
]