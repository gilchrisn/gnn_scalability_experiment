"""
Verification script to check if the refactored codebase is properly set up.
Run this after migration to ensure all modules are importable.
"""
import sys
import os

def check_imports():
    """Verify all modules can be imported."""
    print("Checking module imports...\n")
    
    checks = [
        ("Config", "from src.config import config"),
        ("Data Factory", "from src.data import DatasetFactory"),
        ("Data Loaders", "from src.data.loaders import HGBLoader, OGBLoader, PyGStandardLoader, HNELoader"),
        ("Exact Kernel", "from src.kernels import ExactMaterializationKernel"),
        ("KMV Kernel", "from src.kernels import KMVSketchingKernel"),
        ("C++ Bridge", "from src.bridge import CppBridge, PyGToCppAdapter"),
        ("AnyBURL", "from src.bridge import AnyBURLRunner"),
        ("Utils", "from src.utils import generate_random_metapath"),
        ("Models", "from src.models import get_model"),
    ]
    
    failed = []
    
    for name, import_stmt in checks:
        try:
            exec(import_stmt)
            print(f"✓ {name:20s} - OK")
        except Exception as e:
            print(f"✗ {name:20s} - FAILED: {e}")
            failed.append((name, str(e)))
    
    return failed

def check_directories():
    """Verify required directories exist."""
    print("\n\nChecking directories...\n")
    
    required_dirs = [
        "src/data",
        "src/kernels",
        "src/bridge",
        "output",
        "output/intermediate",
        "output/models",
        "output/results"
    ]
    
    missing = []
    
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        status = "✓" if exists else "✗"
        print(f"{status} {dir_path}")
        if not exists:
            missing.append(dir_path)
    
    return missing

def check_files():
    """Verify required files exist."""
    print("\n\nChecking required files...\n")
    
    required_files = [
        "main.py",
        "src/__init__.py",
        "src/config.py",
        "src/utils.py",
        "src/models.py",
        "src/data/__init__.py",
        "src/data/base.py",
        "src/data/loaders.py",
        "src/data/factory.py",
        "src/kernels/__init__.py",
        "src/kernels/exact.py",
        "src/kernels/kmv.py",
        "src/bridge/__init__.py",
        "src/bridge/cpp_adapter.py",
        "src/bridge/anyburl.py",
    ]
    
    missing = []
    
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        if not exists:
            missing.append(file_path)
    
    return missing

def main():
    print("="*60)
    print("REFACTORED CODEBASE VERIFICATION")
    print("="*60 + "\n")
    
    # Check files
    missing_files = check_files()
    
    # Check directories
    missing_dirs = check_directories()
    
    # Check imports
    failed_imports = check_imports()
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_ok = not (missing_files or missing_dirs or failed_imports)
    
    if missing_files:
        print(f"\n✗ Missing {len(missing_files)} files:")
        for f in missing_files:
            print(f"  - {f}")
    
    if missing_dirs:
        print(f"\n✗ Missing {len(missing_dirs)} directories:")
        for d in missing_dirs:
            print(f"  - {d}")
    
    if failed_imports:
        print(f"\n✗ {len(failed_imports)} import failures:")
        for name, error in failed_imports:
            print(f"  - {name}: {error}")
    
    if all_ok:
        print("\n✓ All checks passed! The refactored codebase is ready.")
        print("\nNext steps:")
        print("  1. Run: python main.py list")
        print("  2. Try: python main.py benchmark --dataset HGB_DBLP --method exact")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())