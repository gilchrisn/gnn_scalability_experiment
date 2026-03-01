"""
Artifact caching logic to enable zero-copy execution.
Implements the Proxy Pattern for large dataset handling.
"""
import os
import torch
import json
from typing import Dict, Any, Optional
import hashlib
from ..config import config  # FIX: was `from .. import config` which relies on __init__ re-export


class BaselineCache:
    """
    Persists 'Exact' baseline results (Embeddings/Logits) to disk.
    Auto-invalidates if the model weights file is modified.
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = os.path.join(cache_dir, "baselines")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_hash(self, dataset: str, model_name: str, metapath: str) -> str:
        """
        Creates a signature based on:
        1. Config (Dataset/Metapath)
        2. Model Weights Timestamp (Auto-invalidation)
        """
        # Get path to model weights
        model_path = config.get_model_path(dataset, model_name)
        
        # Get timestamp (0.0 if file doesn't exist yet)
        timestamp = os.path.getmtime(model_path) if os.path.exists(model_path) else 0.0
        
        # Create unique string: "Dataset|Model|Time|Path"
        raw_str = f"{dataset}|{model_name}|{timestamp}|{metapath.strip().replace(' ', '')}"
        return hashlib.md5(raw_str.encode()).hexdigest()

    def save(self, dataset: str, model: str, metapath: str, data: Dict[str, torch.Tensor]) -> None:
        """Saves baseline tensors."""
        filename = f"{self._get_hash(dataset, model, metapath)}.pt"
        path = os.path.join(self.cache_dir, filename)
        torch.save(data, path)
        print(f"[Cache] Baseline saved: {path}")

    def load(self, dataset: str, model: str, metapath: str) -> Optional[Dict[str, torch.Tensor]]:
        """Loads baseline tensors if they match the current model version."""
        filename = f"{self._get_hash(dataset, model, metapath)}.pt"
        path = os.path.join(self.cache_dir, filename)
        
        if os.path.exists(path):
            print(f"[Cache] Baseline HIT: {path}")
            return torch.load(path)
        
        print(f"[Cache] Baseline MISS (Model changed or new path)")
        return None