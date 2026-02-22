# ---------- File: ./src/experiments/config.py ----------
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ExperimentConfig:
    """
    Configuration DTO for the Master Protocol.
    Adheres to Open/Closed principle: Add fields here, don't change logic consumers.
    """
    dataset: str
    model_arch: str = "SAGE"  # Base architecture (GCN, SAGE)
    
    mining_strategy: str = "stratified" # 'stratified' or 'top_k'
    stratified_buckets: List[int] = field(default_factory=lambda: [2, 4, 6, 8])
    
    # Phase 1: Mining Settings (NEW)
    min_confidence: float = 0.05  # 5% threshold
    top_n_paths: int = 3          # Bound the experiment size

    # Phase 2: Model Zoo Settings
    model_depths: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])  # Depths to train for Model Zoo
    standard_depth: int = 2   # The 'Teacher' depth for Phase 3
    
    # Phase 3: Robustness Settings
    k_values: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    
    # Phase 4: Depth Study Settings
    depth_study_k: int = 32   # Fixed high fidelity for depth check
    
    # General Settings
    epochs: int = 50
    force_retrain: bool = False
    force_remine: bool = False
    seed: int = 42  # Global seed for reproducibility
    
    # Runtime State (Managed by Orchestrator)
    current_model_name: Optional[str] = None  # e.g., "SAGE_L2"