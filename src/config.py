"""
Unified configuration management using the Singleton pattern.
Provides centralized access to all system parameters.
"""
import os
import torch
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    source: str  # 'HGB', 'OGB', 'PyG', 'HNE'
    dataset_name: str  # 'DBLP', 'IMDB', etc.
    target_node: str  # Target node type for prediction
    
    def __str__(self) -> str:
        return f"{self.source}_{self.dataset_name}"


class Config:
    """
    Singleton configuration class.
    Implements lazy initialization and validation.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if Config._initialized:
            return
        
        # System Configuration
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, "datasets")
        self.OUTPUT_DIR = os.path.join(self.PROJECT_ROOT, "output")
        self.TEMP_DIR = os.path.join(self.OUTPUT_DIR, "intermediate")
        self.MODEL_DIR = os.path.join(self.OUTPUT_DIR, "models")
        self.RESULTS_DIR = os.path.join(self.OUTPUT_DIR, "results")
        
        # External Tools
        self.CPP_EXECUTABLE = os.path.join(self.PROJECT_ROOT, "bin", "graph_prep.exe")
        self.ANYBURL_JAR = os.path.join(self.PROJECT_ROOT, "tools", "AnyBURL-23-1x.jar")
        
        # Benchmark Settings
        self.METAPATH_LENGTH = 3
        self.K_VALUES = [2, 4, 8, 16, 32]
        self.N_RUNS = 3
        
        # Model Hyperparameters
        self.HIDDEN_DIM = 64
        self.GAT_HEADS = 8
        self.LEARNING_RATE = 0.01
        self.WEIGHT_DECAY = 5e-4
        self.EPOCHS = 50
        
        # Dataset Registry
        self.DATASETS = self._init_datasets()
        
        # Create necessary directories
        self._ensure_directories()
        
        Config._initialized = True
    
    def _init_datasets(self) -> Dict[str, DatasetConfig]:
        """Initialize dataset registry."""
        return {
            'HGB_DBLP': DatasetConfig('HGB', 'DBLP', 'author'),
            'HGB_ACM': DatasetConfig('HGB', 'ACM', 'paper'),
            'HGB_IMDB': DatasetConfig('HGB', 'IMDB', 'movie'),
            'HGB_Freebase': DatasetConfig('HGB', 'Freebase', 'book'),
            'OGB_MAG': DatasetConfig('OGB', 'ogbn-mag', 'paper'),
            'PyG_IMDB': DatasetConfig('PyG', 'IMDB', 'movie'),
            'PyG_DBLP': DatasetConfig('PyG', 'DBLP', 'author'),
            'PyG_AMiner': DatasetConfig('PyG', 'AMiner', 'author'),
            'HNE_DBLP': DatasetConfig('HNE', 'DBLP', 'author'),
            'HNE_Yelp': DatasetConfig('HNE', 'Yelp', 'user'),
            'HNE_PubMed': DatasetConfig('HNE', 'PubMed', 'paper'),
            'HNE_Freebase': DatasetConfig('HNE', 'Freebase', 'entity'),
        }
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [self.DATA_DIR, self.OUTPUT_DIR, self.TEMP_DIR, 
                        self.MODEL_DIR, self.RESULTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_dataset_config(self, key: str) -> DatasetConfig:
        """
        Retrieve dataset configuration by key.
        
        Args:
            key: Dataset identifier (e.g., 'HGB_DBLP')
            
        Returns:
            DatasetConfig object
            
        Raises:
            KeyError: If dataset key not found
        """
        if key not in self.DATASETS:
            available = ', '.join(list(self.DATASETS.keys())[:5])
            raise KeyError(
                f"Dataset '{key}' not found. "
                f"Available datasets: {available}, ..."
            )
        return self.DATASETS[key]
    
    def list_datasets(self) -> List[str]:
        """Returns list of available dataset keys."""
        return list(self.DATASETS.keys())
    
    def register_dataset(self, key: str, config: DatasetConfig) -> None:
        """
        Register a new dataset configuration.
        
        Args:
            key: Unique identifier for the dataset
            config: DatasetConfig object
        """
        if key in self.DATASETS:
            print(f"[Config] Warning: Overwriting existing dataset config '{key}'")
        self.DATASETS[key] = config
        print(f"[Config] Registered dataset: {key}")
    
    def validate(self) -> bool:
        """
        Validates configuration settings.
        
        Returns:
            True if all validations pass
        """
        issues = []
        
        # Check CUDA availability
        if not torch.cuda.is_available() and self.DEVICE.type == 'cuda':
            issues.append("CUDA requested but not available")
        
        # Check external tools
        if not os.path.exists(self.CPP_EXECUTABLE):
            issues.append(f"C++ executable not found: {self.CPP_EXECUTABLE}")
        
        if not os.path.exists(self.ANYBURL_JAR):
            issues.append(f"AnyBURL JAR not found: {self.ANYBURL_JAR}")
        
        if issues:
            print("[Config] Validation warnings:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("[Config] Validation passed")
        return True
    
    def __repr__(self) -> str:
        return (
            f"Config(device={self.DEVICE}, "
            f"datasets={len(self.DATASETS)}, "
            f"k_values={self.K_VALUES})"
        )


# Global configuration instance
config = Config()