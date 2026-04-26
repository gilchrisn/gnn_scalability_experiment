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
    suggested_paths: List[str] = field(default_factory=list)
    
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

        # Training Configuration
        self.IGNORE_LABEL_INDEX = -100
        
        # Paths
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, "datasets")
        self.STAGING_DIR = os.path.join(self.PROJECT_ROOT, "staging")
        self.OUTPUT_DIR = os.path.join(self.PROJECT_ROOT, "output")
        self.TEMP_DIR = os.path.join(self.OUTPUT_DIR, "intermediate")
        self.MODEL_DIR = os.path.join(self.OUTPUT_DIR, "models")
        self.RESULTS_DIR = os.path.join(self.OUTPUT_DIR, "results")
        
        # External Tools
        _exe = "graph_prep.exe" if os.name == "nt" else "graph_prep"
        self.CPP_EXECUTABLE = os.path.join(self.PROJECT_ROOT, "bin", _exe)
        self.ANYBURL_JAR = os.path.join(self.PROJECT_ROOT, "tools", "AnyBURL-23-1x.jar")
        
        # Benchmark Settings
        self.METAPATH_LENGTH = 3
        self.K_VALUES = [2, 4, 8, 16, 32]
        self.N_RUNS = 3
        
        # Model Hyperparameters
        self.HIDDEN_DIM = 64
        self.GAT_HEADS = 8
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 5e-4
        self.EPOCHS = 50
        
        # Dataset Registry
        self.DATASETS = self._init_datasets()
        
        # Create necessary directories
        self._ensure_directories()
        
        Config._initialized = True
    
    def _init_datasets(self) -> Dict[str, DatasetConfig]:
        """
        Initialize dataset registry with schema-verified metapaths.
        
        Corrections made based on SCHEMA files:
        - HGB_DBLP: 'conf' -> 'venue'
        - HGB_Freebase: 'author'/'genre' -> 'people' (BPB)
        - OGB_MAG: Fixed edge names ('rev_writes') and removed invalid 'venue' path
        - HNE_Yelp: Removed invalid 'service' link; added 'level' path
        """
        return {
            # === HGB DATASETS ===
            'HGB_DBLP': DatasetConfig('HGB', 'DBLP', 'author', [
                "author_to_paper,paper_to_author",                              # APA
                "author_to_paper,paper_to_term,term_to_paper,paper_to_author",  # APTPA
                "author_to_paper,paper_to_venue,venue_to_paper,paper_to_author",# APVPA
            ]),
            'HGB_ACM': DatasetConfig('HGB', 'ACM', 'paper', [
                "paper_to_author,author_to_paper",              # PAP
                "paper_to_subject,subject_to_paper",            # PSP
                "paper_to_term,term_to_paper",                  # PTP
            ]),
            'HGB_IMDB': DatasetConfig('HGB', 'IMDB', 'movie', [
                "movie_to_keyword,keyword_to_movie",                                    # MKM
                "movie_to_actor,actor_to_movie,movie_to_keyword,keyword_to_movie",      # MAMKM
                "movie_to_director,director_to_movie,movie_to_keyword,keyword_to_movie",# MDMKM
                "movie_to_keyword,keyword_to_movie,movie_to_actor,actor_to_movie",      # MKMAM
            ]),

            # === OGB DATASETS ===
            'OGB_MAG': DatasetConfig('OGB', 'ogbn-mag', 'paper', [
                "rev_writes,writes",                                    # PAP — 7.1M edges, dense
                "has_topic,rev_has_topic",                               # PFP — 7.5M edges, very dense
                "cites,rev_cites",                                       # Bib coupling — 5.4M edges
                "rev_cites,cites",                                       # Co-citation — 5.4M edges
                "rev_writes,affiliated_with,rev_affiliated_with,writes", # PAIAP — 4-hop
            ]),

            # === OAG (Open Academic Graph, 768-dim XLNet features, 546K papers) ===
            'OAG_CS': DatasetConfig('OAG', 'cs', 'paper', [
                "rev_AP_write_first,AP_write_first",             # PAP first-author — 457K edges, sparse
                "rev_AP_write_other,AP_write_other",             # PAP co-authors — 664K edges, denser
                "PP_cite,rev_PP_cite",                           # Bib coupling — 5.9M edges, very dense
                "rev_PP_cite,PP_cite",                           # Co-citation — 5.9M edges, very dense
                "PF_in_L2,rev_PF_in_L2",                        # PFP L2 fields — 2.3M edges, dense
                "PF_in_L3,rev_PF_in_L3",                        # PFP L3 fields — 870K edges
            ]),

            # === RCDD (Alibaba risk detection, 13.8M nodes, all types featurized) ===
            # Run scripts/inspect_rcdd.py after download to fill in real metapaths.
            'RCDD_AliRCD': DatasetConfig('RCDD', 'AliRCD', 'item', []),

            # === PyG DATASETS ===
            'PyG_IMDB': DatasetConfig('PyG', 'IMDB', 'movie', [
                "movie_to_actor,actor_to_movie", 
                "movie_to_director,director_to_movie"
            ]),
            'PyG_DBLP': DatasetConfig('PyG', 'DBLP', 'author', [
                "author_to_paper,paper_to_author", 
                "author_to_paper,paper_to_conference,conference_to_paper,paper_to_author"
            ]),
            'PyG_AMiner': DatasetConfig('PyG', 'AMiner', 'paper', [
                "written_by,writes",                           # PAP
                "published_in,publishes"                       # PVP
            ]),

            # === HNE DATASETS ===
            'HNE_DBLP': DatasetConfig('HNE', 'DBLP', 'author', [
                "author_to_paper,paper_to_author", 
                "author_to_paper,paper_to_conf,conf_to_paper,paper_to_author"
            ]),
            'HNE_Yelp': DatasetConfig('HNE', 'Yelp', 'user', [
                "user_to_business,business_to_user",           # UBU (Co-reviews)
                "user_to_level,level_to_user"                  # ULU (Shared Level)
            ]),
            'HNE_PubMed': DatasetConfig('HNE', 'PubMed', 'disease', [
                "disease_to_gene,gene_to_disease",
                "disease_to_chemical,chemical_to_disease",
                
                "disease_to_species,species_to_disease",
            ]),
            'HNE_Freebase': DatasetConfig('HNE', 'Freebase', 'entity', [
                "entity_to_relation,relation_to_entity",       # ERE
                "entity_to_type,type_to_entity"                # ETE
            ]),

            # === MINI (sampled) DATASETS for local pipeline testing ===
            # Generated by: python scripts/make_mini_datasets.py --datasets OGB_MAG OAG_CS
            'MINI_OGB_MAG': DatasetConfig('MINI', 'OGB_MAG', 'paper', [
                "rev_writes,writes",
                "has_topic,rev_has_topic",
            ]),
            'MINI_OAG_CS': DatasetConfig('MINI', 'OAG_CS', 'paper', [
                "rev_AP_write_first,AP_write_first",
                "PF_in_L1,rev_PF_in_L1",
            ]),
            'MINI_RCDD': DatasetConfig('MINI', 'RCDD_AliRCD', 'item', []),

            # === CUSTOM DATASETS ===
            'CUSTOM_ACM': DatasetConfig('CUSTOM', 'ACM', 'paper', [
                "paper_to_author,author_to_paper",
                "paper_to_subject,subject_to_paper"
            ]),
            'CUSTOM_IMDB': DatasetConfig('CUSTOM', 'IMDB', 'movie', [
                "movie_to_actor,actor_to_movie",
                "movie_to_director,director_to_movie"
            ]),
            'CUSTOM_Yelp': DatasetConfig('CUSTOM', 'Yelp', 'user', [
                "user_to_business,business_to_user",
                "user_to_level,level_to_user"
            ]),
            'CUSTOM_Freebase': DatasetConfig('CUSTOM', 'Freebase', 'book', [
                "book_to_people,people_to_book",
                "book_to_film,film_to_book"
            ]),
        }
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [self.DATA_DIR, self.STAGING_DIR, self.OUTPUT_DIR, self.TEMP_DIR,
                        self.MODEL_DIR, self.RESULTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)

    def get_staging_dir(self, dataset_key: str) -> str:
        """Absolute path to the C++ working directory for a dataset.

        All staging folders live under ``<project_root>/staging/``.
        """
        return os.path.join(self.STAGING_DIR, self.get_folder_name(dataset_key))
    
    def get_folder_name(self, dataset_key: str) -> str:
        """Return the C++ working-directory name for a dataset.

        HGB datasets use the legacy ``HGBn-<NAME>`` convention (matching the
        original paper's directory layout).  All other datasets use the dataset
        key itself as the folder name.
        """
        if dataset_key.startswith("HGB_"):
            return f"HGBn-{dataset_key.split('_', 1)[1]}"
        return dataset_key

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
        if self.DEVICE.type == 'cuda' and not torch.cuda.is_available():
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

    def get_model_path(self, dataset_key: str, model_type: str) -> str:
        """Standardized path for saving/loading models."""
        filename = f"{dataset_key}_{model_type.lower()}.pt"
        return os.path.join(self.MODEL_DIR, filename)

# Global configuration instance
config = Config()