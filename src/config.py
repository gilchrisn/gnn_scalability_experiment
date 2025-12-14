import torch
import os

# System
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPP_EXECUTABLE = "./bin/graph_prep"
TEMP_DIR = "exp_data/intermediate"

# --- SELECT YOUR EXPERIMENT HERE ---
# Options: 'HGB_DBLP', 'OGB_MAG', 'PyG_IMDB', etc.
DATASET_NAME = 'HGB_DBLP' 

# --- BENCHMARK SETTINGS ---
METAPATH_LENGTH = 3  # Length of the random metapath to generate
K_VALUES = [2, 4, 8, 16, 32] # Sketch sizes to test

DATASET_CONFIGS = {
    'HGB_DBLP': {
        'source': 'HGB',
        'dataset_name': 'DBLP',
        'target_node': 'author',
    },
    'HGB_ACM': {
        'source': 'HGB',
        'dataset_name': 'ACM',
        'target_node': 'paper',
    },
    'HGB_IMDB': {
        'source': 'HGB',
        'dataset_name': 'IMDB',
        'target_node': 'movie',
    },
    'HGB_Freebase': {
        'source': 'HGB',
        'dataset_name': 'Freebase',
        'target_node': 'book',
    },
    'OGB_MAG': {
        'source': 'OGB',
        'dataset_name': 'ogbn-mag',
        'target_node': 'paper',
    },
    'PyG_IMDB': {
        'source': 'PyG',
        'dataset_name': 'IMDB',
        'target_node': 'movie',
    },
    'PyG_DBLP': {
        'source': 'PyG',
        'dataset_name': 'DBLP',
        'target_node': 'author',
    },
    'PyG_AMiner': {
        'source': 'PyG',
        'dataset_name': 'AMiner',
        'target_node': 'author',
    },
    'HNE_DBLP': {
        'source': 'HNE',
        'dataset_name': 'DBLP',
        'target_node': 'author',
    },
    'HNE_Yelp': {
        'source': 'HNE',
        'dataset_name': 'Yelp',
        'target_node': 'user',
    },
    'HNE_PubMed': {
        'source': 'HNE',
        'dataset_name': 'PubMed',
        'target_node': 'paper',
    },
    'HNE_Freebase': {
        'source': 'HNE',
        'dataset_name': 'Freebase',
        'target_node': 'entity',
    },
}

# Active Config
CURRENT_CONFIG = DATASET_CONFIGS[DATASET_NAME]
TARGET_NODE_TYPE = CURRENT_CONFIG['target_node']

# Model Hyperparams
HIDDEN_DIM = 64
GAT_HEADS = 8
MODEL_DIR = "results/models"