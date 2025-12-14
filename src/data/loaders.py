"""
Concrete implementations of graph data loaders.
Each loader implements the Strategy pattern for specific dataset sources.
"""
import os
import pandas as pd
import torch
from typing import Tuple, Dict, Any
import torch_geometric.data as tg_data
from torch_geometric.datasets import HGBDataset, OGB_MAG, DBLP, IMDB, AMiner

from .base import BaseGraphLoader


class HGBLoader(BaseGraphLoader):
    """Loader for THUDM's Heterogeneous Graph Benchmark datasets."""
    
    def load(self, dataset_name: str, target_ntype: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        print(f"[HGBLoader] Loading {dataset_name}...")
        
        dataset = HGBDataset(root=f'./data/HGB_{dataset_name}', name=dataset_name)
        g = dataset[0]
        
        # Create or extract masks
        train_mask = g[target_ntype].train_mask if hasattr(g[target_ntype], 'train_mask') else None
        train_mask, val_mask, test_mask = self._create_random_masks(
            g[target_ntype].num_nodes, train_mask
        )
        
        # Extract labels and features
        labels, num_classes = self._extract_labels(g, target_ntype)
        features = self._ensure_features(g, target_ntype)
        
        info = self.create_info_dict(features, labels, train_mask, val_mask, test_mask, num_classes)
        return g, info


class OGBLoader(BaseGraphLoader):
    """Loader for Open Graph Benchmark datasets."""
    
    def load(self, dataset_name: str, target_ntype: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        print(f"[OGBLoader] Loading {dataset_name}...")
        
        if dataset_name.lower() not in ['ogbn-mag', 'mag']:
            raise ValueError(f"OGBLoader only supports 'ogbn-mag', got '{dataset_name}'")
        
        dataset = OGB_MAG(root='./data/OGB', preprocess='metapath2vec')
        g = dataset[0]
        
        if not hasattr(g[target_ntype], 'train_mask'):
            raise ValueError(f"Target node '{target_ntype}' in {dataset_name} has no masks!")
        
        # OGB provides masks and labels
        labels, num_classes = self._extract_labels(g, target_ntype)
        features = g[target_ntype].x
        
        info = self.create_info_dict(
            features, labels,
            g[target_ntype].train_mask,
            g[target_ntype].val_mask,
            g[target_ntype].test_mask,
            num_classes
        )
        return g, info


class PyGStandardLoader(BaseGraphLoader):
    """Loader for standard PyTorch Geometric datasets (DBLP, IMDB, AMiner)."""
    
    DATASET_MAPPING = {
        'DBLP': DBLP,
        'IMDB': IMDB,
        'AMiner': AMiner
    }
    
    def load(self, dataset_name: str, target_ntype: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        print(f"[PyGLoader] Loading {dataset_name}...")
        
        if dataset_name not in self.DATASET_MAPPING:
            raise ValueError(
                f"Unknown PyG dataset: {dataset_name}. "
                f"Available: {list(self.DATASET_MAPPING.keys())}"
            )
        
        path = f'./data/PyG_{dataset_name}'
        dataset_class = self.DATASET_MAPPING[dataset_name]
        dataset = dataset_class(root=path)
        g = dataset[0]
        
        # Create random masks
        train_mask, val_mask, test_mask = self._create_random_masks(g[target_ntype].num_nodes)
        
        # Extract labels and features
        labels, num_classes = self._extract_labels(g, target_ntype)
        features = self._ensure_features(g, target_ntype)
        
        info = self.create_info_dict(features, labels, train_mask, val_mask, test_mask, num_classes)
        return g, info


class HNELoader(BaseGraphLoader):
    """
    Loader for HNE (Heterogeneous Network Embedding) format datasets.
    Handles the custom .dat file format (node.dat, link.dat, meta.dat).
    """
    
    # Dataset-specific type mappings
    TYPE_MAPPINGS = {
        'DBLP': {0: 'author', 1: 'paper', 2: 'conf', 3: 'term'},
        'Yelp': {0: 'user', 1: 'business', 2: 'service', 3: 'level'},
        'PubMed': {0: 'paper', 1: 'author', 2: 'journal', 3: 'keyword'},
        'Freebase': {0: 'entity', 1: 'type', 2: 'relation'}
    }
    
    def load(self, dataset_name: str, target_ntype: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        # Handle the case where we loaded a sampled dataset (which might have a different folder name)
        # In main.py, we register sampled datasets with source='HNE', so they come here.
        # We assume the directory name matches the dataset_name passed in config.
        data_dir = f'./data/HNE_{dataset_name}'
        
        print(f"[HNELoader] Loading from {data_dir}...")
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"HNE dataset directory not found: {data_dir}")
        
        # Load nodes
        g = self._load_nodes(data_dir, dataset_name)
        
        # Load edges
        self._load_edges(data_dir, g)
        
        # Generate features and labels (HNE datasets typically lack these)
        features = self._ensure_features(g, target_ntype)
        labels, num_classes = self._extract_labels(g, target_ntype)
        
        # Create masks
        train_mask, val_mask, test_mask = self._create_random_masks(g[target_ntype].num_nodes)
        
        info = self.create_info_dict(features, labels, train_mask, val_mask, test_mask, num_classes)
        return g, info
    
    def _load_nodes(self, data_dir: str, dataset_name: str) -> tg_data.HeteroData:
        """Load node.dat file and create graph structure."""
        node_file = os.path.join(data_dir, "node.dat")
        print(f"    Reading nodes from {node_file}...")
        
        df_nodes = pd.read_csv(
            node_file, sep='\t', header=None,
            names=['id', 'name', 'type', 'info'],
            encoding='utf-8', quoting=3
        )
        
        # Detect base dataset name (remove _Sampled suffix) to look up types
        base_name = dataset_name.replace("_Sampled", "").replace("_SAMPLED", "")
        type_map = self.TYPE_MAPPINGS.get(base_name, {})
        
        if not type_map:
            unique_types = df_nodes['type'].unique()
            type_map = {t: f"type_{t}" for t in unique_types}
        
        g = tg_data.HeteroData()
        self._global_to_local = {} 
        
        # Process nodes by type
        for type_id, group in df_nodes.groupby('type'):
            type_name = type_map.get(type_id, f"type_{type_id}")
            g[type_name].num_nodes = len(group)
            
            # Create ID mapping efficiently
            # Vectorized creation of dict entries is hard, but this loop 
            # over 500k integers is fast enough (sub-second)
            local_indices = range(len(group))
            global_ids = group['id'].values
            
            # Update global map
            for glob, loc in zip(global_ids, local_indices):
                self._global_to_local[glob] = (type_name, loc)
                
        print(f"    Loaded {len(df_nodes)} nodes.")
        return g
    
    def _load_edges(self, data_dir: str, g: tg_data.HeteroData) -> None:
        """Load link.dat file and populate edge indices (VECTORIZED)."""
        link_file = os.path.join(data_dir, "link.dat")
        print(f"    Reading edges from {link_file}...")
        
        # Read only necessary columns to save memory
        df_links = pd.read_csv(
            link_file, sep='\t', header=None,
            names=['src', 'dst'],
            usecols=[0, 1]
        )
        
        print("    Mapping edge IDs (Vectorized)...")
        
        # 1. Filter edges to ensure both nodes exist in the graph 
        # (Crucial if link.dat contains edges to nodes not in node.dat)
        # Using map is strictly faster than iterating
        valid_src = df_links['src'].isin(self._global_to_local)
        valid_dst = df_links['dst'].isin(self._global_to_local)
        df_links = df_links[valid_src & valid_dst].copy()
        
        if df_links.empty:
            print("    [Warning] No valid edges found matching loaded nodes.")
            return

        # 2. Map global IDs to (type, local_id) tuples
        # pd.Series.map with a dict is highly optimized
        src_tuples = df_links['src'].map(self._global_to_local)
        dst_tuples = df_links['dst'].map(self._global_to_local)
        
        # 3. Extract types and indices into separate columns
        # List comprehension is faster than apply/lambda here
        df_links['src_type'] = [t[0] for t in src_tuples]
        df_links['src_idx']  = [t[1] for t in src_tuples]
        df_links['dst_type'] = [t[0] for t in dst_tuples]
        df_links['dst_idx']  = [t[1] for t in dst_tuples]
        
        print("    Building PyG edge tensors...")
        
        # 4. Group by (src_type, dst_type) to create specific edge stores
        grouped = df_links.groupby(['src_type', 'dst_type'])
        
        count = 0
        for (src_type, dst_type), group in grouped:
            rel_name = f"{src_type}_to_{dst_type}"
            
            # Convert directly to torch tensors
            src = torch.tensor(group['src_idx'].values, dtype=torch.long)
            dst = torch.tensor(group['dst_idx'].values, dtype=torch.long)
            
            edge_index = torch.stack([src, dst], dim=0)
            
            # Assign to graph
            if rel_name not in g.edge_types:
                # Basic check to avoid weird relation names
                pass
                
            g[src_type, rel_name, dst_type].edge_index = edge_index
            count += len(group)
            
        print(f"    Loaded {count} edges successfully.")