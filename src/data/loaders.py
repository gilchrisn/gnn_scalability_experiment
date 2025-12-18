"""
Concrete implementations of graph data loaders.
Standardizes edge names and ensures bidirectional connectivity.
"""
import os
import pandas as pd
import torch
from typing import Tuple, Dict, Any, List
import torch_geometric.data as tg_data
from torch_geometric.datasets import HGBDataset, OGB_MAG, DBLP, IMDB, AMiner

from .base import BaseGraphLoader

class GraphStandardizer:
    """Helper to rename generic edges and add reverse edges."""
    
    @staticmethod
    def standardize(g: tg_data.HeteroData) -> None:
        # 1. Collect all edges first to avoid runtime modification issues
        original_edges = list(g.edge_types)
        new_edge_store = {}

        # 2. Rename generic edges (e.g. 'to' -> 'author_to_paper')
        for src, rel, dst in original_edges:
            edge_index = g[src, rel, dst].edge_index
            
            # Determine standard name
            if rel in ['to', 'adj', 'link'] or rel == 'to':
                std_rel = f"{src}_to_{dst}"
            else:
                std_rel = rel
            
            # Store forward edge with standard name
            new_edge_store[(src, std_rel, dst)] = edge_index

            # 3. Create Reverse Edge immediately
            if "_to_" in std_rel:
                parts = std_rel.split("_to_")
                if len(parts) == 2:
                    rev_rel = f"{parts[1]}_to_{parts[0]}"
                else:
                    rev_rel = f"rev_{std_rel}"
            else:
                rev_rel = f"rev_{std_rel}"

            rev_key = (dst, rev_rel, src)
            rev_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            
            # Store reverse edge (if not duplicate)
            if rev_key not in new_edge_store:
                new_edge_store[rev_key] = rev_index

        # 4. Clear and Rebuild Graph
        # We delete old generic edges and replace with the standardized set
        for src, rel, dst in original_edges:
            del g[src, rel, dst]
            
        for (src, rel, dst), index in new_edge_store.items():
            g[src, rel, dst].edge_index = index
            
        print(f"    [Standardized] Graph now has {len(g.edge_types)} edge types (Forward + Reverse).")


class HGBLoader(BaseGraphLoader):
    def load(self, dataset_name: str, target_ntype: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        print(f"[HGBLoader] Loading {dataset_name}...")
        dataset = HGBDataset(root=f'./datasets/HGB_{dataset_name}', name=dataset_name)
        g = dataset[0]
        
        # FIX: Standardize Names & Add Reverse Edges
        GraphStandardizer.standardize(g)
        
        train_mask = g[target_ntype].train_mask if hasattr(g[target_ntype], 'train_mask') else None
        train_mask, val_mask, test_mask = self._create_random_masks(g[target_ntype].num_nodes, train_mask)
        labels, num_classes = self._extract_labels(g, target_ntype)
        features = self._ensure_features(g, target_ntype)
        
        info = self.create_info_dict(features, labels, train_mask, val_mask, test_mask, num_classes)
        return g, info


class OGBLoader(BaseGraphLoader):
    def load(self, dataset_name: str, target_ntype: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        print(f"[OGBLoader] Loading {dataset_name}...")
        if dataset_name.lower() not in ['ogbn-mag', 'mag']:
            raise ValueError(f"OGBLoader only supports 'ogbn-mag'")
        
        dataset = OGB_MAG(root='./datasets/OGB', preprocess='metapath2vec')
        g = dataset[0]
        
        # FIX: Standardize Names & Add Reverse Edges
        GraphStandardizer.standardize(g)
        
        labels, num_classes = self._extract_labels(g, target_ntype)
        features = g[target_ntype].x
        
        info = self.create_info_dict(features, labels, g[target_ntype].train_mask, 
                                   g[target_ntype].val_mask, g[target_ntype].test_mask, num_classes)
        return g, info


class PyGStandardLoader(BaseGraphLoader):
    DATASET_MAPPING = {'DBLP': DBLP, 'IMDB': IMDB, 'AMiner': AMiner}
    
    def load(self, dataset_name: str, target_ntype: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        print(f"[PyGLoader] Loading {dataset_name}...")
        if dataset_name not in self.DATASET_MAPPING:
            raise ValueError(f"Unknown PyG dataset: {dataset_name}")
        
        path = f'./datasets/PyG_{dataset_name}'
        dataset = self.DATASET_MAPPING[dataset_name](root=path)
        g = dataset[0]
        
        # FIX: Standardize Names & Add Reverse Edges
        GraphStandardizer.standardize(g)
        
        train_mask, val_mask, test_mask = self._create_random_masks(g[target_ntype].num_nodes)
        labels, num_classes = self._extract_labels(g, target_ntype)
        features = self._ensure_features(g, target_ntype)
        
        info = self.create_info_dict(features, labels, train_mask, val_mask, test_mask, num_classes)
        return g, info


class HNELoader(BaseGraphLoader):
    TYPE_MAPPINGS = {
        'DBLP': {0: 'author', 1: 'paper', 2: 'conf', 3: 'term'},
        'Yelp': {0: 'user', 1: 'business', 2: 'service', 3: 'level'},
        'PubMed': {0: 'paper', 1: 'author', 2: 'journal', 3: 'keyword'},
        'Freebase': {0: 'entity', 1: 'type', 2: 'relation'}
    }
    
    def load(self, dataset_name: str, target_ntype: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        data_dir = f'./datasets/HNE_{dataset_name}'
        print(f"[HNELoader] Loading from {data_dir}...")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"HNE dataset directory not found: {data_dir}")
        
        g = self._load_nodes(data_dir, dataset_name)
        self._load_edges(data_dir, g)

        # FIX: Standardize Names & Add Reverse Edges
        GraphStandardizer.standardize(g)
        
        features = self._ensure_features(g, target_ntype)
        labels, num_classes = self._extract_labels(g, target_ntype)
        train_mask, val_mask, test_mask = self._create_random_masks(g[target_ntype].num_nodes)
        
        info = self.create_info_dict(features, labels, train_mask, val_mask, test_mask, num_classes)
        return g, info

    def _load_nodes(self, data_dir: str, dataset_name: str) -> tg_data.HeteroData:
        node_file = os.path.join(data_dir, "node.dat")
        df_nodes = pd.read_csv(node_file, sep='\t', header=None, names=['id', 'name', 'type', 'info'], encoding='utf-8', quoting=3)
        base_name = dataset_name.replace("_Sampled", "").replace("_SAMPLED", "")
        type_map = self.TYPE_MAPPINGS.get(base_name, {})
        if not type_map:
            unique_types = df_nodes['type'].unique()
            type_map = {t: f"type_{t}" for t in unique_types}
        g = tg_data.HeteroData()
        self._global_to_local = {}
        for type_id, group in df_nodes.groupby('type'):
            type_name = type_map.get(type_id, f"type_{type_id}")
            g[type_name].num_nodes = len(group)
            for glob, loc in zip(group['id'].values, range(len(group))):
                self._global_to_local[glob] = (type_name, loc)
        return g
    
    def _load_edges(self, data_dir: str, g: tg_data.HeteroData) -> None:
        link_file = os.path.join(data_dir, "link.dat")
        df_links = pd.read_csv(link_file, sep='\t', header=None, names=['src', 'dst'], usecols=[0, 1])
        valid = df_links['src'].isin(self._global_to_local) & df_links['dst'].isin(self._global_to_local)
        df_links = df_links[valid].copy()
        
        src_tuples = df_links['src'].map(self._global_to_local)
        dst_tuples = df_links['dst'].map(self._global_to_local)
        
        df_links['src_type'] = [t[0] for t in src_tuples]
        df_links['src_idx'] = [t[1] for t in src_tuples]
        df_links['dst_type'] = [t[0] for t in dst_tuples]
        df_links['dst_idx'] = [t[1] for t in dst_tuples]
        
        for (src_type, dst_type), group in df_links.groupby(['src_type', 'dst_type']):
            # Initial generic name (will be fixed by Standardizer later)
            rel_name = f"{src_type}_to_{dst_type}" 
            edge_index = torch.stack([
                torch.tensor(group['src_idx'].values, dtype=torch.long),
                torch.tensor(group['dst_idx'].values, dtype=torch.long)
            ], dim=0)
            g[src_type, rel_name, dst_type].edge_index = edge_index