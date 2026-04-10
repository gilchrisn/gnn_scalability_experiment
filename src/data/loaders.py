"""
Concrete implementations of graph data loaders.
Standardizes edge names and ensures bidirectional connectivity.
"""
import os
import pandas as pd
import torch
from typing import Tuple, Dict, Any, List
import torch_geometric.data as tg_data
from torch_geometric.datasets import HGBDataset, OGB_MAG, DBLP, IMDB, AMiner, RCDD
try:
    # H2GB's __init__ imports its full model zoo (layers, networks) which pulls in
    # goat_model.py → turtle → tkinter (unavailable on headless servers).
    # Stub out turtle itself so the import chain doesn't break.
    import sys as _sys
    if 'tkinter' not in _sys.modules:
        _tk_stub = type(_sys)('tkinter')
        _tk_stub.Frame = type('Frame', (), {})
        _sys.modules['tkinter'] = _tk_stub
        _turtle_stub = type(_sys)('turtle')
        _turtle_stub.xcor = lambda: 0
        _sys.modules['turtle'] = _turtle_stub
    from H2GB.datasets import OAGDataset as _OAGDataset
    _H2GB_AVAILABLE = True
except ImportError:
    _H2GB_AVAILABLE = False

from .base import BaseGraphLoader

class GraphStandardizer:
    """Utility to normalize edge relationship names and enforce bi-directionality."""
    
    @staticmethod
    def standardize(g: tg_data.HeteroData) -> None:
        # Materialize edge list to avoid mutation errors during iteration
        original_edges = list(g.edge_types)
        new_edge_store = {}

        for src, rel, dst in original_edges:
            edge_index = g[src, rel, dst].edge_index
            
            # Map generic relation names to a canonical format
            if rel in ['to', 'adj', 'link']:
                std_rel = f"{src}_to_{dst}"
            else:
                std_rel = rel
            
            new_edge_store[(src, std_rel, dst)] = edge_index

            # Generate symmetric reverse edges
            if "_to_" in std_rel:
                parts = std_rel.split("_to_")
                rev_rel = f"{parts[1]}_to_{parts[0]}" if len(parts) == 2 else f"rev_{std_rel}"
            else:
                rev_rel = f"rev_{std_rel}"

            rev_key = (dst, rev_rel, src)
            rev_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            
            # Prevent overwriting if reverse relation already exists
            if rev_key not in new_edge_store:
                new_edge_store[rev_key] = rev_index

        # Rebuild graph schema with standardized keys
        for src, rel, dst in original_edges:
            del g[src, rel, dst]
            
        for (src, rel, dst), index in new_edge_store.items():
            g[src, rel, dst].edge_index = index
            


class HGBLoader(BaseGraphLoader):
    def load(self, dataset_name: str, target_ntype: str, root_dir: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:

        dataset_root = os.path.join(root_dir, f'HGB_{dataset_name}')    
        dataset = HGBDataset(root=dataset_root, name=dataset_name)
        g = dataset[0]
        
        GraphStandardizer.standardize(g)
        
        train_mask = g[target_ntype].train_mask if hasattr(g[target_ntype], 'train_mask') else None
        train_mask, val_mask, test_mask = self._create_random_masks(g[target_ntype].num_nodes, train_mask)
        labels, num_classes = self._extract_labels(g, target_ntype)
        features = self._ensure_features(g, target_ntype)
        
        info = self.create_info_dict(g, target_ntype, features, labels, train_mask, val_mask, test_mask, num_classes)
        return g, info


class OGBLoader(BaseGraphLoader):
    def load(self, dataset_name: str, target_ntype: str, root_dir: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        print(f"[OGBLoader] Loading {dataset_name}...")
        if dataset_name.lower() not in ['ogbn-mag', 'mag']:
            raise ValueError(f"OGBLoader only supports 'ogbn-mag'")
        
        dataset_root = os.path.join(root_dir, 'OGB')
        dataset = OGB_MAG(root=dataset_root, preprocess=None)
        g = dataset[0]
        
        GraphStandardizer.standardize(g)
        
        labels, num_classes = self._extract_labels(g, target_ntype)
        features = g[target_ntype].x
        
        info = self.create_info_dict(g, target_ntype, features, labels, g[target_ntype].train_mask, 
                                   g[target_ntype].val_mask, g[target_ntype].test_mask, num_classes)
        return g, info


class PyGStandardLoader(BaseGraphLoader):
    DATASET_MAPPING = {'DBLP': DBLP, 'IMDB': IMDB, 'AMiner': AMiner}
    
    def load(self, dataset_name: str, target_ntype: str, root_dir: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        print(f"[PyGLoader] Loading {dataset_name}...")
        if dataset_name not in self.DATASET_MAPPING:
            raise ValueError(f"Unknown PyG dataset: {dataset_name}")
        
        path = os.path.join(root_dir, f'PyG_{dataset_name}')
        dataset = self.DATASET_MAPPING[dataset_name](root=path)
        g = dataset[0]
        
        GraphStandardizer.standardize(g)
        
        train_mask, val_mask, test_mask = self._create_random_masks(g[target_ntype].num_nodes)
        labels, num_classes = self._extract_labels(g, target_ntype)
        features = self._ensure_features(g, target_ntype)
        
        info = self.create_info_dict(g, target_ntype, features, labels, train_mask, val_mask, test_mask, num_classes)
        return g, info


class HNELoader(BaseGraphLoader):
    TYPE_MAPPINGS = {
        'DBLP': {0: 'author', 1: 'paper', 2: 'conf', 3: 'term'},
        'Yelp': {0: 'user', 1: 'business', 2: 'service', 3: 'level'},
        'PubMed': {0: 'gene', 1: 'disease', 2: 'chemical', 3: 'species'},
        'Freebase': {0: 'entity', 1: 'type', 2: 'relation'}
    }
    
    def load(self, dataset_name: str, target_ntype: str, root_dir: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        data_dir = os.path.join(root_dir, f'HNE_{dataset_name}')
        print(f"[HNELoader] Loading from {data_dir}...")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"HNE dataset directory not found: {data_dir}")
        
        g = self._load_nodes(data_dir, dataset_name)
        self._load_edges(data_dir, g)
        self._load_labels(data_dir, g)

        GraphStandardizer.standardize(g)

        features = self._ensure_features(g, target_ntype)
        labels, num_classes = self._extract_labels(g, target_ntype)
        train_mask, val_mask, test_mask = self._create_random_masks(g[target_ntype].num_nodes)

        info = self.create_info_dict(g, target_ntype, features, labels, train_mask, val_mask, test_mask, num_classes)
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
            # Parse comma-separated features from the info column if present
            if 'info' in group.columns:
                sample = group['info'].dropna()
                if len(sample) > 0 and isinstance(sample.iloc[0], str) and ',' in sample.iloc[0]:
                    try:
                        feats = torch.tensor(
                            group['info'].apply(lambda s: list(map(float, s.split(',')))).tolist(),
                            dtype=torch.float32
                        )
                        g[type_name].x = feats
                    except (ValueError, AttributeError):
                        pass
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
            # Temp relation name; finalized by GraphStandardizer
            rel_name = f"{src_type}_to_{dst_type}" 
            edge_index = torch.stack([
                torch.tensor(group['src_idx'].values, dtype=torch.long),
                torch.tensor(group['dst_idx'].values, dtype=torch.long)
            ], dim=0)
            g[src_type, rel_name, dst_type].edge_index = edge_index

    def _load_labels(self, data_dir: str, g: tg_data.HeteroData) -> None:
        """Parse label.dat and set g[type_name].y as multi-hot [N, C] or int64 [N]."""
        label_file = os.path.join(data_dir, "label.dat")
        if not os.path.exists(label_file):
            return
        df = pd.read_csv(label_file, sep='\t', header=None,
                         names=['id', 'name', 'type', 'label'], usecols=[0, 2, 3])
        for type_id, group in df.groupby('type'):
            if type_id not in {t for (t, _) in self._global_to_local.values()}:
                # resolve via global_to_local
                pass
            # map global IDs to local indices
            local_rows = []
            for glob_id, lbl in zip(group['id'].values, group['label'].values):
                if glob_id in self._global_to_local:
                    type_name, loc = self._global_to_local[glob_id]
                    local_rows.append((type_name, loc, int(lbl)))
            if not local_rows:
                continue
            type_name = local_rows[0][0]
            n_nodes = g[type_name].num_nodes
            classes = sorted({r[2] for r in local_rows})
            class_to_idx = {c: i for i, c in enumerate(classes)}
            # check if multi-label (same local idx appears with multiple classes)
            from collections import defaultdict
            node_labels: dict = defaultdict(set)
            for _, loc, lbl in local_rows:
                node_labels[loc].add(class_to_idx[lbl])
            n_classes = len(classes)
            if n_classes > 1 and any(len(v) > 1 for v in node_labels.values()):
                # multi-hot
                y = torch.zeros(n_nodes, n_classes, dtype=torch.float32)
                for loc, lbls in node_labels.items():
                    for c in lbls:
                        y[loc, c] = 1.0
            else:
                y = torch.full((n_nodes,), -1, dtype=torch.long)
                for loc, lbls in node_labels.items():
                    y[loc] = next(iter(lbls))
            g[type_name].y = y

class OAGLoader(BaseGraphLoader):
    def load(self, dataset_name: str, target_ntype: str, root_dir: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        if not _H2GB_AVAILABLE:
            raise ImportError("H2GB is not installed. Run: pip install H2GB")
        name = dataset_name.lower()  # 'cs', 'engineering', 'chemistry'
        print(f"[OAGLoader] Loading OAG-{name} (768-dim XLNet features, ~1.1M nodes)...")
        dataset_root = os.path.join(root_dir, 'OAG')
        dataset = _OAGDataset(root=dataset_root, name=name)
        g = dataset[0]

        GraphStandardizer.standardize(g)

        labels, num_classes = self._extract_labels(g, target_ntype)
        features = self._ensure_features(g, target_ntype)

        train_mask = g[target_ntype].get('train_mask', None)
        train_mask, val_mask, test_mask = self._create_random_masks(
            g[target_ntype].num_nodes, train_mask
        )

        info = self.create_info_dict(g, target_ntype, features, labels,
                                     train_mask, val_mask, test_mask, num_classes)
        return g, info


class RCDDLoader(BaseGraphLoader):
    def load(self, dataset_name: str, target_ntype: str, root_dir: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        print("[RCDDLoader] Loading RCDD (Alibaba risk-detection, 13.8M nodes)...")
        dataset_root = os.path.join(root_dir, 'RCDD')
        dataset = RCDD(root=dataset_root)
        g = dataset[0]

        GraphStandardizer.standardize(g)

        labels, num_classes = self._extract_labels(g, target_ntype)
        features = self._ensure_features(g, target_ntype)

        train_mask = g[target_ntype].get('train_mask', None)
        test_mask  = g[target_ntype].get('test_mask',  None)
        train_mask, val_mask, test_mask = self._create_random_masks(
            g[target_ntype].num_nodes, train_mask
        )

        info = self.create_info_dict(g, target_ntype, features, labels,
                                     train_mask, val_mask, test_mask, num_classes)
        return g, info


class MiniLoader(BaseGraphLoader):
    """Loads a pre-sampled mini graph saved by scripts/make_mini_datasets.py."""
    def load(self, dataset_name: str, target_ntype: str, root_dir: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        path = os.path.join(root_dir, f"{dataset_name}_mini", "data.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Mini dataset not found at {path}. "
                f"Run: python scripts/make_mini_datasets.py --datasets {dataset_name}"
            )
        print(f"[MiniLoader] Loading {dataset_name} mini sample from {path}")
        g = torch.load(path, weights_only=False)
        # Graph was already standardized when the mini was created — skip re-standardization.

        features = self._ensure_features(g, target_ntype)
        labels, num_classes = self._extract_labels(g, target_ntype)
        train_mask, val_mask, test_mask = self._create_random_masks(g[target_ntype].num_nodes)

        info = self.create_info_dict(g, target_ntype, features, labels,
                                     train_mask, val_mask, test_mask, num_classes)
        return g, info


class CustomLoader(HNELoader):
    """
    Loader for custom datasets (ACM, IMDB, Yelp, Freebase) provided externally.
    Inherits parsing logic from HNELoader but enforces strict schema mappings
    specific to this data source.
    """
    
    # Explicit schema mapping for the friend's datasets
    TYPE_MAPPINGS = {
        'ACM': {0: 'paper', 1: 'author', 2: 'subject'},
        'IMDB': {0: 'movie', 1: 'actor', 2: 'director'},
        'Yelp': {0: 'user', 1: 'business', 2: 'service', 3: 'level'},
        'Freebase': {0: 'entity', 1: 'type', 2: 'relation'}
    }

    def load(self, dataset_name: str, target_ntype: str, root_dir: str) -> Tuple[tg_data.HeteroData, Dict[str, Any]]:
        print(f"[CustomLoader] Loading {dataset_name}...")
        
        # Enforce directory structure: datasets/CUSTOM_<name>
        folder_name = f"CUSTOM_{dataset_name}"
        data_dir = os.path.join(root_dir, folder_name)
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Custom dataset not found at: {data_dir}\n"
                f"Expected format: node.dat, link.dat inside {folder_name}/"
            )

        # Reuse HNELoader's logic for parsing tab-separated files
        # We access the parent method directly to avoid recursion issues
        g = self._load_nodes(data_dir, dataset_name)
        self._load_edges(data_dir, g)

        GraphStandardizer.standardize(g)
        
        # Standard feature/label extraction (Template Method Pattern from BaseGraphLoader)
        features = self._ensure_features(g, target_ntype)
        labels, num_classes = self._extract_labels(g, target_ntype)
        train_mask, val_mask, test_mask = self._create_random_masks(g[target_ntype].num_nodes)
        
        info = self.create_info_dict(g, target_ntype, features, labels, 
                                   train_mask, val_mask, test_mask, num_classes)
        return g, info