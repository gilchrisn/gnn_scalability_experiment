from turtle import pd
import pandas as pd
import torch
import os
from abc import ABC, abstractmethod
from torch_geometric.datasets import HGBDataset, OGB_MAG, DBLP, IMDB, AMiner
import torch_geometric.data as tg_data
import torch_geometric.transforms as T

class BaseGraphLoader(ABC):
    """
    The Contract: All dataset loaders must implement this.
    """
    @abstractmethod
    def load(self, dataset_name: str, target_ntype: str):
        pass

    def _create_random_masks(self, num_nodes, train_mask=None):
        """Helper: Creates split masks if the dataset doesn't provide them."""
        if train_mask is None:
            # Create fully random 60/20/20 split
            indices = torch.randperm(num_nodes)
            num_train = int(num_nodes * 0.6)
            num_val = int(num_nodes * 0.2)
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[indices[:num_train]] = True
            
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask[indices[num_train:num_train+num_val]] = True
            
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask[indices[num_train+num_val:]] = True
        else:
            # Split remaining nodes 50/50 val/test
            non_train_indices = (~train_mask).nonzero(as_tuple=False).view(-1)
            non_train_indices = non_train_indices[torch.randperm(non_train_indices.size(0))]
            
            num_val = non_train_indices.size(0) // 2
            val_indices = non_train_indices[:num_val]
            test_indices = non_train_indices[num_val:]
            
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask[val_indices] = True
            
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask[test_indices] = True
            
        return train_mask, val_mask, test_mask

    def _ensure_features(self, g, target_ntype):
        """
        Checks if the target node type has features ('x'). 
        If not, generates structural features (One-Hot or Random).
        """
        if hasattr(g[target_ntype], 'x') and g[target_ntype].x is not None:
            return g[target_ntype].x
        
        print(f"[Loader] Warning: Node type '{target_ntype}' has no features. Generating dummy features...")
        num_nodes = g[target_ntype].num_nodes
        
        # Strategy: Use a small random embedding if N is huge, else One-Hot
        # For efficiency on large graphs like Freebase, we avoid full One-Hot (N x N).
        # We'll use a fixed width random embedding (N x 64).
        dummy_dim = 64
        return torch.randn((num_nodes, dummy_dim))

class HGBLoader(BaseGraphLoader):
    """Loader for THUDM's HGB datasets"""
    def load(self, dataset_name: str, target_ntype: str):
        print(f"[Loader] Fetching HGB Dataset: {dataset_name}...")
        dataset = HGBDataset(root=f'./data/HGB_{dataset_name}', name=dataset_name)
        g = dataset[0]
        
        train_mask = g[target_ntype].train_mask
        train_mask, val_mask, test_mask = self._create_random_masks(
            g[target_ntype].num_nodes, train_mask
        )
        
        # Calculate num_classes
        if hasattr(g[target_ntype], 'y') and g[target_ntype].y is not None:
            num_classes = int(g[target_ntype].y.max()) + 1
            labels = g[target_ntype].y
        else:
            # Fallback for datasets without labels (Freebase often lacks downstream labels)
            print(f"[Loader] Warning: No labels found for '{target_ntype}'. Creating dummy labels.")
            num_classes = 2
            labels = torch.randint(0, 2, (g[target_ntype].num_nodes,))

        # Ensure features exist
        features = self._ensure_features(g, target_ntype)

        return g, {
            "features": features,
            "labels": labels,
            "masks": {
                'train': train_mask, 
                'val': val_mask, 
                'test': test_mask
            },
            "num_classes": num_classes,
            "in_dim": features.shape[1],
            "out_dim": num_classes
        }

class OGBLoader(BaseGraphLoader):
    """Loader for OGB datasets"""
    def load(self, dataset_name: str, target_ntype: str):
        print(f"[Loader] Fetching OGB Dataset: {dataset_name}...")
        dataset = OGB_MAG(root='./data/OGB', preprocess='metapath2vec')
        g = dataset[0]
        
        if not hasattr(g[target_ntype], 'train_mask'):
             raise ValueError(f"Target node {target_ntype} in {dataset_name} has no masks!")

        labels = g[target_ntype].y
        if labels.dim() > 1:
            labels = labels.view(-1)

        num_classes = dataset.num_classes
        features = g[target_ntype].x

        return g, {
            "features": features,
            "labels": labels,
            "masks": {
                'train': g[target_ntype].train_mask,
                'val': g[target_ntype].val_mask,
                'test': g[target_ntype].test_mask
            },
            "num_classes": num_classes,
            "in_dim": features.shape[1],
            "out_dim": num_classes
        }

class PyGStandardLoader(BaseGraphLoader):
    """Loader for standard PyG datasets"""
    def load(self, dataset_name: str, target_ntype: str):
        print(f"[Loader] Fetching PyG Standard Dataset: {dataset_name}...")
        path = f'./data/PyG_{dataset_name}'
        
        if dataset_name == 'DBLP':
            dataset = DBLP(root=path)
        elif dataset_name == 'IMDB':
            dataset = IMDB(root=path)
        elif dataset_name == 'AMiner':
            dataset = AMiner(root=path)
        else:
            raise ValueError(f"Unknown PyG Standard dataset: {dataset_name}")
            
        g = dataset[0]
        
        train_mask, val_mask, test_mask = self._create_random_masks(
            g[target_ntype].num_nodes
        )
        
        labels = g[target_ntype].y
        num_classes = int(labels.max()) + 1
        
        # Ensure features exist (Critical for AMiner)
        features = self._ensure_features(g, target_ntype)

        return g, {
            "features": features,
            "labels": labels,
            "masks": {
                'train': train_mask, 
                'val': val_mask, 
                'test': test_mask
            },
            "num_classes": num_classes,
            "in_dim": features.shape[1],
            "out_dim": num_classes
        }


class HNELoader(BaseGraphLoader):
    def load(self, dataset_name: str, target_ntype: str):
        data_dir = f'./data/HNE_{dataset_name}'
        print(f"[Loader] Loading HNE dataset from {data_dir}...")

        # 1. Load Nodes
        # Format: node_id \t node_name \t node_type_id \t node_info
        node_file = os.path.join(data_dir, "node.dat")
        df_nodes = pd.read_csv(node_file, sep='\t', header=None, names=['id', 'name', 'type', 'info'], encoding='utf-8', quoting=3)

        # Map HNE integer types to string names (Specific for DBLP)
        # HNE DBLP Schema: 0:Author, 1:Paper, 2:Conf, 3:Term
        type_map = {0: 'author', 1: 'paper', 2: 'conf', 3: 'term'}
        
        g = tg_data.HeteroData()
        global_to_local = {} # Map global_id -> (type_name, local_idx)

        # Process nodes per type
        for t_id, group in df_nodes.groupby('type'):
            t_name = type_map.get(t_id, f"t{t_id}")
            g[t_name].num_nodes = len(group)
            
            # Create mapping for edges later
            for local_idx, global_id in enumerate(group['id'].values):
                global_to_local[global_id] = (t_name, local_idx)

        # 2. Load Links
        # Format: src_id \t dst_id \t link_type \t weight
        link_file = os.path.join(data_dir, "link.dat")
        df_links = pd.read_csv(link_file, sep='\t', header=None, names=['src', 'dst', 'type', 'w'])

        # Group edges by (SrcType, DstType)
        edges_store = {} 

        for _, row in df_links.iterrows():
            if row['src'] not in global_to_local or row['dst'] not in global_to_local:
                continue
            
            src_t, src_id = global_to_local[row['src']]
            dst_t, dst_id = global_to_local[row['dst']]
            
            # Create a relation name, e.g., "author_to_paper"
            rel_name = f"{src_t}_to_{dst_t}"
            key = (src_t, rel_name, dst_t)
            
            if key not in edges_store:
                edges_store[key] = {'src': [], 'dst': []}
            
            edges_store[key]['src'].append(src_id)
            edges_store[key]['dst'].append(dst_id)

        # Assign to PyG Data
        for (src_t, rel, dst_t), lists in edges_store.items():
            edge_index = torch.tensor([lists['src'], lists['dst']], dtype=torch.long)
            g[src_t, rel, dst_t].edge_index = edge_index

        # 3. Generate Features (HNE DBLP has no features, so we make dummy ones)
        features = self._ensure_features(g, target_ntype)
        
        # 4. Create Dummy Labels (if label.dat is missing or complex)
        # You can expand this to read label.dat if needed
        labels = torch.randint(0, 2, (g[target_ntype].num_nodes,))
        
        train_mask, val_mask, test_mask = self._create_random_masks(g[target_ntype].num_nodes)

        return g, {
            "features": features,
            "labels": labels,
            "masks": {'train': train_mask, 'val': val_mask, 'test': test_mask},
            "in_dim": features.shape[1],
            "out_dim": 2
        }

class DatasetFactory:
    _loaders = {
        'HGB': HGBLoader,
        'OGB': OGBLoader,
        'PyG': PyGStandardLoader,
        'HNE': HNELoader, 
    }

    @staticmethod
    def get_data(source_type: str, dataset_name: str, target_ntype: str):
        if source_type not in DatasetFactory._loaders:
            raise ValueError(f"Unknown Source: {source_type}")
        return DatasetFactory._loaders[source_type]().load(dataset_name, target_ntype)
    