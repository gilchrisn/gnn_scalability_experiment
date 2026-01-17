import torch
import pandas as pd
from torch_geometric.utils import degree
from ..config import config
from ..data import DatasetFactory
from ..backend import BackendFactory
from ..utils import SchemaMatcher

class RankAnalyzer:
    def __init__(self, dataset, metapath):
        self.dataset = dataset
        self.metapath = metapath
        self.device = torch.device('cpu')

    def run(self):
        print("Computing Effective Rank of Neighborhoods...")
        cfg = config.get_dataset_config(self.dataset)
        g_hetero, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        
        backend = BackendFactory.create('python', device=self.device)
        path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in self.metapath.split(',')]
        backend.initialize(g_hetero, path_list, info)
        
        g_exact = backend.materialize_exact()
        
        # Identify Hubs (Top 200)
        row, col = g_exact.edge_index
        degs = degree(row, num_nodes=g_exact.num_nodes)
        _, top_indices = torch.topk(degs, k=200)
        
        features = info['features']
        ranks = []
        
        for hub_idx in top_indices:
            neighbor_indices = col[row == hub_idx]
            if len(neighbor_indices) < 5: continue
            
            neighbor_feats = features[neighbor_indices]
            rank = self._stable_rank(neighbor_feats)
            ranks.append(rank)
            
        avg_rank = sum(ranks) / len(ranks)
        print(f"Average Stable Rank (Top 200 Hubs): {avg_rank:.4f}")
        backend.cleanup()

    def _stable_rank(self, matrix):
        # Center data
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        try:
            _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
            frob_sq = (S ** 2).sum()
            spec_sq = S[0] ** 2
            return (frob_sq / spec_sq).item() if spec_sq > 0 else 0.0
        except:
            return 0.0