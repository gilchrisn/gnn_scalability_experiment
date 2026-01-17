import torch
import torch.nn.functional as F
from ..config import config
from ..data import DatasetFactory
from ..backend import BackendFactory
from ..utils import SchemaMatcher

class DistortionAnalyzer:
    def __init__(self, dataset, metapath, k):
        self.dataset = dataset
        self.metapath = metapath
        self.k = k
        self.device = torch.device('cpu')

    def run(self):
        print(f"Measuring Semantic Distortion (KL Divergence) for K={self.k}...")
        cfg = config.get_dataset_config(self.dataset)
        g_hetero, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        
        backend = BackendFactory.create('python', device=self.device)
        path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in self.metapath.split(',')]
        backend.initialize(g_hetero, path_list, info)
        
        g_exact = backend.materialize_exact()
        g_kmv = backend.materialize_kmv(k=self.k)
        
        num_classes = info['num_classes']
        labels = info['labels']
        
        p_dist = self._get_class_distribution(g_exact, labels, num_classes)
        q_dist = self._get_class_distribution(g_kmv, labels, num_classes)
        
        # KL Divergence
        epsilon = 1e-8
        kl = torch.sum(p_dist * torch.log((p_dist + epsilon) / (q_dist + epsilon)), dim=1)
        
        avg_kl = kl.mean().item()
        print(f"Average KL Divergence: {avg_kl:.6f}")
        backend.cleanup()

    def _get_class_distribution(self, g, labels, num_classes):
        src, dst = g.edge_index
        neighbor_labels = labels[src]
        
        valid_mask = neighbor_labels >= 0
        valid_dst = dst[valid_mask]
        valid_labels = neighbor_labels[valid_mask]
        
        one_hot = F.one_hot(valid_labels, num_classes=num_classes).float()
        
        dist = torch.zeros((g.num_nodes, num_classes), device=self.device)
        dist.index_add_(0, valid_dst, one_hot)
        
        # Normalize
        row_sum = dist.sum(dim=1, keepdim=True)
        return dist / (row_sum + 1e-9)