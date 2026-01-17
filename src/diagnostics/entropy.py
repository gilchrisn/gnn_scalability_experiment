import torch
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch_geometric.utils import scatter
from ..config import config
from ..data import DatasetFactory
from ..backend import BackendFactory
from ..models import get_model
from ..utils import SchemaMatcher

class EntropyAnalyzer:
    def __init__(self, dataset, model_name, metapath, k):
        self.dataset = dataset
        self.model_name = model_name
        self.metapath = metapath
        self.k = k
        self.device = torch.device('cpu') # Force CPU for analysis

    def run(self):
        print(f"Analyzing Attention Entropy for K={self.k}...")
        
        # Load Data
        cfg = config.get_dataset_config(self.dataset)
        g_hetero, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        
        # Load Model
        model_path = config.get_model_path(self.dataset, self.model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        with open(model_path.replace('.pt', '_mapper.json'), 'r') as f:
            mapper_cfg = json.load(f)

        model = get_model(self.model_name, mapper_cfg['global_max_dim'], info['num_classes'], config.HIDDEN_DIM)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        # Materialize Graphs
        backend = BackendFactory.create('python', device=self.device)
        path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in self.metapath.split(',')]
        backend.initialize(g_hetero, path_list, info)
        
        g_exact = backend.materialize_exact()
        g_kmv = backend.materialize_kmv(k=self.k)
        
        # Run Analysis
        results = []
        graphs = [("Exact", g_exact), ("KMV", g_kmv)]
        
        for label, g in graphs:
            print(f" Processing {label}...")
            entropies = self._compute_entropy(model, g, mapper_cfg['global_max_dim'])
            test_mask = info['masks']['test'].cpu()
            test_entropies = entropies[test_mask]
            
            for e in test_entropies:
                results.append({"Condition": label, "Entropy": e.item()})
                
        self._plot(pd.DataFrame(results))
        backend.cleanup()

    def _compute_entropy(self, model, g, target_dim):
        # Feature padding
        x = g.x
        if x.size(1) != target_dim:
            x = torch.nn.functional.pad(x, (0, target_dim - x.size(1)))
            
        # Hook into GAT (assumes GAT model structure)
        if not hasattr(model, 'conv1'):
             raise ValueError("Model must be GAT with conv1 layer")
             
        with torch.no_grad():
             _, (att_edge_index, att_weights) = model.conv1(x, g.edge_index, return_attention_weights=True)
             
        # Calculate Entropy: -sum(p * log(p))
        att_mean = att_weights.mean(dim=1) # Avg over heads
        epsilon = 1e-10
        entropy_comp = -att_mean * torch.log(att_mean + epsilon)
        
        node_entropies = scatter(entropy_comp, att_edge_index[1], dim=0, dim_size=g.num_nodes, reduce='sum')
        return node_entropies

    def _plot(self, df):
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x="Entropy", hue="Condition", fill=True)
        plt.title(f"Attention Entropy Shift (K={self.k})")
        save_path = os.path.join(config.OUTPUT_DIR, f"entropy_{self.dataset}.png")
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")