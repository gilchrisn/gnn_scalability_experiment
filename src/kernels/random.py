import torch
import time
import torch_geometric.utils as pyg_utils
from typing import Tuple, List, Optional, Dict, Any  # Added 'Any'
from torch_geometric.data import Data, HeteroData

class RandomSamplingKernel:
    """
    Baseline Kernel: Approximates graph via uniform random sampling.
    
    Used to prove that KMV (structural sampling) outperforms blind 
    random sampling, particularly in low-homophily regions.
    """

    def __init__(self, k: int = 32, device: torch.device = None):
        self.k = k
        self.device = device or torch.device('cpu')

    def sketch_and_sample(self,
                          g_hetero: HeteroData,
                          metapath: List[Tuple[str, str, str]],
                          target_ntype: str,
                          features: Optional[torch.Tensor] = None,
                          labels: Optional[torch.Tensor] = None,
                          masks: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[Data, float, float]:
        """
        Performs random neighbor selection along the metapath.
        """
        start_time = time.perf_counter()

        # 1. Dependency Injection: Initialize a transient Backend
        # We need the Exact graph structure to sample from.
        # Ideally, this kernel would be injected with a backend, but instantiating
        # a lightweight PythonBackend here is acceptable for this baseline logic.
        from ..backend import PythonBackend
        temp_backend = PythonBackend(self.device)
        
        # --- FIX: Construct Metadata Contract ---
        # The Backend expects a standardized dictionary containing features/labels.
        # We explicitly map the arguments to this contract.
        temp_info: Dict[str, Any] = {
            'features': features,
            'labels': labels,
            'masks': masks
        }
        
        # Initialize backend with valid state (Contract fulfillment)
        temp_backend.initialize(g_hetero, metapath, temp_info) 
        
        # Materialize edges (Delegation to Backend)
        g_exact: Data = temp_backend.materialize_exact()
        
        # 2. Random Sampling Logic
        row, col = g_exact.edge_index
        num_nodes = g_exact.num_nodes
        
        # Get degrees
        deg = pyg_utils.degree(row, num_nodes=num_nodes, dtype=torch.long)
        
        # Identify nodes that need sampling (degree > k)
        mask_high_deg = deg > self.k
        nodes_to_sample = mask_high_deg.nonzero(as_tuple=False).view(-1)
        
        keep_edges_list: List[torch.Tensor] = []
        
        # Add edges for low-degree nodes (keep all)
        mask_keep = ~mask_high_deg[row]
        keep_edges_list.append(g_exact.edge_index[:, mask_keep])
        
        # Sample for high-degree nodes
        for u_tensor in nodes_to_sample:
            u = u_tensor.item()
            # Find neighbors
            mask_u = (row == u)
            neighbors = col[mask_u]
            
            # Random selection
            perm = torch.randperm(neighbors.size(0))[:self.k]
            selected_neighbors = neighbors[perm]
            
            # Create edge tensor
            u_repeated = torch.full((self.k,), u, dtype=torch.long, device=self.device)
            keep_edges_list.append(torch.stack([u_repeated, selected_neighbors]))

        # Reconstruct Graph
        final_edge_index = torch.cat(keep_edges_list, dim=1)
        final_edge_index = pyg_utils.to_undirected(final_edge_index)
        
        duration = time.perf_counter() - start_time
        
        g_sampled = Data(edge_index=final_edge_index, num_nodes=num_nodes)
        
        # Attach Metadata (Forwarding context)
        if features is not None: g_sampled.x = features
        if labels is not None: g_sampled.y = labels
        if masks:
            g_sampled.train_mask = masks.get('train')
            g_sampled.val_mask = masks.get('val')
            g_sampled.test_mask = masks.get('test')

        return g_sampled, 0.0, duration