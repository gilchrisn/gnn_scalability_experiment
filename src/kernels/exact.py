"""
Exact graph materialization kernel using sparse matrix multiplication.
Implements the exact metapath-based graph transformation.
"""
import torch
import time
from typing import Tuple, List
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data, HeteroData
from torch_sparse import spspmm


class ExactMaterializationKernel:
    """
    Kernel for exact graph materialization via sparse matrix multiplication.
    Implements the Strategy pattern for graph transformation.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Args:
            device: PyTorch device for computations (defaults to CPU)
        """
        self.device = device or torch.device('cpu')
    
    def materialize(self,
                   g_hetero: HeteroData,
                   metapath: List[Tuple[str, str, str]],
                   target_ntype: str,
                   features: torch.Tensor = None,
                   labels: torch.Tensor = None,
                   masks: dict = None) -> Tuple[Data, float]:
        """
        Materializes a homogeneous graph from a metapath via exact multiplication.
        
        Args:
            g_hetero: Input heterogeneous graph
            metapath: List of edge type tuples defining the path
            target_ntype: Target node type for the resulting graph
            features: Optional node features (will be fetched if None)
            labels: Optional node labels
            masks: Optional train/val/test masks
            
        Returns:
            Tuple of (homogeneous graph, computation time)
        """
        start_time = time.perf_counter()
        
        # Validate metapath
        if not metapath:
            raise ValueError("Metapath cannot be empty")
        
        # Initialize with first edge type
        edge_index = self._get_edge_index(g_hetero, metapath[0])
        values = torch.ones(edge_index.size(1), device=edge_index.device)
        
        num_src_nodes = g_hetero[metapath[0][0]].num_nodes
        num_dst_nodes = g_hetero[metapath[0][2]].num_nodes
        
        # Iterative sparse matrix multiplication
        for i in range(1, len(metapath)):
            edge_index, values, num_dst_nodes = self._multiply_step(
                g_hetero, metapath[i], edge_index, values, num_src_nodes, num_dst_nodes
            )
        
        # Post-process: symmetrize, coalesce, add self-loops
        num_nodes = g_hetero[target_ntype].num_nodes
        edge_index = self._postprocess_edges(edge_index, num_nodes)
        
        # Create homogeneous graph
        g_homo = self._create_graph_object(
            edge_index, num_nodes, features, labels, masks
        )
        
        duration = time.perf_counter() - start_time
        return g_homo, duration
    
    def _get_edge_index(self, g: HeteroData, edge_tuple: Tuple[str, str, str]) -> torch.Tensor:
        """Safely extract edge index from heterogeneous graph."""
        if edge_tuple not in g.edge_types:
            available = [f"({s},{r},{d})" for s, r, d in g.edge_types]
            raise ValueError(
                f"Edge type {edge_tuple} not found in graph. "
                f"Available: {', '.join(available[:5])}"
            )
        return g[edge_tuple].edge_index
    
    def _multiply_step(self,
                      g: HeteroData,
                      edge_tuple: Tuple[str, str, str],
                      prev_edges: torch.Tensor,
                      prev_values: torch.Tensor,
                      num_src: int,
                      num_mid: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Single matrix multiplication step: (N x M) @ (M x P) -> (N x P)
        """
        next_edges = self._get_edge_index(g, edge_tuple)
        next_values = torch.ones(next_edges.size(1), device=next_edges.device)
        num_dst = g[edge_tuple[2]].num_nodes
        
        result_edges, result_values = spspmm(
            prev_edges, prev_values,
            next_edges, next_values,
            num_src, num_mid, num_dst
        )
        
        return result_edges, result_values, num_dst
    
    def _postprocess_edges(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Applies standard GNN preprocessing:
        1. Handle empty graphs
        2. Convert to undirected
        3. Remove duplicates (coalesce)
        4. Add self-loops
        """
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        
        edge_index = pyg_utils.to_undirected(edge_index, num_nodes=num_nodes)
        edge_index = pyg_utils.coalesce(edge_index, num_nodes=num_nodes)
        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=num_nodes)
        
        return edge_index
    
    def _create_graph_object(self,
                            edge_index: torch.Tensor,
                            num_nodes: int,
                            features: torch.Tensor = None,
                            labels: torch.Tensor = None,
                            masks: dict = None) -> Data:
        """Creates PyG Data object with optional attributes."""
        data = Data(edge_index=edge_index, num_nodes=num_nodes)
        
        if features is not None:
            data.x = features
        if labels is not None:
            data.y = labels
        if masks is not None:
            data.train_mask = masks.get('train')
            data.val_mask = masks.get('val')
            data.test_mask = masks.get('test')
        
        return data


# Convenience function for backward compatibility
def materialize_graph(g_hetero: HeteroData,
                     data_info: dict,
                     metapath: List[Tuple[str, str, str]],
                     target_ntype: str) -> Tuple[Data, float]:
    """
    Convenience wrapper for exact materialization.
    
    Args:
        g_hetero: Heterogeneous graph
        data_info: Dictionary with features, labels, masks
        metapath: Metapath definition
        target_ntype: Target node type
        
    Returns:
        Tuple of (materialized graph, time taken)
    """
    kernel = ExactMaterializationKernel()
    return kernel.materialize(
        g_hetero, metapath, target_ntype,
        features=data_info.get('features'),
        labels=data_info.get('labels'),
        masks=data_info.get('masks')
    )