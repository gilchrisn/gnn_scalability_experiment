"""
K-Minimum Values (KMV) sketching kernel for approximate graph materialization.
Implements probabilistic graph sampling using min-hash sketches.
"""
import torch
import time
from typing import Tuple, List
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data, HeteroData


INFINITY = torch.iinfo(torch.int64).max


class KMVSketchingKernel:
    """
    Kernel for KMV-based graph approximation.
    Uses min-hash sketches to efficiently approximate metapath connectivity.
    """
    
    def __init__(self, k: int = 32, nk: int = 1, device: torch.device = None):
        """
        Args:
            k: Sketch size (number of minimum hashes to keep)
            nk: Expansion factor for sketch arrays
            device: Computation device
        """
        self.k = k
        self.nk = nk
        self.device = device or torch.device('cpu')
    
    def sketch_and_sample(self,
                         g_hetero: HeteroData,
                         metapath: List[Tuple[str, str, str]],
                         target_ntype: str,
                         features: torch.Tensor = None,
                         labels: torch.Tensor = None,
                         masks: dict = None) -> Tuple[Data, float, float]:
        """
        Creates an approximate graph using KMV sketching.
        
        Args:
            g_hetero: Input heterogeneous graph
            metapath: Metapath definition
            target_ntype: Target node type
            features: Node features
            labels: Node labels
            masks: Train/val/test masks
            
        Returns:
            Tuple of (sampled graph, propagation time, graph building time)
        """
        # Phase 1: Sketch propagation
        final_sketches, sorted_hashes, sort_indices, t_prop = self._propagate_sketches(
            g_hetero, metapath
        )
        
        # Phase 2: Graph construction from sketches
        num_nodes = g_hetero[target_ntype].num_nodes
        g_sampled, t_build = self._build_graph_from_sketches(
            final_sketches, sorted_hashes, sort_indices, num_nodes
        )
        
        # Attach attributes
        if features is not None:
            g_sampled.x = features
        if labels is not None:
            g_sampled.y = labels
        if masks is not None:
            g_sampled.train_mask = masks.get('train')
            g_sampled.val_mask = masks.get('val')
            g_sampled.test_mask = masks.get('test')
        
        return g_sampled, t_prop, t_build
    
    def _propagate_sketches(self,
                           g_hetero: HeteroData,
                           metapath: List[Tuple[str, str, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Propagates KMV sketches along the metapath.
        
        Returns:
            Tuple of (final_sketches, sorted_hashes, sort_indices, time_taken)
        """
        start_time = time.perf_counter()
        
        # Collect all node types in metapath
        all_ntypes = set()
        for src_type, _, dst_type in metapath:
            all_ntypes.add(src_type)
            all_ntypes.add(dst_type)
        
        start_ntype = metapath[0][0]
        end_ntype = metapath[-1][2]
        
        # Initialize random hashes for start nodes
        num_start_nodes = g_hetero[start_ntype].num_nodes
        all_hashes = torch.randint(
            0, 2**63 - 1, (num_start_nodes,),
            device=self.device, dtype=torch.int64
        )
        
        # Create hash lookup table
        sorted_hashes, sort_indices = torch.sort(all_hashes)
        
        # Initialize sketches for all node types
        for ntype in all_ntypes:
            num_nodes = g_hetero[ntype].num_nodes
            sketches = torch.full(
                (num_nodes, self.nk, self.k), INFINITY,
                dtype=torch.int64, device=self.device
            )
            
            # Start nodes get their own hash in first position
            if ntype == start_ntype:
                sketches[:, :, 0] = all_hashes.unsqueeze(1)
            
            g_hetero[ntype].sketch = sketches
        
        # Propagate along metapath
        for src_type, rel, dst_type in metapath:
            edge_index = g_hetero[src_type, rel, dst_type].edge_index
            src_sketches = g_hetero[src_type].sketch
            dst_sketches_own = g_hetero[dst_type].sketch
            
            new_dst_sketches = self._propagate_and_merge(
                edge_index, src_sketches, dst_sketches_own
            )
            g_hetero[dst_type].sketch = new_dst_sketches
        
        final_sketches = g_hetero[end_ntype].sketch
        
        # Cleanup temporary sketch attributes
        for ntype in all_ntypes:
            if hasattr(g_hetero[ntype], 'sketch'):
                delattr(g_hetero[ntype], 'sketch')
        
        duration = time.perf_counter() - start_time
        return final_sketches, sorted_hashes, sort_indices, duration
    
    def _propagate_and_merge(self,
                            edge_index: torch.Tensor,
                            src_sketches: torch.Tensor,
                            dst_sketches_own: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized KMV merge operation.
        
        Args:
            edge_index: Edge connectivity [2, num_edges]
            src_sketches: Source node sketches [num_src, nk, k]
            dst_sketches_own: Current destination sketches [num_dst, nk, k]
            
        Returns:
            Updated destination sketches [num_dst, nk, k]
        """
        num_dst_nodes = dst_sketches_own.shape[0]
        
        # Gather sketches from source nodes
        edge_sketches = src_sketches[edge_index[0]]
        
        # Flatten and concatenate own + incoming sketches
        own_vals = dst_sketches_own.flatten()
        own_indices = torch.arange(num_dst_nodes, device=self.device).repeat_interleave(self.nk * self.k)
        
        incoming_vals = edge_sketches.flatten()
        incoming_indices = edge_index[1].repeat_interleave(self.nk * self.k)
        
        all_vals = torch.cat([own_vals, incoming_vals])
        all_indices = torch.cat([own_indices, incoming_indices])
        
        # Filter out infinity values
        mask = all_vals != INFINITY
        all_vals = all_vals[mask]
        all_indices = all_indices[mask]
        
        if all_vals.numel() == 0:
            return torch.full((num_dst_nodes, self.nk, self.k), INFINITY, dtype=torch.int64, device=self.device)
        
        # Sort by (value, node)
        sort_idx_val = torch.argsort(all_vals)
        all_vals_sorted = all_vals[sort_idx_val]
        all_indices_sorted = all_indices[sort_idx_val]
        
        sort_idx_node = torch.argsort(all_indices_sorted, stable=True)
        final_vals = all_vals_sorted[sort_idx_node]
        final_indices = all_indices_sorted[sort_idx_node]
        
        # Deduplicate (value, node) pairs
        vals_shifted = torch.cat([torch.tensor([-1], device=self.device), final_vals[:-1]])
        node_shifted = torch.cat([torch.tensor([-1], device=self.device), final_indices[:-1]])
        is_distinct = (final_vals != vals_shifted) | (final_indices != node_shifted)
        
        distinct_vals = final_vals[is_distinct]
        distinct_indices = final_indices[is_distinct]
        
        # Select top-k per node
        unique_nodes, counts = torch.unique_consecutive(distinct_indices, return_counts=True)
        cumsum = torch.cumsum(counts, dim=0)
        
        # Vectorized rank calculation
        idx_in_group = torch.arange(len(distinct_indices), device=self.device) - torch.repeat_interleave(
            torch.cat([torch.tensor([0], device=self.device), cumsum[:-1]]),
            counts
        )
        
        mask_k = idx_in_group < self.k
        valid_vals = distinct_vals[mask_k]
        valid_nodes = distinct_indices[mask_k]
        valid_ranks = idx_in_group[mask_k]
        
        # Write output
        output = torch.full((num_dst_nodes, self.k), INFINITY, dtype=torch.int64, device=self.device)
        out_flat_idx = valid_nodes * self.k + valid_ranks
        output_flat = output.view(-1)
        output_flat[out_flat_idx] = valid_vals
        
        output = output.view(num_dst_nodes, 1, self.k)
        if self.nk > 1:
            output = output.repeat(1, self.nk, 1)
        
        return output
    
    def _build_graph_from_sketches(self,
                                   final_sketches: torch.Tensor,
                                   sorted_hashes: torch.Tensor,
                                   sort_indices: torch.Tensor,
                                   num_nodes: int) -> Tuple[Data, float]:
        """
        Constructs similarity graph from final sketches.
        
        Returns:
            Tuple of (graph, time_taken)
        """
        start_time = time.perf_counter()
        
        nk, k = final_sketches.shape[1], final_sketches.shape[2]
        
        # Create source node indices (repeated for each sketch element)
        src_nodes = torch.arange(num_nodes, device=self.device).repeat_interleave(nk * k)
        flat_sketches = final_sketches.flatten()
        
        # Filter valid hashes
        valid_mask = (flat_sketches != INFINITY)
        valid_hashes = flat_sketches[valid_mask]
        
        # Map hashes back to node IDs using binary search
        indices = torch.searchsorted(sorted_hashes, valid_hashes)
        indices = indices.clamp(max=len(sorted_hashes) - 1)
        
        # Verify exact matches
        found_hashes = sorted_hashes[indices]
        exact_match_mask = (found_hashes == valid_hashes)
        valid_indices = indices[exact_match_mask]
        
        # Map sorted indices back to original node IDs
        dst_nodes_valid = sort_indices[valid_indices]
        src_nodes_valid = src_nodes[valid_mask][exact_match_mask]
        
        # Filter self-loops
        valid_edge_mask = (src_nodes_valid != dst_nodes_valid)
        src_final = src_nodes_valid[valid_edge_mask]
        dst_final = dst_nodes_valid[valid_edge_mask]
        
        edge_index = torch.stack([src_final, dst_final])
        
        # Post-process: undirected, coalesce, self-loops
        edge_index = pyg_utils.to_undirected(edge_index, num_nodes=num_nodes)
        edge_index = pyg_utils.coalesce(edge_index, num_nodes=num_nodes)
        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=num_nodes)
        
        g_sampled = Data(edge_index=edge_index, num_nodes=num_nodes)
        
        duration = time.perf_counter() - start_time
        return g_sampled, duration


# Convenience functions for backward compatibility
def run_kmv_propagation(g_hetero, metapath, k, nk, device):
    """Wrapper for KMV propagation phase."""
    kernel = KMVSketchingKernel(k=k, nk=nk, device=device)
    return kernel._propagate_sketches(g_hetero, metapath)


def build_graph_from_sketches(sketches, sorted_hashes, sort_indices, num_nodes, device):
    """Wrapper for graph building phase."""
    kernel = KMVSketchingKernel(k=sketches.shape[2], device=device)
    return kernel._build_graph_from_sketches(sketches, sorted_hashes, sort_indices, num_nodes)