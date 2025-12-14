import torch
import time
import os
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import coalesce
from torch_geometric.data import Data

# 

INFINITY = torch.iinfo(torch.int64).max

def _pyg_propagate_and_merge(edge_index, src_sketches, dst_sketches_own, k, nk):
    """
    FULLY VECTORIZED KMV MERGE.
    """
    device = src_sketches.device
    num_dst_nodes = dst_sketches_own.shape[0]

    # 1. Gather sketches
    edge_sketches = src_sketches[edge_index[0]] 
    
    # 2. Flatten and Merge
    own_vals = dst_sketches_own.flatten()
    own_indices = torch.arange(num_dst_nodes, device=device).repeat_interleave(nk * k)
    
    incoming_vals = edge_sketches.flatten()
    incoming_indices = edge_index[1].repeat_interleave(nk * k)

    all_vals = torch.cat([own_vals, incoming_vals])
    all_indices = torch.cat([own_indices, incoming_indices])

    # 3. Filter Infinity
    mask = all_vals != INFINITY
    all_vals = all_vals[mask]
    all_indices = all_indices[mask]

    if all_vals.numel() == 0:
         return torch.full((num_dst_nodes, nk, k), INFINITY, dtype=torch.int64, device=device)

    # 4. Sort by (Value, Node)
    sort_idx_val = torch.argsort(all_vals)
    all_vals_sorted = all_vals[sort_idx_val]
    all_indices_sorted = all_indices[sort_idx_val]
    
    sort_idx_node = torch.argsort(all_indices_sorted, stable=True)
    final_vals = all_vals_sorted[sort_idx_node]
    final_indices = all_indices_sorted[sort_idx_node]

    # 5. Deduplicate (Value, Node) pairs
    vals_shifted = torch.cat([torch.tensor([-1], device=device), final_vals[:-1]])
    node_shifted = torch.cat([torch.tensor([-1], device=device), final_indices[:-1]])
    is_distinct = (final_vals != vals_shifted) | (final_indices != node_shifted)
    
    distinct_vals = final_vals[is_distinct]
    distinct_indices = final_indices[is_distinct]

    # 6. Select Top-K
    unique_nodes, counts = torch.unique_consecutive(distinct_indices, return_counts=True)
    cumsum = torch.cumsum(counts, dim=0)
    
    # Vectorized Rank Calculation
    idx_in_group = torch.arange(len(distinct_indices), device=device) - torch.repeat_interleave(
        torch.cat([torch.tensor([0], device=device), cumsum[:-1]]), 
        counts
    )
    
    mask_k = idx_in_group < k
    valid_vals = distinct_vals[mask_k]
    valid_nodes = distinct_indices[mask_k]
    valid_ranks = idx_in_group[mask_k]
    
    # 7. Write Output
    output = torch.full((num_dst_nodes, k), INFINITY, dtype=torch.int64, device=device)
    out_flat_idx = valid_nodes * k + valid_ranks
    
    output_flat = output.view(-1)
    output_flat[out_flat_idx] = valid_vals
    
    output = output.view(num_dst_nodes, 1, k)
    if nk > 1:
        output = output.repeat(1, nk, 1)
        
    return output


def _run_kmv_propagation(g_hetero, metapath, k, nk, device):
    """
    Iterative KMV Propagation.
    """
    # print("  Starting KMV propagation...")
    start_time = time.time()
    
    all_ntypes = set()
    for step in metapath:
        all_ntypes.add(step[0])
        all_ntypes.add(step[2])
        
    start_ntype = metapath[0][0]
    end_ntype = metapath[-1][-1]
    
    N_nodes = g_hetero[start_ntype].num_nodes
    
    # 1. Initialize Hashes
    all_hashes = torch.randint(0, 2**63 - 1, (N_nodes,), 
                               device=device, dtype=torch.int64)
    
    # 2. Prepare Lookup (Hash -> NodeID)
    sorted_hashes, sort_indices = torch.sort(all_hashes)
    
    # Note: We do NOT need reverse_map here anymore. 
    # sort_indices maps [Rank -> NodeID] which is exactly what we need.
    
    for ntype in all_ntypes:
        num_nodes = g_hetero[ntype].num_nodes
        sketches = torch.full((num_nodes, nk, k), INFINITY, 
                             dtype=torch.int64, device=device)
        
        if ntype == start_ntype:
            sketches[:, :, 0] = all_hashes.unsqueeze(1)
            
        g_hetero[ntype].sketch = sketches

    # 3. Propagate
    for step_edges in metapath:
        src_ntype, rel, dst_ntype = step_edges
        edge_index = g_hetero[src_ntype, rel, dst_ntype].edge_index
        src_sketches = g_hetero[src_ntype].sketch
        dst_sketches_own = g_hetero[dst_ntype].sketch
        
        new_dst_sketches = _pyg_propagate_and_merge(
            edge_index, src_sketches, dst_sketches_own, k, nk
        )
        g_hetero[dst_ntype].sketch = new_dst_sketches
    
    final_sketches = g_hetero[end_ntype].sketch
    
    # Cleanup
    for ntype in all_ntypes:
        if 'sketch' in g_hetero[ntype]:
             del g_hetero[ntype].sketch
    
    duration = time.time() - start_time
    # Passing 'sort_indices' instead of 'reverse_map'
    return final_sketches, sorted_hashes, sort_indices, duration


def _build_graph_from_sketches(final_sketches, sorted_hashes, sort_indices, 
                               N_nodes, device):
    """
    Constructs the similarity graph from sketches.
    FIXED: Uses sort_indices (Rank->Node) instead of reverse_map.
    """
    # print(f"  Building graph from sketches (K={final_sketches.shape[2]})...")
    start_time = time.time()
    
    nk, k = final_sketches.shape[1], final_sketches.shape[2]
    src_nodes = torch.arange(N_nodes, device=device).repeat_interleave(nk * k)
    flat_sketches = final_sketches.flatten()
    
    # 1. Valid Hashes Only
    valid_hashes_mask = (flat_sketches != INFINITY)
    valid_hashes = flat_sketches[valid_hashes_mask]
    
    # 2. Map Hashes back to Nodes
    # searchsorted returns the Rank (index in sorted array)
    indices = torch.searchsorted(sorted_hashes, valid_hashes)
    indices = indices.clamp(max=len(sorted_hashes) - 1)
    
    # Verify exact match (filter false positives)
    found_hashes = sorted_hashes[indices]
    exact_match_mask = (found_hashes == valid_hashes)
    
    valid_indices = indices[exact_match_mask]
    
    # CRITICAL FIX: Use sort_indices to map Rank -> NodeID
    dst_nodes_valid = sort_indices[valid_indices]
    
    # Filter source nodes to match
    src_nodes_valid = src_nodes[valid_hashes_mask][exact_match_mask]

    # 3. Filter self-loops immediately
    valid_edge_mask = (src_nodes_valid != dst_nodes_valid)
    src_final = src_nodes_valid[valid_edge_mask]
    dst_final = dst_nodes_valid[valid_edge_mask]
    
    edge_index = torch.stack([src_final, dst_final])
    
    # 4. Force Undirected & Coalesce
    edge_index = pyg_utils.to_undirected(edge_index, num_nodes=N_nodes)
    edge_index = coalesce(edge_index, num_nodes=N_nodes)
    
    # 5. Add Self Loops (Standard GNN Requirement)
    edge_index_final, _ = pyg_utils.add_self_loops(edge_index, num_nodes=N_nodes)
    
    g_sampled = Data(edge_index=edge_index_final, num_nodes=N_nodes)
    
    duration = time.time() - start_time
    return g_sampled, duration