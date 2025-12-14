import torch
import time
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from torch_sparse import spspmm 

def _materialize_graph(g_hetero, data_info, metapath, target_ntype):
    """
    Exact matrix multiplication to materialize a homogeneous graph.
    Uses data_info for features/labels to ensure consistency with the Factory.
    """
    start_time = time.perf_counter()
    
    # 1. Initialize with the first edge type
    edge_index_tuple = metapath[0]
    edge_index = g_hetero[edge_index_tuple].edge_index 
    values = torch.ones(edge_index.size(1), device=edge_index.device)
    
    num_src_nodes = g_hetero[edge_index_tuple[0]].num_nodes
    num_dst_nodes = g_hetero[edge_index_tuple[2]].num_nodes

    # 2. Matrix Multiplication Loop (Sparse x Sparse)
    for i in range(1, len(metapath)):
        next_edge_tuple = metapath[i]
        next_edge_index = g_hetero[next_edge_tuple].edge_index
        next_values = torch.ones(next_edge_index.size(1), device=next_edge_index.device)
        
        num_next_dst_nodes = g_hetero[next_edge_tuple[2]].num_nodes
        
        # spspmm: (N x M) @ (M x P) -> (N x P)
        edge_index, values = spspmm(
            edge_index, values, 
            next_edge_index, next_values,
            num_src_nodes, num_dst_nodes, num_next_dst_nodes
        )
        
        num_dst_nodes = num_next_dst_nodes
    
    num_nodes = g_hetero[target_ntype].num_nodes

    # 3. Post-Processing: Handle Empty Graphs
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    # 4. Symmetrize and Coalesce
    # We treat the resulting graph as undirected for GCN/GAT
    edge_index = pyg_utils.to_undirected(edge_index, num_nodes=num_nodes)
    edge_index = pyg_utils.coalesce(edge_index, num_nodes=num_nodes)
    
    # 5. Add Self Loops (Standard for GNNs)
    edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=num_nodes)
    
    # 6. Create Data Object
    g_homo = Data(
        x=data_info['features'],
        edge_index=edge_index,
        y=data_info['labels'],
        train_mask=data_info['masks']['train'],
        val_mask=data_info['masks']['val'],
        test_mask=data_info['masks']['test'],
        num_nodes=num_nodes
    )
    
    duration = time.perf_counter() - start_time
    return g_homo, duration