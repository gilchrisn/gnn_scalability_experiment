import torch
import random
from typing import Dict, Any, List, Set
from torch_geometric.data import HeteroData
from .base import GraphSampler

class SnowballSampler(GraphSampler):
    """
    Implements Snowball Sampling (n-hop BFS traversal).
    Guarantees connectivity of the sampled graph.
    """

    def sample(self, g: HeteroData, config: Dict[str, Any]) -> HeteroData:
        seeds: int = config.get('seeds', 1000)
        hops: int = config.get('hops', 2)
        target_ntype: str = config.get('target_ntype')

        if not target_ntype:
            raise ValueError("Snowball sampling requires 'target_ntype' in config")

        print(f"[Sampler] Running Snowball (seeds={seeds}, hops={hops}, start={target_ntype})...")

        # 1. Initialize Sampling State
        sampled_nodes: Dict[str, Set[int]] = {nt: set() for nt in g.node_types}
        
        # 2. Select Seeds
        num_target = g[target_ntype].num_nodes
        seed_indices = random.sample(range(num_target), min(seeds, num_target))
        sampled_nodes[target_ntype].update(seed_indices)
        
        current_frontier: Dict[str, Set[int]] = {target_ntype: set(seed_indices)}

        # 3. BFS Expansion
        for h in range(hops):
            print(f"  -> Hop {h+1}...")
            next_frontier: Dict[str, Set[int]] = {nt: set() for nt in g.node_types}
            
            # Iterate over all edge types to find neighbors of current frontier
            for src_type, rel, dst_type in g.edge_types:
                # Check forward direction (src -> dst)
                if src_type in current_frontier and current_frontier[src_type]:
                    self._expand_frontier(
                        g, src_type, rel, dst_type, 
                        current_frontier[src_type], 
                        sampled_nodes[dst_type], 
                        next_frontier[dst_type]
                    )
                
                # Check reverse direction (dst -> src) 
                # (Graph is undirected for sampling purposes to ensure connectivity)
                if dst_type in current_frontier and current_frontier[dst_type]:
                    self._expand_frontier(
                        g, dst_type, rel, src_type, # Note: logical reversal
                        current_frontier[dst_type],
                        sampled_nodes[src_type], 
                        next_frontier[src_type],
                        reverse_edge=True
                    )
            
            current_frontier = next_frontier
        
        # 4. Materialize Subgraph
        # Convert sets to tensors for subgraph method
        final_nodes = {nt: torch.tensor(list(nodes)) for nt, nodes in sampled_nodes.items() if nodes}
        print(f"[Sampler] Subgraph nodes: {sum(len(n) for n in final_nodes.values())}")
        
        # Use PyG's built-in subgraph method which handles edge filtering efficiently
        return g.subgraph(final_nodes)

    def _expand_frontier(self, g: HeteroData, 
                        src_type: str, rel: str, dst_type: str,
                        frontier_nodes: Set[int], 
                        sampled_dst_set: Set[int],
                        next_frontier_dst_set: Set[int],
                        reverse_edge: bool = False):
        """Helper to find neighbors for a specific edge type."""
        edge_index = g[src_type, rel, dst_type].edge_index
        
        if reverse_edge:
            src_indices, dst_indices = edge_index[1], edge_index[0]
        else:
            src_indices, dst_indices = edge_index[0], edge_index[1]

        # Mask edges starting from frontier
        # Note: For massive graphs, this mask might be slow. 
        # A more optimized approach uses searchsorted if edges are sorted.
        mask = torch.isin(src_indices, torch.tensor(list(frontier_nodes)))
        found_neighbors = dst_indices[mask].tolist()

        for nbr in found_neighbors:
            if nbr not in sampled_dst_set:
                sampled_dst_set.add(nbr)
                next_frontier_dst_set.add(nbr)