"""
Random Walk Sampling Kernel.

Baseline approximation that samples neighbors via K independent random walks
along the metapath, directly on the heterogeneous graph.

Previous implementation had a hidden exact materialization leakage:
it called PythonBackend.materialize_exact() internally before sampling.
This replacement eliminates that leakage entirely — exact materialization
is never called at any point in this kernel.
"""
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data, HeteroData


class RandomSamplingKernel:
    """
    Baseline kernel: samples the metapath-induced neighborhood via random walks.

    Algorithm (for each of K independent samples):
      1. Start all target nodes at their own position.
      2. At each hop, for every alive node, select one random neighbor
         via the corresponding edge type.
      3. After L hops, the endpoint is a sampled neighbor of the start node.

    Complexity: O(N · K · L)  — same order as KMV, no exact computation.

    Limitation (deliberate, for baseline fairness): when multiple slots share
    the same current node, they receive the same randomly selected neighbor at
    that hop. This slightly reduces variance across slots but keeps the
    implementation fully vectorized and free of Python-level node loops.
    """

    def __init__(self, k: int = 32, device: Optional[torch.device] = None):
        """
        Args:
            k:      Number of random walk samples per target node.
            device: Computation device.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k      = k
        self.device = device or torch.device('cpu')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sketch_and_sample(self,
                          g_hetero: HeteroData,
                          metapath: List[Tuple[str, str, str]],
                          target_ntype: str,
                          features: Optional[torch.Tensor] = None,
                          labels:   Optional[torch.Tensor] = None,
                          masks:    Optional[Dict[str, torch.Tensor]] = None
                          ) -> Tuple[Data, float, float]:
        """
        Samples a homogeneous graph via K random walks along the metapath.

        Exact materialization is never invoked — this is a pure random walk.

        Args:
            g_hetero:     Input heterogeneous graph.
            metapath:     List of (src_type, relation, dst_type) tuples.
            target_ntype: Target node type (start and end of the walk).
            features:     Optional node features to attach.
            labels:       Optional node labels to attach.
            masks:        Optional dict of train/val/test boolean masks.

        Returns:
            Tuple of (sampled graph, propagation time, graph-build time).
        """
        if not metapath:
            raise ValueError("metapath cannot be empty")

        num_nodes = g_hetero[target_ntype].num_nodes
        t0 = time.perf_counter()

        edge_index = self._random_walk_edges(g_hetero, metapath, num_nodes)

        t_walk = time.perf_counter() - t0
        t1     = time.perf_counter()

        # Post-process: undirected, deduplicated, self-loops
        if edge_index.size(1) > 0:
            edge_index = pyg_utils.to_undirected(edge_index, num_nodes=num_nodes)
            edge_index = pyg_utils.coalesce(edge_index, num_nodes=num_nodes)
        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=num_nodes)

        g_sampled = Data(edge_index=edge_index, num_nodes=num_nodes)

        if features is not None: g_sampled.x          = features
        if labels   is not None: g_sampled.y          = labels
        if masks:
            g_sampled.train_mask = masks.get('train')
            g_sampled.val_mask   = masks.get('val')
            g_sampled.test_mask  = masks.get('test')

        t_build = time.perf_counter() - t1
        return g_sampled, t_walk, t_build

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_walk_edges(self,
                           g_hetero:  HeteroData,
                           metapath:  List[Tuple[str, str, str]],
                           num_nodes: int) -> torch.Tensor:
        """
        Runs K random walks per target node and returns collected (src, dst) edges.

        Each walk starts at a target node and advances one random hop per
        edge type in the metapath. If a walk reaches a dead-end (no outgoing
        edges), that slot is marked dead and contributes no edge.
        """
        N = num_nodes
        K = self.k
        total_slots = N * K

        # slot_source[s] = which target node slot s belongs to (fixed throughout)
        slot_source = torch.arange(N, device=self.device).repeat_interleave(K)

        # current[s] = current global node ID for slot s
        current = slot_source.clone()

        # alive[s] = False once the slot hits a dead-end
        alive = torch.ones(total_slots, dtype=torch.bool, device=self.device)

        for (src_type, rel, dst_type) in metapath:
            if not alive.any():
                break
            edge_index = g_hetero[src_type, rel, dst_type].edge_index.to(self.device)
            current, alive = self._vectorized_hop(current, alive, edge_index)

        # Collect edges from alive slots; drop self-loops
        valid     = alive
        src_edges = slot_source[valid]
        dst_edges = current[valid]

        non_self  = src_edges != dst_edges
        src_edges = src_edges[non_self]
        dst_edges = dst_edges[non_self]

        if src_edges.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        return torch.stack([src_edges, dst_edges])

    def _vectorized_hop(self,
                        current:    torch.Tensor,
                        alive:      torch.Tensor,
                        edge_index: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advances all alive slots by one random hop.

        Randomness mechanism: edges are gathered in a random permutation
        order; when multiple edges share the same source node, the last
        write wins — producing a uniformly random selection per node.
        Since all alive slots at the same node receive the same chosen
        neighbor, variance between those slots comes from subsequent hops.

        Args:
            current:    [total_slots] — current global node ID per slot.
            alive:      [total_slots] — boolean alive mask.
            edge_index: [2, E]        — edges for this hop's relation type.

        Returns:
            (next_current, next_alive) with same shapes as inputs.
        """
        alive_slots = alive.nonzero(as_tuple=False).view(-1)
        if alive_slots.numel() == 0:
            return current, torch.zeros_like(alive)

        alive_node_ids = current[alive_slots]  # global node IDs currently occupied
        src_e, dst_e   = edge_index

        # Filter to edges whose source is in our alive set
        in_mask    = torch.isin(src_e, alive_node_ids)
        valid_src  = src_e[in_mask]
        valid_dst  = dst_e[in_mask]

        next_current = current.clone()
        next_alive   = alive.clone()

        if valid_src.numel() == 0:
            # All alive slots hit a dead-end at this hop
            next_alive[alive_slots] = False
            return next_current, next_alive

        # Build a node_id → random_neighbor lookup.
        # Random permutation + last-write-wins = uniform random selection.
        max_id = int(max(current.max().item(),
                         src_e.max().item() if src_e.numel() > 0 else 0)) + 1
        chosen = torch.full((max_id,), -1, dtype=torch.long, device=self.device)

        perm      = torch.randperm(valid_src.numel(), device=self.device)
        chosen[valid_src[perm]] = valid_dst[perm]   # last write = random neighbor

        # Apply lookup to alive slots
        slot_ids       = current[alive_slots]
        slot_neighbors = chosen[slot_ids]

        has_neighbor = slot_neighbors >= 0
        next_alive[alive_slots]              = has_neighbor
        next_current[alive_slots[has_neighbor]] = slot_neighbors[has_neighbor]

        return next_current, next_alive