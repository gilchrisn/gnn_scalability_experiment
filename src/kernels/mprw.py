"""
Metapath-based Personalized Random Walk (MPRW) materialization kernel.

Algorithm
---------
For each target node u, launch k independent random walkers.  Each walker
starts at u and follows the metapath edge sequence step-by-step, selecting
a uniformly-random neighbor at each hop.  The terminal nodes of all walkers
from u become candidate neighbors of u in the materialized graph H̃.

An edge (u, v) is created if v was the terminal node of ≥ min_visits walks
from u.  The default (min_visits=1) maximises recall; raising it filters
low-confidence edges and reduces graph density.

Design notes
------------
* Fully vectorised — all N_target × k walkers advance in parallel per hop.
  No Python loop over individual nodes.
* CPU-only — intentionally mirrors exp3 inference which is also CPU-only for
  fair timing.
* Memory: O(N_target × k) walker state + O(nnz of each edge type).
* Walkers that reach a node with no forward neighbours (degree-0 dead-end)
  are marked invalid and excluded from the output.

Comparison to KMV
-----------------
Both KMV and MPRW are parameterised by k:
  KMV   — k min-hash values per node; larger k → smaller sketch error
  MPRW  — k random walks per node;    larger k → higher edge recall

The methods are evaluated on the same frozen SAGE model with the same
edge-count / RAM budget, making the comparison directly meaningful.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import torch
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data, HeteroData


class MPRWKernel:
    """
    Metapath random-walk materialization kernel.

    Args:
        k:           Number of random walks launched per source node.
        seed:        RNG seed for reproducibility.
        min_visits:  Minimum number of walks that must reach v for edge (u,v)
                     to be included.  Default 1.
        device:      Torch device.  Should remain CPU for fair benchmarking.
    """

    def __init__(
        self,
        k: int,
        seed: int = 42,
        min_visits: int = 1,
        device: Optional[torch.device] = None,
    ):
        if k < 1:
            raise ValueError(f"k must be ≥ 1, got {k}")
        if min_visits < 1:
            raise ValueError(f"min_visits must be ≥ 1, got {min_visits}")
        self.k = k
        self.seed = seed
        self.min_visits = min_visits
        self.device = device or torch.device("cpu")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def peak_mb_estimate(self, n_target: int) -> float:
        """
        Analytical upper bound on peak tensor memory (MB) for one materialize() call.

        Dominant allocations (all [N_target × k] tensors):
          source_ids, current          — int64, 8 bytes each
          alive                        — bool,  1 byte
          Per walk step (reused):
            walker_counts, rand, safe_counts,
            choice, walker_offsets, next_nodes  — int64, 8 bytes each  (×6)
          _build_edge_index worst case:
            src, dst arrays            — int64, 8 bytes each            (×2)

        Total = (2 + 0.125 + 6 + 2) × N × k × 8 bytes
              ≈ 10 × N × k × 8 bytes  (conservative; walk-step tensors are reused)

        CSR tensors (sorted_dst, offsets, counts) scale with nnz, not N×k,
        and are built once and reused across k values in exp3 — excluded here.
        """
        bytes_per_walker = (
            8   # source_ids int64
            + 8   # current int64
            + 1   # alive bool
            + 8 * 6  # _walk_step intermediates (walker_counts, rand, safe_counts,
                     #                            choice, walker_offsets, next_nodes)
            + 8 * 2  # _build_edge_index src/dst worst case
        )
        return (n_target * self.k * bytes_per_walker) / (1024 ** 2)

    def materialize(
        self,
        g_hetero: HeteroData,
        metapath_triples: List[Tuple[str, str, str]],
        target_ntype: str,
    ) -> Tuple[Data, float]:
        """
        Materialise a homogeneous graph over *target_ntype* nodes using MPRW.

        Args:
            g_hetero:         Loaded PyG HeteroData.
            metapath_triples: Ordered list of (src_type, edge_name, dst_type).
                              The first src_type and last dst_type must both
                              equal target_ntype for a symmetric metapath.
            target_ntype:     Node type of source/target nodes.

        Returns:
            (Data, elapsed_seconds)
            Data has .edge_index (undirected, coalesced, with self-loops)
            and .num_nodes == g_hetero[target_ntype].num_nodes.
        """
        if not metapath_triples:
            raise ValueError("metapath_triples is empty")

        n_target = g_hetero[target_ntype].num_nodes
        rng = torch.Generator(device=self.device)
        rng.manual_seed(self.seed)

        t0 = time.perf_counter()

        # Pre-build CSR for every edge type in the metapath (done once).
        csr_cache: dict = {}
        for (src_t, edge_name, dst_t) in metapath_triples:
            key = (src_t, edge_name, dst_t)
            if key not in csr_cache:
                ei = g_hetero[src_t, edge_name, dst_t].edge_index.to(self.device)
                n_src = g_hetero[src_t].num_nodes
                csr_cache[key] = self._build_csr(ei, n_src)

        # Initialise walkers: each target node spawns k walkers.
        # source_ids: [n_target * k]  — constant, used at the end for edges.
        # current:    [n_target * k]  — walker positions, updated each hop.
        source_ids = torch.arange(n_target, device=self.device).repeat_interleave(self.k)
        current    = source_ids.clone()
        alive      = torch.ones(n_target * self.k, dtype=torch.bool, device=self.device)

        for (src_t, edge_name, dst_t) in metapath_triples:
            if alive.sum() == 0:
                break
            sorted_dst, offsets, counts = csr_cache[(src_t, edge_name, dst_t)]
            current, alive = self._walk_step(
                current, alive, sorted_dst, offsets, counts, rng
            )

        edge_index = self._build_edge_index(source_ids, current, alive, n_target)
        elapsed = time.perf_counter() - t0

        return Data(edge_index=edge_index, num_nodes=n_target), elapsed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_csr(
        self,
        edge_index: torch.Tensor,
        n_src: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert edge_index to CSR-style tensors for O(1) neighbour lookup.

        Returns:
            sorted_dst: [nnz]      destination nodes sorted by source ID
            offsets:    [n_src+1]  offsets[i] = start of node i's neighbours
            counts:     [n_src]    counts[i]  = degree of node i
        """
        src = edge_index[0]
        dst = edge_index[1]

        perm       = torch.argsort(src, stable=True)
        sorted_dst = dst[perm]

        counts  = torch.zeros(n_src, dtype=torch.long, device=self.device)
        counts.scatter_add_(0, src, torch.ones_like(src))
        offsets = torch.zeros(n_src + 1, dtype=torch.long, device=self.device)
        offsets[1:] = counts.cumsum(0)

        return sorted_dst, offsets, counts

    def _walk_step(
        self,
        current: torch.Tensor,
        alive: torch.Tensor,
        sorted_dst: torch.Tensor,
        offsets: torch.Tensor,
        counts: torch.Tensor,
        rng: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advance every alive walker by one hop.

        Walkers at degree-0 nodes are marked dead and excluded from
        the final edge set.

        Args:
            current: [n_walkers]  current node IDs for each walker
            alive:   [n_walkers]  bool mask of walkers still active
            sorted_dst / offsets / counts: CSR for this edge type

        Returns:
            (new_current, new_alive)
        """
        # Clamp dead walkers' positions to a valid index so tensor lookups don't
        # go out of bounds.  Dead walkers carry node IDs from the previous edge
        # type (e.g. paper IDs while counts is sized for authors).  They are
        # gated out by new_alive before any result is produced.
        safe_current   = current.clamp(0, counts.size(0) - 1)

        walker_counts  = counts[safe_current]     # [n_walkers]
        dead_this_step = walker_counts == 0
        new_alive      = alive & ~dead_this_step

        # Random index within each node's neighbour list.
        # For dead walkers we still compute a value (0) but it's masked out.
        rand = torch.randint(
            0, 1 << 31, (len(current),), generator=rng, device=self.device
        )
        safe_counts    = walker_counts.clamp(min=1)
        choice         = rand % safe_counts                    # [n_walkers]

        walker_offsets = offsets[safe_current]                 # [n_walkers]
        # Dead walkers have offset pointing past the end of sorted_dst.
        # Zero them out — the resulting index is harmless because dead
        # walkers are excluded before the output is used.
        walker_offsets = walker_offsets * new_alive.long()
        next_nodes     = sorted_dst[walker_offsets + choice]   # [n_walkers]

        # Dead walkers keep their current position (won't be used anyway).
        next_nodes = torch.where(new_alive, next_nodes, current)

        return next_nodes, new_alive

    def _build_edge_index(
        self,
        source_ids: torch.Tensor,
        terminal: torch.Tensor,
        alive: torch.Tensor,
        n_target: int,
    ) -> torch.Tensor:
        """
        Convert (source, terminal) walker pairs → coalesced, undirected
        edge_index with self-loops.

        If min_visits > 1, edges with fewer visit counts are dropped.
        """
        if alive.sum() == 0:
            ei = torch.empty((2, 0), dtype=torch.long, device=self.device)
            ei, _ = pyg_utils.add_self_loops(ei, num_nodes=n_target)
            return ei

        src = source_ids[alive]
        dst = terminal[alive]

        # Drop self-loops before counting visits.
        not_self = src != dst
        src = src[not_self]
        dst = dst[not_self]

        if len(src) == 0:
            ei = torch.empty((2, 0), dtype=torch.long, device=self.device)
            ei, _ = pyg_utils.add_self_loops(ei, num_nodes=n_target)
            return ei

        if self.min_visits > 1:
            # Count occurrences of each (src, dst) pair.
            # Encode as src * n_target + dst (fits in int64 for typical graphs).
            encoded = src * n_target + dst
            unique_enc, visit_counts = torch.unique(encoded, return_counts=True)
            keep = unique_enc[visit_counts >= self.min_visits]
            src  = keep // n_target
            dst  = keep %  n_target
            if len(src) == 0:
                ei = torch.empty((2, 0), dtype=torch.long, device=self.device)
                ei, _ = pyg_utils.add_self_loops(ei, num_nodes=n_target)
                return ei

        edge_index = torch.stack([src, dst])
        edge_index = pyg_utils.to_undirected(edge_index, num_nodes=n_target)
        edge_index = pyg_utils.coalesce(edge_index, num_nodes=n_target)
        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=n_target)
        return edge_index


# ---------------------------------------------------------------------------
# Metapath string → triples helper
# ---------------------------------------------------------------------------

def parse_metapath_triples(
    metapath_str: str,
    g_hetero: HeteroData,
) -> List[Tuple[str, str, str]]:
    """
    Convert a comma-separated metapath string to a list of
    (src_type, edge_name, dst_type) triples.

    Uses SchemaMatcher to resolve each edge name against the graph schema,
    then chains consecutive triples by matching dst_type of step i to
    src_type of step i+1.

    Args:
        metapath_str: e.g. "author_to_paper,paper_to_author"
        g_hetero:     The loaded graph (needed for schema lookup).

    Returns:
        Ordered list of (src_type, rel_name, dst_type).

    Raises:
        ValueError if an edge name is not found or the chain is broken.
    """
    from src.utils import SchemaMatcher

    edge_names = [e.strip() for e in metapath_str.split(",")]
    triples: List[Tuple[str, str, str]] = []

    for edge_name in edge_names:
        triple = SchemaMatcher.match(edge_name, g_hetero)
        triples.append(triple)

    # Validate chain continuity.
    for i in range(len(triples) - 1):
        if triples[i][2] != triples[i + 1][0]:
            raise ValueError(
                f"Broken metapath chain at step {i}: "
                f"'{triples[i]}' dst_type='{triples[i][2]}' "
                f"!= '{triples[i+1]}' src_type='{triples[i+1][0]}'"
            )

    return triples
