"""
Streaming simulation for heterogeneous graphs.

Produces cumulative edge snapshots of a HeteroData graph to simulate a
setting where the knowledge graph grows over time. Only the edges along
a given meta-path are partitioned; all node features and labels remain
fixed across snapshots.

Two partitioning strategies:
  - Temporal: filter edges by a timestamp attribute on the intermediate
    node type (e.g. paper.year for DBLP). Requires the relevant node
    type to have a 'year' attribute.
  - Random stratified: shuffle edges per edge type and split by fraction,
    preserving per-edge-type proportions.
"""
import torch
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch_geometric.data import HeteroData


MetaPathEdge = Tuple[str, str, str]


@dataclass
class Snapshot:
    graph: HeteroData
    fraction: float
    label: str  # e.g. "t=0 (70%)"


class TemporalPartitioner:
    """
    Produces cumulative snapshots of a HeteroData graph.

    Usage:
        snapshots = TemporalPartitioner.partition(
            g, meta_path, fractions=[0.7, 0.8, 0.9, 1.0]
        )
        # snapshots[0] is the initial training graph (70% edges)
        # snapshots[1..] are the evolved graphs for inference
    """

    @staticmethod
    def partition(
        g: HeteroData,
        meta_path: List[MetaPathEdge],
        fractions: List[float],
        time_node_type: Optional[str] = None,
        time_attr: str = "year",
    ) -> List[Snapshot]:
        """
        Partition edges along the meta-path into cumulative snapshots.

        Args:
            g: Full HeteroData graph.
            meta_path: List of (src_type, rel, dst_type) for the meta-path.
                Only the edge types named in meta_path are partitioned.
                All other edges remain unchanged in every snapshot.
            fractions: Cumulative fractions, e.g. [0.7, 0.8, 0.9, 1.0].
                Must be strictly increasing and end at 1.0.
            time_node_type: Node type that carries a temporal attribute
                (e.g. "paper"). If None, falls back to random stratified.
            time_attr: Name of the integer timestamp attribute on
                time_node_type (e.g. "year").

        Returns:
            List of Snapshot objects, one per fraction.
        """
        assert fractions == sorted(fractions), "fractions must be increasing"
        assert fractions[-1] == 1.0, "last fraction must be 1.0"

        meta_path_edge_types = {(s, r, d) for s, r, d in meta_path}

        if time_node_type is not None and hasattr(g[time_node_type], time_attr):
            order_per_type = TemporalPartitioner._temporal_order(
                g, meta_path_edge_types, time_node_type, time_attr
            )
        else:
            order_per_type = TemporalPartitioner._random_order(
                g, meta_path_edge_types
            )

        snapshots = []
        for frac in fractions:
            g_snap = TemporalPartitioner._build_snapshot(
                g, meta_path_edge_types, order_per_type, frac
            )
            snapshots.append(Snapshot(
                graph=g_snap,
                fraction=frac,
                label=f"t={len(snapshots)} ({int(frac*100)}%)",
            ))

        return snapshots

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _temporal_order(
        g: HeteroData,
        meta_path_edge_types: set,
        time_node_type: str,
        time_attr: str,
    ) -> dict:
        """
        For each meta-path edge type, produce a sorted permutation of edge
        indices ordered by the timestamp of the destination node.

        For edges (src_type, rel, dst_type) where dst_type == time_node_type,
        we sort by g[time_node_type].year[dst_node_id].
        For edges where src_type == time_node_type, we sort by src timestamp.
        """
        timestamps = getattr(g[time_node_type], time_attr)  # [N_time_nodes]
        order = {}

        for (src_type, rel, dst_type) in meta_path_edge_types:
            edge_index = g[src_type, rel, dst_type].edge_index  # [2, E]

            if dst_type == time_node_type:
                edge_times = timestamps[edge_index[1]]
            elif src_type == time_node_type:
                edge_times = timestamps[edge_index[0]]
            else:
                # Neither endpoint is the time node — fall back to random.
                edge_times = torch.randperm(edge_index.size(1))

            order[(src_type, rel, dst_type)] = torch.argsort(edge_times)

        return order

    @staticmethod
    def _random_order(g: HeteroData, meta_path_edge_types: set) -> dict:
        """Random permutation per edge type (stratified fallback)."""
        return {
            (s, r, d): torch.randperm(g[s, r, d].edge_index.size(1))
            for s, r, d in meta_path_edge_types
        }

    @staticmethod
    def _build_snapshot(
        g: HeteroData,
        meta_path_edge_types: set,
        order_per_type: dict,
        fraction: float,
    ) -> HeteroData:
        """
        Build a HeteroData snapshot that keeps the first `fraction` of
        meta-path edges (by temporal or random order) and all other edges.

        Node features, labels, and masks are shared (not copied) with the
        original graph to save memory.
        """
        g_snap = HeteroData()

        # Copy all node stores (features, labels, masks) by reference.
        for ntype in g.node_types:
            for key, val in g[ntype].items():
                g_snap[ntype][key] = val

        # For each edge type: keep fraction of meta-path edges, all others.
        for edge_type in g.edge_types:
            src_type, rel, dst_type = edge_type
            edge_index = g[src_type, rel, dst_type].edge_index

            if edge_type in meta_path_edge_types:
                n_keep = max(1, int(fraction * edge_index.size(1)))
                keep_idx = order_per_type[edge_type][:n_keep]
                g_snap[src_type, rel, dst_type].edge_index = edge_index[:, keep_idx]
            else:
                g_snap[src_type, rel, dst_type].edge_index = edge_index

        return g_snap
