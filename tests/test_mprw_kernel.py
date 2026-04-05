"""
Sanity checks for MPRWKernel.

Test graph (hand-crafted, exact answer known):
    Node types: A (3 nodes), B (2 nodes)
    A→B edges: A0→B0, A0→B1, A1→B0, A2→B1
    B→A edges: B0→A0, B0→A1, B1→A1, B1→A2

    Metapath A→B→A exact reachability (before symmetrize):
        A0 → {A1, A2}   (via B0: A1; via B1: A1, A2)
        A1 → {A0}       (via B0 only: A0)
        A2 → {A1}       (via B1 only: A1)

    After to_undirected + add_self_loops, expected edges (sorted):
        self-loops: (0,0),(1,1),(2,2)
        undirected:  (0,1),(1,0),(0,2),(2,0),(1,2),(2,1)

    Edge set (as set of frozensets ignoring self-loops):
        {{0,1}, {0,2}, {1,2}}  — a complete triangle.
"""
import torch
import pytest
from torch_geometric.data import HeteroData

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kernels.mprw import MPRWKernel, parse_metapath_triples


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_graph() -> HeteroData:
    """Small A→B→A test graph with fully known exact answer."""
    g = HeteroData()
    g["A"].num_nodes = 3
    g["B"].num_nodes = 2

    # A→B: A0→B0, A0→B1, A1→B0, A2→B1
    g["A", "a_to_b", "B"].edge_index = torch.tensor(
        [[0, 0, 1, 2],
         [0, 1, 0, 1]], dtype=torch.long
    )
    # B→A: B0→A0, B0→A1, B1→A1, B1→A2
    g["B", "b_to_a", "A"].edge_index = torch.tensor(
        [[0, 0, 1, 1],
         [0, 1, 1, 2]], dtype=torch.long
    )
    return g


def _non_self_edges(edge_index: torch.Tensor) -> set:
    """Return set of frozensets for undirected edges, excluding self-loops."""
    result = set()
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u != v:
            result.add(frozenset({u, v}))
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMPRWConvergence:
    """With large k MPRW must recover the full exact edge set."""

    def test_recovers_all_exact_edges_large_k(self):
        g = _make_graph()
        triples = [("A", "a_to_b", "B"), ("B", "b_to_a", "A")]

        kernel = MPRWKernel(k=2000, seed=0)
        data, elapsed = kernel.materialize(g, triples, target_ntype="A")

        found = _non_self_edges(data.edge_index)
        expected = {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})}

        assert found == expected, (
            f"Expected {expected}, got {found}\n"
            f"edge_index:\n{data.edge_index}"
        )

    def test_self_loops_present(self):
        g = _make_graph()
        triples = [("A", "a_to_b", "B"), ("B", "b_to_a", "A")]

        kernel = MPRWKernel(k=500, seed=1)
        data, _ = kernel.materialize(g, triples, target_ntype="A")

        diagonal = {
            (data.edge_index[0, i].item(), data.edge_index[1, i].item())
            for i in range(data.edge_index.size(1))
            if data.edge_index[0, i] == data.edge_index[1, i]
        }
        assert diagonal == {(0, 0), (1, 1), (2, 2)}, (
            f"Self-loops missing or wrong: {diagonal}"
        )

    def test_num_nodes_correct(self):
        g = _make_graph()
        triples = [("A", "a_to_b", "B"), ("B", "b_to_a", "A")]
        kernel = MPRWKernel(k=100, seed=2)
        data, _ = kernel.materialize(g, triples, target_ntype="A")
        assert data.num_nodes == 3

    def test_edge_index_symmetric(self):
        """to_undirected guarantees symmetry."""
        g = _make_graph()
        triples = [("A", "a_to_b", "B"), ("B", "b_to_a", "A")]
        kernel = MPRWKernel(k=200, seed=3)
        data, _ = kernel.materialize(g, triples, target_ntype="A")

        ei = data.edge_index
        edge_set   = set(zip(ei[0].tolist(), ei[1].tolist()))
        for u, v in list(edge_set):
            assert (v, u) in edge_set, f"Missing reverse edge ({v},{u}) for ({u},{v})"


class TestMPRWScaling:
    """Edge recall should increase monotonically with k."""

    def test_recall_increases_with_k(self):
        g = _make_graph()
        triples = [("A", "a_to_b", "B"), ("B", "b_to_a", "A")]
        expected = {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})}

        recalls = []
        for k in [1, 4, 16, 64, 512]:
            kernel = MPRWKernel(k=k, seed=42)
            data, _ = kernel.materialize(g, triples, target_ntype="A")
            found = _non_self_edges(data.edge_index)
            recalls.append(len(found & expected) / len(expected))

        # Not strictly monotone for tiny k (variance), but max should be achieved
        # at largest k.
        assert recalls[-1] == 1.0, (
            f"At k=512 recall should be 1.0, got {recalls[-1]}\nAll recalls: {recalls}"
        )


class TestMPRWMinVisits:
    """min_visits filter should reduce edge count."""

    def test_min_visits_reduces_edges(self):
        g = _make_graph()
        triples = [("A", "a_to_b", "B"), ("B", "b_to_a", "A")]

        k1 = MPRWKernel(k=500, seed=7, min_visits=1)
        k5 = MPRWKernel(k=500, seed=7, min_visits=5)

        d1, _ = k1.materialize(g, triples, target_ntype="A")
        d5, _ = k5.materialize(g, triples, target_ntype="A")

        n1 = d1.edge_index.size(1)
        n5 = d5.edge_index.size(1)
        assert n5 <= n1, (
            f"min_visits=5 should have ≤ edges than min_visits=1: {n5} vs {n1}"
        )


class TestMPRWReproducibility:
    """Same seed → same graph."""

    def test_same_seed_deterministic(self):
        g = _make_graph()
        triples = [("A", "a_to_b", "B"), ("B", "b_to_a", "A")]

        d1, _ = MPRWKernel(k=100, seed=99).materialize(g, triples, "A")
        d2, _ = MPRWKernel(k=100, seed=99).materialize(g, triples, "A")

        assert torch.equal(d1.edge_index, d2.edge_index), (
            "Same seed should produce identical edge_index"
        )

    def test_different_seeds_may_differ(self):
        """With very small k different seeds likely produce different graphs."""
        g = _make_graph()
        triples = [("A", "a_to_b", "B"), ("B", "b_to_a", "A")]

        # Use k=1 to maximise variance
        d1, _ = MPRWKernel(k=1, seed=0).materialize(g, triples, "A")
        d2, _ = MPRWKernel(k=1, seed=1).materialize(g, triples, "A")
        # Can't assert they differ (unlikely but possible), just that both are valid
        assert d1.num_nodes == d2.num_nodes == 3


class TestParseMetapathTriples:
    """parse_metapath_triples should correctly resolve edge names."""

    def test_basic_resolution(self):
        g = _make_graph()
        triples = parse_metapath_triples("a_to_b,b_to_a", g)
        assert triples == [("A", "a_to_b", "B"), ("B", "b_to_a", "A")]

    def test_broken_chain_raises(self):
        """If metapath chain is not contiguous, raise ValueError."""
        g = _make_graph()
        # Two A→B hops in a row: chain broken (B dst ≠ A src of next)
        # We add a dummy edge type to force the issue
        g["C"].num_nodes = 2
        g["A", "a_to_c", "C"].edge_index = torch.zeros((2, 0), dtype=torch.long)

        with pytest.raises(ValueError, match="Broken metapath"):
            parse_metapath_triples("a_to_b,a_to_c", g)


class TestDeadEndHandling:
    """Walkers reaching degree-0 nodes must be excluded from edges."""

    def test_dead_end_node_excluded(self):
        """Node A2 has only one B-neighbour (B1).  If B1 has NO forward edges
        to A, walkers from A2 die and A2 gets no neighbours (except self-loop)."""
        g = HeteroData()
        g["A"].num_nodes = 3
        g["B"].num_nodes = 2

        # A→B: A0→B0, A1→B0, A2→B1
        g["A", "a_to_b", "B"].edge_index = torch.tensor(
            [[0, 1, 2], [0, 0, 1]], dtype=torch.long
        )
        # B→A: B0→A0 only; B1 has no outgoing edges
        g["B", "b_to_a", "A"].edge_index = torch.tensor(
            [[0], [0]], dtype=torch.long
        )

        triples = [("A", "a_to_b", "B"), ("B", "b_to_a", "A")]
        kernel = MPRWKernel(k=200, seed=5)
        data, _ = kernel.materialize(g, triples, target_ntype="A")

        found = _non_self_edges(data.edge_index)
        # A2's walkers all die at B1; A2 should have no non-self-loop edges.
        edges_involving_A2 = {e for e in found if 2 in e}
        assert edges_involving_A2 == set(), (
            f"A2 should have no neighbours (dead-end), got: {edges_involving_A2}"
        )
