"""
Test C++ MPRW kernel against the pure-Python fallback.

Run:
    pytest tests/test_mprw_cpp.py -v

Requires the C++ extension to be built:
    python setup_mprw.py build_ext --inplace
"""
from __future__ import annotations

import time

import pytest
import torch
from torch_geometric.data import HeteroData

from src.kernels.mprw import MPRWKernel, parse_metapath_triples

# ---------------------------------------------------------------------------
# Fixtures: small synthetic heterogeneous graph
# ---------------------------------------------------------------------------

def _make_small_hetero() -> HeteroData:
    """
    Build a tiny author--paper--venue graph:
        4 authors, 5 papers, 2 venues
        author->paper, paper->venue, venue->paper, paper->author edges
    """
    g = HeteroData()
    g["author"].num_nodes = 4
    g["paper"].num_nodes = 5
    g["venue"].num_nodes = 2

    # author -> paper (6 edges)
    g["author", "author_to_paper", "paper"].edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 3],
         [0, 1, 1, 2, 3, 4]], dtype=torch.long
    )
    # paper -> venue (5 edges)
    g["paper", "paper_to_venue", "venue"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4],
         [0, 0, 1, 1, 0]], dtype=torch.long
    )
    # venue -> paper (5 edges)
    g["venue", "venue_to_paper", "paper"].edge_index = torch.tensor(
        [[0, 0, 1, 1, 0],
         [0, 1, 2, 3, 4]], dtype=torch.long
    )
    # paper -> author (6 edges)
    g["paper", "paper_to_author", "author"].edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4],
         [0, 0, 1, 1, 2, 3]], dtype=torch.long
    )
    return g


METAPATH_STR = "author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"


# ---------------------------------------------------------------------------
# Test: C++ import
# ---------------------------------------------------------------------------

def test_cpp_import():
    """The C++ extension should be importable after build."""
    try:
        import mprw_cpp
        assert hasattr(mprw_cpp, "materialize_mprw")
    except ImportError:
        pytest.skip("mprw_cpp not built — run: python setup_mprw.py build_ext --inplace")


# ---------------------------------------------------------------------------
# Test: output shape and properties
# ---------------------------------------------------------------------------

@pytest.fixture
def small_graph():
    return _make_small_hetero()


def test_cpp_output_shape(small_graph):
    """Edge index should be [2, E] with valid node IDs."""
    try:
        import mprw_cpp  # noqa: F401
    except ImportError:
        pytest.skip("mprw_cpp not built")

    kernel = MPRWKernel(k=64, seed=42, min_visits=1)
    # Force-check that C++ path is taken.
    from src.kernels.mprw import _HAS_CPP
    assert _HAS_CPP, "C++ backend not available"

    triples = [
        ("author", "author_to_paper", "paper"),
        ("paper", "paper_to_venue", "venue"),
        ("venue", "venue_to_paper", "paper"),
        ("paper", "paper_to_author", "author"),
    ]
    data, elapsed = kernel.materialize(small_graph, triples, "author")

    assert data.edge_index.shape[0] == 2
    assert data.edge_index.dtype == torch.int64
    assert data.num_nodes == 4
    # All node IDs in [0, num_nodes)
    assert data.edge_index.min() >= 0
    assert data.edge_index.max() < 4
    # Should have self-loops (at least 4 edges for 4 nodes)
    assert data.edge_index.size(1) >= 4
    # Should be undirected: for every (u,v), (v,u) exists
    edges_set = set()
    ei = data.edge_index
    for i in range(ei.size(1)):
        edges_set.add((ei[0, i].item(), ei[1, i].item()))
    for u, v in list(edges_set):
        if u != v:
            assert (v, u) in edges_set, f"Edge ({u},{v}) present but ({v},{u}) missing"


# ---------------------------------------------------------------------------
# Test: determinism — same seed → same output
# ---------------------------------------------------------------------------

def test_cpp_determinism(small_graph):
    """Same seed must produce identical edge_index."""
    try:
        import mprw_cpp  # noqa: F401
    except ImportError:
        pytest.skip("mprw_cpp not built")

    triples = [
        ("author", "author_to_paper", "paper"),
        ("paper", "paper_to_venue", "venue"),
        ("venue", "venue_to_paper", "paper"),
        ("paper", "paper_to_author", "author"),
    ]
    k1 = MPRWKernel(k=128, seed=7)
    d1, _ = k1.materialize(small_graph, triples, "author")

    k2 = MPRWKernel(k=128, seed=7)
    d2, _ = k2.materialize(small_graph, triples, "author")

    assert torch.equal(d1.edge_index, d2.edge_index)


# ---------------------------------------------------------------------------
# Test: different seeds → (likely) different output
# ---------------------------------------------------------------------------

def test_cpp_seed_variation(small_graph):
    """Different seeds should produce different edge counts (with high prob)."""
    try:
        import mprw_cpp  # noqa: F401
    except ImportError:
        pytest.skip("mprw_cpp not built")

    triples = [
        ("author", "author_to_paper", "paper"),
        ("paper", "paper_to_venue", "venue"),
        ("venue", "venue_to_paper", "paper"),
        ("paper", "paper_to_author", "author"),
    ]
    # Use small k so the graph doesn't saturate.
    results = []
    for seed in range(10):
        kern = MPRWKernel(k=4, seed=seed)
        d, _ = kern.materialize(small_graph, triples, "author")
        results.append(d.edge_index.size(1))

    # Not all the same (topology is small, so some might match, but not all 10).
    assert len(set(results)) > 1, f"All 10 seeds produced the same edge count: {results[0]}"


# ---------------------------------------------------------------------------
# Test: C++ vs Python produce same edge set (not necessarily same order)
# ---------------------------------------------------------------------------

def test_cpp_vs_python_agreement(small_graph):
    """C++ and Python backends should produce the same edge set."""
    try:
        import mprw_cpp  # noqa: F401
    except ImportError:
        pytest.skip("mprw_cpp not built")

    from src.kernels import mprw as mprw_mod

    triples = [
        ("author", "author_to_paper", "paper"),
        ("paper", "paper_to_venue", "venue"),
        ("venue", "venue_to_paper", "paper"),
        ("paper", "paper_to_author", "author"),
    ]

    # Run with C++ backend.
    old_flag = mprw_mod._HAS_CPP
    mprw_mod._HAS_CPP = True
    try:
        k_cpp = MPRWKernel(k=256, seed=42, min_visits=1)
        d_cpp, _ = k_cpp.materialize(small_graph, triples, "author")
    finally:
        mprw_mod._HAS_CPP = old_flag

    # Run with Python backend (force it off).
    mprw_mod._HAS_CPP = False
    try:
        k_py = MPRWKernel(k=256, seed=42, min_visits=1)
        d_py, _ = k_py.materialize(small_graph, triples, "author")
    finally:
        mprw_mod._HAS_CPP = old_flag

    # Both should produce the same coalesced, undirected edge set.
    # NOTE: the RNG sequences differ (PCG32 vs torch), so we can't compare
    # element-wise.  Instead we check that with k=256 on a tiny graph,
    # both should saturate to the same topology.
    def _edge_set(ei):
        return set(
            (ei[0, i].item(), ei[1, i].item()) for i in range(ei.size(1))
        )

    cpp_edges = _edge_set(d_cpp.edge_index)
    py_edges = _edge_set(d_py.edge_index)
    assert cpp_edges == py_edges, (
        f"Edge sets differ.\n"
        f"  C++ only: {cpp_edges - py_edges}\n"
        f"  Python only: {py_edges - cpp_edges}"
    )


# ---------------------------------------------------------------------------
# Test: performance (informational, not a pass/fail assertion)
# ---------------------------------------------------------------------------

def test_cpp_performance(small_graph):
    """Benchmark C++ vs Python on a slightly larger workload."""
    try:
        import mprw_cpp  # noqa: F401
    except ImportError:
        pytest.skip("mprw_cpp not built")

    from src.kernels import mprw as mprw_mod

    triples = [
        ("author", "author_to_paper", "paper"),
        ("paper", "paper_to_venue", "venue"),
        ("venue", "venue_to_paper", "paper"),
        ("paper", "paper_to_author", "author"),
    ]
    k = 1024

    # C++ timing
    old_flag = mprw_mod._HAS_CPP
    mprw_mod._HAS_CPP = True
    try:
        kern = MPRWKernel(k=k, seed=0)
        _, t_cpp = kern.materialize(small_graph, triples, "author")
    finally:
        mprw_mod._HAS_CPP = old_flag

    # Python timing
    mprw_mod._HAS_CPP = False
    try:
        kern = MPRWKernel(k=k, seed=0)
        _, t_py = kern.materialize(small_graph, triples, "author")
    finally:
        mprw_mod._HAS_CPP = old_flag

    print(f"\n  k={k}: C++={t_cpp*1000:.2f}ms  Python={t_py*1000:.2f}ms  "
          f"speedup={t_py/max(t_cpp, 1e-9):.1f}x")
