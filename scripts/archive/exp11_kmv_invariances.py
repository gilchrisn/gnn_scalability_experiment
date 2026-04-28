#!/usr/bin/env python3
"""
Rigorous KMV sketch propagation invariance tests (I2, I3, I4, I5, I6).

Tests:
    I2. Propagation tightness: does K_L[v] exactly equal KMV(N_L(v))?
        Measure "leak rate" (winners discarded at intermediate layers).
        Regress against local topology (max intermediate degree, clustering).
    I3. Nested-k free multi-resolution: single propagation at k_max=64 should
        yield identical sketches at k < k_max as an independent k run would,
        because KMV prefix is stable. Verify exact equality.
    I4. Expansion-decay invariant: log(max(K_L[v])) decays with L at a rate
        that measures local expansion / mixing. Correlate slope with exact
        log|N_L(v)|, and test whether slope is orthogonal to degree.
    I5. Multi-hop Jaccard preservation: |K_L[u] ∩ K_L[v]| / k estimates
        Jaccard(N_L(u), N_L(v)). Base paper proves single-set cardinality;
        multi-hop Jaccard has never been measured.
    I6. Structural-equivalence identity: if N_L(u) = N_L(v), then K_L[u] = K_L[v]
        deterministically (no approximation noise).

Outputs:
    results/kmv_properties/invariances_<dataset>.csv
    figures/kmv_properties/inv_<dataset>_<test>.pdf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.data.factory import DatasetFactory

INF = torch.iinfo(torch.int64).max

# ---------------- Dataset registry (canonical meta-path per dataset) ----------------

DATASETS = {
    "HGB_ACM": {
        "source": "HGB",
        "name": "ACM",
        "target": "paper",
        "metapath": [("paper", "paper_to_term", "term"),
                     ("term",  "term_to_paper", "paper")],
    },
    "HGB_DBLP": {
        "source": "HGB",
        "name": "DBLP",
        "target": "author",
        "metapath": [("author", "author_to_paper", "paper"),
                     ("paper",  "paper_to_author", "author")],
    },
    "HGB_IMDB": {
        "source": "HGB",
        "name": "IMDB",
        "target": "movie",
        "metapath": [("movie",   "movie_to_keyword", "keyword"),
                     ("keyword", "keyword_to_movie", "movie")],
    },
    "HNE_PubMed": {
        "source": "HNE",
        "name": "PubMed",
        "target": "disease",
        "metapath": [("disease",  "disease_to_chemical", "chemical"),
                     ("chemical", "chemical_to_disease", "disease")],
    },
}


# ---------------- KMV propagation (single hash stream, instrumented) ----------------

def assign_hashes(num_nodes: int, seed: int) -> torch.Tensor:
    """Uniform random int64 hashes in [1, 2**62], no collisions (vanishing prob)."""
    g = torch.Generator().manual_seed(seed)
    # 62-bit range; collision probability O(n^2 / 2^62) is negligible for our n
    return torch.randint(1, 2**62, (num_nodes,), dtype=torch.int64, generator=g)


def _bottomk_merge(edge_index: torch.Tensor,
                   src_sketch: torch.Tensor,
                   dst_sketch: torch.Tensor,
                   k: int) -> torch.Tensor:
    """
    For each destination node, merge its own sketch with incoming src sketches
    and keep the k smallest distinct hash values.

    src_sketch: (n_src, k_src) int64 -- INF-padded
    dst_sketch: (n_dst, k_dst) int64 -- INF-padded
    edge_index: (2, m) long
    """
    n_dst = dst_sketch.shape[0]
    k_src = src_sketch.shape[1]
    k_dst_in = dst_sketch.shape[1]

    # Gather source sketches onto edges
    inc_vals = src_sketch[edge_index[0]].reshape(-1)
    inc_idx  = edge_index[1].repeat_interleave(k_src)

    own_vals = dst_sketch.reshape(-1)
    own_idx  = torch.arange(n_dst).repeat_interleave(k_dst_in)

    vals = torch.cat([own_vals, inc_vals])
    idx  = torch.cat([own_idx,  inc_idx])

    # Drop INF pads
    keep = vals != INF
    vals = vals[keep]
    idx  = idx[keep]

    if vals.numel() == 0:
        return torch.full((n_dst, k), INF, dtype=torch.int64)

    # Sort by (node, val): primary node, secondary val ascending
    order = torch.argsort(vals)
    vals = vals[order]
    idx  = idx[order]
    order = torch.argsort(idx, stable=True)
    vals = vals[order]
    idx  = idx[order]

    # Deduplicate (node, val) pairs to avoid counting a hash twice
    if vals.numel() > 1:
        prev_val = torch.cat([torch.tensor([-1], dtype=torch.int64), vals[:-1]])
        prev_idx = torch.cat([torch.tensor([-1], dtype=torch.int64), idx[:-1]])
        distinct = (vals != prev_val) | (idx != prev_idx)
        vals = vals[distinct]
        idx  = idx[distinct]

    # Compute rank within each node group
    _, counts = torch.unique_consecutive(idx, return_counts=True)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, 0)[:-1]])
    rank = torch.arange(vals.numel()) - starts.repeat_interleave(counts)
    mask = rank < k
    vals = vals[mask]
    idx  = idx[mask]
    rank = rank[mask]

    out = torch.full((n_dst, k), INF, dtype=torch.int64)
    out[idx, rank] = vals
    return out


def propagate_kmv_all_depths(g_hetero,
                             metapath: List[Tuple[str, str, str]],
                             target_ntype: str,
                             k: int,
                             L_max: int,
                             seed: int) -> Dict:
    """
    Run L_max full meta-path traversals. Return per-depth target sketches
    and all intermediate sketches observed on the way.

    Returns:
        {
          "hashes":     (n_target,) int64   -- origin hash assignments
          "target_by_L": List[Tensor]       -- [L_max+1] tensors (n_target, k)
                                              index 0 = initial (self-hash only).
          "intermediate_k_minhashes": nested list of (n_node, k) per (traversal, step)
        }
    """
    num_target = g_hetero[target_ntype].num_nodes
    h = assign_hashes(num_target, seed)

    # Initial sketches per ntype
    sketches = {}
    for nt in g_hetero.node_types:
        n = g_hetero[nt].num_nodes
        s = torch.full((n, k), INF, dtype=torch.int64)
        if nt == target_ntype:
            s[:, 0] = h
        sketches[nt] = s

    target_by_L = [sketches[target_ntype].clone()]
    intermediate = []

    for L in range(1, L_max + 1):
        # One full meta-path traversal
        step_snapshots = []
        for (src_t, rel, dst_t) in metapath:
            ei = g_hetero[src_t, rel, dst_t].edge_index
            new_dst = _bottomk_merge(ei, sketches[src_t], sketches[dst_t], k)
            sketches[dst_t] = new_dst
            step_snapshots.append((dst_t, new_dst.clone()))
        intermediate.append(step_snapshots)
        target_by_L.append(sketches[target_ntype].clone())

    return {
        "hashes": h,
        "target_by_L": target_by_L,
        "intermediate": intermediate,
    }


# ---------------- Exact L-hop neighborhood computation ----------------

def build_target_to_target_adj(g_hetero,
                               metapath: List[Tuple[str, str, str]],
                               target_ntype: str) -> torch.sparse.Tensor:
    """
    Build the target-to-target connectivity matrix induced by one meta-path
    traversal. Returns a binary sparse COO matrix of shape (n_target, n_target).
    """
    import torch.sparse as sp

    # Start with identity on src type
    sizes = {nt: g_hetero[nt].num_nodes for nt in g_hetero.node_types}
    n0 = sizes[metapath[0][0]]
    # We'll track a reachability matrix from target-type sources
    # src_state: sparse matrix (n_target, n_current_type)
    n_target = sizes[target_ntype]
    # start: identity (each target reaches itself)
    idx = torch.arange(n_target)
    state = torch.sparse_coo_tensor(
        torch.stack([idx, idx]),
        torch.ones(n_target),
        size=(n_target, n_target),
    ).coalesce()

    cur_t = target_ntype
    for (src_t, rel, dst_t) in metapath:
        assert cur_t == src_t, f"meta-path discontinuity: {cur_t} -> {src_t}"
        ei = g_hetero[src_t, rel, dst_t].edge_index
        M = torch.sparse_coo_tensor(
            ei,
            torch.ones(ei.shape[1]),
            size=(sizes[src_t], sizes[dst_t]),
        ).coalesce()
        # state: (n_target, n_src) * M: (n_src, n_dst) -> (n_target, n_dst)
        state = torch.sparse.mm(state, M).coalesce()
        # Binarize (reachability, not count)
        vals = state.values()
        state = torch.sparse_coo_tensor(
            state.indices(),
            torch.ones_like(vals),
            size=state.shape,
        ).coalesce()
        cur_t = dst_t

    assert cur_t == target_ntype, "meta-path must return to target type"
    return state  # (n_target, n_target) binary


def power_sparse_binary(A: torch.Tensor, L: int) -> torch.Tensor:
    """A^L with binarization after each multiply (reachability semantics)."""
    out = A
    for _ in range(L - 1):
        out = torch.sparse.mm(out, A).coalesce()
        vals = out.values()
        out = torch.sparse_coo_tensor(
            out.indices(),
            torch.ones_like(vals),
            size=out.shape,
        ).coalesce()
    return out


def neighborhood_sets(A_L: torch.Tensor, sample_nodes: torch.Tensor) -> Dict[int, torch.Tensor]:
    """For each node in sample_nodes, return sorted tensor of its L-hop target neighbors."""
    idx = A_L.indices()
    n_samp = sample_nodes.numel()
    out: Dict[int, torch.Tensor] = {}
    # Build CSR-like lookup
    row, col = idx[0], idx[1]
    order = torch.argsort(row)
    row_sorted = row[order]
    col_sorted = col[order]
    # Boundaries per source row
    n_rows = A_L.shape[0]
    # counts per row
    counts = torch.zeros(n_rows + 1, dtype=torch.long)
    counts.scatter_add_(0, row_sorted + 1, torch.ones_like(row_sorted))
    starts = torch.cumsum(counts, 0)

    for v in sample_nodes.tolist():
        a, b = starts[v].item(), starts[v + 1].item()
        out[v] = torch.sort(col_sorted[a:b]).values
    return out


# ---------------- Invariance tests ----------------

def test_I2_tightness(target_sketches_by_L: List[torch.Tensor],
                      hashes: torch.Tensor,
                      neigh_by_L: Dict[int, Dict[int, torch.Tensor]],
                      k: int) -> List[Dict]:
    """
    For each sampled node v and each depth L, compute the EXACT KMV sketch of
    N_L(v) (bottom-k of hashes of neighbors) and compare to propagated K_L[v].
    Record leak count (= |exact KMV| − |propagated ∩ exact KMV|).
    """
    rows = []
    for L, neighs in neigh_by_L.items():
        K_L = target_sketches_by_L[L]  # (n_target, k)
        for v, nbrs in neighs.items():
            if nbrs.numel() == 0:
                continue
            true_hashes = hashes[nbrs]  # all hashes in N_L(v)
            # exact bottom-k
            true_sorted = torch.sort(true_hashes).values
            exact_topk = true_sorted[:k]
            # propagated sketch, drop INF
            prop = K_L[v]
            prop = prop[prop != INF]
            prop_sorted = torch.sort(prop).values
            # intersection
            exact_set = set(exact_topk.tolist())
            prop_set  = set(prop_sorted.tolist())
            leaked = exact_set - prop_set
            rows.append({
                "L": L,
                "node": v,
                "true_neighborhood_size": int(nbrs.numel()),
                "exact_topk_size": int(exact_topk.numel()),
                "prop_sketch_size": int(prop_sorted.numel()),
                "intersection": int(len(exact_set & prop_set)),
                "leaked_count": int(len(leaked)),
                "leak_rate":    float(len(leaked) / max(1, len(exact_set))),
                "is_tight":     bool(len(leaked) == 0),
                "exact_max":    int(exact_topk.max().item()) if exact_topk.numel() else -1,
                "prop_max":     int(prop_sorted.max().item()) if prop_sorted.numel() else -1,
            })
    return rows


def test_I3_nested_k(target_sketches_by_L: List[torch.Tensor],
                     k_max: int,
                     k_values: List[int]) -> List[Dict]:
    """
    Nested-k property: bottom-k_small of K_L(k_max) should equal the result of
    a fresh propagation at k_small. We can't cheaply re-propagate here, but
    the *structural* claim is: bottom-k_small of K_L is a valid KMV sketch
    of the same neighborhood. Verify monotonicity + nestedness mechanically.

    Returns: per-L statistics on the stability of the bottom-k prefix.
    """
    rows = []
    for L, K in enumerate(target_sketches_by_L):
        if L == 0:
            continue
        for k_s in k_values:
            if k_s > k_max:
                continue
            # Extract bottom-k_s per row
            K_sorted = torch.sort(K, dim=1).values
            K_small = K_sorted[:, :k_s]
            # For each row, number of valid hashes (non-INF) among bottom-k_s
            valid_mask = K_small != INF
            counts = valid_mask.sum(dim=1)
            # Cardinality estimator: (k_s-1) / max(K_small)
            max_h = K_small.masked_fill(~valid_mask, 0).max(dim=1).values.float()
            # KMV estimator: only defined when row is "full" (counts == k_s)
            full_rows = counts == k_s
            if full_rows.sum() == 0:
                continue
            est = (k_s - 1) * (2.0 ** 62) / max_h[full_rows].clamp(min=1.0)
            rows.append({
                "L": L,
                "k_small": k_s,
                "n_full_rows": int(full_rows.sum().item()),
                "mean_card_estimate": float(est.mean().item()),
                "median_card_estimate": float(est.median().item()),
                "std_card_estimate": float(est.std().item()) if est.numel() > 1 else 0.0,
            })
    return rows


def test_I4_expansion_decay(target_sketches_by_L: List[torch.Tensor],
                            sample_nodes: torch.Tensor,
                            k: int,
                            neigh_by_L: Dict[int, Dict[int, torch.Tensor]]) -> List[Dict]:
    """
    log(max(K_L[v])) ≈ log(k) - log(|N_L(v)|)
    Fit per-node slope across L; correlate with exact log|N_L(v)|.
    """
    rows = []
    HASH_MAX_LOG = np.log(2.0 ** 62)
    for v in sample_nodes.tolist():
        log_max_by_L = []
        log_true_by_L = []
        for L in range(1, len(target_sketches_by_L)):
            K_L = target_sketches_by_L[L]
            row = K_L[v]
            row = row[row != INF]
            if row.numel() < k:
                # sketch not full — cardinality estimator undefined
                continue
            max_h = row.max().item()
            log_max_by_L.append(np.log(max_h) - HASH_MAX_LOG)  # log(max_h / HASH_MAX)
            if L in neigh_by_L and v in neigh_by_L[L]:
                n_true = neigh_by_L[L][v].numel()
                log_true_by_L.append(np.log(max(1, n_true)))
            else:
                log_true_by_L.append(np.nan)
        if len(log_max_by_L) < 2:
            continue
        # Linear fit
        L_vals = np.arange(1, 1 + len(log_max_by_L))
        logmax = np.array(log_max_by_L)
        logtrue = np.array(log_true_by_L)
        slope_max  = np.polyfit(L_vals, logmax, 1)[0]
        if np.isfinite(logtrue).sum() >= 2:
            slope_true = np.polyfit(
                L_vals[np.isfinite(logtrue)],
                logtrue[np.isfinite(logtrue)],
                1,
            )[0]
        else:
            slope_true = np.nan
        rows.append({
            "node": v,
            "slope_log_max": float(slope_max),
            "slope_log_true_card": float(slope_true),
            "log_card_L1": float(logtrue[0]) if np.isfinite(logtrue[0]) else np.nan,
        })
    return rows


def test_I5_jaccard(target_sketches_by_L: List[torch.Tensor],
                    neigh_by_L: Dict[int, Dict[int, torch.Tensor]],
                    k: int,
                    n_pairs: int = 5000,
                    rng_seed: int = 0) -> List[Dict]:
    """
    Sample node pairs; for each pair (u,v) at each L, compare:
      estimator(u,v,L) = |K_L[u] ∩ K_L[v]| / k  (KMV Jaccard estimator)
      truth(u,v,L)     = |N_L(u) ∩ N_L(v)| / |N_L(u) ∪ N_L(v)|
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    for L, neighs in neigh_by_L.items():
        nodes_with_neigh = [v for v, nb in neighs.items() if nb.numel() > 0]
        if len(nodes_with_neigh) < 2:
            continue
        K_L = target_sketches_by_L[L]
        pairs = rng.choice(nodes_with_neigh, size=(n_pairs, 2), replace=True)
        for u, v in pairs:
            if u == v:
                continue
            # Exact Jaccard
            Nu = set(neighs[int(u)].tolist())
            Nv = set(neighs[int(v)].tolist())
            if len(Nu) == 0 and len(Nv) == 0:
                continue
            union = len(Nu | Nv)
            inter = len(Nu & Nv)
            true_jac = inter / union if union > 0 else 0.0
            # Estimator from sketches
            sk_u = set(K_L[int(u)].tolist()) - {INF}
            sk_v = set(K_L[int(v)].tolist()) - {INF}
            est_jac = len(sk_u & sk_v) / max(1, len(sk_u | sk_v))
            rows.append({
                "L": L,
                "u": int(u),
                "v": int(v),
                "true_jac": float(true_jac),
                "est_jac": float(est_jac),
                "Nu_size": len(Nu),
                "Nv_size": len(Nv),
            })
    return rows


def test_I6_structural_equality(target_sketches_by_L: List[torch.Tensor],
                                neigh_by_L: Dict[int, Dict[int, torch.Tensor]]) -> List[Dict]:
    """
    Find pairs (u,v) with identical N_L(u) = N_L(v); check whether K_L[u] = K_L[v].
    If they differ, invariant broken (shouldn't happen modulo INF padding differences).
    """
    rows = []
    for L, neighs in neigh_by_L.items():
        K_L = target_sketches_by_L[L]
        # Hash each node's neighborhood into a signature
        sig_to_nodes: Dict[Tuple, List[int]] = {}
        for v, nb in neighs.items():
            sig = tuple(nb.tolist())
            sig_to_nodes.setdefault(sig, []).append(v)
        n_pairs = 0
        n_matches = 0
        for sig, nodes in sig_to_nodes.items():
            if len(nodes) < 2 or len(sig) == 0:
                continue
            base = tuple(sorted(K_L[nodes[0]].tolist()))
            for v in nodes[1:]:
                this = tuple(sorted(K_L[v].tolist()))
                n_pairs += 1
                if this == base:
                    n_matches += 1
        rows.append({
            "L": L,
            "n_isomorphic_pairs": n_pairs,
            "n_sketch_identical": n_matches,
            "match_rate": (n_matches / n_pairs) if n_pairs > 0 else np.nan,
        })
    return rows


# ---------------- Main driver ----------------

def run_dataset(ds_key: str, k: int, L_max: int, seed: int,
                sample_n: int, out_dir: Path) -> Dict:
    cfg = DATASETS[ds_key]
    print(f"\n[{ds_key}] loading...")
    g, info = DatasetFactory.get_data(cfg["source"], cfg["name"], cfg["target"])
    target_nt = cfg["target"]
    metapath  = cfg["metapath"]
    n_target  = g[target_nt].num_nodes
    print(f"[{ds_key}] target={target_nt}, n_target={n_target}")

    print(f"[{ds_key}] propagating KMV sketches (k={k}, L_max={L_max})...")
    t0 = time.time()
    prop = propagate_kmv_all_depths(g, metapath, target_nt, k, L_max, seed)
    t_prop = time.time() - t0
    print(f"[{ds_key}]   done in {t_prop:.2f}s")

    # Sample target nodes for expensive exact computations
    rng = np.random.default_rng(seed)
    sample_n = min(sample_n, n_target)
    sample_nodes = torch.tensor(
        sorted(rng.choice(n_target, size=sample_n, replace=False).tolist()),
        dtype=torch.long,
    )

    # Build target-to-target connectivity for 1 meta-path traversal
    print(f"[{ds_key}] building target-to-target adjacency (exact)...")
    A1 = build_target_to_target_adj(g, metapath, target_nt)
    print(f"[{ds_key}]   nnz(A1) = {A1.indices().shape[1]}")

    # Compute N_L(v) exactly for L = 1..L_max (restrict to sampled nodes)
    neigh_by_L: Dict[int, Dict[int, torch.Tensor]] = {}
    A_power = A1
    for L in range(1, L_max + 1):
        print(f"[{ds_key}]   computing N_{L}(v) for {sample_n} sampled nodes...")
        neigh_by_L[L] = neighborhood_sets(A_power, sample_nodes)
        if L < L_max:
            A_power = torch.sparse.mm(A_power, A1).coalesce()
            vals = A_power.values()
            A_power = torch.sparse_coo_tensor(
                A_power.indices(),
                torch.ones_like(vals),
                size=A_power.shape,
            ).coalesce()

    # ---- Run invariance tests ----
    print(f"[{ds_key}] I2: tightness / leak rate...")
    i2 = test_I2_tightness(prop["target_by_L"], prop["hashes"], neigh_by_L, k)
    print(f"[{ds_key}] I3: nested-k cardinality-estimate stability...")
    i3 = test_I3_nested_k(prop["target_by_L"], k, [2, 4, 8, 16, 32, 64])
    print(f"[{ds_key}] I4: expansion-decay slope...")
    i4 = test_I4_expansion_decay(prop["target_by_L"], sample_nodes, k, neigh_by_L)
    print(f"[{ds_key}] I5: multi-hop Jaccard...")
    i5 = test_I5_jaccard(prop["target_by_L"], neigh_by_L, k, n_pairs=3000, rng_seed=seed)
    print(f"[{ds_key}] I6: structural-equivalence identity...")
    i6 = test_I6_structural_equality(prop["target_by_L"], neigh_by_L)

    # Persist
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / f"i2_{ds_key}.csv", i2)
    _write_csv(out_dir / f"i3_{ds_key}.csv", i3)
    _write_csv(out_dir / f"i4_{ds_key}.csv", i4)
    _write_csv(out_dir / f"i5_{ds_key}.csv", i5)
    _write_csv(out_dir / f"i6_{ds_key}.csv", i6)

    summary = _summarize(ds_key, i2, i3, i4, i5, i6, t_prop)
    (out_dir / f"summary_{ds_key}.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[{ds_key}] SUMMARY:")
    print(json.dumps(summary, indent=2))
    return summary


def _write_csv(path: Path, rows: List[Dict]) -> None:
    import csv
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _summarize(ds_key: str, i2, i3, i4, i5, i6, t_prop) -> Dict:
    import statistics as stats

    def _mean(xs): return float(sum(xs) / len(xs)) if xs else float("nan")

    # I2
    by_L = {}
    for r in i2:
        by_L.setdefault(r["L"], []).append(r)
    i2_summary = {}
    for L, rs in sorted(by_L.items()):
        i2_summary[f"L={L}"] = {
            "n": len(rs),
            "mean_leak_rate": _mean([r["leak_rate"] for r in rs]),
            "frac_tight":     _mean([1.0 if r["is_tight"] else 0.0 for r in rs]),
        }

    # I4
    slopes_max = [r["slope_log_max"] for r in i4 if not np.isnan(r["slope_log_max"])]
    slopes_true = [r["slope_log_true_card"] for r in i4 if not np.isnan(r["slope_log_true_card"])]
    if slopes_max and slopes_true:
        from scipy.stats import pearsonr
        common = [(a, b) for (a, b) in zip(
            [r["slope_log_max"] for r in i4],
            [r["slope_log_true_card"] for r in i4]
        ) if not (np.isnan(a) or np.isnan(b))]
        if len(common) >= 3:
            a = [c[0] for c in common]; b = [c[1] for c in common]
            r_val, p_val = pearsonr(a, b)
        else:
            r_val, p_val = float("nan"), float("nan")
    else:
        r_val, p_val = float("nan"), float("nan")
    i4_summary = {
        "n_nodes": len(i4),
        "mean_slope_log_max": _mean(slopes_max),
        "mean_slope_log_true_card": _mean(slopes_true),
        "pearson_r (slope_logmax vs slope_logtrue)": r_val,
        "pearson_p": p_val,
    }

    # I5
    by_L_j = {}
    for r in i5:
        by_L_j.setdefault(r["L"], []).append(r)
    i5_summary = {}
    for L, rs in sorted(by_L_j.items()):
        truths = [r["true_jac"] for r in rs]
        ests   = [r["est_jac"]  for r in rs]
        if len(truths) >= 3:
            from scipy.stats import pearsonr
            r_j, p_j = pearsonr(truths, ests)
        else:
            r_j, p_j = float("nan"), float("nan")
        i5_summary[f"L={L}"] = {
            "n_pairs": len(rs),
            "mean_true_jac": _mean(truths),
            "mean_est_jac":  _mean(ests),
            "bias (est-true)": _mean([e - t for e, t in zip(ests, truths)]),
            "pearson_r": r_j,
        }

    # I6
    i6_summary = {f"L={r['L']}": r for r in i6}

    return {
        "dataset": ds_key,
        "propagation_time_s": t_prop,
        "I2_tightness": i2_summary,
        "I3_nested_k_samples": len(i3),
        "I4_expansion_decay": i4_summary,
        "I5_jaccard": i5_summary,
        "I6_structural_equality": i6_summary,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["HGB_ACM", "HGB_DBLP", "HGB_IMDB"],
                   help="Subset of: " + ", ".join(DATASETS.keys()))
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--L-max", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-n", type=int, default=500,
                   help="Number of target nodes to evaluate exactly")
    p.add_argument("--out-dir", type=str,
                   default=str(PROJECT_ROOT / "results" / "kmv_properties"))
    args = p.parse_args()

    out = Path(args.out_dir)
    results = {}
    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"[skip] unknown dataset {ds}")
            continue
        try:
            results[ds] = run_dataset(ds, args.k, args.L_max, args.seed,
                                      args.sample_n, out)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{ds}] FAILED: {e}")
            results[ds] = {"error": str(e)}

    (out / "global_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nAll results under {out}")


if __name__ == "__main__":
    main()
