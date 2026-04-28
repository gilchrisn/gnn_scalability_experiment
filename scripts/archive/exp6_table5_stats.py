"""
exp6_table5_stats.py — Compute Table 5 (HGNN evaluation graph statistics).

For each dataset, picks the densest available meta-path (maximises |E_exact|),
then reports:
    |V|       : peer count (target nodes participating in the meta-path BFS)
    |E*|      : matching graph edges (active edges in G*, forward+backward pruned)
    |E_exact| : exact relational graph edges (materialized H, the OOM bottleneck)
    |E_kmv|   : KMV-reconstructed edges at k=32 (≤ |V|·32)
    rho_exact : exact graph density = |E_exact| / (|V| × (|V| - 1))

Methodology:
    - |V| and |E_exact| are counted from the C++ graph_prep materialize output.
    - |E*| uses a forward+backward BFS through the meta-path hops on the PyG
      HeteroData graph, counting only edges between nodes that are active at
      both endpoints (i.e., nodes that lie on at least one complete meta-path
      instance from a peer to a peer).
    - |E_kmv| is counted from graph_prep sketch output at k=32.
    - Global↔local ID conversion is handled via sorted-alphabetical offsets
      (matching PyGToCppAdapter's concatenation order).

Usage:
    python scripts/exp6_table5_stats.py [--datasets HGB_ACM HGB_DBLP ...]

Output:
    results/table5_hgnn_stats.csv
    Also prints a formatted table to stdout.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.factory import DatasetFactory
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes

GRAPH_PREP = "./bin/graph_prep"

# All candidate metapaths per dataset — densest is selected automatically.
CANDIDATES: Dict[str, List[str]] = {
    "HGB_ACM": [
        "paper_to_term,term_to_paper",
        "paper_to_author,author_to_paper",
        "paper_to_subject,subject_to_paper",
    ],
    "HGB_DBLP": [
        "author_to_paper,paper_to_author",
        "author_to_paper,paper_to_term,term_to_paper,paper_to_author",
    ],
    "HGB_IMDB": [
        "movie_to_keyword,keyword_to_movie",
        "movie_to_actor,actor_to_movie",
        "movie_to_director,director_to_movie",
    ],
    "HNE_PubMed": [
        "disease_to_chemical,chemical_to_disease",
        "disease_to_gene,gene_to_disease",
        "disease_to_species,species_to_disease",
    ],
}

# DatasetFactory (source, name, target_type) and C++ folder name
META = {
    "HGB_ACM":    ("HGB", "ACM",    "paper",   "HGBn-ACM"),
    "HGB_DBLP":   ("HGB", "DBLP",   "author",  "HGBn-DBLP"),
    "HGB_IMDB":   ("HGB", "IMDB",   "movie",   "HGBn-IMDB"),
    "HNE_PubMed": ("HNE", "PubMed", "disease", "HNE_PubMed"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_adj(path: str) -> Tuple[int, Set[int]]:
    """Count total directed edges and collect peer node IDs from adj file."""
    n_edges = 0
    peers: Set[int] = set()
    if not os.path.exists(path):
        return 0, set()
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 1:
                peers.add(int(parts[0]))
                n_edges += len(parts) - 1
    return n_edges, peers


def _global_offsets(g) -> Dict[str, int]:
    """Global ID offset per node type (sorted alphabetically, matching C++)."""
    sorted_types = sorted(g.node_types)
    offsets: Dict[str, int] = {}
    running = 0
    for nt in sorted_types:
        offsets[nt] = running
        running += g[nt].num_nodes
    return offsets


def _compute_e_star(g, metapath_str: str, peer_global_ids: Set[int]) -> Tuple[int, List[Tuple[str, int]]]:
    """
    BFS through meta-path hops counting only active edges.

    Forward pass:  from peers, follow each edge type, track reachable nodes.
    Backward pass: from peers (terminal), reverse-follow, track reachable nodes.
    Intersect at each layer to get truly active nodes.
    Count edges between active nodes at consecutive layers.

    Returns (total_e_star, per_hop_details).
    """
    steps = metapath_str.split(",")
    offsets = _global_offsets(g)

    # Resolve edge types
    step_info = []
    for step in steps:
        for (s, r, d), ei in g.edge_index_dict.items():
            if r == step:
                step_info.append((s, r, d, ei))
                break
        else:
            raise ValueError(f"Edge type '{step}' not found in graph")

    # Layer node types
    layer_types = [step_info[0][0]]
    for _, _, d, _ in step_info:
        layer_types.append(d)

    n_layers = len(steps) + 1

    # Convert peer global IDs → local IDs of layer 0 type
    offset_0 = offsets[layer_types[0]]
    fwd_active: List[Set[int]] = [set() for _ in range(n_layers)]
    fwd_active[0] = {pid - offset_0 for pid in peer_global_ids}

    # Forward BFS
    for i, (s, r, d, ei) in enumerate(step_info):
        src_np = ei[0].numpy()
        dst_np = ei[1].numpy()
        prev = fwd_active[i]
        nxt: Set[int] = set()
        for j in range(len(src_np)):
            if src_np[j] in prev:
                nxt.add(int(dst_np[j]))
        fwd_active[i + 1] = nxt

    # Backward BFS: terminal = peers in local IDs of last layer type
    offset_L = offsets[layer_types[-1]]
    bwd_active: List[Set[int]] = [set() for _ in range(n_layers)]
    bwd_active[-1] = {pid - offset_L for pid in peer_global_ids}

    for i in range(len(steps) - 1, -1, -1):
        _, _, _, ei = step_info[i]
        src_np = ei[0].numpy()
        dst_np = ei[1].numpy()
        nxt_active = bwd_active[i + 1]
        prev: Set[int] = set()
        for j in range(len(src_np)):
            if dst_np[j] in nxt_active:
                prev.add(int(src_np[j]))
        bwd_active[i] = prev

    # Intersect forward ∩ backward
    active = [fwd_active[i] & bwd_active[i] for i in range(n_layers)]

    # Count active edges per hop
    total = 0
    details: List[Tuple[str, int]] = []
    for i, (s, r, d, ei) in enumerate(step_info):
        src_np = ei[0].numpy()
        dst_np = ei[1].numpy()
        a_src = active[i]
        a_dst = active[i + 1]
        count = 0
        for j in range(len(src_np)):
            if src_np[j] in a_src and dst_np[j] in a_dst:
                count += 1
        total += count
        details.append((r, count))

    return total, details


def _stage(g, mp: str, target: str, folder: str) -> str:
    """Stage rule file and qnodes. Returns rule file path."""
    rule_path = os.path.join(folder, f"cod-rules_{folder}.limit")
    compile_rule_for_cpp(metapath_str=mp, g_hetero=g, data_dir=folder, folder_name=folder)
    generate_qnodes(data_dir=folder, folder_name=folder,
                    target_node_type=target, g_hetero=g)
    return rule_path


def _materialize(folder: str, rule: str, adj_path: str, timeout: int = 600) -> bool:
    """Run graph_prep materialize. Returns True on success."""
    os.makedirs(os.path.dirname(adj_path), exist_ok=True)
    res = subprocess.run(
        [GRAPH_PREP, "materialize", folder, rule, adj_path],
        capture_output=True, text=True, timeout=timeout,
    )
    return res.returncode == 0


def _sketch(folder: str, rule: str, out_prefix: str, k: int = 32,
            seed: int = 0, timeout: int = 600) -> str:
    """Run graph_prep sketch. Returns path to output file."""
    subprocess.run(
        [GRAPH_PREP, "sketch", folder, rule, out_prefix, str(k), "1", str(seed)],
        capture_output=True, text=True, timeout=timeout,
    )
    return out_prefix + "_0"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+",
                        default=["HGB_ACM", "HGB_DBLP", "HGB_IMDB", "HNE_PubMed"])
    parser.add_argument("--k", type=int, default=32, help="KMV sketch size")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", default="results/table5_hgnn_stats.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []

    for ds_key in args.datasets:
        if ds_key not in META:
            print(f"[WARN] Unknown dataset: {ds_key}, skipping")
            continue

        src, name, target, folder = META[ds_key]
        g, info = DatasetFactory.get_data(src, name, target)
        target_v_total = g[target].num_nodes

        print(f"\n{'='*65}")
        print(f"  {ds_key}  (total {target} nodes: {target_v_total:,})")
        print(f"{'='*65}")

        # --- Find densest metapath ---
        best_mp = None
        best_exact = 0
        best_adj = None

        for mp in CANDIDATES.get(ds_key, []):
            rule = _stage(g, mp, target, folder)
            adj = f"results/{ds_key}/exact_{mp.replace(',', '_')}.adj"

            if not _materialize(folder, rule, adj, args.timeout):
                print(f"  {mp}: MATERIALIZE FAILED")
                continue

            ee, peers = _count_adj(adj)
            print(f"  {mp}: |E_exact|={ee:,}  peers={len(peers):,}")

            if ee > best_exact:
                best_exact = ee
                best_mp = mp
                best_adj = adj

        if best_mp is None:
            print(f"  NO VALID METAPATH for {ds_key}")
            continue

        print(f"  >>> Densest: {best_mp}")

        # --- Compute all stats for densest metapath ---
        # Reuse the candidate adj file (already materialized correctly).
        e_exact, peer_set = _count_adj(best_adj)
        v = len(peer_set)

        # E* via BFS
        e_star, hop_details = _compute_e_star(g, best_mp, peer_set)
        for rel, cnt in hop_details:
            print(f"      {rel}: {cnt:,} active edges")

        # KMV sketch — re-stage the densest metapath rule for sketch
        rule = _stage(g, best_mp, target, folder)
        sk_path = _sketch(folder, rule, f"results/{ds_key}/sketch_k{args.k}_densest",
                          k=args.k, timeout=args.timeout)
        e_kmv, _ = _count_adj(sk_path) if os.path.exists(sk_path) else (0, set())

        rho = e_exact / (v * (v - 1)) if v > 1 else 0.0

        rows.append({
            "Dataset":   ds_key,
            "MetaPath":  best_mp,
            "V":         v,
            "E_star":    e_star,
            "E_exact":   e_exact,
            "E_kmv":     e_kmv,
            "k":         args.k,
            "rho_exact": rho,
        })

        print(f"\n  |V|={v:,}  |E*|={e_star:,}  |E_exact|={e_exact:,}  "
              f"|E_kmv|={e_kmv:,}  rho={rho:.4e}")

    # --- Write CSV ---
    fields = ["Dataset", "MetaPath", "V", "E_star", "E_exact", "E_kmv", "k", "rho_exact"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written to: {args.out}")

    # --- Summary table ---
    print(f"\n{'='*90}")
    print(f"{'Dataset':<12} {'MetaPath':<50} {'|V|':>8} {'|E*|':>12} {'|E_exact|':>14} {'|E_kmv|':>10} {'rho':>10}")
    print(f"{'-'*90}")
    for r in rows:
        print(f"{r['Dataset']:<12} {r['MetaPath']:<50} {r['V']:>8,} {r['E_star']:>12,} "
              f"{r['E_exact']:>14,} {r['E_kmv']:>10,} {r['rho_exact']:>10.4e}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
