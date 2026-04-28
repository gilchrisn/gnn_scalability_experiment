"""
analysis_is_bias.py — Importance-sampling bias test for MPRW vs KGRW.

For each edge (s, t) in the exact induced graph, counts the number of
L-hop paths supporting it (path_count).  Then measures whether MPRW
and KGRW preferentially discover high-path-count edges (IS bias).

MPRW IS bias prediction: discovered edges should have higher mean
path_count than the exact graph, because random walks are more likely
to traverse multi-path edges (proportional to path count).

KMV Phase 1 prediction: KGRW should be closer to the exact distribution
because min-hash assigns each source a uniform random value, independent
of how many paths connect it to any midpoint.

Usage
-----
python scripts/analysis_is_bias.py --dataset HGB_DBLP --sample 2000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import types as _t
_ts = _t.ModuleType("torch_sparse"); _ts.spspmm = None
sys.modules.setdefault("torch_sparse", _ts)

import warnings
warnings.filterwarnings("ignore")

from src.config import config
from src.data import DatasetFactory


def build_coauthor_index(g, target_type: str):
    """Build author->papers and paper->authors lookups."""
    # Find edge type from target to intermediate
    a2p = g[target_type, f"{target_type}_to_paper", "paper"].edge_index
    p2a = g["paper", f"paper_to_{target_type}", target_type].edge_index

    paper_to_nodes = defaultdict(set)
    node_to_papers = defaultdict(set)
    for a, p in zip(a2p[0].tolist(), a2p[1].tolist()):
        paper_to_nodes[p].add(a)
        node_to_papers[a].add(p)
    return node_to_papers, paper_to_nodes


def path_count_2hop(s: int, t: int,
                    node_to_papers: dict, paper_to_nodes: dict) -> int:
    """Number of (paper1, midpoint, paper2) triples connecting s and t via 4-hop path."""
    # Midpoints = 2-hop neighbors of s
    mids_s: dict[int, int] = defaultdict(int)
    for p in node_to_papers[s]:
        for m in paper_to_nodes[p]:
            if m != s:
                mids_s[m] += 1
    mids_t: dict[int, int] = defaultdict(int)
    for p in node_to_papers[t]:
        for m in paper_to_nodes[p]:
            if m != t:
                mids_t[m] += 1
    return sum(mids_s[m] * mids_t[m] for m in set(mids_s) & set(mids_t))


def load_adj_edges(adj_file: str, offset: int, n_target: int) -> list[tuple[int, int]]:
    edges = []
    with open(adj_file) as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if not parts:
                continue
            u = parts[0] - offset
            if not (0 <= u < n_target):
                continue
            for vg in parts[1:]:
                v = vg - offset
                if 0 <= v < n_target:
                    edges.append((u, v))
    return edges


def measure_bias(edges: list[tuple[int, int]],
                 pc_lookup: dict[tuple[int, int], int],
                 label: str) -> None:
    found = []
    for u, v in edges:
        if (u, v) in pc_lookup:
            found.append(pc_lookup[(u, v)])
        elif (v, u) in pc_lookup:
            found.append(pc_lookup[(v, u)])
    if not found:
        print(f"  {label:<28}  (no sampled edges found)")
        return
    m   = statistics.mean(found)
    med = statistics.median(found)
    sd  = statistics.stdev(found) if len(found) > 1 else 0.0
    sp  = 100 * sum(1 for x in found if x == 1) / len(found)
    hp  = 100 * sum(1 for x in found if x > 10) / len(found)
    n   = len(edges)
    print(f"  {label:<28}  edges={n:>7,}  mean_pc={m:>5.2f}  "
          f"stdev={sd:>5.2f}  single%={sp:>5.1f}  high%={hp:>5.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="HGB_DBLP")
    p.add_argument("--sample", type=int, default=2000,
                   help="Number of exact edges to sample for path-count computation")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = config.get_dataset_config(args.dataset)
    g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    n_target = g[cfg.target_node].num_nodes
    offset   = sum(g[nt].num_nodes for nt in sorted(g.node_types) if nt < cfg.target_node)

    print(f"Dataset: {args.dataset}  target={cfg.target_node}  n={n_target}  offset={offset}")

    # Build co-authorship index (works for author or paper target)
    target = cfg.target_node
    # Try to find the right edge types
    try:
        node_to_papers, paper_to_nodes = build_coauthor_index(g, target)
    except KeyError:
        print(f"Cannot build co-authorship index for target={target}. "
              f"Supported: 'author' (DBLP), 'paper' (ACM), 'movie' (IMDB via director).")
        # Fall back to generic 2-hop via first edge type
        first_etype = [et for et in g.edge_types if et[0] == target][0]
        mid_type = first_etype[2]
        e1 = g[first_etype].edge_index
        e2 = g[mid_type, f"{mid_type}_to_{target}", target].edge_index
        node_to_papers = defaultdict(set)
        paper_to_nodes = defaultdict(set)
        for a, p in zip(e1[0].tolist(), e1[1].tolist()):
            node_to_papers[a].add(p); paper_to_nodes[p].add(a)
        for a, p in zip(e2[0].tolist(), e2[1].tolist()):
            node_to_papers[a].add(p); paper_to_nodes[p].add(a)

    # Load exact adj
    exact_adj = ROOT / "results" / args.dataset / f"exact_{target[:4]}a{target[:2]}.adj"
    # Find by pattern
    result_dir = ROOT / "results" / args.dataset
    exact_files = list(result_dir.glob("exact_*.adj"))
    if not exact_files:
        print(f"No exact adj file found in {result_dir}. Run bench first.")
        return
    exact_adj = exact_files[0]
    print(f"Exact adj: {exact_adj.name}")

    # Load and sample exact edges
    exact_edges = load_adj_edges(str(exact_adj), offset, n_target)
    print(f"Exact edges: {len(exact_edges):,}")

    random.seed(args.seed)
    sample = random.sample(exact_edges, min(args.sample, len(exact_edges)))
    print(f"Computing path counts for {len(sample)} sampled edges...")

    pc_lookup: dict[tuple[int, int], int] = {}
    for s, t in sample:
        pc = path_count_2hop(s, t, node_to_papers, paper_to_nodes)
        pc_lookup[(s, t)] = pc

    pc_vals = list(pc_lookup.values())
    mean_pc  = statistics.mean(pc_vals)
    med_pc   = statistics.median(pc_vals)
    sd_pc    = statistics.stdev(pc_vals) if len(pc_vals) > 1 else 0.0
    sp_exact = 100 * sum(1 for x in pc_vals if x == 1) / len(pc_vals)

    print(f"\nExact graph path-count distribution:")
    print(f"  mean={mean_pc:.2f}  median={med_pc:.1f}  stdev={sd_pc:.2f}  "
          f"single-path={sp_exact:.1f}%")

    print(f"\n{'Method':<28}  {'edges':>7}  {'mean_pc':>7}  {'stdev':>6}  "
          f"{'single%':>7}  {'high%':>6}  (path-count of discovered edges)")
    print("-" * 80)
    print(f"  {'Exact (reference)':<28}  {'':>7}  {mean_pc:>7.2f}  {sd_pc:>6.2f}  "
          f"{sp_exact:>7.1f}  {'':>6}")
    print()

    # Measure MPRW adj files
    mprw_files = sorted(result_dir.glob("mprw_*_w*.adj"))
    for adj_file in mprw_files:
        edges = load_adj_edges(str(adj_file), offset, n_target)
        measure_bias(edges, pc_lookup, adj_file.name.replace(".adj", ""))

    print()
    # Measure KGRW adj files from bench CSV
    import csv
    bench_csv = result_dir / "kgrw_bench.csv"
    if bench_csv.exists():
        # Re-generate a few KGRW adj files for analysis
        print("KGRW (re-materializing for path-count analysis):")
        folder = config.get_folder_name(args.dataset)
        staging_rel = f"staging/{folder}"
        rule   = f"{staging_rel}/cod-rules_{folder}.limit"
        import subprocess
        for k, wp in [(4, 1), (4, 4), (8, 4), (16, 4), (16, 16)]:
            out = f"/tmp/bias_kgrw_k{k}_wp{wp}.adj"
            cmd = (f"cd /mnt/c/Users/Gilchris/UNI/not-school/Research/gnn/"
                   f"scalability_experiment && "
                   f"bin/mprw_exec kgrw {staging_rel} {rule} {out} {k} {wp} {args.seed}")
            subprocess.run(["wsl", "--exec", "bash", "-c", cmd],
                           capture_output=True, timeout=60)
            # Copy to Windows
            win_out = str(ROOT / "results" / "kgrw_eval" / f"tmp_bias_k{k}_wp{wp}.adj")
            subprocess.run(["wsl", "--exec", "bash", "-c",
                f"cp {out} /mnt/c/Users/Gilchris/UNI/not-school/Research/gnn/"
                f"scalability_experiment/{win_out}"], capture_output=True)
            if os.path.exists(win_out):
                edges = load_adj_edges(win_out, offset, n_target)
                measure_bias(edges, pc_lookup, f"kgrw_k{k}_wp{wp}")

    print(f"\nInterpretation:")
    print(f"  IS bias = discovered mean_pc >> exact mean_pc ({mean_pc:.2f})")
    print(f"  Uniform = discovered mean_pc ≈ exact mean_pc  (lower stdev = more uniform)")
    print(f"  MPRW over-discovers multi-path edges; KGRW Phase 1 (min-hash) breaks this bias.")


if __name__ == "__main__":
    main()
