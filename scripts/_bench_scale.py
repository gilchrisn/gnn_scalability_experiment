"""Benchmark ExactD vs GloD AND materialize vs sketch at different graph scales.
Stages 20%, 40%, 60%, 100% OGB_MAG subgraphs and times each.

Usage: python scripts/_bench_scale.py
"""
import sys, os, time, subprocess, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import HeteroData
from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from src.bridge.engine import CppEngine
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes

def p(*a): print(*a, flush=True)

def subgraph_by_year(g, fraction):
    years = g['paper'].year.squeeze()
    sorted_years, _ = years.sort()
    cutoff_idx = max(1, int(fraction * len(sorted_years))) - 1
    year_cutoff = sorted_years[cutoff_idx].item()
    keep_mask = years <= year_cutoff
    keep_ids = keep_mask.nonzero(as_tuple=False).squeeze(1)
    id_maps = {}
    id_maps['paper'] = torch.full((g['paper'].num_nodes,), -1, dtype=torch.long)
    id_maps['paper'][keep_ids] = torch.arange(keep_ids.size(0), dtype=torch.long)
    filtered_edges = {}
    referenced = {nt: set() for nt in g.node_types if nt != 'paper'}
    for src_type, rel, dst_type in g.edge_types:
        ei = g[src_type, rel, dst_type].edge_index
        src_ids, dst_ids = ei[0].clone(), ei[1].clone()
        valid = torch.ones(ei.size(1), dtype=torch.bool)
        if src_type == 'paper': valid &= id_maps['paper'][src_ids] >= 0
        if dst_type == 'paper': valid &= id_maps['paper'][dst_ids] >= 0
        src_ids, dst_ids = src_ids[valid], dst_ids[valid]
        filtered_edges[(src_type, rel, dst_type)] = (src_ids, dst_ids)
        if src_type != 'paper' and src_ids.numel() > 0: referenced[src_type].update(src_ids.tolist())
        if dst_type != 'paper' and dst_ids.numel() > 0: referenced[dst_type].update(dst_ids.tolist())
    for ntype in g.node_types:
        if ntype == 'paper': continue
        old_ids = torch.tensor(sorted(referenced[ntype]), dtype=torch.long) if referenced[ntype] else torch.zeros(0, dtype=torch.long)
        mapping = torch.full((g[ntype].num_nodes,), -1, dtype=torch.long)
        mapping[old_ids] = torch.arange(old_ids.size(0), dtype=torch.long)
        id_maps[ntype] = mapping
    g_sub = HeteroData()
    for ntype in g.node_types:
        mapping = id_maps[ntype]
        surviving = (mapping >= 0).nonzero(as_tuple=False).squeeze(1)
        for key, val in g[ntype].items():
            if isinstance(val, torch.Tensor) and val.size(0) == g[ntype].num_nodes:
                g_sub[ntype][key] = val[surviving]
            else:
                g_sub[ntype][key] = val
        g_sub[ntype].num_nodes = surviving.size(0)
    for (src_type, rel, dst_type), (src_ids, dst_ids) in filtered_edges.items():
        if src_ids.numel() == 0:
            g_sub[src_type, rel, dst_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
            continue
        g_sub[src_type, rel, dst_type].edge_index = torch.stack([id_maps[src_type][src_ids], id_maps[dst_type][dst_ids]], dim=0)
    return g_sub

def run_cmd(args, timeout=1800):
    t0 = time.perf_counter()
    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        elapsed = time.perf_counter() - t0
        for line in res.stdout.split('\n'):
            if line.strip().lower().startswith('time:'):
                try: return float(line.split(':')[1].strip())
                except: pass
        return elapsed
    except subprocess.TimeoutExpired:
        return -1
    except subprocess.CalledProcessError as e:
        if 'bad_alloc' in e.stderr:
            return -2  # OOM
        return -3

def fmt(t):
    if t == -1: return "TIMEOUT"
    if t == -2: return "OOM"
    if t < 0: return "CRASH"
    return f"{t:.2f}s"

# ---- Load full graph ----
p("Loading OGB_MAG...")
cfg = config.get_dataset_config('OGB_MAG')
g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)

folder = 'OGB_MAG'
data_dir = os.path.join('.', folder)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(f'global_res/{folder}/df1', exist_ok=True)
bin_path = config.CPP_EXECUTABLE.replace('\\', '/')

TIMEOUT = 600  # 10 min per command

import csv
csv_path = os.path.join('results', 'OGB_MAG', 'bench_scale.csv')
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
csv_fh = open(csv_path, 'w', newline='')
csv_w = csv.DictWriter(csv_fh, fieldnames=[
    'fraction', 'papers', 'total_nodes', 'edges',
    'exactd_s', 'exactd_plus_s', 'glod_k8_s', 'glod_k32_s',
    'materialize_s', 'materialize_mb', 'sketch_k8_s', 'sketch_k8_mb',
])
csv_w.writeheader()

for frac in [0.2, 0.4, 0.6, 1.0]:
    p(f"\n{'='*60}")
    p(f"  Fraction: {int(frac*100)}%")
    p(f"{'='*60}")

    if frac < 1.0:
        g_sub = subgraph_by_year(g, frac)
    else:
        g_sub = g

    total = sum(g_sub[nt].num_nodes for nt in g_sub.node_types)
    n_paper = g_sub['paper'].num_nodes
    edges = sum(g_sub[et].edge_index.size(1) for et in g_sub.edge_types)
    p(f"  papers={n_paper:,}  total_nodes={total:,}  edges={edges:,}")

    # Stage
    PyGToCppAdapter(data_dir).convert(g_sub)
    generate_qnodes(data_dir, folder, target_node_type='paper', g_hetero=g_sub)
    compile_rule_for_cpp('rev_writes,writes', g_sub, data_dir, folder)
    shutil.copy2(
        os.path.join(data_dir, f'cod-rules_{folder}.limit'),
        os.path.join(data_dir, f'{folder}-cod-global-rules.dat')
    )

    # ExactD
    p("  Running ExactD...")
    t_exact = run_cmd([bin_path, 'ExactD', folder, '0.05'], timeout=TIMEOUT)
    p(f"  ExactD:     {fmt(t_exact)}")

    # ExactD+ (needed for GloD)
    p("  Running ExactD+...")
    t_exactp = run_cmd([bin_path, 'ExactD+', folder, '0.05'], timeout=TIMEOUT)
    p(f"  ExactD+:    {fmt(t_exactp)}")

    # GloD k=8
    p("  Running GloD k=8...")
    t_glo8 = run_cmd([bin_path, 'GloD', folder, '0.05', '8'], timeout=TIMEOUT)
    p(f"  GloD k=8:   {fmt(t_glo8)}")

    # GloD k=32
    p("  Running GloD k=32...")
    t_glo32 = run_cmd([bin_path, 'GloD', folder, '0.05', '32'], timeout=TIMEOUT)
    p(f"  GloD k=32:  {fmt(t_glo32)}")

    # materialize (exact adj list)
    p("  Running materialize...")
    rule_file = os.path.join(data_dir, f'cod-rules_{folder}.limit')
    adj_file = os.path.join(data_dir, 'mat_exact.adj')
    engine = CppEngine(config.CPP_EXECUTABLE, data_dir)
    try:
        t_mat = engine.run_command('materialize', rule_file, adj_file, timeout=TIMEOUT)
        mb = os.path.getsize(adj_file) / 1024 / 1024
        p(f"  materialize: {t_mat:.2f}s  ({mb:.0f} MB)")
    except MemoryError:
        t_mat = -2
        p(f"  materialize: OOM")
    except RuntimeError:
        t_mat = -1
        p(f"  materialize: TIMEOUT")

    # sketch k=8
    p("  Running sketch k=8...")
    try:
        t_sk8 = engine.run_command('sketch', rule_file,
            os.path.join(data_dir, 'mat_sketch'), k=8, timeout=TIMEOUT)
        sk_path = os.path.join(data_dir, 'mat_sketch_0')
        sk_mb = os.path.getsize(sk_path) / 1024 / 1024 if os.path.exists(sk_path) else 0
        p(f"  sketch k=8:  {t_sk8:.2f}s  ({sk_mb:.0f} MB)")
    except MemoryError:
        p(f"  sketch k=8:  OOM")
    except RuntimeError:
        p(f"  sketch k=8:  TIMEOUT")

    # Write CSV row
    row = {
        'fraction': frac,
        'papers': g_sub['paper'].num_nodes if frac < 1.0 else g['paper'].num_nodes,
        'total_nodes': sum(g_sub[nt].num_nodes for nt in g_sub.node_types) if frac < 1.0 else sum(g[nt].num_nodes for nt in g.node_types),
        'edges': sum(g_sub[et].edge_index.size(1) for et in g_sub.edge_types) if frac < 1.0 else sum(g[et].edge_index.size(1) for et in g.edge_types),
        'exactd_s': t_exactd if t_exactd and t_exactd > 0 else '',
        'exactd_plus_s': t_exactd_plus if t_exactd_plus and t_exactd_plus > 0 else '',
        'glod_k8_s': t_glo8 if t_glo8 and t_glo8 > 0 else '',
        'glod_k32_s': t_glo32 if t_glo32 and t_glo32 > 0 else '',
        'materialize_s': t_mat if t_mat and t_mat > 0 else '',
        'materialize_mb': mb if t_mat and t_mat > 0 else '',
        'sketch_k8_s': t_sk8 if 't_sk8' in dir() and t_sk8 and t_sk8 > 0 else '',
        'sketch_k8_mb': sk_mb if 't_sk8' in dir() and t_sk8 and t_sk8 > 0 else '',
    }
    csv_w.writerow(row)
    csv_fh.flush()
    p("")

csv_fh.close()
p(f"CSV saved: {csv_path}")
p("Done.")
