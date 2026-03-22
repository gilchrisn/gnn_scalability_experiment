"""Quick benchmark: sketch K=2,4,8 vs exact on 20% OGB_MAG."""
import sys, os, torch, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from src.bridge.engine import CppEngine
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes
from torch_geometric.data import HeteroData

def subgraph_20(g):
    years = g['paper'].year.squeeze()
    sorted_years, _ = years.sort()
    cutoff_idx = max(1, int(0.2 * len(sorted_years))) - 1
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

def p(*args): print(*args, flush=True)

cfg = config.get_dataset_config('OGB_MAG')
p("Loading OGB_MAG...")
g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
p("Subgraphing to 20%...")
g_sub = subgraph_20(g)
total = sum(g_sub[nt].num_nodes for nt in g_sub.node_types)
p(f"  Total nodes: {total:,}")

folder = 'OGB_MAG'
data_dir = os.path.join('.', folder)
os.makedirs(data_dir, exist_ok=True)
PyGToCppAdapter(data_dir).convert(g_sub)
generate_qnodes(data_dir, folder, target_node_type='paper', g_hetero=g_sub)
engine = CppEngine(config.CPP_EXECUTABLE, data_dir)
rule_file = os.path.join(data_dir, f'cod-rules_{folder}.limit')

for mp_name, mp_str in [("PAP", "rev_writes,writes"), ("PCP", "cites,rev_cites")]:
    p(f"\n{'='*50}")
    p(f"  {mp_name} ({mp_str}) on 20% OGB_MAG")
    p(f"{'='*50}")
    compile_rule_for_cpp(mp_str, g_sub, data_dir, folder)

    p("  Exact...")
    t_exact = engine.run_command('materialize', rule_file,
        os.path.join(data_dir, 'mat_exact.adj'), timeout=300)
    mb = os.path.getsize(os.path.join(data_dir, 'mat_exact.adj')) / 1024 / 1024
    p(f"  Exact: {t_exact:.2f}s  ({mb:.1f} MB)")

    for k in [2, 4, 8]:
        p(f"  Sketch K={k}...")
        try:
            t_sk = engine.run_command('sketch', rule_file,
                os.path.join(data_dir, 'mat_sketch'), k=k, timeout=300)
            sk_path = os.path.join(data_dir, 'mat_sketch_0')
            sk_mb = os.path.getsize(sk_path) / 1024 / 1024 if os.path.exists(sk_path) else 0
            p(f"  K={k}: {t_sk:.2f}s  ({sk_mb:.1f} MB)  ratio={t_sk/t_exact:.2f}x")
        except Exception as e:
            p(f"  K={k}: FAILED — {e}")

p("\nDone.")
