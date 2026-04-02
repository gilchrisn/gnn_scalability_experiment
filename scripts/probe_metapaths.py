"""Probe candidate metapaths on OGB_MAG and OAG_CS to find which ones materialize.

For each candidate metapath:
  1. Stage the data (converter + qnodes + rule bytecode)
  2. Try materialize (timeout 30 min)
  3. If materialize works: run ExactD, ExactD+, GloD (timeout 30 min each)
  4. Log results to CSV

Usage:
    python scripts/probe_metapaths.py [--timeout 1800] [--datasets OGB_MAG OAG_CS]

Run on server after recompiling:
    cd HUB && g++ -O2 -o ../bin/graph_prep main.cpp param.cpp -std=c++17 && cd ..
    python scripts/probe_metapaths.py
"""
import sys, os, time, csv, subprocess, shutil, argparse, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub torch_sparse if missing (not needed for staging/C++ calls)
if 'torch_sparse' not in sys.modules:
    try:
        import torch_sparse
    except ImportError:
        sys.modules['torch_sparse'] = types.ModuleType('torch_sparse')
        sys.modules['torch_sparse'].spspmm = None

from src.config import config
from src.data.factory import DatasetFactory
from src.bridge.converter import PyGToCppAdapter
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes

# ---- Candidate metapaths to probe ----
CANDIDATES = {
    'OGB_MAG': [
        # Already known to work (2.86B edges, 1226s)
        ("rev_writes,writes",                "PAP (co-author)"),
        # Estimated ~301M edges each — should be fast
        ("rev_cites,cites",                  "Co-citation"),
        ("cites,rev_cites",                  "Bib-coupling"),
        # Known to timeout — include for completeness
        ("has_topic,rev_has_topic",          "PFP (field)"),
    ],
    'OAG_CS': [
        # Already known to work (2M edges, 30s)
        ("rev_AP_write_first,AP_write_first", "PAP write_first"),
        # New candidates — similar fanout expected
        ("rev_AP_write_last,AP_write_last",   "PAP write_last"),
        ("rev_AP_write_other,AP_write_other", "PAP write_other"),
        # Citation-based
        ("rev_PP_cite,PP_cite",               "Co-citation"),
        ("PP_cite,rev_PP_cite",               "Bib-coupling"),
        # Known to timeout — include for completeness
        ("PF_in_L3,rev_PF_in_L3",            "PFP L3"),
        # Try finer field levels
        ("PF_in_L0,rev_PF_in_L0",            "PFP L0 (finest)"),
        ("PF_in_L1,rev_PF_in_L1",            "PFP L1"),
    ],
}


def p(*a):
    print(*a, flush=True)


def run_cmd(args, timeout=1800):
    """Run a subprocess, return (time_s, stdout) or (error_code, stderr)."""
    t0 = time.perf_counter()
    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        elapsed = time.perf_counter() - t0
        # Try to extract time from stdout
        for line in res.stdout.split('\n'):
            if line.strip().lower().startswith('time:'):
                try:
                    return float(line.split(':')[1].strip()), res.stdout
                except:
                    pass
        return elapsed, res.stdout
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -2, str(e)


def check_edge_type_exists(g, metapath):
    """Check if all edge types in the metapath exist in the graph schema."""
    edge_names = {r for _, r, _ in g.edge_types}
    for edge in metapath.split(','):
        if edge not in edge_names:
            return False, edge
    return True, None


def probe_one(dataset, folder, metapath, label, g, bin_path, timeout, csv_w, csv_fh):
    """Probe a single metapath. Returns dict of results."""
    p(f"\n  --- {label}: {metapath} ---")

    # Check schema
    ok, bad_edge = check_edge_type_exists(g, metapath)
    if not ok:
        p(f"    SKIP: edge type '{bad_edge}' not in graph schema")
        row = {'dataset': dataset, 'metapath': metapath, 'label': label,
               'status': f'SCHEMA_MISS({bad_edge})', 'materialize_s': '', 'materialize_edges': '',
               'exactd_s': '', 'exactd_plus_s': '', 'glod_s': '', 'glod_f1': ''}
        csv_w.writerow(row)
        csv_fh.flush()
        return

    # Stage rule
    try:
        compile_rule_for_cpp(metapath, g, folder, folder)
    except Exception as e:
        p(f"    SKIP: compile_rule_for_cpp failed: {e}")
        row = {'dataset': dataset, 'metapath': metapath, 'label': label,
               'status': f'COMPILE_FAIL', 'materialize_s': '', 'materialize_edges': '',
               'exactd_s': '', 'exactd_plus_s': '', 'glod_s': '', 'glod_f1': ''}
        csv_w.writerow(row)
        csv_fh.flush()
        return

    # Copy rule file to global rules location
    rule_file = os.path.join(folder, f'cod-rules_{folder}.limit')
    global_rule = os.path.join(folder, f'{folder}-cod-global-rules.dat')
    shutil.copy2(rule_file, global_rule)

    adj_file = os.path.join(folder, 'mat_probe.adj')

    # 1. Materialize
    p(f"    materialize...")
    t_mat, out_mat = run_cmd(
        [bin_path, 'materialize', folder, rule_file, adj_file],
        timeout=timeout)

    if t_mat < 0:
        status = "TIMEOUT" if t_mat == -1 else "ERROR"
        p(f"    materialize: {status}")
        row = {'dataset': dataset, 'metapath': metapath, 'label': label,
               'status': f'MAT_{status}', 'materialize_s': '',
               'materialize_edges': '', 'exactd_s': '', 'exactd_plus_s': '',
               'glod_s': '', 'glod_f1': ''}
        csv_w.writerow(row)
        csv_fh.flush()
        return

    # Count edges in adj file
    n_edges = 0
    try:
        with open(adj_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    n_edges += len(parts) - 1  # first is source node
    except:
        pass

    mat_mb = os.path.getsize(adj_file) / 1024 / 1024 if os.path.exists(adj_file) else 0
    p(f"    materialize: {t_mat:.1f}s, {n_edges:,} edges, {mat_mb:.0f} MB")

    # 2. ExactD (ground truth for degree centrality)
    p(f"    ExactD...")
    t_exact, _ = run_cmd([bin_path, 'ExactD', folder, '0.05'], timeout=timeout)
    p(f"    ExactD: {t_exact:.1f}s" if t_exact > 0 else f"    ExactD: {'TIMEOUT' if t_exact == -1 else 'ERROR'}")

    # 3. ExactD+ (ground truth for F1)
    p(f"    ExactD+...")
    t_exactp, _ = run_cmd([bin_path, 'ExactD+', folder, '0.05', '0'], timeout=timeout)
    p(f"    ExactD+: {t_exactp:.1f}s" if t_exactp > 0 else f"    ExactD+: {'TIMEOUT' if t_exactp == -1 else 'ERROR'}")

    # 4. GloD k=32
    glod_f1 = ''
    t_glo = -3
    if t_exactp > 0:  # Only run GloD if ground truth succeeded
        p(f"    GloD k=32...")
        t_glo, out_glo = run_cmd([bin_path, 'GloD', folder, '0.05', '0', '32'], timeout=timeout)
        if t_glo > 0:
            for line in out_glo.split('\n'):
                if 'goodness:' in line.lower() and 'time' not in line.lower():
                    try:
                        glod_f1 = line.split(':')[1].strip()
                    except:
                        pass
            p(f"    GloD: {t_glo:.1f}s, F1={glod_f1}")
        else:
            p(f"    GloD: {'TIMEOUT' if t_glo == -1 else 'ERROR'}")

    row = {
        'dataset': dataset,
        'metapath': metapath,
        'label': label,
        'status': 'OK',
        'materialize_s': f"{t_mat:.2f}" if t_mat > 0 else '',
        'materialize_edges': n_edges,
        'exactd_s': f"{t_exact:.2f}" if t_exact > 0 else ('TIMEOUT' if t_exact == -1 else 'ERROR'),
        'exactd_plus_s': f"{t_exactp:.2f}" if t_exactp > 0 else ('TIMEOUT' if t_exactp == -1 else 'ERROR'),
        'glod_s': f"{t_glo:.2f}" if t_glo > 0 else ('TIMEOUT' if t_glo == -1 else ''),
        'glod_f1': glod_f1,
    }
    csv_w.writerow(row)
    csv_fh.flush()


def main():
    parser = argparse.ArgumentParser(description="Probe candidate metapaths")
    parser.add_argument('--timeout', type=int, default=1800, help='Timeout per command in seconds (default 1800)')
    parser.add_argument('--datasets', nargs='+', default=['OGB_MAG', 'OAG_CS'],
                        help='Datasets to probe (default: OGB_MAG OAG_CS)')
    args = parser.parse_args()

    bin_path = config.CPP_EXECUTABLE
    if not os.path.exists(bin_path):
        p(f"ERROR: C++ binary not found at {bin_path}")
        p("Compile first: cd HUB && g++ -O2 -o ../bin/graph_prep main.cpp param.cpp -std=c++17")
        sys.exit(1)

    csv_path = os.path.join('results', 'probe_metapaths.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_fh = open(csv_path, 'w', newline='')
    csv_w = csv.DictWriter(csv_fh, fieldnames=[
        'dataset', 'metapath', 'label', 'status',
        'materialize_s', 'materialize_edges',
        'exactd_s', 'exactd_plus_s', 'glod_s', 'glod_f1',
    ])
    csv_w.writeheader()
    csv_fh.flush()

    for dataset in args.datasets:
        if dataset not in CANDIDATES:
            p(f"SKIP: {dataset} not in CANDIDATES list")
            continue

        p(f"\n{'='*60}")
        p(f"  Dataset: {dataset}")
        p(f"  Timeout: {args.timeout}s per command")
        p(f"{'='*60}")

        cfg = config.get_dataset_config(dataset)
        p(f"  Loading graph...")
        g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)

        folder = dataset
        os.makedirs(folder, exist_ok=True)
        os.makedirs(f'global_res/{folder}/df1', exist_ok=True)
        os.makedirs(f'global_res/{folder}/hf1', exist_ok=True)

        # Stage graph (once per dataset)
        p(f"  Staging graph to {folder}/...")
        PyGToCppAdapter(folder).convert(g)
        generate_qnodes(folder, folder, target_node_type=cfg.target_node, g_hetero=g)

        p(f"  Node types: {g.node_types}")
        p(f"  Edge types: {[r for _,r,_ in g.edge_types]}")
        total_nodes = sum(g[nt].num_nodes for nt in g.node_types)
        total_edges = sum(g[et].edge_index.size(1) for et in g.edge_types)
        p(f"  Total: {total_nodes:,} nodes, {total_edges:,} edges")

        for metapath, label in CANDIDATES[dataset]:
            probe_one(dataset, folder, metapath, label, g, bin_path, args.timeout, csv_w, csv_fh)

    csv_fh.close()
    p(f"\n{'='*60}")
    p(f"  Done. Results saved to {csv_path}")
    p(f"{'='*60}")


if __name__ == '__main__':
    main()
