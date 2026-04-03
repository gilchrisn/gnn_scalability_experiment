"""Probe candidate metapaths on OGB_MAG and OAG_CS.

Phase 1: Test variable metapaths (materialize + ExactD + GloD)
Phase 2: Mine AnyBURL instance rules, compile, run ExactD+ + GloD

Usage:
    python scripts/probe_metapaths.py [--timeout 1800] [--datasets OGB_MAG OAG_CS]
"""
import sys, os, time, csv, subprocess, shutil, argparse, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if 'torch_sparse' not in sys.modules:
    try:
        import torch_sparse
    except Exception:
        sys.modules['torch_sparse'] = types.ModuleType('torch_sparse')
        sys.modules['torch_sparse'].spspmm = None

from src.config import config
from src.data.factory import DatasetFactory
from src.bridge.converter import PyGToCppAdapter
from src.bridge.anyburl import AnyBURLRunner, load_validated_rules
from scripts.bench_utils import (
    compile_rule_for_cpp, compile_all_rules_for_cpp, generate_qnodes,
)

# ---- Variable metapaths to probe ----
VARIABLE_CANDIDATES = {
    'OGB_MAG': [
        ("rev_writes,writes",       "PAP"),
        ("rev_cites,cites",         "Co-citation"),
        ("cites,rev_cites",         "Bib-coupling"),
    ],
    'OAG_CS': [
        ("rev_AP_write_first,AP_write_first", "PAP write_first"),
        ("rev_AP_write_last,AP_write_last",   "PAP write_last"),
        ("rev_AP_write_other,AP_write_other", "PAP write_other"),
        ("rev_PP_cite,PP_cite",               "Co-citation"),
        ("PP_cite,rev_PP_cite",               "Bib-coupling"),
    ],
}


def p(*a):
    print(*a, flush=True)


def run_cmd(args, timeout=1800):
    t0 = time.perf_counter()
    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        elapsed = time.perf_counter() - t0
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


def extract_field(stdout, field):
    """Extract a field like 'goodness:0.98' from C++ stdout."""
    for line in stdout.split('\n'):
        if field in line.lower() and 'time' not in line.lower():
            try:
                return line.split(':')[1].strip()
            except:
                pass
    return ''


def check_edge_type_exists(g, metapath):
    edge_names = {r for _, r, _ in g.edge_types}
    for edge in metapath.split(','):
        if edge not in edge_names:
            return False, edge
    return True, None


def write_row(csv_w, csv_fh, **kwargs):
    csv_w.writerow(kwargs)
    csv_fh.flush()


# ===================== PHASE 1: Variable metapaths =====================

def probe_variable(dataset, folder, metapath, label, g, bin_path, timeout, csv_w, csv_fh):
    p(f"\n  [VAR] {label}: {metapath}")

    ok, bad_edge = check_edge_type_exists(g, metapath)
    if not ok:
        p(f"    SKIP: edge '{bad_edge}' not in schema")
        write_row(csv_w, csv_fh, dataset=dataset, metapath=metapath, label=label,
                  rule_type='variable', status=f'SCHEMA_MISS({bad_edge})',
                  n_rules='', materialize_s='', materialize_edges='',
                  exactd_plus_s='', glod_s='', glod_f1='', rule_count='')
        return

    try:
        compile_rule_for_cpp(metapath, g, folder, folder)
    except Exception as e:
        p(f"    SKIP: compile failed: {e}")
        write_row(csv_w, csv_fh, dataset=dataset, metapath=metapath, label=label,
                  rule_type='variable', status='COMPILE_FAIL',
                  n_rules='', materialize_s='', materialize_edges='',
                  exactd_plus_s='', glod_s='', glod_f1='', rule_count='')
        return

    rule_file = os.path.join(folder, f'cod-rules_{folder}.limit')
    shutil.copy2(rule_file, os.path.join(folder, f'{folder}-cod-global-rules.dat'))
    adj_file = os.path.join(folder, 'mat_probe.adj')

    # Materialize
    p(f"    materialize...")
    t_mat, _ = run_cmd([bin_path, 'materialize', folder, rule_file, adj_file], timeout=timeout)
    if t_mat < 0:
        p(f"    materialize: {'TIMEOUT' if t_mat == -1 else 'ERROR'}")
        write_row(csv_w, csv_fh, dataset=dataset, metapath=metapath, label=label,
                  rule_type='variable', status='MAT_TIMEOUT' if t_mat == -1 else 'MAT_ERROR',
                  n_rules=1, materialize_s='', materialize_edges='',
                  exactd_plus_s='', glod_s='', glod_f1='', rule_count='')
        return

    n_edges = 0
    try:
        with open(adj_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    n_edges += len(parts) - 1
    except:
        pass
    p(f"    materialize: {t_mat:.1f}s, {n_edges:,} edges")

    # ExactD+
    p(f"    ExactD+...")
    t_ep, _ = run_cmd([bin_path, 'ExactD+', folder, '0.05', '0'], timeout=timeout)
    p(f"    ExactD+: {t_ep:.1f}s" if t_ep > 0 else f"    ExactD+: FAIL")

    # GloD
    glod_f1 = ''
    t_glo = -3
    rc = ''
    if t_ep > 0:
        p(f"    GloD k=32...")
        t_glo, out = run_cmd([bin_path, 'GloD', folder, '0.05', '0', '32'], timeout=timeout)
        glod_f1 = extract_field(out, 'goodness')
        rc = extract_field(out, 'rule_count')
        p(f"    GloD: {t_glo:.1f}s, F1={glod_f1}, rules={rc}" if t_glo > 0 else f"    GloD: FAIL")

    write_row(csv_w, csv_fh, dataset=dataset, metapath=metapath, label=label,
              rule_type='variable', status='OK', n_rules=1,
              materialize_s=f"{t_mat:.2f}",
              materialize_edges=n_edges,
              exactd_plus_s=f"{t_ep:.2f}" if t_ep > 0 else 'FAIL',
              glod_s=f"{t_glo:.2f}" if t_glo > 0 else '',
              glod_f1=glod_f1, rule_count=rc)


# ===================== PHASE 2: AnyBURL instance rules =====================

def probe_instance_rules(dataset, folder, g, cfg, bin_path, timeout, csv_w, csv_fh):
    p(f"\n  [INST] Mining AnyBURL rules...")

    data_dir = os.path.join('datasets', dataset)
    os.makedirs(data_dir, exist_ok=True)
    jar_path = os.path.join('tools', 'AnyBURL-23-1x.jar')
    runner = AnyBURLRunner(data_dir, jar_path)

    # Export triples + mine
    runner.export_for_mining(g)
    runner.run_mining(timeout=300, max_length=4)  # 5 min mining

    rules_file = runner.rules_file
    if not os.path.exists(rules_file) or os.path.getsize(rules_file) == 0:
        p(f"    No rules mined")
        write_row(csv_w, csv_fh, dataset=dataset, metapath='(AnyBURL mining)', label='instance',
                  rule_type='instance', status='NO_RULES',
                  n_rules=0, materialize_s='', materialize_edges='',
                  exactd_plus_s='', glod_s='', glod_f1='', rule_count='')
        return

    rules, stats = load_validated_rules(rules_file, g, cfg.target_node,
                                        min_conf=0.1, max_n=500)
    n_var = sum(1 for _, iid in rules if iid == -1)
    n_inst = sum(1 for _, iid in rules if iid != -1)
    p(f"    Mined: {n_var} variable + {n_inst} instance rules")
    p(f"    Stats: {stats}")

    if not rules:
        write_row(csv_w, csv_fh, dataset=dataset, metapath='(AnyBURL)', label='instance',
                  rule_type='instance', status='NO_VALID_RULES',
                  n_rules=0, materialize_s='', materialize_edges='',
                  exactd_plus_s='', glod_s='', glod_f1='', rule_count='')
        return

    # Show sample rules
    from collections import Counter
    path_counts = Counter(path for path, iid in rules if iid != -1)
    p(f"    Instance rules by metapath:")
    for path, cnt in path_counts.most_common(10):
        p(f"      {path}: {cnt} instances")

    # Compile all rules into one file
    n = compile_all_rules_for_cpp(rules, g, folder, folder)
    p(f"    Compiled {n} rules")

    # ExactD+ (ground truth)
    p(f"    ExactD+...")
    t_ep, out_ep = run_cmd([bin_path, 'ExactD+', folder, '0.05', '0'], timeout=timeout)
    rc_ep = extract_field(out_ep, 'rule_count')
    p(f"    ExactD+: {t_ep:.1f}s, rule_count={rc_ep}" if t_ep > 0 else f"    ExactD+: FAIL")

    # GloD
    glod_f1 = ''
    t_glo = -3
    rc_glo = ''
    if t_ep > 0:
        p(f"    GloD k=32...")
        t_glo, out_glo = run_cmd([bin_path, 'GloD', folder, '0.05', '0', '32'], timeout=timeout)
        glod_f1 = extract_field(out_glo, 'goodness')
        rc_glo = extract_field(out_glo, 'rule_count')
        p(f"    GloD: {t_glo:.1f}s, F1={glod_f1}, rule_count={rc_glo}" if t_glo > 0 else f"    GloD: FAIL")

    # Summarize unique metapaths
    metapaths_str = '; '.join(f"{p}({c})" for p, c in path_counts.most_common(5))

    write_row(csv_w, csv_fh, dataset=dataset, metapath=metapaths_str, label='AnyBURL instance',
              rule_type='instance', status='OK', n_rules=n,
              materialize_s='',
              materialize_edges='',
              exactd_plus_s=f"{t_ep:.2f}" if t_ep > 0 else 'FAIL',
              glod_s=f"{t_glo:.2f}" if t_glo > 0 else '',
              glod_f1=glod_f1, rule_count=rc_glo)


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(description="Probe metapaths (variable + instance)")
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--datasets', nargs='+', default=['OGB_MAG', 'OAG_CS'])
    parser.add_argument('--skip-variable', action='store_true', help='Skip variable metapath probing')
    parser.add_argument('--skip-instance', action='store_true', help='Skip AnyBURL instance probing')
    args = parser.parse_args()

    bin_path = config.CPP_EXECUTABLE
    if not os.path.exists(bin_path):
        p(f"ERROR: binary not found at {bin_path}")
        p("Compile: cd HUB && g++ -O2 -o ../bin/graph_prep main.cpp param.cpp -std=c++17")
        sys.exit(1)

    csv_path = os.path.join('results', 'probe_metapaths.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_fh = open(csv_path, 'w', newline='')
    csv_w = csv.DictWriter(csv_fh, fieldnames=[
        'dataset', 'metapath', 'label', 'rule_type', 'status',
        'n_rules', 'materialize_s', 'materialize_edges',
        'exactd_plus_s', 'glod_s', 'glod_f1', 'rule_count',
    ])
    csv_w.writeheader()
    csv_fh.flush()

    for dataset in args.datasets:
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

        p(f"  Staging graph to {folder}/...")
        PyGToCppAdapter(folder).convert(g)
        generate_qnodes(folder, folder, target_node_type=cfg.target_node, g_hetero=g)

        total_nodes = sum(g[nt].num_nodes for nt in g.node_types)
        total_edges = sum(g[et].edge_index.size(1) for et in g.edge_types)
        p(f"  {total_nodes:,} nodes, {total_edges:,} edges")

        # Phase 1: Variable metapaths
        if not args.skip_variable and dataset in VARIABLE_CANDIDATES:
            p(f"\n  === Phase 1: Variable metapaths ===")
            for metapath, label in VARIABLE_CANDIDATES[dataset]:
                probe_variable(dataset, folder, metapath, label, g, bin_path,
                               args.timeout, csv_w, csv_fh)

        # Phase 2: AnyBURL instance rules
        if not args.skip_instance:
            p(f"\n  === Phase 2: AnyBURL instance rules ===")
            probe_instance_rules(dataset, folder, g, cfg, bin_path,
                                 args.timeout, csv_w, csv_fh)

    csv_fh.close()
    p(f"\n{'='*60}")
    p(f"  Done. Results in {csv_path}")
    p(f"{'='*60}")


if __name__ == '__main__':
    main()
