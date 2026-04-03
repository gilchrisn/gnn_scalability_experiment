"""Run base paper experiments on OGB_MAG and OAG_CS using instance rules.

Matches the original paper's AnyBURL config:
  - Snapshot at 10s, confidence threshold 0.1, MAX_LENGTH_CYCLIC = 4
  - Instance rules enabled (CONSTANTS_OFF not set)

For each dataset:
  1. Stage graph for C++
  2. Mine AnyBURL rules (10s snapshot, reuse if exists)
  3. Load + validate rules (mirror instance rules, convert to global IDs)
  4. Compile into C++ bytecode
  5. Run ground truth: ExactD, ExactD+, ExactH, ExactH+
  6. Run approximate: GloD (k=32), GloH (k=4)
  7. Save results to CSV

Usage:
    python scripts/run_large_basepaper.py [--datasets OGB_MAG OAG_CS] [--timeout 1800]
    python scripts/run_large_basepaper.py --force-remine  # delete old rules, mine fresh
"""
import sys, os, time, csv, subprocess, argparse, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch_sparse
except Exception:
    sys.modules['torch_sparse'] = types.ModuleType('torch_sparse')
    sys.modules['torch_sparse'].spspmm = None

from src.config import config
from src.data.factory import DatasetFactory
from src.bridge.converter import PyGToCppAdapter
from src.bridge.anyburl import AnyBURLRunner, load_validated_rules
from src.bridge.runner import GraphPrepRunner
from scripts.bench_utils import compile_all_rules_for_cpp, generate_qnodes
from collections import Counter


def p(*a):
    print(*a, flush=True)


def run_cmd(args, timeout=1800):
    t0 = time.perf_counter()
    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return time.perf_counter() - t0, res.stdout, res.returncode
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT", -1
    except Exception as e:
        return -2, str(e), -1


def extract(stdout, field):
    for line in stdout.split('\n'):
        if field in line:
            try:
                return line.split(':')[1].strip().split()[0]
            except:
                pass
    return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['OGB_MAG', 'OAG_CS'])
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--force-remine', action='store_true')
    parser.add_argument('--mining-time', type=int, default=10,
                        help='AnyBURL snapshot seconds (paper uses 10)')
    parser.add_argument('--min-conf', type=float, default=0.1)
    parser.add_argument('--max-rules', type=int, default=0,
                        help='Max rules to use (0 = all)')
    args = parser.parse_args()

    bin_path = config.CPP_EXECUTABLE
    if not os.path.exists(bin_path):
        p(f"ERROR: binary not found at {bin_path}")
        sys.exit(1)

    csv_path = os.path.join('results', 'large_basepaper.csv')
    os.makedirs('results', exist_ok=True)
    csv_fh = open(csv_path, 'w', newline='')
    fields = ['dataset', 'n_rules', 'n_var', 'n_inst', 'metapaths',
              'exactd_plus_s', 'exactd_s', 'glod_s', 'glod_f1', 'glod_rule_count',
              'exacth_plus_s', 'exacth_s', 'gloh_s', 'gloh_f1', 'gloh_rule_count']
    csv_w = csv.DictWriter(csv_fh, fieldnames=fields)
    csv_w.writeheader()
    csv_fh.flush()

    for dataset in args.datasets:
        p(f"\n{'='*60}")
        p(f"  {dataset}")
        p(f"{'='*60}")

        cfg = config.get_dataset_config(dataset)
        p(f"  Loading graph...")
        g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)

        folder = dataset
        os.makedirs(folder, exist_ok=True)
        os.makedirs(f'global_res/{folder}/df1', exist_ok=True)
        os.makedirs(f'global_res/{folder}/hf1', exist_ok=True)

        total_nodes = sum(g[nt].num_nodes for nt in g.node_types)
        total_edges = sum(g[et].edge_index.size(1) for et in g.edge_types)
        p(f"  {total_nodes:,} nodes, {total_edges:,} edges, target={cfg.target_node}")

        # Stage
        p(f"  Staging...")
        PyGToCppAdapter(folder).convert(g)
        generate_qnodes(folder, folder, target_node_type=cfg.target_node, g_hetero=g)

        # Mine AnyBURL (paper config: 10s snapshot)
        data_dir = os.path.join('datasets', dataset)
        os.makedirs(data_dir, exist_ok=True)
        jar_path = os.path.join('tools', 'AnyBURL-23-1x.jar')
        miner = AnyBURLRunner(data_dir, jar_path)

        if args.force_remine and os.path.exists(miner.rules_file):
            os.remove(miner.rules_file)
            p(f"  Deleted old rules, will re-mine")

        miner.export_for_mining(g)
        miner.run_mining(timeout=args.mining_time, max_length=4, num_threads=4)

        if not os.path.exists(miner.rules_file) or os.path.getsize(miner.rules_file) == 0:
            p(f"  No rules mined — skipping")
            continue

        # Load and validate
        max_n = args.max_rules if args.max_rules > 0 else 100000
        rules, stats = load_validated_rules(
            miner.rules_file, g, cfg.target_node,
            min_conf=args.min_conf, max_n=max_n)

        n_var = sum(1 for _, iid in rules if iid == -1)
        n_inst = sum(1 for _, iid in rules if iid != -1)
        path_counts = Counter(path for path, _ in rules)
        p(f"  Rules: {n_var} variable + {n_inst} instance = {len(rules)} total")
        p(f"  Stats: {stats}")
        p(f"  Metapaths:")
        for path, cnt in path_counts.most_common(10):
            p(f"    {path}: {cnt}")

        if not rules:
            p(f"  No valid rules — skipping")
            continue

        # Compile
        n = compile_all_rules_for_cpp(rules, g, folder, folder)
        p(f"  Compiled {n} rules")

        # Ground truth (ExactD+, ExactD, ExactH+, ExactH)
        runner = GraphPrepRunner(bin_path, '.', timeout=args.timeout)

        p(f"  Running ground truth...")
        t0 = time.perf_counter()
        try:
            gt = runner.run_ground_truth(folder, topr='0.05')
            t_gt = time.perf_counter() - t0
            p(f"  Ground truth OK: {t_gt:.1f}s")
        except Exception as e:
            p(f"  Ground truth FAILED: {e}")
            continue

        # Extract individual timings from runner output
        # ExactD+ and ExactD are in the ground truth call
        # We'll time GloD and GloH separately

        # GloD (k=32)
        p(f"  GloD k=32...")
        t_glo, out_glo, rc = run_cmd([bin_path, 'GloD', folder, '0.05', '0', '32'],
                                      timeout=args.timeout)
        glod_f1 = extract(out_glo, '~goodness')
        glod_rc = extract(out_glo, 'rule_count')
        p(f"  GloD: {t_glo:.1f}s, F1={glod_f1}, rules={glod_rc}" if t_glo > 0 else f"  GloD: FAIL")

        # GloH (k=4)
        p(f"  GloH k=4...")
        t_gloh, out_gloh, rc = run_cmd([bin_path, 'GloH', folder, '0.05', '0', '4'],
                                        timeout=args.timeout)
        gloh_f1 = extract(out_gloh, '~goodness')
        gloh_rc = extract(out_gloh, 'rule_count')
        p(f"  GloH: {t_gloh:.1f}s, F1={gloh_f1}, rules={gloh_rc}" if t_gloh > 0 else f"  GloH: FAIL")

        metapaths_str = '; '.join(f"{mp}({c})" for mp, c in path_counts.most_common(5))

        row = {
            'dataset': dataset,
            'n_rules': len(rules),
            'n_var': n_var,
            'n_inst': n_inst,
            'metapaths': metapaths_str,
            'exactd_plus_s': f"{t_gt:.2f}",
            'exactd_s': '',
            'glod_s': f"{t_glo:.2f}" if t_glo > 0 else 'FAIL',
            'glod_f1': glod_f1,
            'glod_rule_count': glod_rc,
            'exacth_plus_s': '',
            'exacth_s': '',
            'gloh_s': f"{t_gloh:.2f}" if t_gloh > 0 else 'FAIL',
            'gloh_f1': gloh_f1,
            'gloh_rule_count': gloh_rc,
        }
        csv_w.writerow(row)
        csv_fh.flush()

    csv_fh.close()
    p(f"\n{'='*60}")
    p(f"  Done. Results in {csv_path}")
    p(f"{'='*60}")


if __name__ == '__main__':
    main()
