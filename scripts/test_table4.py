"""
==============================================================================
Table IV — F1 Scores: Global + Personalized Centrality Approximation
==============================================================================
"""
import os
import sys
import argparse

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter, GraphPrepRunner
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes, setup_global_res_dirs


def main():
    parser = argparse.ArgumentParser(description="Table IV: Centrality Approximation F1")
    parser.add_argument('dataset',    type=str, help="Dataset name (e.g., HGB_DBLP)")
    parser.add_argument('--metapath', type=str, required=True)
    parser.add_argument('--topr',     type=str, default="0.1")
    parser.add_argument('--binary',   type=str, default=config.CPP_EXECUTABLE)
    parser.add_argument('--k_glo',    type=int, default=32,  help="K for GloD (= SEED)")
    parser.add_argument('--k_glo_h',  type=int, default=4,   help="K for GloH (= SEED)")
    parser.add_argument('--k_per',    type=int, default=32,  help="K for PerD/PerD+ (= SEED)")
    parser.add_argument('--k_per_h',  type=int, default=4,   help="K for PerH/PerH+ (= SEED)")
    parser.add_argument('--beta',     type=float, default=0.1, help="Beta for PerD+/PerH+")
    args = parser.parse_args()

    folder_name = f"HGBn-{args.dataset.split('_')[1]}"
    data_dir    = os.path.join(project_root, folder_name)

    cfg = config.get_dataset_config(args.dataset)
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    PyGToCppAdapter(data_dir).convert(g_hetero)
    generate_qnodes(data_dir, folder_name, target_node_type=cfg.target_node, g_hetero=g_hetero)
    compile_rule_for_cpp(args.metapath, g_hetero, data_dir, folder_name)
    setup_global_res_dirs(folder_name, project_root)

    runner = GraphPrepRunner(binary=args.binary, working_dir=project_root, verbose=True)
    gt = runner.run_ground_truth(folder_name, topr=args.topr)

    def parse_exact_time(exact_result):
        """Extract time: token from ExactResult.stdout."""
        for line in exact_result.stdout.split('\n'):
            line = line.strip().lower()
            if line.startswith("time:"):
                try:
                    return float(line.split(":")[1].strip().replace(" s", ""))
                except ValueError:
                    pass
        return None

    exact_dp = runner.run_exact("ExactD+", folder_name, topr=args.topr)
    exact_d  = runner.run_exact("ExactD",  folder_name, topr=args.topr)
    exact_hp = runner.run_exact("ExactH+", folder_name, topr=args.topr)
    exact_h  = runner.run_exact("ExactH",  folder_name, topr=args.topr)

    t_wcd = parse_exact_time(exact_d)
    t_wch = parse_exact_time(exact_h)

    glo_d  = runner.run_glo("GloD",  folder_name, topr=args.topr, k=args.k_glo)
    glo_h  = runner.run_glo("GloH",  folder_name, topr=args.topr, k=args.k_glo_h)
    per_d  = runner.run_per("PerD",  folder_name, topr=args.topr, k=args.k_per)
    per_h  = runner.run_per("PerH",  folder_name, topr=args.topr, k=args.k_per_h)
    per_dp = runner.run_per("PerD+", folder_name, topr=args.topr, k=args.k_per,   beta=args.beta)
    per_hp = runner.run_per("PerH+", folder_name, topr=args.topr, k=args.k_per_h, beta=args.beta)

    print(f"\n[TABLE IV]  {args.dataset}  topr={args.topr}  metapath={args.metapath}")

    print(f"\n  Exact baselines (WcD / WcH):")
    if t_wcd is not None:
        print(f"  {'WcD (ExactD)':<14}  time={t_wcd:.6f}s")
    else:
        print(f"  {'WcD (ExactD)':<14}  time=N/A (no time: token in stdout)")
    if t_wch is not None:
        print(f"  {'WcH (ExactH)':<14}  time={t_wch:.6f}s")
    else:
        print(f"  {'WcH (ExactH)':<14}  time=N/A (no time: token in stdout)")

    print(f"\n  {'Method':<8}  {'F1/Acc':>8}  {'Time/rule':>10}  {'Rules':>5}  {'Speedup':>8}")
    print(f"  {'-'*50}")
    for r in [glo_d, glo_h, per_d, per_h, per_dp, per_hp]:
        score = r.avg_f1 if hasattr(r, "avg_f1") else r.avg_accuracy
        # Compute speedup vs appropriate exact baseline
        if r.task in ("GloD", "PerD", "PerD+") and t_wcd is not None and r.avg_time_s > 0:
            speedup = t_wcd / r.avg_time_s
            sp_str = f"{speedup:.1f}x"
        elif r.task in ("GloH", "PerH", "PerH+") and t_wch is not None and r.avg_time_s > 0:
            speedup = t_wch / r.avg_time_s
            sp_str = f"{speedup:.1f}x"
        else:
            sp_str = "N/A"
        print(f"  {r.task:<8}  {score:>8.4f}  {r.avg_time_s:>10.4f}  {r.rule_count:>5}  {sp_str:>8}")


if __name__ == "__main__":
    main()