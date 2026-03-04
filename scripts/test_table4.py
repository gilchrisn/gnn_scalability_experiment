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

    print(f"\n{'='*60}")
    print(f"TABLE IV: {args.dataset}  topr={args.topr}")
    print(f"{'='*60}")

    # Step 1: Stage dataset
    print("\n>>> [Step 1] Staging Data...")
    cfg = config.get_dataset_config(args.dataset)
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    PyGToCppAdapter(data_dir).convert(g_hetero)
    generate_qnodes(data_dir, folder_name)
    compile_rule_for_cpp(args.metapath, g_hetero, data_dir, folder_name)
    setup_global_res_dirs(folder_name, project_root)

    # Step 2: Ground truth (typed, topr-safe, prereq-validated)
    print("\n>>> [Step 2] Generating Ground Truth...")
    runner = GraphPrepRunner(
        binary      = args.binary,
        working_dir = project_root,
        verbose     = True,
    )
    runner.run_ground_truth(folder_name, topr=args.topr)

    # Step 3: Approximate methods
    print("\n>>> [Step 3] Running Approximate Methods...")
    glo_d  = runner.run_glo("GloD",  folder_name, topr=args.topr, k=args.k_glo)
    glo_h  = runner.run_glo("GloH",  folder_name, topr=args.topr, k=args.k_glo_h)
    per_d  = runner.run_per("PerD",  folder_name, topr=args.topr, k=args.k_per)
    per_h  = runner.run_per("PerH",  folder_name, topr=args.topr, k=args.k_per_h)
    per_dp = runner.run_per("PerD+", folder_name, topr=args.topr, k=args.k_per,   beta=args.beta)
    per_hp = runner.run_per("PerH+", folder_name, topr=args.topr, k=args.k_per_h, beta=args.beta)

    # Step 4: Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS — {args.dataset}  topr={args.topr}")
    print(f"{'='*60}")
    print(f"  {'Method':<12}  {'Avg F1/Acc':>10}  {'Time/rule (s)':>14}  {'Rules':>6}")
    print(f"  {'-'*50}")
    for result in [glo_d, glo_h, per_d, per_h, per_dp, per_hp]:
        score = result.avg_f1 if hasattr(result, "avg_f1") else result.avg_accuracy
        print(
            f"  {result.task:<12}  {score:>10.4f}  "
            f"{result.avg_time_s:>14.4f}  {result.rule_count:>6}"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()