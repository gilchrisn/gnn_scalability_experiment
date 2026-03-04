"""
Phase 1 for Figure 4: Raw Data Extraction.
Executes approximate centrality methods and captures SCATTER_DATA from stdout.
"""
import os
import sys
import argparse
import re

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter, GraphPrepRunner
from scripts.bench_utils import (
    compile_rule_for_cpp,
    generate_qnodes,
    setup_global_res_dirs,
)


def _parse_scatter(stdout: str, method: str) -> list:
    """Returns list of (rule_id, edges, time_s) tuples from SCATTER_DATA lines."""
    matches = re.findall(r"SCATTER_DATA:\s*([0-9.]+),([0-9.]+),([0-9.]+)", stdout)
    if not matches:
        print(f"\n[WARNING] No SCATTER_DATA found for {method}. Did you recompile with the tag?")
    return matches


def main():
    parser = argparse.ArgumentParser(description="Raw Data Extraction for Figure 4")
    parser.add_argument('dataset',    type=str)
    parser.add_argument('--metapath', type=str, required=True)
    parser.add_argument('--topr',     type=str, default="0.05")
    args = parser.parse_args()

    folder_name = f"HGBn-{args.dataset.split('_')[1]}"
    data_dir    = os.path.join(project_root, folder_name)

    print(f"\n--- INITIATING RIGOROUS EXTRACTION: {args.dataset} ---")

    # 1. Stage
    cfg = config.get_dataset_config(args.dataset)
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    PyGToCppAdapter(data_dir).convert(g_hetero)
    generate_qnodes(data_dir, folder_name)
    compile_rule_for_cpp(args.metapath, g_hetero, data_dir, folder_name)
    setup_global_res_dirs(folder_name, project_root)

    runner = GraphPrepRunner(
        binary      = config.CPP_EXECUTABLE,
        working_dir = project_root,
        verbose     = True,
    )

    # 2. Ground truths (mandatory before any approximate method)
    print("\n[Phase 1] Generating Exact Baselines...")
    runner.run_ground_truth(folder_name, topr=args.topr)

    # 3. Target methods — typed calls with prereq validation
    print("\n[Phase 2] Executing Target Methods and capturing Scatter Data...")

    results = [
        runner.run_glo("GloD",  folder_name, topr=args.topr, k=32),
        runner.run_per("PerD",  folder_name, topr=args.topr, k=32),
        runner.run_per("PerD+", folder_name, topr=args.topr, k=32, beta=0.1),
        runner.run_glo("GloH",  folder_name, topr=args.topr, k=4),
        runner.run_per("PerH",  folder_name, topr=args.topr, k=4),
        runner.run_per("PerH+", folder_name, topr=args.topr, k=4, beta=0.1),
    ]

    all_scatter_data = {}
    for result in results:
        scatter = _parse_scatter(result.stdout, result.task)
        all_scatter_data[result.task] = scatter
        print(f"\n--- {result.task} RAW SCATTER DATA ---")
        for rule_id, edges, time_s in scatter:
            print(f"  Rule ID: {rule_id},  Edges: {edges},  Time(s): {time_s}")

    print("\n--- EXTRACTION COMPLETE ---")


if __name__ == "__main__":
    main()