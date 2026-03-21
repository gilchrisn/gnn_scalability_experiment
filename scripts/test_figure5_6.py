"""
Figures 5 & 6 Raw Data Extraction.
Evaluates F1/Accuracy scores against varying lambda (Top-R) and k (Sketch Size).
Outputs raw CSV for downstream validation.
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
from scripts.bench_utils import (
    compile_rule_for_cpp,
    generate_qnodes,
    setup_global_res_dirs,
)


LAMBDAS        = ["0.02", "0.03", "0.04", "0.05"]
K_VALUES       = [2, 4, 8, 16, 32]
DEFAULT_LAMBDA = "0.05"
DEFAULT_K_DEG  = 32
DEFAULT_K_HIDX = 4



def main():
    parser = argparse.ArgumentParser(description="Raw Extraction for Figures 5 & 6")
    parser.add_argument('dataset',    type=str)
    parser.add_argument('--metapath', type=str, required=True)
    args = parser.parse_args()

    folder_name = f"HGBn-{args.dataset.split('_')[1]}"
    data_dir    = os.path.join(project_root, folder_name)

    # 1. Stage
    cfg = config.get_dataset_config(args.dataset)
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    PyGToCppAdapter(data_dir).convert(g_hetero)
    generate_qnodes(data_dir, folder_name, target_node_type=cfg.target_node, g_hetero=g_hetero)
    compile_rule_for_cpp(args.metapath, g_hetero, data_dir, folder_name)
    setup_global_res_dirs(folder_name, project_root)

    runner = GraphPrepRunner(
        binary      = config.CPP_EXECUTABLE,
        working_dir = project_root,
        verbose     = False,  # suppress per-run C++ stdout in sweep mode
    )

    # Result accumulators: {method_name: [f1_per_lambda_or_k, ...]}
    res_lam_deg  = {m: [] for m in ["GloD", "PerD", "PerD+"]}
    res_lam_hidx = {m: [] for m in ["GloH", "PerH", "PerH+"]}
    res_k_deg    = {m: [] for m in ["GloD", "PerD", "PerD+"]}
    res_k_hidx   = {m: [] for m in ["GloH", "PerH", "PerH+"]}

    print(f"\n[System] Starting F1 Extraction for {args.dataset} | Metapath: {args.metapath}")

    # --- Experiment 1: Varying Lambda ---
    print(f"\n[System] Lambda Sweep (k_deg={DEFAULT_K_DEG}, k_hidx={DEFAULT_K_HIDX})...")
    for lam in LAMBDAS:
        # Ground truth generated fresh for each lambda; topr canonicalization
        # inside run_ground_truth prevents "0.02" vs "0.020" filename divergence.
        runner.run_ground_truth(folder_name, topr=lam)

        res_lam_deg["GloD"].append( runner.run_glo("GloD",  folder_name, topr=lam, k=DEFAULT_K_DEG).avg_f1)
        res_lam_deg["PerD"].append( runner.run_per("PerD",  folder_name, topr=lam, k=DEFAULT_K_DEG).avg_accuracy)
        res_lam_deg["PerD+"].append(runner.run_per("PerD+", folder_name, topr=lam, k=DEFAULT_K_DEG, beta=0.1).avg_accuracy)

        res_lam_hidx["GloH"].append( runner.run_glo("GloH",  folder_name, topr=lam, k=DEFAULT_K_HIDX).avg_f1)
        res_lam_hidx["PerH"].append( runner.run_per("PerH",  folder_name, topr=lam, k=DEFAULT_K_HIDX).avg_accuracy)
        res_lam_hidx["PerH+"].append(runner.run_per("PerH+", folder_name, topr=lam, k=DEFAULT_K_HIDX, beta=0.1).avg_accuracy)

    # --- Experiment 2: Varying K ---
    print(f"\n[System] K Sweep (lambda={DEFAULT_LAMBDA})...")
    runner.run_ground_truth(folder_name, topr=DEFAULT_LAMBDA)

    for k in K_VALUES:
        res_k_deg["GloD"].append( runner.run_glo("GloD",  folder_name, topr=DEFAULT_LAMBDA, k=k).avg_f1)
        res_k_deg["PerD"].append( runner.run_per("PerD",  folder_name, topr=DEFAULT_LAMBDA, k=k).avg_accuracy)
        res_k_deg["PerD+"].append(runner.run_per("PerD+", folder_name, topr=DEFAULT_LAMBDA, k=k, beta=0.1).avg_accuracy)

        res_k_hidx["GloH"].append( runner.run_glo("GloH",  folder_name, topr=DEFAULT_LAMBDA, k=k).avg_f1)
        res_k_hidx["PerH"].append( runner.run_per("PerH",  folder_name, topr=DEFAULT_LAMBDA, k=k).avg_accuracy)
        res_k_hidx["PerH+"].append(runner.run_per("PerH+", folder_name, topr=DEFAULT_LAMBDA, k=k, beta=0.1).avg_accuracy)

    # --- Raw Output ---
    print("\n" + "=" * 60)
    print("--- RAW F1 DATA EXTRACTION RESULTS ---")
    print("=" * 60)

    print("\n[EXPERIMENT 1: VARYING LAMBDA]")
    print(f"LAMBDAS,{','.join(LAMBDAS)}")
    for meth in ["GloD", "PerD", "PerD+"]:
        print(f"{meth},{','.join(map(str, res_lam_deg[meth]))}")
    for meth in ["GloH", "PerH", "PerH+"]:
        print(f"{meth},{','.join(map(str, res_lam_hidx[meth]))}")

    print("\n[EXPERIMENT 2: VARYING K]")
    print(f"K_VALUES,{','.join(map(str, K_VALUES))}")
    for meth in ["GloD", "PerD", "PerD+"]:
        print(f"{meth},{','.join(map(str, res_k_deg[meth]))}")
    for meth in ["GloH", "PerH", "PerH+"]:
        print(f"{meth},{','.join(map(str, res_k_hidx[meth]))}")

    print("=" * 60)


if __name__ == "__main__":
    main()