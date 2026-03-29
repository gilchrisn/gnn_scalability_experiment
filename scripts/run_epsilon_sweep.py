"""
Fig 7: Epsilon (approximation ratio) vs K sweep.

Runs the Epsilon C++ command for each dataset × metapath × K × seed.
Paper uses 10 seeds per K, reports mean ± variance.

Usage:
    python scripts/run_epsilon_sweep.py HGB_ACM HGB_DBLP HGB_IMDB
    python scripts/run_epsilon_sweep.py HGB_ACM --k-values 2 4 8 16 32 --n-seeds 10
    python scripts/run_epsilon_sweep.py OAG_CS --timeout 600

Output: results/<DATASET>/figure7.csv
"""
import argparse
import csv
import os
import subprocess
import sys
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes, setup_global_res_dirs


def run_epsilon(binary, folder, topr, k, seed, timeout):
    cmd = [binary, "Epsilon", folder, str(topr), str(k), str(seed)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        stdout = result.stdout
    except subprocess.TimeoutExpired:
        return None, None
    except Exception:
        return None, None

    eps_d = eps_h = None
    m = re.search(r"EPSILON_df1:([-0-9.eE+]+)", stdout)
    if m:
        eps_d = float(m.group(1))
    m = re.search(r"EPSILON_hf1:([-0-9.eE+]+)", stdout)
    if m:
        eps_h = float(m.group(1))
    return eps_d, eps_h


def main():
    parser = argparse.ArgumentParser(description="Fig 7: Epsilon vs K sweep")
    parser.add_argument("datasets", nargs="+", help="Dataset names (e.g. HGB_ACM HGB_DBLP)")
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--topr", type=str, default="0.05")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    fields = ["dataset", "metapath", "k", "seed",
              "epsilon_degree", "epsilon_hindex"]

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"  Epsilon sweep: {dataset}")
        print(f"  K={args.k_values}  seeds=1..{args.n_seeds}  topr={args.topr}")
        print(f"{'='*60}")

        cfg = config.get_dataset_config(dataset)
        g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        folder = config.get_folder_name(dataset)
        data_dir = os.path.join(".", folder)

        # Stage graph
        os.makedirs(data_dir, exist_ok=True)
        setup_global_res_dirs(folder, ".")
        PyGToCppAdapter(data_dir).convert(g)
        generate_qnodes(data_dir, folder, target_node_type=cfg.target_node, g_hetero=g)

        # Get metapaths
        if cfg.suggested_paths:
            metapaths = list(cfg.suggested_paths)
        else:
            from src.bridge.anyburl import load_validated_metapaths, AnyBURLRunner
            work_dir = os.path.join(config.DATA_DIR, f"mining_{dataset}")
            anyburl = AnyBURLRunner(work_dir, config.ANYBURL_JAR)
            anyburl.export_for_mining(g)
            rules_file = os.path.join(work_dir, "anyburl_rules.txt")
            if not os.path.exists(rules_file):
                anyburl.run_mining(timeout=10, max_length=6, num_threads=4)
            metapaths, _ = load_validated_metapaths(
                rules_file, g, cfg.target_node, min_conf=0.1, max_n=500)
            metapaths = [mp for mp in metapaths if len(mp.split(",")) <= 4]

        # Output CSV
        out_dir = os.path.join("results", dataset)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "figure7.csv")
        fh = open(out_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for mp in metapaths:
            print(f"\n  Metapath: {mp}")
            compile_rule_for_cpp(mp, g, data_dir, folder)

            for k in args.k_values:
                eps_d_list = []
                eps_h_list = []
                for seed in range(1, args.n_seeds + 1):
                    t0 = time.perf_counter()
                    eps_d, eps_h = run_epsilon(
                        config.CPP_EXECUTABLE, folder, args.topr, k, seed, args.timeout)
                    elapsed = time.perf_counter() - t0

                    row = {
                        "dataset": dataset,
                        "metapath": mp,
                        "k": k,
                        "seed": seed,
                        "epsilon_degree": round(eps_d, 6) if eps_d is not None else "",
                        "epsilon_hindex": round(eps_h, 6) if eps_h is not None else "",
                    }
                    writer.writerow(row)
                    fh.flush()

                    if eps_d is not None:
                        eps_d_list.append(eps_d)
                    if eps_h is not None:
                        eps_h_list.append(eps_h)

                # Print summary for this K
                if eps_d_list:
                    mean_d = sum(eps_d_list) / len(eps_d_list)
                    print(f"    K={k:3d}: eps_degree={mean_d:.4f} ({len(eps_d_list)} seeds)", end="")
                    if eps_h_list:
                        mean_h = sum(eps_h_list) / len(eps_h_list)
                        print(f"  eps_hindex={mean_h:.4f}", end="")
                    print()
                else:
                    print(f"    K={k:3d}: FAILED")

        fh.close()
        print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
