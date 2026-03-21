"""
Master Protocol: Dataset Staging and Metapath Mining.

Stages C++ graph files for a dataset and discovers metapaths via AnyBURL.
Output is a ranked list of metapaths ready to pass to test_table*.py / test_figure*.py.

Usage:
    python scripts/run_master_protocol.py --dataset HGB_ACM
    python scripts/run_master_protocol.py --dataset HGB_DBLP --top-paths 5 --force-remine
"""
import argparse
import os
import sys

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter, AnyBURLRunner
from scripts.bench_utils import generate_qnodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage C++ data and mine metapaths via AnyBURL."
    )
    parser.add_argument("--dataset",      type=str, required=True,
                        help="Dataset key (e.g. HGB_ACM, HGB_DBLP)")
    parser.add_argument("--min-conf",     type=float, default=0.001,
                        help="Minimum AnyBURL confidence threshold")
    parser.add_argument("--top-paths",    type=int, default=3,
                        help="Number of top metapaths to report")
    parser.add_argument("--force-remine", action="store_true",
                        help="Ignore cached rules and re-run AnyBURL")
    parser.add_argument("--cpu",          action="store_true",
                        help="Force CPU (no effect on staging/mining; passed for parity)")
    args = parser.parse_args()

    ds_name     = args.dataset
    folder_name = f"HGBn-{ds_name.split('_')[1]}"
    data_dir    = os.path.join(project_root, folder_name)
    work_dir    = os.path.join(config.DATA_DIR, f"mining_{ds_name}")

    print(f"\n{'='*60}")
    print(f"  Master Protocol: {ds_name}")
    print(f"{'='*60}")

    print(f"\n[1/3] Loading graph...")
    cfg = config.get_dataset_config(ds_name)
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    print(f"      Node types : {g_hetero.node_types}")
    print(f"      Edge types : {[et[1] for et in g_hetero.edge_types]}")
    print(f"      Target     : {cfg.target_node}")

    print(f"\n[2/3] Staging C++ files → {data_dir}/")
    PyGToCppAdapter(data_dir).convert(g_hetero)
    generate_qnodes(
        data_dir,
        folder_name,
        target_node_type=cfg.target_node,
        g_hetero=g_hetero,
    )

    print(f"\n[3/3] Mining metapaths via AnyBURL (work_dir={work_dir})...")
    runner = AnyBURLRunner(work_dir, config.ANYBURL_JAR)
    runner.export_for_mining(g_hetero)

    rules_file = os.path.join(work_dir, "rules", "rules-final")
    if args.force_remine or not os.path.exists(rules_file):
        runner.run_mining(timeout=3600, max_length=6, num_threads=4)
    else:
        print(f"   [Skip] Cached rules found — use --force-remine to re-run.")

    top_paths = runner.get_top_k_paths(k=args.top_paths, min_conf=args.min_conf)

    print(f"\n{'='*60}")
    print(f"  TOP {args.top_paths} METAPATHS — {ds_name}")
    print(f"{'='*60}")
    if not top_paths:
        print("  [WARNING] No metapaths found. Lower --min-conf or re-run with --force-remine.")
    else:
        for rank, (conf, path, _) in enumerate(top_paths, start=1):
            print(f"  [{rank}] conf={conf:.4f}  path={path}")

    print(f"\n  To reproduce Table III (hg_stats):")
    for conf, path, _ in top_paths:
        print(f"    python scripts/test_table3.py {ds_name} --metapath \"{path}\"")

    print(f"\n  To reproduce Table IV (F1 scores):")
    for conf, path, _ in top_paths:
        print(f"    python scripts/test_table4.py {ds_name} --metapath \"{path}\"")

    print()


if __name__ == "__main__":
    main()
