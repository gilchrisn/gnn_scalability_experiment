"""
==============================================================================
C++ Engine I/O Protocol & Failure Modes (`effectiveness.cpp`)
==============================================================================
The C++ backend utilises highly rigid, hardcoded file paths for its statistics
and materialisation logic. If these exact conditions are not met, the binary
will fail silently—evaluating 0 nodes and returning empty metrics—instead of
throwing standard file I/O exceptions.

1. The Query Nodes Constraint:
   - Expected Path: `<data_dir>/qnodes_<folder_name>.dat`
   - Silent Failure Mode: Missing file → `peer_size = 0` → loop bypassed.
     Result: `~|peer|:0` and `~dens:0`.

2. The Rule Bytecode Constraint:
   - Expected Path: `<data_dir>/cod-rules_<folder_name>.limit`
   - Silent Failure Mode: Wrong filename → `getline` returns false immediately.
     Result: Zero rules processed, zero output.

3. Node Homogenisation Offset:
   - meta.dat must start with exactly "Node Number is : " (17 chars).
   - Wrong prefix → `std::out_of_range` fatal exception.
==============================================================================
"""
import os
import sys
import argparse
import re
import time

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from scripts.bench_utils import compile_rule_for_cpp, generate_qnodes, run_cpp


def main():
    parser = argparse.ArgumentParser(description="Isolated Test: Table III Statistics Extraction")
    parser.add_argument('dataset',     type=str, help="Dataset name (e.g., HGB_DBLP)")
    parser.add_argument('--metapath',  type=str, required=True,
                        help="Metapath to analyse (e.g., 'author_to_paper,paper_to_author')")
    parser.add_argument('--binary',    type=str, default=config.CPP_EXECUTABLE,
                        help="Path to the C++ binary")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"TABLE III ISOLATED TEST: {args.dataset}")
    print(f"{'='*60}")

    folder_name = f"HGBn-{args.dataset.split('_')[1]}"
    data_dir    = os.path.join(project_root, folder_name)

    # 1. Load and stage data
    print("\n>>> [Step 1] Loading and Staging Data...")
    cfg = config.get_dataset_config(args.dataset)
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)

    PyGToCppAdapter(data_dir).convert(g_hetero)
    generate_qnodes(data_dir, folder_name)
    compile_rule_for_cpp(args.metapath, g_hetero, data_dir, folder_name)

    # 2. Execute C++ engine
    print(f"\n>>> [Step 2] Executing C++ Engine...")
    start_time = time.perf_counter()
    stdout     = run_cpp(args.binary, ["hg_stats", folder_name])
    exec_time  = time.perf_counter() - start_time

    # 3. Parse output
    print("\n>>> [Step 3] Parsing Output...")

    # One RAW_EDGES_E* entry is emitted per evaluated query node; we average them.
    edges_matches = re.findall(r"RAW_EDGES_E\*:\s*([0-9.]+)", stdout)
    density_match = re.search(r"~dens:\s*([0-9.]+)",           stdout)

    if not edges_matches or not density_match:
        print("\n[FATAL] Regex parsing failed. Expected tags missing.")
        print("\n--- RAW STDOUT DUMP FOR DEBUGGING ---")
        print(stdout.strip())
        sys.exit(1)

    raw_edges = sum(float(m) for m in edges_matches) / len(edges_matches)
    density   = float(density_match.group(1))

    print(f"\n[SUCCESS] Extraction Validated for {args.dataset} / {args.metapath}")
    print(f"  Queries Evaluated : {len(edges_matches)}")
    print(f"  |E*| (Avg Edges)  : {raw_edges:,.2f}")
    print(f"  rho* (Density)    : {density:.6f}")
    print(f"  Extraction Time   : {exec_time:.4f} seconds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()