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

    folder_name = f"HGBn-{args.dataset.split('_')[1]}"
    data_dir    = os.path.join(project_root, folder_name)

    cfg = config.get_dataset_config(args.dataset)
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    PyGToCppAdapter(data_dir).convert(g_hetero)
    generate_qnodes(data_dir, folder_name, target_node_type=cfg.target_node, g_hetero=g_hetero)
    compile_rule_for_cpp(args.metapath, g_hetero, data_dir, folder_name)

    start_time = time.perf_counter()
    stdout     = run_cpp(args.binary, ["hg_stats", folder_name])
    exec_time  = time.perf_counter() - start_time

    edges_matches = re.findall(r"RAW_EDGES_E\*:\s*([0-9.eE+\-]+)", stdout)
    peer_match    = re.search(r"~\|peer\|:\s*([0-9.eE+\-]+)",       stdout)
    density_match = re.search(r"~dens:\s*([0-9.eE+\-]+)",           stdout)

    if not edges_matches or not density_match:
        print("\n[FATAL] Regex parsing failed — expected tags missing.")
        sys.exit(1)

    raw_edges = sum(float(m) for m in edges_matches) / len(edges_matches)
    peer_size = float(peer_match.group(1)) if peer_match else float("nan")
    density   = float(density_match.group(1))

    print(f"\n[TABLE III]  {args.dataset}  metapath={args.metapath}")
    print(f"  |E*| (avg per qnode) : {raw_edges:,.2f}")
    print(f"  ~|peer|              : {peer_size:.2f}")
    print(f"  ~rho*                : {density:.8f}")
    print(f"  time                 : {exec_time:.4f}s")


if __name__ == "__main__":
    main()