"""
==============================================================================
C++ Engine I/O Protocol & Failure Modes (`effectiveness.cpp`)
==============================================================================
The C++ backend utilizes highly rigid, hardcoded file paths for its statistics 
and materialization logic. If these exact conditions are not met, the binary 
will fail silently—evaluating 0 nodes and returning empty metrics—instead of 
throwing standard file I/O exceptions.

1. The Query Nodes Constraint:
   - Expected Path: `<data_dir>/qnodes_<folder_name>.dat`
   - Purpose: Defines the subset of node IDs to use as the initial frontier.
   - Silent Failure Mode: If this file is missing, the `qnodes` array initializes 
     empty. The variable `peer_size` evaluates to 0. The internal `for` loop 
     that computes the matching graph edges is bypassed entirely.
     Result: `~|peer|:0` and `~dens:0`.

2. The Rule Bytecode Constraint:
   - Expected Path: `<data_dir>/cod-rules_<folder_name>.limit`
   - Purpose: Contains the stack machine bytecode mapping PyG edge IDs to C++ operations.
   - Silent Failure Mode: The C++ `ifstream` does not verify file existence. 
     If you pass `.dat` instead of `.limit`, `getline(rules_in, rules_line)` 
     instantly evaluates to false. The execution loop exits immediately.
     Result: Zero rules processed, zero output.

3. Node Homogenization Offset:
   - The C++ engine expects globally contiguous node IDs. 
   - Ensure PyGToCppAdapter correctly writes `node.dat` and `link.dat` with 
     the `Node Number is : <N>` metadata header precisely 17 characters long 
     in `meta.dat`, otherwise the C++ string parser will throw a `std::out_of_range` 
     (`substr: __pos > this->size()`) fatal exception.
==============================================================================
"""

import os
import sys
import argparse
import subprocess
import re
import time
import random

# Ensure project root is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from src.utils import SchemaMatcher

def generate_qnodes(data_dir: str, folder_name: str) -> None:
    """Selects 100 random nodes for personalized queries."""
    node_file = os.path.join(data_dir, "node.dat")
    valid_ids = []
    
    with open(node_file, 'r') as f:
        for i, line in enumerate(f):
            if i > 5000: break
            parts = line.strip().split('\t')
            if parts: valid_ids.append(parts[0])
            
    if not valid_ids: valid_ids = ["0"]
    selected = random.sample(valid_ids, min(100, len(valid_ids)))
    
    qnode_path = os.path.join(data_dir, f"qnodes_{folder_name}.dat")
    with open(qnode_path, 'w') as f:
        f.write("\n".join(selected))
    print(f"   [Staging] Generated {len(selected)} query nodes at: {qnode_path}")

def compile_rule_for_cpp(metapath_str: str, g_hetero, data_dir: str, folder_name: str) -> None:
    """Compiles rule and writes it to the exact filename C++ expects."""
    print("   [Staging] Compiling Metapath Rule...")
    
    sorted_edges = sorted(list(g_hetero.edge_types))
    edge_map = {et: i for i, et in enumerate(sorted_edges)}
    
    path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in metapath_str.split(',')]
    
    try:
        rule_ids = [edge_map[edge] for edge in path_list]
    except KeyError as e:
        raise RuntimeError(f"Mined rule contains edge {e} not found in schema.")
        
    parts = []
    for i, eid in enumerate(rule_ids):
        parts.append("-2")
        if i == len(rule_ids) - 1:
            parts.append("-1") 
        parts.append(str(eid))
        
    parts.extend(["-5", "-1"])
    for _ in rule_ids: parts.append("-4")
    
    rule_content = " ".join(parts)
    
    # CRITICAL FIX: Match the exact hardcoded path in effectiveness.cpp
    filename = f"cod-rules_{folder_name}.limit"
    file_path = os.path.join(data_dir, filename)
    
    with open(file_path, "w") as f:
        f.write(rule_content + "\n") # Adding newline just to be safe for getline
        
    print(f"   [Staging] Wrote rule bytecode to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Isolated Test: Table III Statistics Extraction")
    parser.add_argument('dataset', type=str, help="Dataset name (e.g., HGB_DBLP)")
    parser.add_argument('--metapath', type=str, required=True, 
                        help="Metapath to analyze (e.g., 'author_to_paper,paper_to_author')")
    parser.add_argument('--binary', type=str, default=config.CPP_EXECUTABLE, 
                        help="Path to the C++ binary")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"TABLE III ISOLATED TEST: {args.dataset}")
    print(f"{'='*60}")

    folder_name = f"HGBn-{args.dataset.split('_')[1]}"
    data_dir = os.path.join(project_root, folder_name)
    
    print("\n>>> [Step 1] Loading and Staging Data...")
    cfg = config.get_dataset_config(args.dataset)
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    
    adapter = PyGToCppAdapter(data_dir)
    adapter.convert(g_hetero)
    
    generate_qnodes(data_dir, folder_name)
    compile_rule_for_cpp(args.metapath, g_hetero, data_dir, folder_name)

    cmd = [args.binary, "hg_stats", folder_name]
    print(f"\n>>> [Step 2] Executing C++ Engine...")
    print(f"Executing: {' '.join(cmd)}")
    
    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=300
        )
        stdout = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL] C++ Execution Failed! Exit Code: {e.returncode}")
        print(f"--- STDERR ---\n{e.stderr.strip()}")
        print(f"--- STDOUT ---\n{e.stdout.strip()}")
        sys.exit(1)
        
    execution_time = time.perf_counter() - start_time

    # --- INJECTED RAW OUTPUT SECTION ---
    print("\n" + "-"*40)
    print("--- RAW C++ STDOUT (hg_stats) ---")
    print("-" * 40)
    print(stdout.strip())
    print("-" * 40 + "\n")

    print("\n>>> [Step 3] Parsing Output...")
    
    # Extract all occurrences of RAW_EDGES_E*: (one per valid query node)
    edges_matches = re.findall(r"RAW_EDGES_E\*:\s*([0-9.]+)", stdout)
    density_match = re.search(r"~dens:\s*([0-9.]+)", stdout)

    if not density_match or not edges_matches:
        print("\n[FATAL] Regex parsing failed. Expected tags missing.")
        print("\n--- RAW STDOUT DUMP FOR DEBUGGING ---")
        print(stdout.strip())
        sys.exit(1)

    # Calculate average edges over all evaluated queries
    raw_edges = sum(float(m) for m in edges_matches) / len(edges_matches)
    density = float(density_match.group(1))

    print(f"\n[SUCCESS] Extraction Validated for {args.dataset} / {args.metapath}")
    print(f"  Queries Evaluated: {len(edges_matches)}")
    print(f"  |E*| (Avg Edges) : {raw_edges:,.2f}")
    print(f"  rho* (Density)   : {density:.6f}")
    print(f"  Extraction Time  : {execution_time:.4f} seconds")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()