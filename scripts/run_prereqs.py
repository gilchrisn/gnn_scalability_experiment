#!/usr/bin/env python3
"""
Isolated test script for Table III generation.
Properly stages data using the established Python-C++ Bridge architecture.
"""
import os
import sys
import argparse
import subprocess
import re
import time

# Ensure project root is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from src.utils import SchemaMatcher

def compile_rule_for_cpp(metapath_str: str, instance_id: int, g_hetero, data_dir: str, folder_name: str) -> None:
    """Compiles rule bytecode to both .limit and .dat to satisfy C++ I/O."""
    sorted_edges = sorted(list(g_hetero.edge_types))
    edge_map = {et: i for i, et in enumerate(sorted_edges)}
    
    path_list = [s.strip() for s in metapath_str.split(',')]
    parts = []
    
    for rel_str in path_list:
        # Determine Traversal Opcode
        direction = "-3" if rel_str.startswith("rev_") else "-2"
        
        # Match Edge ID
        matched_edge = SchemaMatcher.match(rel_str, g_hetero)
        try:
            eid = edge_map[matched_edge]
        except KeyError as e:
            raise RuntimeError(f"Mined rule contains edge {e} not found in schema.")
            
        parts.extend([direction, str(eid)])
        
    # Append the correct C++ flag
    if instance_id == -1:
        parts.append("-1") # Variable mode
    else:
        parts.extend(["-5", str(instance_id)]) # Instance mode
        
    for _ in path_list: 
        parts.append("-4") # Pop stack
        
    rule_content = " ".join(parts) + "\n"
    
    # Write to files
    file_limit = os.path.join(data_dir, f"cod-rules_{folder_name}.limit")
    file_dat = os.path.join(data_dir, f"{folder_name}-cod-global-rules.dat")
    
    with open(file_limit, "w") as f: f.write(rule_content)
    with open(file_dat, "w") as f: f.write(rule_content)
    
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

    # 1. Resolve Folders (Matching run_prereqs.py logic)
    folder_name = f"HGBn-{args.dataset.split('_')[1]}"
    data_dir = os.path.join(project_root, folder_name)
    
    # 2. Stage Data using the established Adapter
    print("\n>>> [Step 1] Loading and Staging Data...")
    cfg = config.get_dataset_config(args.dataset)
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    
    adapter = PyGToCppAdapter(data_dir)
    adapter.convert(g_hetero)
    
    # 3. Compile Rule
    compile_rule_for_cpp(args.metapath, g_hetero, data_dir, folder_name)

    # 4. Execute C++ Engine
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

    # 5. Strict Parsing
    print("\n>>> [Step 3] Parsing Output...")
    
    edges_match = re.search(r"~raw_edges:\s*([0-9.]+)", stdout)
    density_match = re.search(r"~dens:\s*([0-9.]+)", stdout)

    if not edges_match or not density_match:
        print("\n[FATAL] Regex parsing failed. The C++ output did not contain the expected tags.")
        print("Did you insert the exact `std::cout << \"~raw_edges: \" << temp_density << std::endl;` tag into effectiveness.cpp and recompile?")
        print("\n--- RAW STDOUT DUMP FOR DEBUGGING ---")
        print(stdout.strip())
        sys.exit(1)

    raw_edges = float(edges_match.group(1))
    density = float(density_match.group(1))

    # 6. Results
    print(f"\n[SUCCESS] Extraction Validated for {args.dataset} / {args.metapath}")
    print(f"  |E*| (Raw Edges) : {raw_edges:,.0f}")
    print(f"  rho* (Density)   : {density:.6f}")
    print(f"  Extraction Time  : {execution_time:.4f} seconds")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()