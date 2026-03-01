#!/usr/bin/env python3
"""
Phase 1 for Figure 4: Raw Data Extraction.
Executes Approximate Centrality methods and captures SCATTER_DATA from stdout.
"""
import os
import sys
import argparse
import subprocess
import re
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Implicit dependency on your internal framework based on test_table4.py
from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from src.utils import SchemaMatcher

def generate_qnodes(data_dir: str, folder_name: str) -> None:
    """Selects 100 random query nodes. Required by C++ to prevent peer_size=0 silent failures."""
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

def compile_rule_for_cpp(metapath_str: str, g_hetero, data_dir: str, folder_name: str) -> None:
    """Compiles rule bytecode to both .limit and .dat to satisfy C++ schizophrenic I/O."""
    sorted_edges = sorted(list(g_hetero.edge_types))
    edge_map = {et: i for i, et in enumerate(sorted_edges)}
    path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in metapath_str.split(',')]
    
    rule_ids = [edge_map[edge] for edge in path_list]
    parts = []
    for i, eid in enumerate(rule_ids):
        parts.append("-2")
        if i == len(rule_ids) - 1:
            parts.append("-1") 
        parts.append(str(eid))
        
    parts.extend(["-5", "-1"])
    for _ in rule_ids: parts.append("-4")
    rule_content = " ".join(parts) + "\n"
    
    file_limit = os.path.join(data_dir, f"cod-rules_{folder_name}.limit")
    file_dat = os.path.join(data_dir, f"{folder_name}-cod-global-rules.dat")
    
    with open(file_limit, "w") as f: f.write(rule_content)
    with open(file_dat, "w") as f: f.write(rule_content)

def setup_global_res_dirs(folder_name: str):
    """Establishes rigid directory structure for C++ Exact baseline tracking."""
    df1_dir = os.path.join(project_root, "global_res", folder_name, "df1")
    hf1_dir = os.path.join(project_root, "global_res", folder_name, "hf1")
    os.makedirs(df1_dir, exist_ok=True)
    os.makedirs(hf1_dir, exist_ok=True)
    return df1_dir, hf1_dir

def execute_cpp_command(binary: str, args: list, output_redirect: str = None) -> str:
    """Executes C++ binary and returns raw stdout."""
    cmd = [binary] + args
    print(f"[Engine] Executing: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1200)
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL] C++ Crashed! Exit Code: {e.returncode}")
        print(f"Command: {' '.join(cmd)}")
        print(f"--- STDERR ---\n{e.stderr.strip()}")
        print(f"--- STDOUT ---\n{e.stdout.strip()}")
        sys.exit(1)

    if output_redirect:
        with open(output_redirect, 'w') as f:
            f.write(result.stdout)
            
    # --- INJECTED RAW OUTPUT SECTION ---
    print("\n" + "-"*40)
    print(f"--- RAW C++ STDOUT ({args[0]}) ---")
    print("-" * 40)
    print(result.stdout.strip() if result.stdout.strip() else "[NO OUTPUT - REDIRECTED TO FILE]")
    print("-" * 40)
            
    return result.stdout

def main():
    parser = argparse.ArgumentParser(description="Raw Data Extraction for Figure 4")
    parser.add_argument('dataset', type=str)
    parser.add_argument('--metapath', type=str, required=True)
    args = parser.parse_args()

    folder_name = f"HGBn-{args.dataset.split('_')[1]}"
    data_dir = os.path.join(project_root, folder_name)
    
    print(f"\n--- INITIATING RIGOROUS EXTRACTION: {args.dataset} ---")

    # 1. Staging
    cfg = config.get_dataset_config(args.dataset)
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    
    PyGToCppAdapter(data_dir).convert(g_hetero)
    generate_qnodes(data_dir, folder_name)
    compile_rule_for_cpp(args.metapath, g_hetero, data_dir, folder_name)
    
    df1_dir, hf1_dir = setup_global_res_dirs(folder_name)

    # 2. Ground Truth Generation (MANDATORY per test_table4.py contract)
    print("\n[Phase 1] Forcing Exact Baseline generation to prevent C++ F1 Out-of-Bounds crash...")
    exact_plan = [
        ("ExactD+", ["ExactD+", folder_name, "0.05", "0"], os.path.join(df1_dir, "hg_global_r0.05.res")),
        ("ExactD",  ["ExactD",  folder_name, "0.05", "0"], os.path.join(df1_dir, "hg_global_greater_r0.05.res")),
        ("ExactH+", ["ExactH+", folder_name, "0.05", "0"], os.path.join(hf1_dir, "hg_global_r0.05.res")),
        ("ExactH",  ["ExactH",  folder_name, "0.05", "0"], os.path.join(hf1_dir, "hg_global_greater_r0.05.res")),
    ]
    for _, cmd_args, redirect_path in exact_plan:
        execute_cpp_command(config.CPP_EXECUTABLE, cmd_args, redirect_path)

    # 3. Figure 4 Target Execution
    print("\n[Phase 2] Executing Target Methods and capturing Scatter Data...")
    target_plan = [
        ("GloD",  ["GloD",  folder_name, "0.05", "0", "32"]),
        ("PerD",  ["PerD",  folder_name, "0.05", "0", "32"]), 
        ("PerD+", ["PerD+", folder_name, "0.05", "0.1", "32"]), # Beta > 0 required for optimized
        ("GloH",  ["GloH",  folder_name, "0.05", "0", "4"]),
        ("PerH",  ["PerH",  folder_name, "0.05", "0", "4"]), 
        ("PerH+", ["PerH+", folder_name, "0.05", "0.1", "4"]),  # Beta > 0 required for optimized
    ]

    all_scatter_data = {}

    for method, cmd_args in target_plan:
        raw_stdout = execute_cpp_command(config.CPP_EXECUTABLE, cmd_args)
        
        # Regex parse for the specific SCATTER_DATA tag you added to the C++
        scatter_matches = re.findall(r"SCATTER_DATA:\s*([0-9.]+),([0-9.]+),([0-9.]+)", raw_stdout)
        
        all_scatter_data[method] = scatter_matches
        
        print(f"\n--- {method} RAW SCATTER DATA ---")
        if not scatter_matches:
            print(f"[WARNING] No SCATTER_DATA found in stdout for {method}. Did you recompile the C++ binary?")
            print("[DUMPING RAW STDOUT FOR DIAGNOSTICS]:")
            print(raw_stdout.strip())
        else:
            for match in scatter_matches:
                print(f"Rule ID: {match[0]}, Edges: {match[1]}, Time(s): {match[2]}")

    print("\n--- EXTRACTION COMPLETE ---")

if __name__ == "__main__":
    main()