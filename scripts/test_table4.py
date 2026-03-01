"""
==============================================================================
C++ Engine Execution Contract: Table IV (Efficiency & Speedup)
==============================================================================

This document defines the strict, undocumented internal logic of the C++ 
`graph_prep` binary when executing Centrality and Approximation benchmarking.

1. THE EXECUTION ORDER DEPENDENCY (The "Ground Truth" Trap)
------------------------------------------------------------------------------
Approximate methods (GloD, PerD, GloH, PerH) do NOT just run sketching. 
Immediately after computing the sketch, the C++ code forces an F1-score 
validation against the Exact baseline. 
- If you run `GloD` before running `ExactD`, the C++ engine will try to open 
  a non-existent ground truth file, initialize an empty result array, and 
  CRASH with an out-of-bounds error.
- RULE: You MUST execute `Exact` methods and physically pipe their `stdout` 
  to the `global_res/` directory BEFORE running any approximate method.

2. THE RIGID DIRECTORY STRUCTURE
------------------------------------------------------------------------------
The C++ engine hardcodes relative paths for its ground truth files. You must 
create this exact structure in your Python working directory:
  ./global_res/<Folder_Name>/df1/hg_global_r<Lambda>.res          (ExactD+ stdout)
  ./global_res/<Folder_Name>/df1/hg_global_greater_r<Lambda>.res  (ExactD stdout)
  ./global_res/<Folder_Name>/hf1/hg_global_r<Lambda>.res          (ExactH+ stdout)
  ./global_res/<Folder_Name>/hf1/hg_global_greater_r<Lambda>.res  (ExactH stdout)

3. CLI ARGUMENTS & THE BETA PRUNING CRASH
------------------------------------------------------------------------------
Syntax: `./graph_prep <Method> <Dataset> <Lambda> <Beta> <K>`
- <Lambda>: The Top-R threshold (e.g., 0.05 for top 5%).
- <Beta>: The early-stopping/pruning threshold for optimized methods.
- <K>: The sketch size (e.g., 32 for Degree, 4 for H-Index).

*CRITICAL FAILURE MODE (The Beta Trap)*: 
For the optimized methods (`PerD+` and `PerH+`), the C++ code uses the `Beta` 
value to dynamically size a `std::vector<bool>`. If you pass `0` for Beta, 
the vector size evaluates to 1. The engine then tries to access index 1, 
triggering a fatal `std::out_of_range` exception (`__n >= this->size()`). 
- RULE: Standard methods use Beta=0. Optimized methods (+) MUST use Beta > 0 (e.g., 0.1).

4. SCHIZOPHRENIC FILE NAMING
------------------------------------------------------------------------------
The C++ engine is inconsistent in how it loads rule bytecodes:
- `hg_stats` explicitly requires: `cod-rules_<dataset>.limit`
- `ExactD`, `GloD`, etc. require: `<dataset>-cod-global-rules.dat`
- RULE: The Python adapter must write identical bytecode to BOTH filenames 
  to prevent silent bypassing of the execution loops.

5. OUTPUT PARSING INCONSISTENCIES
------------------------------------------------------------------------------
The C++ developer changed the standard output formatting depending on the command:
- Exact & Approx methods print: `time: 0.145` or `~time_per_rule: 0.145` (in seconds).
- `matching_graph_time` prints: `~matching_graph_time_per_rule: 75.6 ms` (in milliseconds).
- RULE: Your regex MUST capture the unit (`ms` vs `s`), and your Python code 
  MUST standardize the division, otherwise Speedup calculations will be off by 1000x.

6. SPEEDUP BASELINE MAPPING
------------------------------------------------------------------------------
To calculate mathematically valid Table IV Speedups ($T_{exact} / T_{approx}$), 
you must pair the methods according to their algorithmic intent:
- GloD, PerD  -> baseline is ExactD
- GloH, PerH  -> baseline is ExactH
- PerD+       -> baseline is ExactD+
- PerH+       -> baseline is ExactH+
==============================================================================
"""
import os
import sys
import argparse
import subprocess
import re
import time
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from src.utils import SchemaMatcher

def generate_qnodes(data_dir: str, folder_name: str) -> None:
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
    
    # Write to both to handle C++ hardcoded inconsistencies
    file_limit = os.path.join(data_dir, f"cod-rules_{folder_name}.limit")
    file_dat = os.path.join(data_dir, f"{folder_name}-cod-global-rules.dat")
    
    with open(file_limit, "w") as f: f.write(rule_content)
    with open(file_dat, "w") as f: f.write(rule_content)

def setup_global_res_dirs(folder_name: str):
    """Creates the rigid directory structure demanded by the README."""
    df1_dir = os.path.join(project_root, "global_res", folder_name, "df1")
    hf1_dir = os.path.join(project_root, "global_res", folder_name, "hf1")
    os.makedirs(df1_dir, exist_ok=True)
    os.makedirs(hf1_dir, exist_ok=True)
    return df1_dir, hf1_dir

def execute_cpp_command(binary: str, args: list, output_redirect: str = None) -> float:
    cmd = [binary] + args
    print(f"\n   -> Executing: {' '.join(cmd)}")
    if output_redirect:
        print(f"      Redirecting stdout to: {output_redirect}")
    
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

    time_match = re.search(r"~?(?:time|time_per_rule|matching_graph_time_per_rule):\s*([0-9.]+)(?:\s*(ms|s))?", result.stdout, re.IGNORECASE)
    
    if not time_match and not output_redirect:
        print("\n[FATAL] Failed to parse execution time.")
        sys.exit(1)
    elif not time_match and output_redirect:
        return 0.0 # Exact baselines
        
    time_val = float(time_match.group(1))
    unit = time_match.group(2)
    
    if unit and unit.lower() == 'ms':
        time_val /= 1000.0
        
    return time_val

def main():
    parser = argparse.ArgumentParser(description="Isolated Test: Table IV")
    parser.add_argument('dataset', type=str)
    parser.add_argument('--metapath', type=str, required=True)
    args = parser.parse_args()

    folder_name = f"HGBn-{args.dataset.split('_')[1]}"
    data_dir = os.path.join(project_root, folder_name)
    
    print(f"{'='*60}")
    print(f"TABLE IV ISOLATED TEST: {args.dataset}")
    print(f"{'='*60}")

    print("\n>>> [Step 1] Loading and Staging Data...")
    cfg = config.get_dataset_config(args.dataset)
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    
    PyGToCppAdapter(data_dir).convert(g_hetero)
    generate_qnodes(data_dir, folder_name)
    compile_rule_for_cpp(args.metapath, g_hetero, data_dir, folder_name)
    
    # CRITICAL: Setup global_res directories
    df1_dir, hf1_dir = setup_global_res_dirs(folder_name)

    # Execution Plan matching the README prerequisite flow exactly.
    # Format: (TaskName, CommandArgs, OutputRedirectionPath)
    execution_plan = [
        # --- Ground Truths (Redirected to global_res/) ---
        ("ExactD+", ["ExactD+", folder_name, "0.05", "0"], os.path.join(df1_dir, "hg_global_r0.05.res")),
        ("ExactD",  ["ExactD",  folder_name, "0.05", "0"], os.path.join(df1_dir, "hg_global_greater_r0.05.res")),
        ("ExactH+", ["ExactH+", folder_name, "0.05", "0"], os.path.join(hf1_dir, "hg_global_r0.05.res")),
        ("ExactH",  ["ExactH",  folder_name, "0.05", "0"], os.path.join(hf1_dir, "hg_global_greater_r0.05.res")),
        
        # --- Degree Methods (K=32, Beta matched to README specs) ---
        ("GloD",    ["GloD",  folder_name, "0.05", "0", "32"], None),
        ("PerD",    ["PerD",  folder_name, "0.05", "0", "32"], None), 
        ("PerD+",   ["PerD+", folder_name, "0.05", "0.1", "32"], None),
        
        # --- H-Index Methods (K=4, Beta matched to README specs) ---
        ("GloH",    ["GloH",  folder_name, "0.05", "0", "4"], None),
        ("PerH",    ["PerH",  folder_name, "0.05", "0", "4"], None), 
        ("PerH+",   ["PerH+", folder_name, "0.05", "0.1", "4"], None),
        
        # --- Extraction Time ---
        ("T(G*)",   ["matching_graph_time", folder_name], None)
    ]

    print("\n>>> [Step 2] Executing C++ Methods (with Ground Truth redirection)...")
    results_time = {}
    
    for method_name, cmd_args, redirect_path in execution_plan:
        exec_time = execute_cpp_command(config.CPP_EXECUTABLE, cmd_args, redirect_path)
        results_time[method_name] = exec_time

    print("\n>>> [Step 3] Calculating Speedups & Generating Table...")
    
    speedups = {
        "GloD":  results_time["ExactD"] / results_time["GloD"] if results_time["GloD"] > 0 else 0,
        "PerD":  results_time["ExactD"] / results_time["PerD"] if results_time["PerD"] > 0 else 0,
        "PerD+": results_time["ExactD+"] / results_time["PerD+"] if results_time["PerD+"] > 0 else 0,
        
        "GloH":  results_time["ExactH"] / results_time["GloH"] if results_time["GloH"] > 0 else 0,
        "PerH":  results_time["ExactH"] / results_time["PerH"] if results_time["PerH"] > 0 else 0,
        "PerH+": results_time["ExactH+"] / results_time["PerH+"] if results_time["PerH+"] > 0 else 0,
    }

    print(f"\n{'='*55}")
    print(f"{'Method':<12} | {'Time (s)':<15} | {'Speedup':<15}")
    print(f"{'-'*55}")
    
    for method in ["ExactD", "ExactD+", "ExactH", "ExactH", "T(G*)"]:
        print(f"{method:<12} | {results_time[method]:<15.4f} | {'1.00x':<15}")
    
    print(f"{'-'*55}")
    
    for method in ["GloD", "PerD", "PerD+", "GloH", "PerH", "PerH+"]:
        print(f"{method:<12} | {results_time[method]:<15.4f} | {speedups[method]:>6.2f}x")
        
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()