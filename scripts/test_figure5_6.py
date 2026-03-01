#!/usr/bin/env python3
"""
Figures 5 & 6 Raw Data Extraction.
Evaluates F1/Accuracy scores against varying lambda (Top-R) and k (Sketch Size).
Outputs strictly in raw text/CSV format for downstream validation.
"""
import os
import sys
import argparse
import subprocess
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Internal dependencies
from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from src.utils import SchemaMatcher

# --- Configuration & Constants ---
LAMBDAS = ["0.02", "0.03", "0.04", "0.05"]
K_VALUES = ["2", "4", "8", "16", "32"]
DEFAULT_LAMBDA = "0.05"
DEFAULT_K_DEG = "32"
DEFAULT_K_HIDX = "4"

def generate_qnodes(data_dir: str, folder_name: str) -> None:
    """Selects 100 random query nodes to prevent C++ silent bypass."""
    import random
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
    """Compiles rule bytecode to both expected C++ filenames."""
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

def execute_and_parse_f1(binary: str, args: list, redirect_path: str = None) -> float:
    """Executes C++ command, handles redirection, prints raw output, and strictly parses ~goodness."""
    cmd = [binary] + args
    print(f"  [Engine] Executing: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1200)
    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL FAILURE] C++ Exception Caught. Exit Code: {e.returncode}")
        print(f"Command: {' '.join(cmd)}")
        print(f"--- STDERR ---\n{e.stderr.strip()}")
        sys.exit(1)

    if redirect_path:
        with open(redirect_path, 'w') as f:
            f.write(result.stdout)

    # --- INJECTED RAW OUTPUT SECTION ---
    print("\n" + "-"*40)
    print(f"--- RAW C++ STDOUT ({args[0]}) ---")
    print("-" * 40)
    print(result.stdout.strip() if result.stdout.strip() else "[NO OUTPUT - REDIRECTED TO FILE]")
    print("-" * 40)

    if redirect_path:
        return -1.0 # Baselines don't output goodness, they generate the truth file.
        
    match = re.search(r"~goodness:\s*([0-9.]+)", result.stdout)
    if not match:
        print(f"\n[FATAL] Failed to parse '~goodness:' from output for command: {' '.join(cmd)}")
        sys.exit(1)
        
    return float(match.group(1))

def generate_exact_baselines(folder_name: str, lam: str, df1_dir: str, hf1_dir: str):
    """Forces the C++ engine to generate exact ground truths for a specific lambda."""
    plan = [
        ("ExactD+", ["ExactD+", folder_name, lam, "0"], os.path.join(df1_dir, f"hg_global_r{lam}.res")),
        ("ExactD",  ["ExactD",  folder_name, lam, "0"], os.path.join(df1_dir, f"hg_global_greater_r{lam}.res")),
        ("ExactH+", ["ExactH+", folder_name, lam, "0"], os.path.join(hf1_dir, f"hg_global_r{lam}.res")),
        ("ExactH",  ["ExactH",  folder_name, lam, "0"], os.path.join(hf1_dir, f"hg_global_greater_r{lam}.res")),
    ]
    for _, args, redirect in plan:
        execute_and_parse_f1(config.CPP_EXECUTABLE, args, redirect)

def main():
    parser = argparse.ArgumentParser(description="Raw Extraction for Figures 5 & 6")
    parser.add_argument('dataset', type=str)
    parser.add_argument('--metapath', type=str, required=True)
    args = parser.parse_args()

    folder_name = f"HGBn-{args.dataset.split('_')[1]}"
    data_dir = os.path.join(project_root, folder_name)
    
    # 1. Staging Data and Rules
    cfg = config.get_dataset_config(args.dataset)
    g_hetero, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    
    PyGToCppAdapter(data_dir).convert(g_hetero)
    generate_qnodes(data_dir, folder_name)
    compile_rule_for_cpp(args.metapath, g_hetero, data_dir, folder_name)
    df1_dir, hf1_dir = setup_global_res_dirs(folder_name)

    methods_deg = [("GloD", "0"), ("PerD", "0"), ("PerD+", "0.1")]
    methods_hidx = [("GloH", "0"), ("PerH", "0"), ("PerH+", "0.1")]

    # Data structures to hold results
    res_lam_deg = {m[0]: [] for m in methods_deg}
    res_lam_hidx = {m[0]: [] for m in methods_hidx}
    res_k_deg = {m[0]: [] for m in methods_deg}
    res_k_hidx = {m[0]: [] for m in methods_hidx}

    print(f"[System] Starting F1 Extraction for {args.dataset} | Metapath: {args.metapath}")

    # --- EXPERIMENT 1: Varying Lambda ---
    print(f"[System] Executing Lambda Sweep (k_deg={DEFAULT_K_DEG}, k_hidx={DEFAULT_K_HIDX})...")
    for lam in LAMBDAS:
        generate_exact_baselines(folder_name, lam, df1_dir, hf1_dir)
        for meth, beta in methods_deg:
            res_lam_deg[meth].append(execute_and_parse_f1(config.CPP_EXECUTABLE, [meth, folder_name, lam, beta, DEFAULT_K_DEG]))
        for meth, beta in methods_hidx:
            res_lam_hidx[meth].append(execute_and_parse_f1(config.CPP_EXECUTABLE, [meth, folder_name, lam, beta, DEFAULT_K_HIDX]))

    # --- EXPERIMENT 2: Varying K ---
    print(f"[System] Executing K Sweep (lambda={DEFAULT_LAMBDA})...")
    generate_exact_baselines(folder_name, DEFAULT_LAMBDA, df1_dir, hf1_dir)
    for k in K_VALUES:
        for meth, beta in methods_deg:
            res_k_deg[meth].append(execute_and_parse_f1(config.CPP_EXECUTABLE, [meth, folder_name, DEFAULT_LAMBDA, beta, k]))
        for meth, beta in methods_hidx:
            res_k_hidx[meth].append(execute_and_parse_f1(config.CPP_EXECUTABLE, [meth, folder_name, DEFAULT_LAMBDA, beta, k]))

    # --- RAW OUTPUT DUMP ---
    print("\n" + "="*60)
    print("--- RAW F1 DATA EXTRACTION RESULTS ---")
    print("="*60)
    
    print("\n[EXPERIMENT 1: VARYING LAMBDA]")
    print(f"LAMBDAS,{','.join(LAMBDAS)}")
    for meth in ["GloD", "PerD", "PerD+"]:
        print(f"{meth},{','.join(map(str, res_lam_deg[meth]))}")
    for meth in ["GloH", "PerH", "PerH+"]:
        print(f"{meth},{','.join(map(str, res_lam_hidx[meth]))}")

    print("\n[EXPERIMENT 2: VARYING K]")
    print(f"K_VALUES,{','.join(K_VALUES)}")
    for meth in ["GloD", "PerD", "PerD+"]:
        print(f"{meth},{','.join(map(str, res_k_deg[meth]))}")
    for meth in ["GloH", "PerH", "PerH+"]:
        print(f"{meth},{','.join(map(str, res_k_hidx[meth]))}")
        
    print("="*60)

if __name__ == "__main__":
    main()