import subprocess
import os
import sys
import csv
import time
import re

def run_benchmark(dataset, path_str, method, k=32):
    """Runs main.py and parses the output for time/edges."""
    cmd = [
        "python", "main.py", "benchmark",
        "--dataset", dataset,
        "--manual-path", path_str,
        "--method", method,
        "--backend", "python" # or 'cpp'
    ]
    
    if method == 'kmv':
        cmd.extend(["--k", str(k)])
        
    try:
        # Run command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout
        
        # Regex to find numbers in the output
        # Looks for "Nodes: 1,234 | Edges: 5,678"
        # Looks for "Prep: 0.1234s"
        
        edges_match = re.search(r"Edges: ([\d,]+)", output)
        time_match = re.search(r"Prep: ([\d\.]+)s", output)
        
        if edges_match and time_match:
            edges = int(edges_match.group(1).replace(",", ""))
            prep_time = float(time_match.group(1))
            return edges, prep_time
        else:
            print(f"   [Error] Could not parse output for {method}")
            return -1, -1
            
    except Exception as e:
        print(f"   [Exception] {e}")
        return -1, -1

def main(dataset):
    # Setup paths
    base_dir = os.path.join("data", dataset)
    input_file = os.path.join(base_dir, "metapaths_clean.txt")
    results_file = os.path.join("output", f"final_results_{dataset}.csv")
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        print("Run 'python convert_rules.py' first!")
        return

    # Prepare CSV
    os.makedirs("output", exist_ok=True)
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metapath", "Exact_Edges", "Exact_Time", "KMV_Edges", "KMV_Time", "Speedup", "Accuracy"])

    print(f"Reading paths from {input_file}...")
    with open(input_file, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]

    print(f"Starting benchmark on {len(paths)} paths...")
    print("-" * 60)

    for i, path in enumerate(paths):
        print(f"[{i+1}/{len(paths)}] Path: {path}")
        
        # 1. Run Exact
        print("   Running Exact...", end="", flush=True)
        ex_edges, ex_time = run_benchmark(dataset, path, "exact")
        print(f" Done ({ex_time:.4f}s)")
        
        # 2. Run KMV (k=32)
        print("   Running KMV...", end="", flush=True)
        km_edges, km_time = run_benchmark(dataset, path, "kmv", k=32)
        print(f" Done ({km_time:.4f}s)")
        
        # 3. Calculate Stats
        speedup = ex_time / km_time if km_time > 0 else 0
        accuracy = km_edges / ex_edges if ex_edges > 0 else 0
        
        # 4. Save
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([path, ex_edges, ex_time, km_edges, km_time, f"{speedup:.2f}", f"{accuracy:.4f}"])
            
    print("-" * 60)
    print(f"✅ Benchmarking complete. Results saved to {results_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_batch.py <Dataset_Key>")
        print("Example: python run_batch.py HNE_DBLP")
    else:
        main(sys.argv[1])