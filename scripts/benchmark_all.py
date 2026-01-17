import subprocess
import sys
import os
from datetime import datetime

# --- CONFIGURATION ---
PYTHON = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_PY = os.path.join(ROOT, "main.py")

# Number of times to repeat each experiment for stability
N_RUNS = 5

# Output file
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
CSV_FILE = os.path.join(ROOT, "output", "results", f"full_benchmark_{TIMESTAMP}.csv")
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

MODELS = ["GCN", "SAGE", "GAT"]
K_VALUES = [2, 4, 8, 16, 32, 64]

DATASETS = {
    "HGB_DBLP": [
        "author_to_paper,paper_to_author",
        "author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"
    ],
    "HGB_ACM": [
        "paper_to_author,author_to_paper",
        "paper_to_subject,subject_to_paper"
    ],
    "HGB_IMDB": [
        "movie_to_actor,actor_to_movie",
        "movie_to_director,director_to_movie"
    ],
    "PyG_IMDB": [
        "movie_to_actor,actor_to_movie",
        "movie_to_director,director_to_movie"
    ],
}

def run():
    print(f"--- STARTING FULL BENCHMARK SUITE ({N_RUNS} runs per config) ---")
    print(f"Results: {CSV_FILE}")
    
    # Calculate total operations for progress tracking
    ops_per_path = (1 + 2 * len(K_VALUES)) * N_RUNS
    total_ops = len(MODELS) * sum(len(p) for p in DATASETS.values()) * ops_per_path
    current_op = 0

    for model in MODELS:
        for ds_name, paths in DATASETS.items():
            
            # 0. Auto-Train (Once per model/dataset - Use GPU)
            print(f"\n[Auto-Setup] Ensuring model exists: {ds_name} | {model}")
            subprocess.run([
                PYTHON, MAIN_PY, 
                "train",
                "--dataset", ds_name,
                "--model", model,
                "--epochs", "50"
            ])

            for metapath in paths:
                short_path = metapath[:30] + "..." if len(metapath) > 30 else metapath
                
                # --- REPEAT BENCHMARKS N TIMES ---
                for i in range(N_RUNS):
                    run_tag = f"[Run {i+1}/{N_RUNS}]"
                    
                    # 1. Run Exact
                    current_op += 1
                    print(f"\n[{current_op}/{total_ops}] {run_tag} EXACT | {ds_name} | {short_path}")
                    
                    cmd = [
                        PYTHON, MAIN_PY, 
                        "--device", "cpu",  
                        "benchmark",
                        "--dataset", ds_name,
                        "--method", "exact",
                        "--backend", "python",
                        "--model", model,
                        "--manual-path", metapath,
                        "--check-fidelity",
                        "--csv-output", CSV_FILE
                    ]
                    subprocess.run(cmd)

                    # 2. Run KMV
                    for k in K_VALUES:
                        current_op += 1
                        print(f"\n[{current_op}/{total_ops}] {run_tag} KMV (k={k}) | {ds_name}")
                        
                        cmd = [
                            PYTHON, MAIN_PY, 
                            "--device", "cpu",  
                            "benchmark",
                            "--method", "kmv",
                            "--k", str(k),
                            "--dataset", ds_name,
                            "--backend", "python",
                            "--model", model,
                            "--manual-path", metapath,
                            "--check-fidelity",
                            "--csv-output", CSV_FILE
                        ]
                        subprocess.run(cmd)

                    # 3. Run Random Sampling
                    for k in K_VALUES:
                        current_op += 1
                        print(f"\n[{current_op}/{total_ops}] {run_tag} RANDOM (k={k}) | {ds_name}")
                        
                        cmd = [
                            PYTHON, MAIN_PY, 
                            "--device", "cpu",  
                            "benchmark",
                            "--method", "random",
                            "--k", str(k),
                            "--dataset", ds_name,
                            "--backend", "python",
                            "--model", model,
                            "--manual-path", metapath,
                            "--check-fidelity",
                            "--csv-output", CSV_FILE
                        ]
                        subprocess.run(cmd)

if __name__ == "__main__":
    run()