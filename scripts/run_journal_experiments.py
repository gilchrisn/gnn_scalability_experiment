
"""
Master Execution Script for Journal Experiments (A, B, C, D).
Validates the 'Stochastic Materialization' hypothesis using the C++ Engine.

Usage:
    python scripts/run_journal_experiments.py
"""
import subprocess
import os
import sys
from typing import List

# --- Configuration ---
PYTHON_EXEC = sys.executable
MAIN_SCRIPT = "main.py"
RESULTS_DIR = "output/journal_experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Standard Metapaths for Experiment A (Fidelity)
STD_BENCHMARKS = [
    ("HGB_ACM",  "paper_to_author,author_to_paper", "paper"),        # PAP
    # ("HGB_DBLP", "author_to_paper,paper_to_author", "author"),       # APA
    # ("HGB_IMDB", "movie_to_actor,actor_to_movie",   "movie"),        # MAM
]

def run_cmd(cmd_list: List[str], log_file: str = None) -> None:
    """Executes a subprocess command safely."""
    cmd_str = " ".join(cmd_list)
    print(f"\n[EXEC] {cmd_str}")
    
    try:
        if log_file:
            with open(log_file, "w") as f:
                subprocess.run(cmd_list, check=True, stdout=f, stderr=subprocess.STDOUT)
        else:
            subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}")
        pass

def construct_dblp_path(length: int) -> str:
    """Constructs A-P-A-P... sequence of specific length for Scalability test."""
    path = []
    rels = ["author_to_paper", "paper_to_author"]
    for i in range(length):
        path.append(rels[i % 2])
    return ",".join(path)

# ---------------------------------------------------------
# Experiment A: Fidelity Test (Effectiveness)
# ---------------------------------------------------------
def run_experiment_a_fidelity() -> None:
    """
    Experiment A: The Core Comparison.
    Goal: Prove C++ KMV (Static) is statistically comparable to C++ Exact.
    """
    print(f"\n{'='*60}\nEXPERIMENT A: FIDELITY TEST (Robustness) - CPP BACKEND\n{'='*60}")
    output_csv = os.path.join(RESULTS_DIR, "exp_a_fidelity_cpp.csv")
    if os.path.exists(output_csv): os.remove(output_csv)

    modes = ["exact", "kmv-static"] # kmv-dynamic typically stays in Python for callback flexibility, or we treat it as static per epoch
    
    for dataset, metapath, _ in STD_BENCHMARKS:
        for mode in modes:
            print(f"\n>> Benchmarking {dataset} | {mode.upper()}")
            
            cmd = [
                PYTHON_EXEC, MAIN_SCRIPT,
                "--device", "cuda", # Training on GPU
                "train_fidelity",
                "--dataset", dataset,
                "--metapath", metapath,
                "--mode", mode,
                "--backend", "cpp",  # Force C++ Backend
                "--model", "GCN",
                "--epochs", "100",
                "--k", "32",
                "--csv-output", output_csv
            ]
            run_cmd(cmd)

# ---------------------------------------------------------
# Experiment B: The "Geometric Explosion" (Scalability)
# ---------------------------------------------------------
def run_experiment_b_scalability() -> None:
    """
    Experiment B: Scalability Stress Test.
    Goal: Demonstrate C++ Exact fails/slows on long paths while C++ KMV is constant.
    """
    print(f"\n{'='*60}\nEXPERIMENT B: SCALABILITY (Geometric Explosion) - CPP BACKEND\n{'='*60}")
    output_csv = os.path.join(RESULTS_DIR, "exp_b_scalability_cpp.csv")
    if os.path.exists(output_csv): os.remove(output_csv)

    lengths = [2, 4, 6] 
    dataset = "HGB_DBLP"

    for L in lengths:
        metapath = construct_dblp_path(L)
        print(f"\n>> Length {L} ({metapath[:30]}...)")

        # 1. Exact (C++ Backend)
        run_cmd([
            PYTHON_EXEC, MAIN_SCRIPT,
            "--device", "cpu", # Use CPU for training to allow RAM to handle massive exact graph if it survives
            "train_fidelity",
            "--dataset", dataset,
            "--metapath", metapath,
            "--mode", "exact",
            "--backend", "cpp", # Force C++
            "--model", "GCN",
            "--epochs", "0", # Only measure prep time
            "--csv-output", output_csv
        ])

        # 2. KMV Static (C++ Backend)
        run_cmd([
            PYTHON_EXEC, MAIN_SCRIPT,
            "--device", "cpu",
            "train_fidelity",
            "--dataset", dataset,
            "--metapath", metapath,
            "--mode", "kmv-static",
            "--backend", "cpp", # Force C++
            "--k", "32",
            "--model", "GCN",
            "--epochs", "0",
            "--csv-output", output_csv
        ])

# ---------------------------------------------------------
# Experiment C: Sketch Sensitivity
# ---------------------------------------------------------
def run_experiment_c_sensitivity() -> None:
    """
    Experiment C: Hyperparameter K using C++ Sketching.
    """
    print(f"\n{'='*60}\nEXPERIMENT C: SENSITIVITY (Sketch Size) - CPP BACKEND\n{'='*60}")
    output_csv = os.path.join(RESULTS_DIR, "exp_c_sensitivity_cpp.csv")
    if os.path.exists(output_csv): os.remove(output_csv)

    k_values = [2, 4, 8, 16, 32, 64]
    dataset = "HGB_DBLP"
    metapath = "author_to_paper,paper_to_author"

    # Baseline (Exact) - Run once
    print(f"\n>> Sensitivity Baseline: EXACT")
    run_cmd([
        PYTHON_EXEC, MAIN_SCRIPT,
        "train_fidelity",
        "--dataset", dataset,
        "--metapath", metapath,
        "--mode", "exact",
        "--backend", "cpp",
        "--model", "GCN",
        "--epochs", "50",
        "--csv-output", output_csv
    ])

    # KMV Sweep
    for k in k_values:
        print(f"\n>> Sensitivity Sweep: K={k}")
        run_cmd([
            PYTHON_EXEC, MAIN_SCRIPT,
            "train_fidelity",
            "--dataset", dataset,
            "--metapath", metapath,
            "--mode", "kmv-static",
            "--backend", "cpp",
            "--k", str(k),
            "--model", "GCN",
            "--epochs", "50",
            "--csv-output", output_csv
        ])

# ---------------------------------------------------------
# Experiment D: Semantic Preservation (The "Why")
# ---------------------------------------------------------
def run_experiment_d_semantics() -> None:
    """
    Experiment D: Representation Topology.
    Uses models trained via C++ backend in Experiment A/C.
    """
    print(f"\n{'='*60}\nEXPERIMENT D: SEMANTIC PRESERVATION (CKA)\n{'='*60}")
    
    dataset = "HGB_DBLP"
    metapath = "author_to_paper,paper_to_author"
    k_val = 32

    print(">> Ensuring pre-trained models exist for DBLP (Using C++ Backend)...")
    
    # Train Exact
    run_cmd([
        PYTHON_EXEC, MAIN_SCRIPT, "train_fidelity",
        "--dataset", dataset, "--metapath", metapath, 
        "--mode", "exact", "--backend", "cpp", "--epochs", "10"
    ], log_file=os.path.join(RESULTS_DIR, "d_prep_exact.log"))

    # Train KMV
    run_cmd([
        PYTHON_EXEC, MAIN_SCRIPT, "train_fidelity",
        "--dataset", dataset, "--metapath", metapath, 
        "--mode", "kmv-static", "--backend", "cpp", "--k", str(k_val), "--epochs", "10"
    ], log_file=os.path.join(RESULTS_DIR, "d_prep_kmv.log"))

    # Execute Visualization
    # Note: visualize_fidelity loads the saved python graph, so backend arg is less critical here, 
    # but we pass it for consistency in loading if needed.
    print(f"\n>> Calculating CKA and generating heatmap...")
    run_cmd([
        PYTHON_EXEC, MAIN_SCRIPT,
        "visualize_fidelity",
        "--dataset", dataset,
        "--metapath", metapath,
        "--model", "GCN",
        "--k", str(k_val)
    ])

# ---------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Starting Complete Journal Experiment Suite (C++ Backend)...")
    
    # 1. Fidelity (The "What")
    run_experiment_a_fidelity()
    
    # 2. Scalability (The "How")
    run_experiment_b_scalability()
    
    # 3. Sensitivity (The "How Much")
    run_experiment_c_sensitivity()
    
    # 4. Semantics (The "Why")
    run_experiment_d_semantics()
    
    print(f"\n[DONE] All experiments completed. Check {RESULTS_DIR}")