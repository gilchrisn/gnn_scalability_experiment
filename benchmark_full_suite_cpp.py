import torch
import time
import pandas as pd
import numpy as np
import gc
import os
import sys
import multiprocessing  # <--- Added for process isolation

# Visualization Imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("[Warning] Matplotlib/Seaborn not found. Plots will not be generated.")

# Import your modules
from src import config, models, utils
from src.data import DatasetFactory
from src.adapter import PyGToCppAdapter
from src.cpp_bridge import CppBridge

# --- EXPERIMENT CONFIGURATION ---
N_RUNS = 3
METAPATH_LENGTHS = [2, 3, 4]
K_VALUES = [2, 4, 8, 16, 32]
OUTPUT_PREFIX = "benchmark_results_cpp"
DATASET_TIMEOUT = 180  # <--- NEW: 1 Minute Timeout per dataset to prevent hanging

# List of datasets to exclude if needed (e.g., if one is known to crash)
EXCLUDE_LIST = [] 

def cleanup():
    """Aggressive memory cleanup."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError:
        pass

def save_length_results(results_list, length):
    """Saves results for a specific length to a distinct CSV file."""
    if not results_list:
        return
        
    filename = f"{OUTPUT_PREFIX}_length_{length}.csv"
    df = pd.DataFrame(results_list)
    
    # If file exists, append without header; else write new with header
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)
    print(f"    [Saved] Appended {len(results_list)} rows to {filename}")

def run_gnn_inference(model, graph, features):
    """
    Runs a forward pass and measures pure inference time.
    """
    # Re-initialize device inside the process to be safe
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    graph = graph.to(device)
    x = features.to(device)
    model.eval()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start = time.perf_counter()
    with torch.no_grad():
        if graph.num_edges > 0:
            _ = model(x, graph.edge_index)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    return time.perf_counter() - start

def run_dataset_experiment_cpp(dataset_key, dataset_cfg):
    print(f"\n{'#'*60}")
    print(f"PROCESSING DATASET (C++ BACKEND): {dataset_key}")
    print(f"Process ID: {os.getpid()}")
    print(f"{'#'*60}")

    # Re-verify device inside the isolated process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Data (Python Side)
    try:
        print(f"Loading {dataset_key}...")
        g_hetero, info = DatasetFactory.get_data(
            dataset_cfg['source'],
            dataset_cfg['dataset_name'],
            dataset_cfg['target_node']
        )
    except Exception as e:
        print(f"[CRITICAL FAIL] Could not load {dataset_key}: {e}")
        return

    # 2. Init Model (One dummy model per dataset is enough for timing)
    model = models.get_model(
        "GCN", info['in_dim'], info['out_dim'], config.HIDDEN_DIM, config.GAT_HEADS
    ).to(device)

    # Warmup
    try:
        dummy_edges = torch.zeros((2, 100), dtype=torch.long, device=device)
        with torch.no_grad():
            model(info['features'].to(device), dummy_edges)
    except:
        pass

    # 3. RUN ADAPTER (Convert to C++ Format ONCE)
    print(f"\n>>> Converting Graph to C++ Format (Adapter)...")
    # Adapter uses local paths, so this is safe in multiprocessing
    adapter = PyGToCppAdapter(config.TEMP_DIR)
    adapter.convert(g_hetero) # Writes node.dat, link.dat, meta.dat
    
    # Calculate offset for target node (needed to map C++ IDs back to PyG IDs)
    target_offset = adapter.type_offsets[dataset_cfg['target_node']]
    num_target_nodes = info['features'].shape[0]

    bridge = CppBridge(config.TEMP_DIR)

    # 4. Iterate Metapath Lengths
    for length in METAPATH_LENGTHS:
        print(f"\n--- Metapath Length: {length} ---")
        length_results = [] 
        
        # Generate Metapath & Rule
        try:
            metapath = utils.generate_random_metapath(
                g_hetero, 
                dataset_cfg['target_node'], 
                length
            )
            path_str = "->".join([x[1] for x in metapath])
            print(f"Path: {path_str}")
            
            # Generate C++ Rule File
            rule_str = adapter.generate_cpp_rule(metapath)
            rule_path = adapter.write_rule_file(rule_str)
            
        except ValueError as e:
            print(f"[Skip] Could not generate path/rule for {dataset_key}: {e}")
            continue

        # =========================================================
        # METHOD 1: EXACT MATERIALIZATION (C++ Backend)
        # =========================================================
        print(f"  > Method: Exact (C++)")
        metrics = {'prep': [], 'infer': [], 'edges': []}
        out_file = os.path.join(config.TEMP_DIR, "mat.txt")
        
        failed = False
        for r in range(N_RUNS):
            cleanup()
            try:
                # A. Run C++ Executable
                t_prep = bridge.run_command("materialize", rule_path, out_file)
                
                # B. Load Resulting Graph back to Python
                g_homo = bridge.load_result_graph(out_file, num_target_nodes, target_offset)
                
                # C. Run Inference
                t_infer = run_gnn_inference(model, g_homo, info['features'])
                
                metrics['prep'].append(t_prep)
                metrics['infer'].append(t_infer)
                metrics['edges'].append(g_homo.num_edges)
                
                del g_homo
            except Exception as e:
                print(f"    [Run {r+1}] Failed: {e}")
                failed = True; break

        if not failed:
            length_results.append({
                "Dataset": dataset_key, "Length": length, "Method": "Exact", "K": 0,
                "Edges_Mean": np.mean(metrics['edges']),
                "Prep_Mean": np.mean(metrics['prep']),
                "Infer_Mean": np.mean(metrics['infer']),
                "Total_Mean": np.mean(metrics['prep']) + np.mean(metrics['infer'])
            })
            print(f"    Avg Time: {np.mean(metrics['prep']) + np.mean(metrics['infer']):.4f}s")
        else:
             length_results.append({
                "Dataset": dataset_key, "Length": length, "Method": "Exact", "K": 0,
                "Edges_Mean": -1, "Prep_Mean": -1, "Infer_Mean": -1, "Total_Mean": -1
            })

        # =========================================================
        # METHOD 2: KMV SKETCHING (C++ Backend)
        # =========================================================
        for k in K_VALUES:
            print(f"  > Method: KMV (K={k})", end="", flush=True)
            metrics = {'prep': [], 'infer': [], 'edges': []}
            out_file = os.path.join(config.TEMP_DIR, f"sketch_k{k}.txt")
            
            failed = False
            for r in range(N_RUNS):
                cleanup()
                try:
                    # A. Run C++ Executable
                    t_prep = bridge.run_command("sketch", rule_path, out_file, k=k)
                    
                    # B. Load Result
                    g_sketch = bridge.load_result_graph(out_file, num_target_nodes, target_offset)
                    
                    # C. Run Inference
                    t_infer = run_gnn_inference(model, g_sketch, info['features'])
                    
                    metrics['prep'].append(t_prep)
                    metrics['infer'].append(t_infer)
                    metrics['edges'].append(g_sketch.num_edges)
                    
                    del g_sketch
                    print(".", end="", flush=True)
                    
                except Exception as e:
                    print(f"\n    [Run {r+1} Fail] {e}")
                    failed = True; break
            
            print() # Newline
            
            if not failed:
                length_results.append({
                    "Dataset": dataset_key, "Length": length, "Method": f"KMV (K={k})", "K": k,
                    "Edges_Mean": np.mean(metrics['edges']),
                    "Prep_Mean": np.mean(metrics['prep']),
                    "Infer_Mean": np.mean(metrics['infer']),
                    "Total_Mean": np.mean(metrics['prep']) + np.mean(metrics['infer'])
                })
            else:
                 length_results.append({
                    "Dataset": dataset_key, "Length": length, "Method": f"KMV (K={k})", "K": k,
                    "Edges_Mean": -1, "Prep_Mean": -1, "Infer_Mean": -1, "Total_Mean": -1
                })
        
        # SAVE RESULTS FOR THIS LENGTH IMMEDIATELY
        save_length_results(length_results, length)

    # Cleanup Graph for next dataset
    del g_hetero, info, model, adapter
    cleanup()

def generate_analysis_report():
    """Reads the generated CSVs and produces plots and markdown tables."""
    print("\n" + "="*60)
    print("GENERATING ANALYSIS REPORT (C++ BACKEND)")
    print("="*60)

    for length in METAPATH_LENGTHS:
        filename = f"{OUTPUT_PREFIX}_length_{length}.csv"
        if not os.path.exists(filename):
            print(f"Skipping report for Length {length} (File not found)")
            continue
            
        df = pd.read_csv(filename)
        print(f"\n--- Summary for Metapath Length {length} ---")
        
        # 1. Print Table
        print(df.to_markdown(index=False))
        
        # 2. Generate Plots
        if VISUALIZATION_AVAILABLE:
            try:
                df_clean = df[df['Total_Mean'] > 0].copy()
                
                # Plot 1: Total Time
                plt.figure(figsize=(12, 6))
                sns.barplot(data=df_clean, x='Dataset', y='Total_Mean', hue='Method')
                plt.title(f'C++ Backend: Total Time (Prep + Infer) - Length {length}')
                plt.ylabel('Time (s)')
                plt.yscale('log')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'{OUTPUT_PREFIX}_time_len_{length}.png')
                plt.close()
                
                # Plot 2: Edges
                plt.figure(figsize=(12, 6))
                sns.barplot(data=df_clean, x='Dataset', y='Edges_Mean', hue='Method')
                plt.title(f'C++ Backend: Resulting Edges - Length {length}')
                plt.ylabel('Number of Edges')
                plt.yscale('log')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'{OUTPUT_PREFIX}_edges_len_{length}.png')
                plt.close()
                
                print(f"Plots saved: {OUTPUT_PREFIX}_time_len_{length}.png")
            except Exception as e:
                print(f"Could not generate plots: {e}")

def main():
    # Set Start Method for CUDA compatibility (Crucial!)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Remove old files
    for length in METAPATH_LENGTHS:
        f = f"{OUTPUT_PREFIX}_length_{length}.csv"
        if os.path.exists(f):
            os.remove(f)

    # Iterate over all datasets in config
    for key, cfg in config.DATASET_CONFIGS.items():
        if key in EXCLUDE_LIST:
            continue
            
        print(f"\n---> Spawning isolated process for dataset: {key}")
        
        # Create a completely separate process for this dataset
        p = multiprocessing.Process(
            target=run_dataset_experiment_cpp, 
            args=(key, cfg)
        )
        p.start()
        # Wait for process with a strict timeout (30 mins)
        p.join(timeout=DATASET_TIMEOUT)
        
        # Check if process is still alive after timeout
        if p.is_alive():
            print(f"\n!!! [TIMEOUT] Process for {key} exceeded {DATASET_TIMEOUT}s. Terminating...")
            p.terminate()
            p.join() # Ensure it's fully dead
            print(f"!!! Skipped {key} due to timeout.")
        elif p.exitcode != 0:
            print(f"!!! Process for {key} exited with code {p.exitcode}")
        
    generate_analysis_report()

if __name__ == "__main__":
    main()