import torch
import time
import pandas as pd
import numpy as np
import gc
import os
import sys

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
from src.methods.method1_materialize import _materialize_graph
from src.methods.method2_kmv_sample import _run_kmv_propagation, _build_graph_from_sketches

# --- EXPERIMENT CONFIGURATION ---
N_RUNS = 3
METAPATH_LENGTHS = [2, 3, 4]
K_VALUES = [2, 4, 8, 16, 32]
NK_FACTOR = 1
OUTPUT_PREFIX = "benchmark_results"

# List of datasets to exclude
EXCLUDE_LIST = [] 

def cleanup():
    """Aggressive memory cleanup with error handling."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"    [Warning] Cleanup failed (likely pending OOM): {e}")

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
    graph = graph.to(config.DEVICE)
    x = features.to(config.DEVICE)
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

def run_dataset_experiment(dataset_key, dataset_cfg):
    print(f"\n{'#'*60}")
    print(f"PROCESSING DATASET: {dataset_key}")
    print(f"{'#'*60}")

    # 1. Load Data
    try:
        print(f"Loading {dataset_key}...")
        g_hetero, info = DatasetFactory.get_data(
            dataset_cfg['source'],
            dataset_cfg['dataset_name'],
            dataset_cfg['target_node']
        )
        g_hetero = g_hetero.to(config.DEVICE)
    except Exception as e:
        print(f"[CRITICAL FAIL] Could not load {dataset_key}: {e}")
        return

    # 2. Init Model (One dummy model per dataset is enough for timing)
    model = models.get_model(
        "GCN", info['in_dim'], info['out_dim'], config.HIDDEN_DIM, config.GAT_HEADS
    ).to(config.DEVICE)

    # Warmup
    try:
        dummy_edges = torch.zeros((2, 100), dtype=torch.long, device=config.DEVICE)
        with torch.no_grad():
            model(info['features'].to(config.DEVICE), dummy_edges)
    except:
        pass 

    # 3. Iterate Metapath Lengths
    for length in METAPATH_LENGTHS:
        print(f"\n--- Metapath Length: {length} ---")
        length_results = [] # Store results ONLY for this length iteration
        
        # Generate Metapath
        try:
            metapath = utils.generate_random_metapath(
                g_hetero, 
                dataset_cfg['target_node'], 
                length
            )
            path_str = "->".join([x[1] for x in metapath])
            print(f"Path: {path_str}")
        except ValueError as e:
            print(f"[Skip] Could not generate path length {length} for {dataset_key}: {e}")
            continue

        # =========================================================
        # METHOD 1: EXACT MATERIALIZATION (Baseline)
        # =========================================================
        print(f"  > Method: Exact Materialization")
        metrics = {'prep': [], 'infer': [], 'edges': []}
        
        failed = False
        for r in range(N_RUNS):
            cleanup()
            try:
                g_homo, t_prep = _materialize_graph(
                    g_hetero, info, metapath, dataset_cfg['target_node']
                )
                t_infer = run_gnn_inference(model, g_homo, info['features'])
                
                metrics['prep'].append(t_prep)
                metrics['infer'].append(t_infer)
                metrics['edges'].append(g_homo.num_edges)
                del g_homo
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"    [Run {r+1}] OOM Error!")
                    failed = True; break
                else:
                    print(f"    [Run {r+1}] Error: {e}")
                    failed = True; break
            except Exception as e:
                print(f"    [Run {r+1}] Unexpected Error: {e}")
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
        # METHOD 2: KMV SKETCHING
        # =========================================================
        stop_k_scaling = False # Safety flag
        
        for k in K_VALUES:
            if stop_k_scaling:
                print(f"  > Method: KMV (K={k}) [SKIPPED due to previous OOM]")
                length_results.append({
                    "Dataset": dataset_key, "Length": length, "Method": f"KMV (K={k})", "K": k,
                    "Edges_Mean": -1, "Prep_Mean": -1, "Infer_Mean": -1, "Total_Mean": -1
                })
                continue

            print(f"  > Method: KMV (K={k})", end="", flush=True)
            metrics = {'prep': [], 'infer': [], 'edges': []}
            
            failed = False
            for r in range(N_RUNS):
                cleanup()
                try:
                    sketches, hashes, sort_indices, t_prop = _run_kmv_propagation(
                        g_hetero, metapath, k, NK_FACTOR, config.DEVICE
                    )
                    g_sketch, t_build = _build_graph_from_sketches(
                        sketches, hashes, sort_indices, 
                        g_hetero[dataset_cfg['target_node']].num_nodes, 
                        config.DEVICE
                    )
                    t_prep = t_prop + t_build
                    t_infer = run_gnn_inference(model, g_sketch, info['features'])
                    
                    metrics['prep'].append(t_prep)
                    metrics['infer'].append(t_infer)
                    metrics['edges'].append(g_sketch.num_edges)
                    del sketches, hashes, sort_indices, g_sketch
                    print(".", end="", flush=True)
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f"\n    [Run {r+1} OOM] Aborting K={k} and larger K values.")
                        failed = True
                        stop_k_scaling = True # STOP FLAG SET HERE
                        # Attempt to clear large vars if they exist
                        if 'sketches' in locals(): del sketches
                        if 'g_sketch' in locals(): del g_sketch
                        break 
                    else:
                         print(f"\n    [Run {r+1} RuntimeErr] {e}")
                         failed = True; break
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
    del g_hetero, info, model
    cleanup()

def generate_analysis_report():
    """Reads the generated CSVs and produces plots and markdown tables."""
    print("\n" + "="*60)
    print("GENERATING ANALYSIS REPORT")
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
                # Filter out failures
                df_clean = df[df['Total_Mean'] > 0].copy()
                
                # Plot 1: Total Time Comparison
                plt.figure(figsize=(12, 6))
                sns.barplot(data=df_clean, x='Dataset', y='Total_Mean', hue='Method')
                plt.title(f'Total Execution Time (Prep + Infer) - Length {length}')
                plt.ylabel('Time (s)')
                plt.yscale('log') # Log scale often helps if Exact is very slow
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'plot_time_length_{length}.png')
                plt.close()
                
                # Plot 2: Edge Count Comparison
                plt.figure(figsize=(12, 6))
                sns.barplot(data=df_clean, x='Dataset', y='Edges_Mean', hue='Method')
                plt.title(f'Resulting Edges - Length {length}')
                plt.ylabel('Number of Edges')
                plt.yscale('log')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'plot_edges_length_{length}.png')
                plt.close()
                
                print(f"Plots saved: plot_time_length_{length}.png, plot_edges_length_{length}.png")
            except Exception as e:
                print(f"Could not generate plots for length {length}: {e}")

def main():
    # remove old files to avoid appending to previous runs
    for length in METAPATH_LENGTHS:
        f = f"{OUTPUT_PREFIX}_length_{length}.csv"
        if os.path.exists(f):
            os.remove(f)

    # Iterate over all datasets in config
    for key, cfg in config.DATASET_CONFIGS.items():
        if key in EXCLUDE_LIST:
            continue
        run_dataset_experiment(key, cfg)
        
    generate_analysis_report()

if __name__ == "__main__":
    main()