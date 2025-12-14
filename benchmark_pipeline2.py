import torch
import time
import pandas as pd
import gc
from src import config, models, utils
from src.data import DatasetFactory
from src.methods.method1_materialize import _materialize_graph
from src.methods.method2_kmv_sample import _run_kmv_propagation, _build_graph_from_sketches

# --- Configuration ---
# Ensure we use the metapath length defined in config, or override here
METAPATH_LENGTH = config.METAPATH_LENGTH 
NK = 1 # Expansion factor for KMV (Internal param)

def warmup(model, info):
    """
    Run dummy data through the GPU to initialize CUDA contexts
    and JIT compilers before measuring time.
    """
    print("   [System] Warming up CUDA...")
    dummy_edge_index = torch.randint(0, info['features'].shape[0], (2, 1000), device=config.DEVICE)
    dummy_x = info['features'].to(config.DEVICE)
    model = model.to(config.DEVICE)
    
    # Warmup Model
    with torch.no_grad():
        _ = model(dummy_x, dummy_edge_index)
    
    # Warmup Allocator (Create some tensors and delete them)
    temp = torch.ones((10000, 10000), device=config.DEVICE)
    del temp
    torch.cuda.synchronize()

def run_gnn_inference(model, graph, features):
    """
    Runs a forward pass and measures pure inference time.
    """
    # Ensure everything is on the correct device
    graph = graph.to(config.DEVICE)
    x = features.to(config.DEVICE)
    
    model.eval()
    
    # GPU Sync for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start = time.perf_counter()
    
    with torch.no_grad():
        # Some methods might produce graphs with 0 edges if K is too small
        if graph.num_edges > 0:
            _ = model(x, graph.edge_index)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    return time.perf_counter() - start

def benchmark():
    print(f"{'='*60}")
    print(f"PURE PYTHON BENCHMARK: {config.DATASET_NAME}")
    print(f"Target Node: {config.TARGET_NODE_TYPE} | Device: {config.DEVICE}")
    print(f"{'='*60}\n")

    # 1. Load Data
    print("1. Loading Data...")
    g_hetero, info = DatasetFactory.get_data(
        config.CURRENT_CONFIG['source'],
        config.CURRENT_CONFIG['dataset_name'],
        config.TARGET_NODE_TYPE
    )
    
    # Move Hetero Graph to GPU immediately for PyTorch-based processing
    # (Note: For very large graphs OGB_MAG, you might keep this on CPU and move chunks)
    g_hetero = g_hetero.to(config.DEVICE)

    # 2. Generate Random Metapath
    print("\n2. Generating Metapath...")
    try:
        # Use the utility to find a valid schema path
        metapath = utils.generate_random_metapath(
            g_hetero, 
            config.TARGET_NODE_TYPE, 
            METAPATH_LENGTH
        )
        path_str = " -> ".join([t[1] for t in metapath])
        print(f"   Path: {path_str}")
    except ValueError as e:
        print(f"   [Error] {e}")
        return

    # 3. Initialize Model (Dummy for inference timing)
    print("\n3. Initializing Model...")
    model = models.get_model(
        "GCN", # You can change this to GAT/SAGE
        info['in_dim'], 
        info['out_dim'], 
        config.HIDDEN_DIM, 
        config.GAT_HEADS
    ).to(config.DEVICE)

    warmup(model, info)

    results = []

    # ---------------------------------------------------------
    # METHOD 1: Exact Matrix Multiplication (Python/Torch)
    # ---------------------------------------------------------
    print(f"\n{'='*20} Method 1: Exact Materialization {'='*20}")
    
    try:
        # Clean memory before heavy op
        torch.cuda.empty_cache()
        
        # This function returns (Data, duration)
        g_homo, t_prep = _materialize_graph(
            g_hetero, 
            info, 
            metapath, 
            config.TARGET_NODE_TYPE
        )
        
        t_infer = run_gnn_inference(model, g_homo, info['features'])
        
        print(f"   Edges: {g_homo.num_edges}")
        print(f"   Prep Time: {t_prep:.4f}s | Infer Time: {t_infer:.4f}s")
        
        results.append({
            "Method": "Exact (Py)",
            "K": "N/A",
            "Edges": g_homo.num_edges,
            "Prep_Time": t_prep,
            "Infer_Time": t_infer,
            "Total_Time": t_prep + t_infer
        })
        
        # Cleanup to free VRAM
        del g_homo
        torch.cuda.empty_cache()

    except RuntimeError as e:
        print(f"   [OOM] Exact method failed: {e}")
        results.append({"Method": "Exact (Py)", "K": "N/A", "Edges": "OOM", "Prep_Time": -1})

    # ---------------------------------------------------------
    # METHOD 2: KMV Sketching (Python/Torch)
    # ---------------------------------------------------------
    print(f"\n{'='*20} Method 2: KMV Sketching {'='*20}")
    
    for k in config.K_VALUES:
        print(f"\n   --- Run K={k} ---")
        try:
            # A. Propagation (Generates sketches on nodes)
            # Returns: (final_sketches, sorted_hashes, reverse_map, duration)
            sketches, hashes, rev_map, t_prop = _run_kmv_propagation(
                g_hetero, metapath, k, NK, config.DEVICE
            )
            
            # B. Graph Building (Connects nodes based on sketches)
            # Returns: (Data, duration)
            g_sketch, t_build = _build_graph_from_sketches(
                sketches, hashes, rev_map, 
                g_hetero[config.TARGET_NODE_TYPE].num_nodes, 
                config.DEVICE
            )
            
            t_prep = t_prop + t_build
            
            # C. Inference
            t_infer = run_gnn_inference(model, g_sketch, info['features'])
            
            print(f"     Edges: {g_sketch.num_edges}")
            print(f"     Prep: {t_prep:.4f}s (Prop:{t_prop:.2f} + Build:{t_build:.2f})")
            print(f"     Infer: {t_infer:.4f}s")
            
            results.append({
                "Method": "KMV (Py)",
                "K": k,
                "Edges": g_sketch.num_edges,
                "Prep_Time": t_prep,
                "Infer_Time": t_infer,
                "Total_Time": t_prep + t_infer
            })
            
            # Cleanup
            del sketches, hashes, rev_map, g_sketch
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"     [Error] KMV K={k} Failed: {e}")
            import traceback
            traceback.print_exc()

    # 4. Final Report
    print(f"\n{'='*60}")
    print("FINAL RESULTS (Python Backend)")
    print(f"{'='*60}")
    df = pd.DataFrame(results)
    # Reorder columns for readability
    cols = ["Method", "K", "Edges", "Prep_Time", "Infer_Time", "Total_Time"]
    print(df[cols].to_markdown(index=False))
    
    # Save
    df.to_csv("benchmark_results_python.csv", index=False)

if __name__ == "__main__":
    benchmark()