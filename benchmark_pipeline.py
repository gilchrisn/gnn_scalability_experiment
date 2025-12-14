import torch
import time
import pandas as pd
import os
from src import config, models, utils
from src.data import DatasetFactory
from src.adapter import PyGToCppAdapter
from src.cpp_bridge import CppBridge

def run_gnn_inference(model, graph, features):
    """
    Runs a forward pass and measures pure inference time.
    """
    graph = graph.to(config.DEVICE)
    x = features.to(config.DEVICE)
    
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(x, graph.edge_index)
    
    # Ensure GPU sync for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    return time.perf_counter() - start

def benchmark():
    print(f"{'='*60}")
    print(f"BENCHMARK SUITE: {config.DATASET_NAME}")
    print(f"Target Node: {config.TARGET_NODE_TYPE} | Length: {config.METAPATH_LENGTH}")
    print(f"{'='*60}\n")

    # 1. Load Data
    print("1. Loading Data...")
    g_hetero, info = DatasetFactory.get_data(
        config.CURRENT_CONFIG['source'],
        config.CURRENT_CONFIG['dataset_name'], # <-- FIXED: Uses 'DBLP' instead of 'HGB_DBLP'
        config.TARGET_NODE_TYPE
    )
    
    # 2. Run Adapter
    print("\n2. Running Adapter...")
    adapter = PyGToCppAdapter(config.TEMP_DIR)
    adapter.convert(g_hetero)
    
    # 3. Generate Dynamic Metapath & Rule
    print("\n3. Generating Metapath...")
    try:
        metapath = utils.generate_random_metapath(
            g_hetero, 
            config.TARGET_NODE_TYPE, 
            config.METAPATH_LENGTH
        )
        rule_str = adapter.generate_cpp_rule(metapath)
        print(f"   Rule: {rule_str}")
        rule_path = adapter.write_rule_file(rule_str)
    except ValueError as e:
        print(f"   [Error] {e}")
        return

    # 4. Initialize Bridge & Model
    bridge = CppBridge(config.TEMP_DIR)
    target_offset = adapter.type_offsets[config.TARGET_NODE_TYPE]
    num_target_nodes = info['features'].shape[0]
    
    # Initialize a dummy model for timing (we care about speed here)
    model = models.get_model(
        "GCN", 
        info['in_dim'], 
        info['out_dim'], 
        config.HIDDEN_DIM, 
        config.GAT_HEADS
    ).to(config.DEVICE)

    results = []

    # --- METHOD 1: EXACT MATERIALIZATION ---
    print(f"\n{'='*20} Running Method 1 (Exact) {'='*20}")
    out_file = os.path.join(config.TEMP_DIR, "mat.txt")
    
    try:
        # A. C++ Execution
        t_prep = bridge.run_command("materialize", rule_path, out_file)
        
        # B. Load Result
        g_homo = bridge.load_result_graph(out_file, num_target_nodes, target_offset)
        print(f"   Graph Loaded: {g_homo.num_edges} edges")
        
        # C. Inference
        if g_homo.num_edges > 0:
            t_infer = run_gnn_inference(model, g_homo, info['features'])
            print(f"   Inference Time: {t_infer:.4f}s")
            
            results.append({
                "Method": "Exact",
                "K": "N/A", 
                "Edges": g_homo.num_edges,
                "Prep_Time": t_prep,
                "Infer_Time": t_infer,
                "Total_Time": t_prep + t_infer
            })
        else:
            print("   [Warning] Exact materialization produced 0 edges.")
            # Record 0 edges but success
            results.append({
                "Method": "Exact", "K": "N/A", "Edges": 0,
                "Prep_Time": t_prep, "Infer_Time": 0.0, "Total_Time": t_prep
            })

    except Exception as e:
        print(f"   [!] Method 1 Crashed or Failed: {e}")
        print("   [!] Skipping Exact Method and proceeding to Sketching...")
        
        # Log failure in results so the CSV isn't empty for this row
        results.append({
            "Method": "Exact",
            "K": "N/A", 
            "Edges": "CRASHED",
            "Prep_Time": "N/A",
            "Infer_Time": "N/A",
            "Total_Time": "N/A"
        })

    # --- METHOD 2: SKETCHING (Iterate K) ---
    print(f"\n{'='*20} Running Method 2 (Sketch) {'='*20}")
    for k in config.K_VALUES:
        out_file = os.path.join(config.TEMP_DIR, f"sketch_k{k}.txt")
        
        # A. C++ Execution
        t_prep = bridge.run_command("sketch", rule_path, out_file, k=k)
        
        # B. Load Result
        g_sketch = bridge.load_result_graph(out_file, num_target_nodes, target_offset)
        print(f"   [K={k}] Graph Loaded: {g_sketch.num_edges} edges")
        
        # C. Inference
        if g_sketch.num_edges > 0:
            t_infer = run_gnn_inference(model, g_sketch, info['features'])
            print(f"   [K={k}] Inference Time: {t_infer:.4f}s")
            
            results.append({
                "Method": "Sketch",
                "K": k,
                "Edges": g_sketch.num_edges,
                "Prep_Time": t_prep,
                "Infer_Time": t_infer,
                "Total_Time": t_prep + t_infer
            })

    # 5. Report
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    
    # Save
    df.to_csv("benchmark_results.csv", index=False)

if __name__ == "__main__":
    benchmark()