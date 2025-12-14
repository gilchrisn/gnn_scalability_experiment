import torch
import time
import pandas as pd
from src import config, models, utils
from src.data import DatasetFactory 
from src.adapter import PyGToCppAdapter
from src.cpp_bridge import CppBridge

def benchmark():
    # 1. Load Config
    cfg = config.CURRENT_CONFIG
    
    # 2. Orchestrate Data Loading
    # The Factory handles the details (HGB vs OGB logic)
    g, info = DatasetFactory.get_data(
        source_type=cfg['source'], 
        dataset_name=config.DATASET_NAME, 
        target_ntype=cfg['target_node']
    )
    
    # 3. Run Adapter (PyG -> C++)
    # The adapter works blindly on 'g', regardless of where 'g' came from
    adapter = PyGToCppAdapter(config.TEMP_DIR)
    adapter.convert(g)
    
    # Create the rule file needed by C++
    rule_path = adapter.create_rule_file(config.CURRENT_CONFIG['rule_str'])
    
    # Calculate offset for the target node type (needed to map back IDs)
    # The C++ IDs are global. We need to subtract the offset of 'author' (or target)
    target_offset = adapter.type_offsets[config.TARGET_NODE_TYPE]
    
    bridge = CppBridge(config.TEMP_DIR)
    model_list = ["GCN", "GAT"] # Add SAGE etc
    results = []

    for model_name in model_list:
        print(f"\n--- Benchmarking {model_name} ---")
        # Load Model (assuming trained)
        ckpt = utils.get_model_checkpoint_path(config.MODEL_DIR, model_name, config.TRAIN_METAPATH)
        # ... load lit_model logic ...
        # dummy model for now
        model = models.get_model(model_name, info['in_dim'], info['out_dim'], config.HIDDEN_DIM, config.GAT_HEADS).to(config.DEVICE)

        # --- Method 1: Materialize ---
        out_file = f"{config.TEMP_DIR}/mat.txt"
        t_prep = bridge.run_command("materialize", rule_path, out_file)
        
        # Load and Infer
        g_homo = bridge.load_result_graph(out_file, info['features'].shape[0], target_offset)
        g_homo = g_homo.to(config.DEVICE)
        g_homo.x = info['features'].to(config.DEVICE)
        
        t_infer_start = time.perf_counter()
        with torch.no_grad():
            _ = model(g_homo.x, g_homo.edge_index)
        t_infer = time.perf_counter() - t_infer_start
        
        results.append({"Model": model_name, "Method": "Materialize", "Prep": t_prep, "Infer": t_infer})

        # --- Method 2: Sketch ---
        for k in [2, 32]:
            out_file = f"{config.TEMP_DIR}/sketch_k{k}.txt"
            t_prep = bridge.run_command("sketch", rule_path, out_file, k=k)
            
            g_sketch = bridge.load_result_graph(out_file, info['features'].shape[0], target_offset)
            g_sketch = g_sketch.to(config.DEVICE)
            g_sketch.x = info['features'].to(config.DEVICE)
            
            t_infer_start = time.perf_counter()
            with torch.no_grad():
                _ = model(g_sketch.x, g_sketch.edge_index)
            t_infer = time.perf_counter() - t_infer_start
            
            results.append({"Model": model_name, "Method": f"Sketch(k={k})", "Prep": t_prep, "Infer": t_infer})

    print(pd.DataFrame(results))

if __name__ == "__main__":
    benchmark()