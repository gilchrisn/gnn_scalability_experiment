import torch
import pandas as pd
import psutil
import os
import gc
import sys

# Import your project modules
from src import config
from src.data import DatasetFactory

def get_memory_usage():
    """Returns current RAM usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def collect_metadata():
    print(f"{'='*80}")
    print(f"DATASET METADATA EXTRACTOR")
    print(f"{'='*80}")
    
    results = []
    
    # Iterate over all datasets defined in config
    keys = list(config.DATASET_CONFIGS.keys())
    
    for i, key in enumerate(keys):
        cfg = config.DATASET_CONFIGS[key]
        print(f"[{i+1}/{len(keys)}] Inspecting {key}...", end=" ", flush=True)
        
        try:
            # 1. Load Data
            # Note: We load onto CPU first to avoid OOM on GPU for inspection
            g, info = DatasetFactory.get_data(
                cfg['source'], 
                cfg['dataset_name'], 
                cfg['target_node']
            )
            
            # 2. Extract Metrics
            # Count total nodes across all types if g.num_nodes is not aggregating properly
            if hasattr(g, 'num_nodes') and g.num_nodes:
                total_nodes = g.num_nodes
            else:
                total_nodes = sum(g[nt].num_nodes for nt in g.node_types)

            # Count total edges
            if hasattr(g, 'num_edges') and g.num_edges:
                total_edges = g.num_edges
            else:
                total_edges = sum(g[et].num_edges for et in g.edge_types)
                
            stats = {
                "Dataset Key": key,
                "Source": cfg['source'],
                "Name": cfg['dataset_name'],
                "Target Node": cfg['target_node'],
                "Total Nodes": total_nodes,
                "Total Edges": total_edges,
                "Node Types": len(g.node_types),
                "Edge Types": len(g.edge_types),
                "Feature Dim": info['in_dim'],
                "Num Classes": info['out_dim'],
                "Schema (Node Types)": ", ".join(g.node_types),
                # "Schema (Edge Types)": ", ".join([rel for _, rel, _ in g.edge_types]) # often too long
            }
            
            results.append(stats)
            print(f"Success. ({get_memory_usage():.2f} GB RAM used)")
            
            # 3. Cleanup
            del g, info
            gc.collect()
            
        except Exception as e:
            print(f"Failed! Error: {e}")
            results.append({
                "Dataset Key": key,
                "Source": cfg['source'],
                "Name": "ERROR",
                "Total Nodes": str(e)
            })
            
    print(f"\n{'='*80}")
    print("METADATA COLLECTION COMPLETE")
    print(f"{'='*80}")

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    cols = [
        "Dataset Key", "Source", "Name", "Target Node", 
        "Total Nodes", "Total Edges", 
        "Node Types", "Edge Types", 
        "Feature Dim", "Num Classes"
    ]
    # Add remaining columns (like Schema) at the end
    remaining_cols = [c for c in df.columns if c not in cols]
    df = df[cols + remaining_cols]

    # Display Table
    print(df.drop(columns=remaining_cols).to_markdown(index=False))
    
    # Save to CSV
    output_file = "dataset_metadata.csv"
    df.to_csv(output_file, index=False)
    print(f"\n[Saved] Full metadata saved to {output_file}")

if __name__ == "__main__":
    collect_metadata()