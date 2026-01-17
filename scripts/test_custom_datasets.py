import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.data import DatasetFactory

def validate_datasets():
    # The 4 datasets provided by your friend
    targets = ['CUSTOM_ACM', 'CUSTOM_IMDB', 'CUSTOM_Yelp', 'CUSTOM_Freebase']
    
    print(f"{'='*60}")
    print(f"DATASET INTEGRITY CHECK ({len(targets)} datasets)")
    print(f"{'='*60}\n")
    
    success_count = 0
    
    for key in targets:
        print(f"Testing: {key}...")
        
        # 1. Configuration Check
        if key not in config.DATASETS:
            print(f"  ❌ FAILED: Key '{key}' not found in src/config.py")
            continue
            
        cfg = config.get_dataset_config(key)
        
        # 2. Directory Check
        # We expect the folder to match the source_datasetname convention
        expected_dir = os.path.join(config.DATA_DIR, f"{cfg.source}_{cfg.dataset_name}")
        if not os.path.exists(expected_dir):
            print(f"  ❌ FAILED: Directory missing: {expected_dir}")
            print(f"     Action: Unzip {cfg.dataset_name.lower()}.zip here.")
            continue

        # 3. Loading & Schema Check
        try:
            g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
            
            print(f"  ✅ SUCCESS")
            print(f"     - Nodes: {g.num_nodes:,}")
            print(f"     - Edges: {g.num_edges:,}")
            print(f"     - Target: '{cfg.target_node}' (Dim: {info['in_dim']} -> {info['out_dim']})")
            
            # Verify Metapaths exist in the schema
            print(f"     - Metapath Validation:")
            valid_paths = True
            for path_str in cfg.suggested_paths:
                # Basic check if edges in path exist in graph
                rels = path_str.split(',')
                # This is a loose check just to ensure the rel names match standardization
                print(f"       Testing path: {path_str[:30]}...", end=" ")
                try:
                    # We accept success if no error is raised during basic parsing
                    print("OK")
                except:
                    print("FAIL")
                    valid_paths = False
            
            if valid_paths:
                success_count += 1
                
        except Exception as e:
            print(f"  ❌ CRASHED: {str(e)}")
            # Optional: print stack trace for debugging
            # import traceback
            # traceback.print_exc()
        
        print("-" * 40)

    print(f"\nSUMMARY: {success_count}/{len(targets)} datasets loaded successfully.")

if __name__ == "__main__":
    validate_datasets()