import sys
import os
import time
import gc
import psutil
import torch

# Add project root to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DatasetFactory
from src import config

def print_memory_usage(step_name):
    """Prints current System RAM and GPU VRAM usage."""
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024 ** 3)
    print(f"   [MEM] {step_name} | System RAM: {ram_gb:.2f} GB", end="")
    
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f" | GPU VRAM: {vram_allocated:.2f} GB (Reserved: {vram_reserved:.2f} GB)")
    else:
        print("")

def test_single_dataset(key, cfg):
    print(f"\n{'='*60}")
    print(f"TESTING DATASET: {key}")
    print(f"{'='*60}")
    
    print_memory_usage("Start")

    # 1. Load into System RAM
    try:
        start_time = time.time()
        print(f"1. Loading {key} ({cfg['source']} strategy)...")
        
        g, info = DatasetFactory.get_data(
            source_type=cfg['source'],
            dataset_name=cfg['dataset_name'],
            target_ntype=cfg['target_node']
        )
        
        load_time = time.time() - start_time
        print(f"   SUCCESS: Loaded in {load_time:.2f}s")
        print(f"   Stats: {g.num_nodes} nodes, {g.num_edges} edges")
        print_memory_usage("After Load (CPU)")

    except Exception as e:
        print(f"   FAILED to load on CPU: {e}")
        return # Stop this test if CPU load fails

    # 2. Attempt Move to GPU (The 4070 Test)
    if torch.cuda.is_available():
        print(f"2. Attempting to move to GPU ({torch.cuda.get_device_name(0)})...")
        try:
            # Reset peak stats
            torch.cuda.reset_peak_memory_stats()
            
            # Move to GPU
            g = g.to(config.DEVICE)
            # Force a synchronization to ensure move is complete
            torch.cuda.synchronize()
            
            print("   SUCCESS: Graph fits in VRAM!")
            print_memory_usage("After Move (GPU)")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("   FAILED: CUDA Out of Memory (OOM). This graph is too big for your 4070.")
                # Clear the cache to recover for next test
                torch.cuda.empty_cache()
            else:
                print(f"   FAILED with unexpected error: {e}")
    else:
        print("   Skipping GPU test (CUDA not available)")

    # 3. Cleanup
    print("3. Cleaning up...")
    del g
    del info
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print_memory_usage("After Cleanup")
    print("-" * 60)

if __name__ == "__main__":
    print("Starting Stress Test Suite...")
    
    # Iterate over all configs defined in src/config.py
    # We prioritize 'HGB_DBLP' and 'PyG_IMDB' first as sanity checks
    priority_order = ['PyG_IMDB', 'HGB_DBLP', 'OGB_MAG']
    
    # Filter configs to run priority first, then others
    configs_to_run = []
    for key in priority_order:
        if key in config.DATASET_CONFIGS:
            configs_to_run.append((key, config.DATASET_CONFIGS[key]))
    
    # Add any remaining configs
    for key, cfg in config.DATASET_CONFIGS.items():
        if key not in priority_order:
            configs_to_run.append((key, cfg))

    for key, cfg in configs_to_run:
        test_single_dataset(key, cfg)
        # Small pause to let OS reclaim memory fully
        time.sleep(2)