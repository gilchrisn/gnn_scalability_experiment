import sys
import os
import time
import torch
import numpy as np
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.data import DatasetFactory
from src.backend import BackendFactory
from src.utils import SchemaMatcher

def test_ensemble(backend_type, k_val, l_val):
    print(f"\n{'='*60}")
    print(f"ENSEMBLE TEST: {backend_type.upper()} BACKEND")
    print(f"Dataset: HGB_ACM | K={k_val} | L={l_val} (Samples)")
    print(f"{'='*60}\n")

    # 1. Load HGB_ACM
    dataset_name = "HGB_ACM"
    target_node = "paper"
    metapath_str = "paper_to_author,author_to_paper" # PAP path

    print(f"[Data] Loading {dataset_name}...")
    cfg = config.get_dataset_config(dataset_name)
    g_hetero, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    
    path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in metapath_str.split(',')]

    # 2. Initialize Backend with L parameter
    print(f"[{backend_type}] Initializing engine with num_sketches={l_val}...")
    
    backend = BackendFactory.create(
        backend_type, 
        executable_path=config.CPP_EXECUTABLE, 
        temp_dir=config.TEMP_DIR,
        device=config.DEVICE,
        num_sketches=l_val  # <--- Configuring L here
    )
    
    backend.initialize(g_hetero, path_list, info)

    # 3. Run Ensemble Materialization
    print(f"[{backend_type}] Running materialize_kmv_ensemble...")
    start_t = time.perf_counter()
    
    # This should return a List[Data] of length L
    graphs = backend.materialize_kmv_ensemble(k=k_val)
    
    total_time = time.perf_counter() - start_t
    
    # 4. Validation
    print(f"\n[Validation]")
    print(f"   -> Time Taken: {total_time:.4f}s")
    print(f"   -> Return Type: {type(graphs)}")
    print(f"   -> List Length: {len(graphs)} (Expected: {l_val})")

    if len(graphs) != l_val:
        print(f"❌ FAIL: Expected {l_val} graphs, got {len(graphs)}")
        return

    # 5. Variance Analysis (Proof of Independence)
    edge_counts = []
    print(f"\n[Sample Statistics]")
    for i, g in enumerate(graphs):
        print(f"   Sample {i}: {g.num_edges} edges | {g.num_nodes} nodes")
        edge_counts.append(g.num_edges)

    edge_counts = np.array(edge_counts)
    mean_edges = np.mean(edge_counts)
    std_edges = np.std(edge_counts)

    print(f"\n[Independence Check]")
    print(f"   Mean Edges: {mean_edges:.2f}")
    print(f"   Std Dev:    {std_edges:.2f}")
    
    if l_val > 1 and std_edges == 0:
        print("⚠️  WARNING: Standard Deviation is 0. Samples might be identical!")
    elif l_val > 1:
        print("✅  SUCCESS: Variance detected. Samples are distinct.")
    
    backend.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='cpp', choices=['cpp', 'python'])
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--l', type=int, default=3) # Default L=3 to prove list functionality
    args = parser.parse_args()

    test_ensemble(args.backend, args.k, args.l)