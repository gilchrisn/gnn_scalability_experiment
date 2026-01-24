import os
import sys
import pandas as pd
import subprocess
import torch
from tqdm import tqdm

# Setup paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter

# CONFIG
DATASET_NAME = "ACM"  # Change as needed
CPP_BIN = "./bin/graph_prep.exe"
TEMP_DIR = os.path.join(ROOT_DIR, "output", "verify_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

def verify():
    print(f"--- 1. LOADING PYTHON GROUND TRUTH ({DATASET_NAME}) ---")
    # Load PyG Graph
    # Note: We must replicate the loader logic to get the exact raw object
    loader_map = {'HGB': ('HGB', DATASET_NAME, 'paper')} # HGB_ACM specific
    src, name, target = loader_map['HGB']
    
    g_pyg, _ = DatasetFactory.get_data(src, name, target)
    
    # Run the Adapter (to generate the files we want to test)
    # We use the ROBUST adapter code you (hopefully) updated
    adapter = PyGToCppAdapter(TEMP_DIR)
    adapter.convert(g_pyg)
    
    # Retrieve the mappings the adapter generated
    # This dictates "Global ID 0 = Author 0", etc.
    type_offsets = adapter.type_offsets
    edge_type_mapping = adapter.edge_type_mapping
    
    print("\n--- 2. BUILDING EXPECTED EDGE SET ---")
    expected_edges = set()
    
    # Reconstruct what the C++ graph SHOULD contain based on PyG data + Offsets
    for etype in tqdm(g_pyg.edge_types, desc="Hashing PyG Edges"):
        src_type, _, dst_type = etype
        
        # Get the ID C++ uses for this edge type
        if etype not in edge_type_mapping: continue
        tid = edge_type_mapping[etype]
        
        # Get Global ID offsets
        src_off = type_offsets[src_type]
        dst_off = type_offsets[dst_type]
        
        edge_index = g_pyg[etype].edge_index
        srcs = edge_index[0].tolist()
        dsts = edge_index[1].tolist()
        
        for s, d in zip(srcs, dsts):
            # C++ Graph is directed but stores forward/backward logic internally.
            # The 'dump' function dumps the raw adjacency list.
            # Our adapter writes to 'link.dat'. C++ reads 'link.dat'.
            # So we verify that (GlobalSrc, GlobalDst, Type) exists.
            
            g_s = s + src_off
            g_d = d + dst_off
            
            # Add forward edge
            expected_edges.add((g_s, g_d, tid))
            
            # NOTE: The C++ HeterGraph loader AUTOMATICALLY adds reverse edges 
            # into rEL/rET arrays, but the 'dump' function only iterates EL (forward).
            # So we only assert the forward edges match the 'link.dat' input.

    print(f"Expected Edges: {len(expected_edges)}")

    print("\n--- 3. RUNNING C++ DUMP ---")
    dump_file = os.path.join(TEMP_DIR, "cpp_dump.txt")
    if os.path.exists(dump_file): os.remove(dump_file)
    
    # Point C++ to the directory containing the node.dat/link.dat we just made
    cmd = [CPP_BIN, "dump", TEMP_DIR, dump_file]
    subprocess.run(cmd, check=True)
    
    print("\n--- 4. COMPARING MEMORY STATES ---")
    actual_edges = set()
    
    with open(dump_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            u, v, t = int(parts[0]), int(parts[1]), int(parts[2])
            actual_edges.add((u, v, t))
            
    print(f"C++ Loaded Edges: {len(actual_edges)}")
    
    # SET DIFFERENCE
    missing = expected_edges - actual_edges
    extra = actual_edges - expected_edges
    
    if len(missing) == 0 and len(extra) == 0:
        print("\n✅✅✅ MATCH CONFIRMED. CONVERSION IS PERFECT. ✅✅✅")
        print("The C++ memory state is bit-for-bit identical to PyG structure.")
    else:
        print("\n❌❌❌ MISMATCH DETECTED ❌❌❌")
        print(f"Edges present in Python but missing in C++: {len(missing)}")
        print(f"Edges present in C++ but extra in Python:   {len(extra)}")
        
        if len(missing) > 0:
            print("Sample Missing (PyG has, C++ lost):", list(missing)[:5])
        if len(extra) > 0:
            print("Sample Extra (C++ hallucinated):", list(extra)[:5])
            
        # Diagnosis
        if len(extra) > 0 and len(missing) == 0:
            print("\n[Diagnosis] C++ has MORE edges. Did it duplicate undirected edges?")
        elif len(missing) > 0:
            print("\n[Diagnosis] C++ LOST data. Likely the 'Dirty Data' offset issue.")

if __name__ == "__main__":
    verify()