import os
import sys
import subprocess
import pandas as pd
from tabulate import tabulate

# Setup paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter
from src.utils import SchemaMatcher
from src.config import config

# CONFIG
DATASET_KEY = "HGB_ACM"  # Must match config key
CPP_BIN = "./bin/graph_prep"
TEMP_DIR = os.path.join(ROOT_DIR, "output", "verify_rule_temp")

def verify_pipeline_logic():
    print(f"--- 1. SETUP: Loading {DATASET_KEY} Semantics ---")
    
    # 1. Load Data to get the Schema
    cfg = config.get_dataset_config(DATASET_KEY)
    g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    
    # 2. Replicate Adapter ID Assignment (Sorted Order)
    # This is the "Truth" for ID mapping
    edge_types = sorted(g.edge_types)
    mapping = {et: i for i, et in enumerate(edge_types)}
    
    # Print ID Map for clarity
    table = [[i, f"{et[0]}->{et[1]}->{et[2]}"] for i, et in enumerate(edge_types)]
    print(tabulate(table, headers=["ID", "Edge Type"], tablefmt="grid"))

    # 3. Define a Multi-Hop Test Case
    # ACM Specific: Paper -> Author -> Paper
    # We use the raw strings AnyBURL would output
    test_chain_str = ["paper_to_author", "author_to_paper"]
    
    print(f"\n--- 2. PYTHON GENERATION ---")
    print(f"Goal Metapath: {test_chain_str}")
    
    expected_ids = []
    
    for rel_str in test_chain_str:
        # Use our FIXED SchemaMatcher
        match = SchemaMatcher.match(rel_str, g)
        if match not in mapping:
            print(f"❌ FAIL: Matcher returned {match} which is not in the graph schema!")
            return
        
        eid = mapping[match]
        expected_ids.append(eid)
        print(f"  Mapped '{rel_str}' -> {match} -> ID {eid}")
        
    print(f"EXPECTED SEQUENCE: {expected_ids}")
    
    # 4. Write Rule File
    adapter = PyGToCppAdapter(TEMP_DIR)
    
    # Construct rule string: -1 -2 ID1 -2 ID2 -4 -4


    # New (Correct)
    rule_parts = []

    # 1. Add all edges EXCEPT the last one (The Body)
    for eid in expected_ids[:-1]:
        rule_parts.append("-2") 
        rule_parts.append(str(eid))

    # 2. Add the LAST edge with the Trigger (The Head)
    rule_parts.append("-2") 
    rule_parts.append("-1")  # <--- Moves here
    rule_parts.append(str(expected_ids[-1]))

    # 3. Add Pops
    for _ in expected_ids:
        rule_parts.append("-4")
        
    rule_str = " ".join(rule_parts)
    print(f"Generated Rule String: '{rule_str}'")
    
    rule_path = adapter.write_rule_file(rule_str, "test.rule")
    
    print("\n--- 3. C++ EXECUTION ---")
    cmd = [CPP_BIN, "debug_rule", rule_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("C++ Output:")
        print(result.stdout.strip())
        
        # Parse C++ output
        # Look for line: DEBUG_PARSED_SEQUENCE: 3 0
        actual_ids = []
        for line in result.stdout.split('\n'):
            if line.startswith("DEBUG_PARSED_SEQUENCE:"):
                parts = line.split(':')[1].strip().split()
                actual_ids = [int(x) for x in parts]
                break
                
        print(f"\n--- 4. VERDICT ---")
        print(f"Python Expected: {expected_ids}")
        print(f"C++ Actual:      {actual_ids}")
        
        if expected_ids == actual_ids:
            print("\n✅✅✅ PARSING IS CORRECT ✅✅✅")
            print("Python and C++ are 100% aligned on graph traversal logic.")
        else:
            print("\n❌❌❌ MISMATCH DETECTED ❌❌❌")
            if len(actual_ids) < len(expected_ids):
                print("Diagnosis: C++ is truncating the rule (Parser Bug).")
            else:
                print("Diagnosis: C++ is reading garbage values.")
                
    except subprocess.CalledProcessError as e:
        print(f"C++ Crashed: {e}")
        print(e.stderr)

if __name__ == "__main__":
    verify_pipeline_logic()