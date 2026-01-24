import os
import sys
import pandas as pd
from tabulate import tabulate

# Setup paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.data import DatasetFactory
from src.utils import SchemaMatcher
from src.config import config

# CONFIG
DATASET_KEY = "HGB_ACM"  # Use the Config Key

def verify_rules():
    print(f"--- 1. LOADING PROCESSED GRAPH ({DATASET_KEY}) ---")
    
    # 1. Load Data via Factory (Applies Standardization)
    cfg = config.get_dataset_config(DATASET_KEY)
    g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    
    # 2. Replicate Adapter ID Assignment (Sorted Order)
    edge_types = sorted(g.edge_types)
    mapping = {et: i for i, et in enumerate(edge_types)}
    
    # Print Table
    table_data = [[i, et[0], et[1], et[2]] for i, et in enumerate(edge_types)]
    print(tabulate(table_data, headers=["ID", "Source", "Rel", "Dest"], tablefmt="grid"))
    
    print("\n--- 2. VERIFYING RULES ---")
    
    # Test cases (ACM specific)
    test_cases = [
        "paper_to_author", # Standard
        "rev_cite",        # Mined rule example
        "inverse_cite"     # Mined rule example
    ]
    
    for rule in test_cases:
        print(f"\nQuery: '{rule}'")
        match = SchemaMatcher.match(rule, g)
        
        if match in mapping:
            eid = mapping[match]
            print(f"  -> Match: {match}")
            print(f"  -> ID:    {eid}")
        else:
            print(f"  -> FAIL: {match} (Not in graph)")

if __name__ == "__main__":
    verify_rules()