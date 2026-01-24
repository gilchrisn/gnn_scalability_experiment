import os
import sys
import pandas as pd
import subprocess
import time
from typing import List, Tuple

# Path setup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.config import config
from src.data import DatasetFactory
from src.bridge import PyGToCppAdapter, AnyBURLRunner
from src.utils import SchemaMatcher

# --- CONFIGURATION ---
CPP_BIN = os.path.join(ROOT_DIR, "bin", "graph_prep.exe")
OUT_CSV = os.path.join(ROOT_DIR, "output", "FINAL_EXPERIMENT_RESULTS.csv")
TARGET_DATASETS = ["HGB_ACM", "HGB_DBLP"]

def parse_and_filter_rules(dataset_key: str, rule_file: str, g_hetero) -> List[Tuple[str, str]]:
    """
    Reads RAW AnyBURL output and converts to Validated C++ Rule Strings.
    Returns: List of (Readable_Path, Cpp_Rule_String)
    """
    valid_benchmarks = []
    seen_paths = set()
    
    # Need adapter to map Schema -> IDs
    # We create a dummy one just to access the ID mapping logic if we had it, 
    # but here we need to rebuild the ID map fresh to ensure consistency with the graph loaded NOW.
    
    # Setup ID Mapping (Standardized)
    edge_types = sorted(g_hetero.edge_types)
    edge_map = {et: i for i, et in enumerate(edge_types)}
    
    with open(rule_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4: continue
            
            rule_str = parts[3]
            # Filter 1: Must be generic (X, Y)
            if "person_" in rule_str or "paper_" in rule_str: continue 
            
            # Parse Body: target(X,Y) <= rel1(X,A), rel2(A,Y)
            try:
                body = rule_str.split(" <= ")[1]
                atoms = body.split(", ")
                
                # Extract relations
                rels = []
                for atom in atoms:
                    rel_name = atom.split("(")[0]
                    rels.append(rel_name)
                
                # Convert to Schema IDs
                path_ids = []
                
                # Validation Loop
                for r in rels:
                    # Use the robust SchemaMatcher
                    matched_tuple = SchemaMatcher.match(r, g_hetero)
                    if matched_tuple not in edge_map:
                        # Try reverse
                        rev = (matched_tuple[2], matched_tuple[1], matched_tuple[0])
                        if rev in edge_map:
                            path_ids.append(edge_map[rev])
                        else:
                            raise ValueError(f"Relation {r} not in graph")
                    else:
                        path_ids.append(edge_map[matched_tuple])
                
                # Construct Readable Name
                readable = ",".join(rels)
                if readable in seen_paths: continue
                
                # Construct C++ String (-2 ID ... -2 -1 ID ...)
                cpp_parts = []
                for i, eid in enumerate(path_ids):
                    cpp_parts.append("-2")
                    if i == len(path_ids) - 1: cpp_parts.append("-1") # Trigger
                    cpp_parts.append(str(eid))
                for _ in path_ids: cpp_parts.append("-4")
                
                rule_string = " ".join(cpp_parts)
                
                valid_benchmarks.append((readable, rule_string))
                seen_paths.add(readable)
                
                if len(valid_benchmarks) >= 5: break # Top 5 per dataset
                
            except Exception as e:
                continue
                
    return valid_benchmarks

def run_experiment():
    results = []
    
    for ds in TARGET_DATASETS:
        print(f"\n>>> EXPERIMENT: {ds} <<<")
        
        # 1. Locate Mining Artifacts
        mining_dir = os.path.join(config.DATA_DIR, f"mining_{ds}")
        rule_file = os.path.join(mining_dir, "anyburl_rules.txt")
        
        if not os.path.exists(rule_file):
            print(f"Skipping {ds}: No mining output found (Run Step 1 first).")
            continue
            
        # 2. Load Graph & Convert for C++
        print("Loading Graph...")
        g, _ = DatasetFactory.get_data(config.get_dataset_config(ds).source, 
                                     config.get_dataset_config(ds).dataset_name, 
                                     config.get_dataset_config(ds).target_node)
        
        # Prepare C++ Data Directory
        cpp_data_dir = os.path.join(ROOT_DIR, ds) # e.g. HGB_ACM/
        adapter = PyGToCppAdapter(cpp_data_dir)
        adapter.convert(g)
        
        # 3. Filter Rules
        print("Filtering Rules...")
        benchmarks = parse_and_filter_rules(ds, rule_file, g)
        print(f"Selected {len(benchmarks)} high-quality metapaths.")
        
        # 4. Run Benchmarks
        res_dir = os.path.join(ROOT_DIR, "global_res", ds)
        os.makedirs(res_dir, exist_ok=True)
        
        for path_name, rule_str in benchmarks:
            print(f"  Running: {path_name}")
            
            # Write rule file
            rule_path = os.path.join(cpp_data_dir, "temp.rule")
            with open(rule_path, 'w') as f: f.write(rule_str)
            
            # Run Exact
            t0 = time.perf_counter()
            subprocess.run([CPP_BIN, "materialize", cpp_data_dir, rule_path, os.path.join(res_dir, "exact.txt")], 
                         check=True, timeout=600)
            t_exact = time.perf_counter() - t0
            
            # Run Sketch
            t0 = time.perf_counter()
            subprocess.run([CPP_BIN, "sketch", cpp_data_dir, rule_path, os.path.join(res_dir, "sketch.txt"), "32", "1"], 
                         check=True, timeout=600)
            t_sketch = time.perf_counter() - t0
            
            results.append({
                "Dataset": ds,
                "Metapath": path_name,
                "ExactTime": t_exact,
                "SketchTime": t_sketch,
                "Speedup": t_exact / t_sketch if t_sketch > 0 else 0
            })
            
    # Save Final Report
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[Done] Final results saved to {OUT_CSV}")
    print(df.to_markdown())

if __name__ == "__main__":
    run_experiment()