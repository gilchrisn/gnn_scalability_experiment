"""
Diagnostic tool for AnyBURL rule output.
Analyzes raw rule files to determine quality, validity, and schema alignment.
"""
import os
import sys
import re
from typing import List, Tuple, Dict

# --- PATH SETUP ---
if os.getcwd().endswith("scripts"):
    os.chdir("..")
ROOT_DIR = os.path.abspath(os.getcwd())
sys.path.append(ROOT_DIR)

from src.config import config
from src.data import DatasetFactory
from src.utils import SchemaMatcher

DATASETS = ["HGB_ACM", "HGB_DBLP", "HGB_IMDB", "HGB_Freebase"]

def parse_atoms(rule_body: str) -> List[str]:
    """Extracts relations from a rule body string."""
    # Example: "rel1(X,A), rel2(A,Y)" -> ["rel1", "rel2"]
    atoms = rule_body.split(", ")
    relations = []
    for atom in atoms:
        # Extract name before '('
        rel = atom.split("(")[0].strip()
        relations.append(rel)
    return relations

def diagnose(dataset_key: str):
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {dataset_key}")
    print(f"{'='*60}")

    # 1. Locate File
    mining_dir = os.path.join(config.DATA_DIR, f"mining_{dataset_key}")
    rule_file = os.path.join(mining_dir, "anyburl_rules.txt")
    
    if not os.path.exists(rule_file):
        print(f"[FAIL] Rule file not found: {rule_file}")
        print(f"       Did you run 'step1_mining.py'?")
        return

    file_size = os.path.getsize(rule_file) / (1024 * 1024)
    print(f"[File] Size: {file_size:.2f} MB")

    # 2. Load Schema for Validation
    try:
        cfg = config.get_dataset_config(dataset_key)
        print(f"[Schema] Loading {dataset_key} graph to verify relations...")
        g, _ = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        print(f"         Graph Edge Types: {len(g.edge_types)}")
        
    except Exception as e:
        print(f"[Error] Could not load graph schema: {e}")
        return

    # 3. Analyze Rules
    total_rules = 0
    grounded_rules = 0
    acyclic_rules = 0
    cyclic_valid = 0
    cyclic_invalid = 0
    
    top_valid_rules = []
    top_invalid_rules = []

    print("[Analysis] Scanning rules...")
    
    with open(rule_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            total_rules += 1
            
            # Expected AnyBURL format:
            # 342   16  0.953   head(X,Y) <= body(X,A), ...
            parts = line.split('\t')
            if len(parts) < 4: continue
            
            try:
                conf = float(parts[2])
                rule_str = parts[3]
            except: continue

            # Check 1: Grounded vs Lifted
            # Heuristic: Grounded rules contain node type prefixes followed by numbers (e.g. author_123)
            # Lifted rules use variables (A, B, X, Y)
            if re.search(r'[a-z]+_\d+', rule_str):
                grounded_rules += 1
                continue

            # Check 2: Structure
            if " <= " not in rule_str: continue
            _, body = rule_str.split(" <= ")
            
            relations = parse_atoms(body)
            
            # Check 3: Schema Match
            is_schema_valid = True
            invalid_reason = ""
            
            # Map each relation in the path to the graph
            matched_path = []
            
            for rel in relations:
                # Use the actual SchemaMatcher to see if it resolves
                matched_tuple = SchemaMatcher.match(rel, g)
                
                # SchemaMatcher returns ('node', rel, 'node') if it fails
                if matched_tuple[0] == 'node' and matched_tuple[2] == 'node':
                    is_schema_valid = False
                    invalid_reason = f"Relation '{rel}' matches nothing in schema"
                    break
                matched_path.append(matched_tuple)

            # Check 4: Connectivity (Type Checking)
            # If A->B, then next must be B->C
            if is_schema_valid:
                for i in range(len(matched_path) - 1):
                    curr_dst = matched_path[i][2]
                    next_src = matched_path[i+1][0]
                    
                    if curr_dst != next_src:
                        is_schema_valid = False
                        invalid_reason = f"Type Mismatch: {matched_path[i]} ends in '{curr_dst}', but next edge {matched_path[i+1]} starts with '{next_src}'"
                        break

            if is_schema_valid:
                cyclic_valid += 1
                if len(top_valid_rules) < 5:
                    top_valid_rules.append((conf, rule_str, matched_path))
            else:
                cyclic_invalid += 1
                if len(top_invalid_rules) < 5:
                    top_invalid_rules.append((conf, rule_str, invalid_reason))

    # 4. Report
    print(f"\n[Report] {dataset_key}")
    print(f"  Total Lines Processed: {total_rules}")
    print(f"  Grounded Rules (Ignored): {grounded_rules}")
    print(f"  Lifted Rules (Variables): {cyclic_valid + cyclic_invalid}")
    print(f"    - Schema Valid:   {cyclic_valid}")
    print(f"    - Schema Invalid: {cyclic_invalid}")

    print("\n[Top 5 Valid Rules]")
    if top_valid_rules:
        for conf, r, path in top_valid_rules:
            print(f"  [{conf:.4f}] {r}")
            # Simplify tuple printing
            path_str = " -> ".join([f"({s},{r},{d})" for s,r,d in path])
            print(f"      Mapped: {path_str}")
    else:
        print("  (None found)")

    print("\n[Top 5 Invalid Rules - Why did they fail?]")
    if top_invalid_rules:
        for conf, r, reason in top_invalid_rules:
            print(f"  [{conf:.4f}] {r}")
            print(f"      Error: {reason}")
    else:
        print("  (None found)")

def main():
    for ds in DATASETS:
        diagnose(ds)

if __name__ == "__main__":
    main()