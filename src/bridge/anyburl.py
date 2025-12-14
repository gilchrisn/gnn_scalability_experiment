"""
AnyBURL interoperability bridge for rule mining.
Manages Java subprocess execution and rule parsing.
"""
import os
import subprocess
from typing import Optional, List
from torch_geometric.data import HeteroData

class AnyBURLRunner:
    """
    Bridge for executing AnyBURL rule mining tool.
    Handles graph export, Java execution, and rule parsing.
    """
    
    def __init__(self, data_dir: str, jar_path: str):
        self.data_dir = data_dir
        self.jar_path = jar_path
        
        os.makedirs(data_dir, exist_ok=True)
        
        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"AnyBURL JAR not found: {jar_path}")
        
        self.triples_file = os.path.join(data_dir, "anyburl_triples.txt")
        self.rules_file = os.path.join(data_dir, "anyburl_rules.txt")
        self.clean_list_file = os.path.join(data_dir, "metapaths_clean.txt")
        self.gnn_rules_file = os.path.join(data_dir, "gnn-rules") 
        self.config_file = os.path.join(data_dir, "config-learn.properties")
    
    def export_graph(self, g_hetero: HeteroData) -> None:
        """Exports PyG HeteroData to AnyBURL triple format."""
        if os.path.exists(self.triples_file) and os.path.getsize(self.triples_file) > 0:
             print(f"[AnyBURL] Found existing triples. Skipping export.")
             return

        print(f"[AnyBURL] Exporting graph to {self.triples_file}...")
        with open(self.triples_file, 'w') as f:
            for edge_type in g_hetero.edge_types:
                src_type, rel, dst_type = edge_type
                edges = g_hetero[edge_type].edge_index
                srcs = edges[0].tolist()
                dsts = edges[1].tolist()
                for u, v in zip(srcs, dsts):
                    # Format: nodeType_ID
                    f.write(f"{src_type}_{u}\t{rel}\t{dst_type}_{v}\n")
        print(f"[AnyBURL] Export complete.")
    
    def run_mining(self, timeout: int = 60, max_length: int = 4, num_threads: int = 8) -> None:
        """Executes AnyBURL rule mining."""
        if os.path.exists(self.rules_file) and os.path.getsize(self.rules_file) > 0:
            print(f"[AnyBURL] Found existing rules. Skipping mining.")
            return

        # Write Config
        safe_triples = self.triples_file.replace(os.sep, '/')
        safe_rules = self.rules_file.replace(os.sep, '/')
        
        config = f"""PATH_TRAINING = {safe_triples}
PATH_OUTPUT = {safe_rules}
SNAPSHOTS_AT = {timeout}
WORKER_THREADS = {num_threads}
MAX_LENGTH_CYCLIC = {max_length}
ZERO_RULES_ACTIVE = false
THRESHOLD_CORRECT_PREDICTIONS = 2
THRESHOLD_CONFIDENCE = 0.0001
"""
        with open(self.config_file, 'w') as f:
            f.write(config)
        
        # Run Java
        print(f"[AnyBURL] Running mining for {timeout} seconds...")
        cmd = ["java", "-Xmx12G", "-cp", self.jar_path, "de.unima.ki.anyburl.Learn", self.config_file]
        try:
            subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout + 300)
        except subprocess.TimeoutExpired:
            print("[AnyBURL] Python timeout reached.")

        # Rename Output
        expected_output = f"{self.rules_file}-{timeout}"
        if os.path.exists(expected_output):
            if os.path.exists(self.rules_file): os.remove(self.rules_file)
            os.rename(expected_output, self.rules_file)

    def _smart_parse_rule(self, rule_body: str, start_var: str) -> List[str]:
        """
        Intelligently traces variables to detect reverse edges.
        If walking B->A using 'rel(A,B)', it generates 'inverse_rel' or 'conf_to_paper'.
        """
        path = []
        curr_var = start_var
        
        # 1. Parse atoms: "rel(A,B)" -> ("rel", "A", "B")
        atoms_raw = rule_body.split("), ")
        atoms = []
        for a in atoms_raw:
            clean = a.replace(")", "").strip()
            if "(" in clean:
                rel = clean.split("(")[0]
                args = clean.split("(")[1].split(",")
                atoms.append((rel, args[0].strip(), args[1].strip()))
        
        # 2. Trace variables to determine direction
        pool = list(atoms)
        while pool:
            found = False
            for i, (rel, arg1, arg2) in enumerate(pool):
                
                # Case 1: Forward (Current -> Arg2)
                # Assumes AnyBURL format is rel(Head, Tail)
                if arg1 == curr_var:
                    path.append(rel)
                    curr_var = arg2
                    pool.pop(i)
                    found = True
                    break
                
                # Case 2: Reverse (Current -> Arg1)
                elif arg2 == curr_var:
                    # Logic: If we are traversing backwards, we need the reverse relation name.
                    # Standard convention: src_to_dst -> dst_to_src
                    if "_to_" in rel:
                        parts = rel.split("_to_")
                        # e.g. "paper_to_conf" becomes "conf_to_paper"
                        inv_rel = f"{parts[1]}_to_{parts[0]}"
                        path.append(inv_rel)
                    else:
                        # Fallback for other naming conventions
                        path.append(f"inverse_{rel}")
                    
                    curr_var = arg1
                    pool.pop(i)
                    found = True
                    break
            
            if not found: break # Broken chain
            
        return path

    def save_clean_list(self) -> str:
        """Parses rules using Smart Logic (Fixes the -1.0s crash)."""
        print(f"[AnyBURL] Generating clean metapath list (Smart Parse)...")
        if not os.path.exists(self.rules_file): return ""

        unique_paths = set()
        
        with open(self.rules_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or "<=" not in line: continue
                try:
                    # Extract Start Variable from Head: target(X,Y) -> X
                    head_part = line.split(" <= ")[0]
                    start_var = head_part.split("(")[1].split(")")[0].split(",")[0].strip()
                    
                    # Extract Body
                    body_part = line.split(" <= ")[1]
                    
                    # Parse using Smart Logic
                    relations = self._smart_parse_rule(body_part, start_var)
                    
                    if len(relations) >= 2:
                        path_str = ",".join(relations)
                        unique_paths.add(path_str)
                except:
                    continue
        
        with open(self.clean_list_file, 'w') as f:
            for p in sorted(list(unique_paths)):
                f.write(p + "\n")
        
        print(f"   ✅ Saved {len(unique_paths)} unique paths to {self.clean_list_file}")
        return self.clean_list_file

    def parse_best_metapath(self, target_head, target_tail, min_confidence=0.001) -> Optional[List[str]]:
        """
        Generates 'gnn-rules' file (Friend's requirement) AND finds best path.
        """
        print(f"[AnyBURL] Parsing rules & generating gnn-rules...")
        if not os.path.exists(self.rules_file): return None
        
        best_path = None
        best_conf = -1.0

        # Generate legacy gnn-rules file
        with open(self.gnn_rules_file, 'w') as wf:
            with open(self.rules_file, 'r') as f:
                for line in f:
                    try:
                        line = line.strip()
                        if not line or "<=" not in line: continue
                        
                        parts = line.split("\t")
                        if len(parts) < 4: continue
                        conf = float(parts[2])
                        rule_str = parts[3]
                        
                        body_part = rule_str.split(" <= ")[1]
                        atoms = body_part.split("), ")
                        
                        if len(atoms) >= 2:
                            wf.write(body_part + '\n')
                        
                        # 2. Find best path (for our display)
                        # Check grounding (skip specific names)
                        head_part = rule_str.split(" <= ")[0]
                        head_args = head_part.split("(")[1].split(")")[0].split(",")
                        is_grounded = False
                        for arg in head_args:
                            if not arg.strip()[0].isupper(): is_grounded = True
                        if is_grounded: continue
                        
                        # The benchmark relies on 'metapaths_clean.txt' generated by save_clean_list above.
                        relations = [a.split("(")[0].strip() for a in atoms]
                        if conf > best_conf:
                            best_path = relations
                            best_conf = conf
                    except: continue

        return best_path

    def export_to_config_format(self, metapath: List[str], output_file: str = "mined_metapath.txt") -> str:
        output_path = os.path.join(self.data_dir, output_file)
        with open(output_path, 'w') as f:
            f.write("# Mined metapath\n")
            f.write(f"METAPATH = {' -> '.join(metapath)}\n")
        return output_path