"""
AnyBURL wrapper for rule mining and metapath extraction.
Handles graph serialization, Java execution, and rule-to-path parsing.
"""
import os
import subprocess
from typing import Optional, List
from torch_geometric.data import HeteroData

class AnyBURLRunner:
    """
    Interoperability layer for AnyBURL.
    Manages data export, subprocess lifecycle, and rule parsing.
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
        """Serializes HeteroData edge indices into a tab-separated triple format."""
        if os.path.exists(self.triples_file) and os.path.getsize(self.triples_file) > 0:
             print(f"[AnyBURL] Using existing triples.")
             return

        print(f"[AnyBURL] Exporting graph to {self.triples_file}...")
        with open(self.triples_file, 'w') as f:
            for edge_type in g_hetero.edge_types:
                src_type, rel, dst_type = edge_type
                edges = g_hetero[edge_type].edge_index
                srcs = edges[0].tolist()
                dsts = edges[1].tolist()
                for u, v in zip(srcs, dsts):
                    # Canonical formatting: nodeType_nodeID
                    f.write(f"{src_type}_{u}\t{rel}\t{dst_type}_{v}\n")
        print(f"[AnyBURL] Export complete.")
    
    def run_mining(self, timeout: int = 60, max_length: int = 4, num_threads: int = 8) -> None:
        """Generates AnyBURL configuration and executes the Java learning process."""
        if os.path.exists(self.rules_file) and os.path.getsize(self.rules_file) > 0:
            print(f"[AnyBURL] Rules exist; skipping mining.")
            return

        # Prepare path strings for property file (standardizing separators)
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
        
        # Execute learner via subprocess
        print(f"[AnyBURL] Learning rules (timeout={timeout}s)...")
        cmd = ["java", "-Xmx12G", "-cp", self.jar_path, "de.unima.ki.anyburl.Learn", self.config_file]
        try:
            subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout + 300)
        except subprocess.TimeoutExpired:
            print("[AnyBURL] Subprocess reached secondary timeout.")

        # Post-process: rename snapshot to canonical output path
        expected_output = f"{self.rules_file}-{timeout}"
        if os.path.exists(expected_output):
            if os.path.exists(self.rules_file): os.remove(self.rules_file)
            os.rename(expected_output, self.rules_file)

    def _smart_parse_rule(self, rule_body: str, start_var: str) -> List[str]:
        """
        Traces variable bindings across atoms to reconstruct directed metapaths.
        Inverts relationship names when traversing from Tail to Head.
        """
        path = []
        curr_var = start_var
        
        # Tokenize body atoms
        atoms_raw = rule_body.split("), ")
        atoms = []
        for a in atoms_raw:
            clean = a.replace(")", "").strip()
            if "(" in clean:
                rel = clean.split("(")[0]
                args = clean.split("(")[1].split(",")
                atoms.append((rel, args[0].strip(), args[1].strip()))
        
        # Chain reconstruction logic
        pool = list(atoms)
        while pool:
            found = False
            for i, (rel, arg1, arg2) in enumerate(pool):
                
                # Standard traversal
                if arg1 == curr_var:
                    path.append(rel)
                    curr_var = arg2
                    pool.pop(i)
                    found = True
                    break
                
                # Inverse traversal: map to canonical reverse relation
                elif arg2 == curr_var:
                    if "_to_" in rel:
                        parts = rel.split("_to_")
                        inv_rel = f"{parts[1]}_to_{parts[0]}"
                        path.append(inv_rel)
                    else:
                        path.append(f"inverse_{rel}")
                    
                    curr_var = arg1
                    pool.pop(i)
                    found = True
                    break
            
            if not found: break 
            
        return path

    def save_clean_list(self) -> str:
        """Parses learned rules into unique, comma-separated metapaths."""
        print(f"[AnyBURL] Extracting paths from rules...")
        if not os.path.exists(self.rules_file): return ""

        unique_paths = set()
        
        with open(self.rules_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or "<=" not in line: continue
                try:
                    # Extract starting anchor variable from head
                    head_part = line.split(" <= ")[0]
                    start_var = head_part.split("(")[1].split(")")[0].split(",")[0].strip()
                    
                    # Parse body chain
                    body_part = line.split(" <= ")[1]
                    relations = self._smart_parse_rule(body_part, start_var)
                    
                    if len(relations) >= 2:
                        path_str = ",".join(relations)
                        unique_paths.add(path_str)
                except:
                    continue
        
        with open(self.clean_list_file, 'w') as f:
            for p in sorted(list(unique_paths)):
                f.write(p + "\n")
        
        print(f"    Exported {len(unique_paths)} paths to {self.clean_list_file}")
        return self.clean_list_file

    def parse_best_metapath(self, target_head, target_tail, min_confidence=0.001) -> Optional[List[str]]:
        """Parses output file to identify path with highest confidence score."""
        print(f"[AnyBURL] Parsing scores and generating ruleset...")
        if not os.path.exists(self.rules_file): return None
        
        best_path = None
        best_conf = -1.0

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
                        
                        # Validate grounding (filter out specific entity rules)
                        head_part = rule_str.split(" <= ")[0]
                        head_args = head_part.split("(")[1].split(")")[0].split(",")
                        is_grounded = False
                        for arg in head_args:
                            if not arg.strip()[0].isupper(): is_grounded = True
                        if is_grounded: continue
                        
                        relations = [a.split("(")[0].strip() for a in atoms]
                        if conf > best_conf:
                            best_path = relations
                            best_conf = conf
                    except: continue

        return best_path

    def export_to_config_format(self, metapath: List[str], output_file: str = "mined_metapath.txt") -> str:
        """Utility for saving specific metapath selections in a standard key-value format."""
        output_path = os.path.join(self.data_dir, output_file)
        with open(output_path, 'w') as f:
            f.write("# Mined metapath\n")
            f.write(f"METAPATH = {' -> '.join(metapath)}\n")
        return output_path