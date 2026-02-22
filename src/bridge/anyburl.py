"""
AnyBURL implementation of the RuleMiner interface.
Handles graph serialization to triples, Java execution, and rule parsing.
"""
import re
import os
import subprocess
from typing import Optional, List, Tuple, Dict, Any
from torch_geometric.data import HeteroData
from .base import RuleMiner

class AnyBURLRunner(RuleMiner):
    """
    Interoperability layer for AnyBURL.
    """
    
    def __init__(self, data_dir: str, jar_path: str):
        self.data_dir = data_dir
        self.jar_path = jar_path
        
        os.makedirs(data_dir, exist_ok=True)
        
        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"AnyBURL JAR not found: {jar_path}")
        
        self.triples_file = os.path.join(data_dir, "anyburl_triples.txt")
        self.rules_file = os.path.join(data_dir, "anyburl_rules.txt")
        self.config_file = os.path.join(data_dir, "config-learn.properties")

    def export_for_mining(self, g_hetero: HeteroData) -> None:
        """Serializes HeteroData edge indices into AnyBURL triple format."""
        if os.path.exists(self.triples_file) and os.path.getsize(self.triples_file) > 0:
             print(f"[AnyBURL] Using existing triples at {self.triples_file}")
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

    def run_mining(self, timeout: int, max_length: int = 4, num_threads: int = 1, seed: int = 42) -> None:
        """
        Generates configuration and executes the Java learning process.
        """
        if os.path.exists(self.rules_file) and os.path.getsize(self.rules_file) > 0:
            print(f"[AnyBURL] Rules exist; skipping mining.")
            return

        # Prepare config
        safe_triples = self.triples_file.replace(os.sep, '/')
        safe_rules = self.rules_file.replace(os.sep, '/')
        
        config_content = f"""PATH_TRAINING = {safe_triples}
            PATH_OUTPUT = {safe_rules}
            SNAPSHOTS_AT = {timeout}
            WORKER_THREADS = {num_threads}
            MAX_LENGTH_CYCLIC = {max_length}
            ZERO_RULES_ACTIVE = false
            THRESHOLD_CORRECT_PREDICTIONS = 2
            THRESHOLD_CONFIDENCE = 0.0001
            RANDOM_SEED = {seed}
        """
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        # Execute
        print(f"[AnyBURL] Learning rules (timeout={timeout}s)...")
        cmd = ["java", "-Xmx12G", "-cp", self.jar_path, "de.unima.ki.anyburl.Learn", self.config_file]
        try:
            subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout + 300)
        except subprocess.TimeoutExpired:
            print("[AnyBURL] Subprocess reached secondary timeout (Expected behavior for snapshots).")

        # Rename snapshot to canonical output path
        expected_output = f"{self.rules_file}-{timeout}"
        if os.path.exists(expected_output):
            if os.path.exists(self.rules_file): os.remove(self.rules_file)
            os.rename(expected_output, self.rules_file)
        else:
            print(f"[AnyBURL] Warning: Expected snapshot {expected_output} not found.")

    def get_top_k_paths(self, k: int = 5, min_conf: float = 0.1) -> List[Tuple[float, str]]:
        """Parses output file to identify top-K generic paths."""
        if not os.path.exists(self.rules_file): return []
        
        valid_rules = []
        with open(self.rules_file, 'r') as f:
            for line in f:
                try:
                    res = self._parse_single_line(line, min_conf)
                    if res: valid_rules.append(res)
                except Exception: continue

        # Deduplicate and Sort
        valid_rules.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        unique = []
        for conf, path in valid_rules:
            if path not in seen:
                unique.append((conf, path))
                seen.add(path)
                if len(unique) >= k: break
        return unique

    def _parse_single_line(self, line: str, min_conf: float) -> Optional[Tuple[float, str]]:
        """
        Parses a single AnyBURL rule line, resolving FOL variable bindings 
        into a strict directed sequential metapath.
        """
        line = line.strip()
        parts = line.split("\t")
        if len(parts) < 4: return None
        
        try:
            conf = float(parts[2])
        except ValueError:
            return None
            
        if conf < min_conf: return None
        
        rule_str = parts[3]
        
        # Filter grounded rules (those with specific node IDs)
        if re.search(r'[a-z]+_\d+', rule_str): return None 
        
        try:
            # Isolate the body: target(X,Y) <= rel1(X,A), rel2(Y,A) -> rel1(X,A), rel2(Y,A)
            body = rule_str.split(" <= ")[1]
        except IndexError:
            return None
            
        atoms = body.split(", ")
        relations = []
        current_var = None
        
        for i, atom in enumerate(atoms):
            # Parse atom: relation_name(var1, var2)
            # Lifted rules use uppercase letters for variables (X, Y, A, B, etc.)
            match = re.match(r'([a-zA-Z0-9_]+)\(([A-Z]),([A-Z])\)', atom)
            if not match: 
                return None
            
            rel, v1, v2 = match.groups()
            
            # Initialize the walk at the source of the first atom (usually 'X')
            if i == 0:
                current_var = v1
                
            # Determine traversal direction based on variable tracking
            if current_var == v1:
                # Forward traversal: current_var -> v2
                relations.append(rel)
                current_var = v2
            elif current_var == v2:
                # Backward traversal: current_var -> v1
                relations.append(f"rev_{rel}")
                current_var = v1
            else:
                # The rule is disjointed or branches in a way that 
                # cannot be represented as a single sequential walk.
                return None
                
        if len(relations) >= 2:
            return (conf, ",".join(relations))
        return None