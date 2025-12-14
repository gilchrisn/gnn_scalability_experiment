import os
import subprocess
import torch

class AnyBURLRunner:
    def __init__(self, data_dir, jar_path):
        self.data_dir = data_dir
        self.jar_path = jar_path
        self.triples_file = os.path.join(data_dir, "anyburl_triples.txt")
        self.rules_file = os.path.join(data_dir, "anyburl_rules.txt")
        self.config_file = os.path.join(data_dir, "config-learn.properties")

    def export_graph(self, g_hetero):
        """Converts PyG Graph -> AnyBURL Triples (Head \t Rel \t Tail)"""
        print(f"[AnyBURL] Exporting graph to {self.triples_file}...")
        with open(self.triples_file, 'w') as f:
            for edge_type in g_hetero.edge_types:
                src_t, rel, dst_t = edge_type
                
                # Retrieve edge index
                edges = g_hetero[edge_type].edge_index
                srcs = edges[0].tolist()
                dsts = edges[1].tolist()
                
                for u, v in zip(srcs, dsts):
                    # Create unique string IDs: "author_0", "paper_5"
                    head = f"{src_t}_{u}"
                    tail = f"{dst_t}_{v}"
                    # Write triple
                    f.write(f"{head}\t{rel}\t{tail}\n")

    def run_mining(self, timeout=30):
        """Runs the Java AnyBURL process"""
        # Create Config File
        config_content = f"""
PATH_TRAINING = {self.triples_file}
PATH_OUTPUT   = {self.rules_file}
SNAPSHOTS_AT  = {timeout}
WORKER_THREADS = 4
MAX_LENGTH_CYCLIC = 4
"""
        with open(self.config_file, 'w') as f:
            f.write(config_content)

        print(f"[AnyBURL] Running mining for {timeout} seconds...")
        cmd = ["java", "-Xmx4G", "-jar", self.jar_path, self.config_file]
        
        try:
            subprocess.run(cmd, check=True)
            print("[AnyBURL] Mining complete.")
        except subprocess.CalledProcessError as e:
            print(f"[AnyBURL] Failed: {e}")

    def parse_best_metapath(self, target_head_type, target_tail_type):
        """Reads rules and finds the best Cyclic Metapath"""
        print(f"[AnyBURL] Parsing rules from {self.rules_file}...")
        
        if not os.path.exists(self.rules_file):
            print("Rules file not found!")
            return None

        best_path = None
        best_conf = -1.0

        with open(self.rules_file, 'r') as f:
            for line in f:
                # Format: Confidence \t Predicted \t Body
                parts = line.strip().split("\t")
                if len(parts) < 4: continue # Skip malformed
                
                conf = float(parts[1]) # Confidence is usually index 1 or 0 depending on version
                # Let's assume standard format: 100(supported)  0.95(conf)  head <= body
                
                # Check actual format of your output file. 
                # Standard AnyBURL output: "NumSupport \t Confidence \t Rule"
                try:
                    conf = float(parts[1])
                    rule_str = parts[2]
                except:
                    continue

                if conf > best_conf:
                    # Parse Rule: target(X,Y) <= rel1(X,A), rel2(A,B)...
                    if "<=" not in rule_str: continue
                    
                    head, body = rule_str.split(" <= ")
                    relations = []
                    
                    # Extract relations from body
                    # Body: rel1(X,A), rel2(A,B)
                    atoms = body.split(", ")
                    for atom in atoms:
                        rel = atom.split("(")[0]
                        relations.append(rel)
                    
                    # Reconstruct Metapath Tuples (simplified)
                    # You need to map relation strings back to (Src, Rel, Dst)
                    # For this snippet, we just return the list of relation names
                    best_path = relations
                    best_conf = conf

        return best_path