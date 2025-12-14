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
        """
        Args:
            data_dir: Directory for input/output files
            jar_path: Path to AnyBURL JAR file
        """
        self.data_dir = data_dir
        self.jar_path = jar_path
        
        os.makedirs(data_dir, exist_ok=True)
        
        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"AnyBURL JAR not found: {jar_path}")
        
        self.triples_file = os.path.join(data_dir, "anyburl_triples.txt")
        self.rules_file = os.path.join(data_dir, "anyburl_rules.txt")
        self.config_file = os.path.join(data_dir, "config-learn.properties")
    
    def export_graph(self, g_hetero: HeteroData) -> None:
        """
        Exports PyG HeteroData to AnyBURL triple format.
        Format: <head>\t<relation>\t<tail>
        
        Args:
            g_hetero: Input heterogeneous graph
        """
        print(f"[AnyBURL] Exporting graph to {self.triples_file}...")
        
        with open(self.triples_file, 'w') as f:
            for edge_type in g_hetero.edge_types:
                src_type, rel, dst_type = edge_type
                edges = g_hetero[edge_type].edge_index
                
                srcs = edges[0].tolist()
                dsts = edges[1].tolist()
                
                for u, v in zip(srcs, dsts):
                    head = f"{src_type}_{u}"
                    tail = f"{dst_type}_{v}"
                    f.write(f"{head}\t{rel}\t{tail}\n")
        
        print(f"[AnyBURL] Exported {sum(g_hetero[et].num_edges for et in g_hetero.edge_types)} triples")
    
    def run_mining(self, 
                   timeout: int = 60,
                   max_length: int = 4,
                   num_threads: int = 4) -> None:
        """
        Executes AnyBURL rule mining.
        
        Args:
            timeout: Mining duration in seconds
            max_length: Maximum rule length
            num_threads: Number of worker threads
        """
        # Create configuration file
        config_content = f"""PATH_TRAINING = {self.triples_file}
PATH_OUTPUT   = {self.rules_file}
SNAPSHOTS_AT  = {timeout}
WORKER_THREADS = {num_threads}
MAX_LENGTH_CYCLIC = {max_length}
"""
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        print(f"[AnyBURL] Running mining for {timeout} seconds...")
        
        cmd = ["java", "-Xmx4G", "-jar", self.jar_path, self.config_file]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout + 30  # Buffer for startup/shutdown
            )
            print("[AnyBURL] Mining complete")
        except subprocess.TimeoutExpired:
            print("[AnyBURL] Mining timed out (expected behavior)")
        except subprocess.CalledProcessError as e:
            print(f"[AnyBURL] Mining failed: {e}")
            print(f"Stderr: {e.stderr}")
    
    def parse_best_metapath(self,
                           target_head_type: str,
                           target_tail_type: str,
                           min_confidence: float = 0.5) -> Optional[List[str]]:
        """
        Parses mined rules and extracts the best cyclic metapath.
        
        Args:
            target_head_type: Expected head node type
            target_tail_type: Expected tail node type
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of relation names forming the metapath, or None if not found
        """
        print(f"[AnyBURL] Parsing rules from {self.rules_file}...")
        
        if not os.path.exists(self.rules_file):
            print("[AnyBURL] Rules file not found!")
            return None
        
        best_path = None
        best_conf = -1.0
        
        with open(self.rules_file, 'r') as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                
                try:
                    # Parse confidence (format varies by AnyBURL version)
                    conf = float(parts[1])
                    rule_str = parts[2] if len(parts) > 2 else parts[1]
                    
                    if conf < min_confidence:
                        continue
                    
                    # Parse rule: target(X,Y) <= body
                    if "<=" not in rule_str:
                        continue
                    
                    head, body = rule_str.split(" <= ")
                    
                    # Extract relations from body
                    relations = []
                    atoms = body.split(", ")
                    for atom in atoms:
                        if "(" in atom:
                            rel = atom.split("(")[0].strip()
                            relations.append(rel)
                    
                    if conf > best_conf and relations:
                        best_path = relations
                        best_conf = conf
                
                except (ValueError, IndexError) as e:
                    continue
        
        if best_path:
            print(f"[AnyBURL] Best metapath (conf={best_conf:.3f}): {' -> '.join(best_path)}")
        else:
            print("[AnyBURL] No valid metapaths found")
        
        return best_path
    
    def export_to_config_format(self, 
                                metapath: List[str],
                                output_file: str = "mined_metapath.txt") -> str:
        """
        Exports mined metapath in a format suitable for config files.
        
        Args:
            metapath: List of relation names
            output_file: Output filename
            
        Returns:
            Path to output file
        """
        output_path = os.path.join(self.data_dir, output_file)
        
        with open(output_path, 'w') as f:
            f.write("# Mined metapath\n")
            f.write(f"METAPATH = {' -> '.join(metapath)}\n")
        
        print(f"[AnyBURL] Metapath exported to {output_path}")
        return output_path