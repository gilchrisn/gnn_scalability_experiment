import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

class PyGToCppAdapter:
    """
    Converts PyG HeteroData objects into the custom 3-file format 
    (meta.dat, node.dat, link.dat) required by the C++ graph_prep tool.
    
    Refactored to use Pandas/NumPy for vectorized I/O (significant speedup).
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.type_offsets = {}
        self.edge_type_mapping = {}
        self.node_type_mapping = {}

    def convert(self, g_hetero):
        print(f"[Adapter] Converting graph to C++ format in '{self.output_dir}'...")
        
        # Paths
        node_file = os.path.join(self.output_dir, "node.dat")
        link_file = os.path.join(self.output_dir, "link.dat")
        meta_file = os.path.join(self.output_dir, "meta.dat")

        # Cleanup existing files (since we append)
        for fpath in [node_file, link_file, meta_file]:
            if os.path.exists(fpath):
                os.remove(fpath)

        # 1. Setup Node Mappings (Deterministic Order)
        node_types = sorted(g_hetero.node_types)
        self.node_type_mapping = {nt: i for i, nt in enumerate(node_types)}
        
        global_id_counter = 0
        self.type_offsets = {} 
        
        # --- Write node.dat (Vectorized) ---
        print(f"   Writing {node_file}...")
        
        for ntype in tqdm(node_types, desc="Processing Nodes"):
            num_nodes = g_hetero[ntype].num_nodes
            self.type_offsets[ntype] = global_id_counter
            type_id = self.node_type_mapping[ntype]
            
            # Generate names vector: "author_0", "author_1", ...
            # Note: List comprehension is faster than numpy string ops for this specific format
            names = [f"{ntype}_{i}" for i in range(num_nodes)]
            
            # Create DataFrame for bulk write
            df_nodes = pd.DataFrame({
                'name': names,
                'label': 'IGNORED',
                'type': type_id
            })
            
            # Append to file (header=False, sep='\t')
            df_nodes.to_csv(node_file, mode='a', sep='\t', header=False, index=False)
            
            global_id_counter += num_nodes

        # --- Write meta.dat ---
        print(f"   Writing {meta_file}...")
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(f"Total Node Num : {global_id_counter}\n")

        # --- Write link.dat (Vectorized) ---
        print(f"   Writing {link_file}...")
        
        # Sort edge types to ensure ID 0, 1, 2... are consistent
        edge_types = sorted(g_hetero.edge_types)
        self.edge_type_mapping = {et: i for i, et in enumerate(edge_types)}
        
        for etype_tuple in tqdm(edge_types, desc="Processing Edges"):
            src_type, rel, dst_type = etype_tuple
            etype_id = self.edge_type_mapping[etype_tuple]
            
            src_offset = self.type_offsets[src_type]
            dst_offset = self.type_offsets[dst_type]
            
            # Extract Edge Index (Move to CPU -> Numpy)
            edge_index = g_hetero[etype_tuple].edge_index.cpu().numpy()
            
            if edge_index.shape[1] == 0:
                continue

            # Vectorized Offset Addition
            # Row 0 is src, Row 1 is dst
            srcs_global = edge_index[0] + src_offset
            dsts_global = edge_index[1] + dst_offset
            etypes_col = np.full(srcs_global.shape, etype_id, dtype=np.int32)
            
            # Stack into (N, 3) matrix
            edge_data = np.column_stack((srcs_global, dsts_global, etypes_col))
            
            # Write to CSV
            df_edges = pd.DataFrame(edge_data)
            df_edges.to_csv(link_file, mode='a', sep='\t', header=False, index=False)
        
        print(f"[Adapter] Conversion Complete. Total Nodes: {global_id_counter}")

    def generate_cpp_rule(self, metapath):
        """
        Converts a PyG metapath [('A','to','B'), ('B','to','A')] 
        into C++ rule string: "-1 -2 0 -2 1 -5 0 -4 -4"
        """
        if not self.edge_type_mapping:
            raise ValueError("Run convert() first to populate edge type mappings!")

        rule_parts = ["-1"] # Start Variable Rule
        
        for edge_tuple in metapath:
            # 1. Try exact match (Forward)
            if edge_tuple in self.edge_type_mapping:
                rule_parts.append("-2") # Forward
                rule_parts.append(str(self.edge_type_mapping[edge_tuple]))
            else:
                # 2. Try reverse match (Backward)
                # Not implemented for this benchmark suite as stated in previous code
                raise ValueError(f"Edge Type {edge_tuple} not found in graph! Available: {list(self.edge_type_mapping.keys())}")

        # Add tail (Instance rule part)
        # -5 (Instance Start) 0 (Dummy Node ID) -4 -4 (Closers)
        rule_parts.extend(["-5", "0", "-4", "-4"])
        
        return " ".join(rule_parts)

    def write_rule_file(self, rule_string, filename="experiment.rule"):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(rule_string)
        print(f"[Adapter] Rule written to {path}: {rule_string}")
        return path