"""
Python-to-C++ bridge for graph processing.
Handles data serialization for C++ ingestion and parsing of generated adjacency lists.
"""
import os
import subprocess
import time
from typing import Tuple, Dict, List
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data, HeteroData


class CppBridge:
    """
    Subprocess wrapper for the C++ graph processing executable.
    """
    
    def __init__(self, executable_path: str, data_dir: str):
        self.executable = executable_path
        self.data_dir = data_dir
        
        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"C++ binary missing: {executable_path}")
    
    def run_command(self,
                    mode: str,
                    rule_file: str,
                    output_file: str,
                    k: int = None) -> float:
        """
        Invokes the C++ engine. 
        Supported modes: 'materialize' (exact), 'sketch' (approximate).
        """
        cmd = [
            self.executable,
            mode,
            self.data_dir,
            rule_file,
            output_file
        ]
        
        if k is not None:
            cmd.append(str(k))
        
        print(f"   [C++] Mode: {mode.upper()}", end=" ", flush=True)
        start = time.perf_counter()
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            duration = time.perf_counter() - start
            print(f"Done ({duration:.4f}s)")
            return duration
        except subprocess.CalledProcessError as e:
            # Check specifically for C++ memory allocation failure
            if "std::bad_alloc" in e.stderr:
                print("\n   [C++] Memory allocation failed.")
                raise MemoryError("C++ backend exhausted available RAM.") from None
                
            print(f"\n   [C++] Error: {e.stderr}")
            raise RuntimeError(f"C++ binary failed with exit code {e.returncode}") from e
    
    def load_result_graph(self,
                          filepath: str,
                          num_nodes: int,
                          node_offset: int) -> Data:
        """
        Parses the C++ adjacency list back into PyG format.
        Converts global IDs to local indices using the provided offset.
        """
        if not os.path.exists(filepath):
            print(f"   [C++] Warning: {filepath} not found. Returning empty graph.")
            return Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=num_nodes)
        
        srcs, dsts = [], []
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = list(map(int, line.strip().split()))
                if not parts:
                    continue
                
                # Format: <src_global_id> <dst1_global_id> <dst2_global_id> ...
                u_global = parts[0]
                u_local = u_global - node_offset
                
                # Bound check relative to target node type slice
                if u_local < 0 or u_local >= num_nodes:
                    continue
                
                for v_global in parts[1:]:
                    v_local = v_global - node_offset
                    if 0 <= v_local < num_nodes:
                        srcs.append(u_local)
                        dsts.append(v_local)
        
        if not srcs:
            return Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=num_nodes)
        
        # Coalesce to remove duplicates and add self-loops for GNN stability
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        edge_index = pyg_utils.coalesce(edge_index, num_nodes=num_nodes)
        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=num_nodes)
        
        return Data(edge_index=edge_index, num_nodes=num_nodes)


class PyGToCppAdapter:
    """
    Format translator for converting HeteroData to C++ 'node.dat' and 'link.dat'.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.type_offsets: Dict[str, int] = {}
        # Load offsets from cache if available to ensure consistency
        self._load_cached_offsets()

    def _load_cached_offsets(self) -> None:
        """Attempts to load type offsets from a previous run."""
        offset_path = os.path.join(self.output_dir, "offsets.json")
        if os.path.exists(offset_path):
            import json
            with open(offset_path, 'r') as f:
                data = json.load(f)
                self.type_offsets = data.get('offsets', {})
                
    def convert(self, g_hetero: HeteroData) -> None:
        """
        Entry point for graph conversion. Generates node, link, and meta files.
        Checks for existing artifacts to enable zero-copy execution.
        """
        node_file = os.path.join(self.output_dir, "node.dat")
        link_file = os.path.join(self.output_dir, "link.dat")
        
        # Short-circuit: If files exist and are non-empty, skip conversion
        if (os.path.exists(node_file) and os.path.getsize(node_file) > 0 and
            os.path.exists(link_file) and os.path.getsize(link_file) > 0):
            print(f"[Adapter] Found existing artifacts. Skipping write.")
            self._recalculate_offsets(g_hetero)
            return

        print(f"[Adapter] Serializing HeteroData to {self.output_dir}...")
        
        if os.path.exists(node_file): os.remove(node_file)
        if os.path.exists(link_file): os.remove(link_file)
        
        total_nodes = self._write_nodes(g_hetero, node_file)
        self._write_edges(g_hetero, link_file)
        
        # Save offsets for future runs (essential for C++ ID mapping)
        import json
        with open(os.path.join(self.output_dir, "offsets.json"), 'w') as f:
            json.dump({'offsets': self.type_offsets}, f)

        print(f"[Adapter] Success. Total graph size: {total_nodes} nodes.")

    def _recalculate_offsets(self, g: HeteroData) -> None:
        """Recalculates global ID offsets without writing to disk."""
        node_types = sorted(g.node_types)
        global_id = 0
        for ntype in node_types:
            self.type_offsets[ntype] = global_id
            global_id += g[ntype].num_nodes
    
    def _write_nodes(self, g: HeteroData, filepath: str) -> int:
        """
        Writes nodes to file. Uses sorted type keys to maintain deterministic global IDs.
        """
        node_types = sorted(g.node_types)
        self.node_type_mapping = {nt: i for i, nt in enumerate(node_types)}
        
        global_id = 0
        for ntype in tqdm(node_types, desc="Processing Node Types"):
            num_nodes = g[ntype].num_nodes
            self.type_offsets[ntype] = global_id
            type_id = self.node_type_mapping[ntype]
            
            # Batch generate names and types for CSV export
            df_nodes = pd.DataFrame({
                'name': [f"{ntype}_{i}" for i in range(num_nodes)],
                'label': 'IGNORED',
                'type': type_id
            })
            
            df_nodes.to_csv(filepath, mode='a', sep='\t', header=False, index=False)
            global_id += num_nodes
        
        return global_id
    
    def _write_edges(self, g: HeteroData, filepath: str) -> None:
        """
        Writes links using vectorized offset additions for performance.
        """
        edge_types = sorted(g.edge_types)
        self.edge_type_mapping = {et: i for i, et in enumerate(edge_types)}
        
        for etype_tuple in tqdm(edge_types, desc="Processing Edge Types"):
            src_type, rel, dst_type = etype_tuple
            etype_id = self.edge_type_mapping[etype_tuple]
            
            src_offset = self.type_offsets[src_type]
            dst_offset = self.type_offsets[dst_type]
            
            edge_index = g[etype_tuple].edge_index.cpu().numpy()
            if edge_index.shape[1] == 0:
                continue
            
            # Compute global IDs via vectorized shift
            srcs_global = edge_index[0] + src_offset
            dsts_global = edge_index[1] + dst_offset
            etypes_col = np.full(srcs_global.shape, etype_id, dtype=np.int32)
            
            edge_data = np.column_stack((srcs_global, dsts_global, etypes_col))
            df_edges = pd.DataFrame(edge_data)
            df_edges.to_csv(filepath, mode='a', sep='\t', header=False, index=False)
    
    def generate_cpp_rule(self, metapath: List[Tuple[str, str, str]]) -> str:
        """
        Maps a PyG metapath to the C++ DSL rule format.
        Format: -1 [forward_flag, relation_id]* -5 0 -4 -4
        """
        if not self.edge_type_mapping:
            raise RuntimeError("Mappings not initialized. Call convert() first.")
        
        rule_parts = ["-1"]
        for edge_tuple in metapath:
            if edge_tuple not in self.edge_type_mapping:
                raise ValueError(f"Unknown edge type: {edge_tuple}")
            
            rule_parts.append("-2") # Forward direction flag
            rule_parts.append(str(self.edge_type_mapping[edge_tuple]))
        
        # Terminator sequence for the C++ parser
        rule_parts.extend(["-5", "0", "-4", "-4"])
        return " ".join(rule_parts)
    
    def write_rule_file(self, rule_string: str, filename: str = "experiment.rule") -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(rule_string)
        return path