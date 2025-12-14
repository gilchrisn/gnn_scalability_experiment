"""
C++ interoperability bridge.
Handles conversion between PyG format and C++ graph format.
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
    Bridge for executing C++ graph processing executable.
    Manages subprocess calls and result parsing.
    """
    
    def __init__(self, executable_path: str, data_dir: str):
        """
        Args:
            executable_path: Path to compiled C++ binary
            data_dir: Directory containing input/output files
        """
        self.executable = executable_path
        self.data_dir = data_dir
        
        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"C++ executable not found: {executable_path}")
    
    def run_command(self,
                   mode: str,
                   rule_file: str,
                   output_file: str,
                   k: int = None) -> float:
        """
        Executes C++ graph processing command.
        
        Args:
            mode: 'materialize' or 'sketch'
            rule_file: Path to rule definition file
            output_file: Path for output adjacency list
            k: Sketch size (only for sketch mode)
            
        Returns:
            Execution time in seconds
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
        
        print(f"   [C++] Running {mode.upper()}...", end=" ", flush=True)
        start = time.perf_counter()
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            duration = time.perf_counter() - start
            print(f"Done ({duration:.4f}s)")
            return duration
        except subprocess.CalledProcessError as e:
            print("\n   [C++] EXECUTION FAILED!")
            print(f"   Error: {e.stderr}")
            raise
    
    def load_result_graph(self,
                         filepath: str,
                         num_nodes: int,
                         node_offset: int) -> Data:
        """
        Parses C++ output adjacency list and converts to PyG Data.
        
        Args:
            filepath: Path to adjacency list file
            num_nodes: Number of nodes in target type
            node_offset: Global ID offset for target node type
            
        Returns:
            PyG Data object with edge_index
        """
        if not os.path.exists(filepath):
            print(f"   [C++] Warning: Output file not found: {filepath}")
            return Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=num_nodes)
        
        srcs, dsts = [], []
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = list(map(int, line.strip().split()))
                if not parts:
                    continue
                
                # Format: <src_global> <dst1_global> <dst2_global> ...
                u_global = parts[0]
                u_local = u_global - node_offset
                
                if u_local < 0 or u_local >= num_nodes:
                    continue
                
                for v_global in parts[1:]:
                    v_local = v_global - node_offset
                    if 0 <= v_local < num_nodes:
                        srcs.append(u_local)
                        dsts.append(v_local)
        
        if not srcs:
            return Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=num_nodes)
        
        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        edge_index = pyg_utils.coalesce(edge_index, num_nodes=num_nodes)
        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=num_nodes)
        
        return Data(edge_index=edge_index, num_nodes=num_nodes)


class PyGToCppAdapter:
    """
    Adapter for converting PyG HeteroData to C++ format.
    Implements the Adapter pattern for format translation.
    """
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Directory for output files (node.dat, link.dat, meta.dat)
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.type_offsets: Dict[str, int] = {}
        self.node_type_mapping: Dict[str, int] = {}
        self.edge_type_mapping: Dict[Tuple[str, str, str], int] = {}
    
    def convert(self, g_hetero: HeteroData) -> None:
        """
        Converts PyG HeteroData to C++ format files.
        
        Args:
            g_hetero: Input heterogeneous graph
        """
        print(f"[Adapter] Converting graph to C++ format...")
        
        node_file = os.path.join(self.output_dir, "node.dat")
        link_file = os.path.join(self.output_dir, "link.dat")
        meta_file = os.path.join(self.output_dir, "meta.dat")
        
        # Clear existing files
        for fpath in [node_file, link_file, meta_file]:
            if os.path.exists(fpath):
                os.remove(fpath)
        
        # Write nodes
        total_nodes = self._write_nodes(g_hetero, node_file)
        
        # Write edges
        self._write_edges(g_hetero, link_file)
        
        # Write metadata
        with open(meta_file, "w") as f:
            f.write(f"Total Node Num : {total_nodes}\n")
        
        print(f"[Adapter] Conversion complete. Total nodes: {total_nodes}")
    
    def _write_nodes(self, g: HeteroData, filepath: str) -> int:
        """Writes node.dat file using vectorized operations."""
        node_types = sorted(g.node_types)
        self.node_type_mapping = {nt: i for i, nt in enumerate(node_types)}
        
        global_id = 0
        
        for ntype in tqdm(node_types, desc="Writing nodes"):
            num_nodes = g[ntype].num_nodes
            self.type_offsets[ntype] = global_id
            type_id = self.node_type_mapping[ntype]
            
            # Generate node names
            names = [f"{ntype}_{i}" for i in range(num_nodes)]
            
            # Create DataFrame
            df_nodes = pd.DataFrame({
                'name': names,
                'label': 'IGNORED',
                'type': type_id
            })
            
            df_nodes.to_csv(filepath, mode='a', sep='\t', header=False, index=False)
            global_id += num_nodes
        
        return global_id
    
    def _write_edges(self, g: HeteroData, filepath: str) -> None:
        """Writes link.dat file using vectorized operations."""
        edge_types = sorted(g.edge_types)
        self.edge_type_mapping = {et: i for i, et in enumerate(edge_types)}
        
        for etype_tuple in tqdm(edge_types, desc="Writing edges"):
            src_type, rel, dst_type = etype_tuple
            etype_id = self.edge_type_mapping[etype_tuple]
            
            src_offset = self.type_offsets[src_type]
            dst_offset = self.type_offsets[dst_type]
            
            edge_index = g[etype_tuple].edge_index.cpu().numpy()
            
            if edge_index.shape[1] == 0:
                continue
            
            # Vectorized offset addition
            srcs_global = edge_index[0] + src_offset
            dsts_global = edge_index[1] + dst_offset
            etypes_col = np.full(srcs_global.shape, etype_id, dtype=np.int32)
            
            edge_data = np.column_stack((srcs_global, dsts_global, etypes_col))
            df_edges = pd.DataFrame(edge_data)
            df_edges.to_csv(filepath, mode='a', sep='\t', header=False, index=False)
    
    def generate_cpp_rule(self, metapath: List[Tuple[str, str, str]]) -> str:
        """
        Converts PyG metapath to C++ rule string.
        
        Args:
            metapath: List of edge type tuples
            
        Returns:
            Rule string for C++ executable
        """
        if not self.edge_type_mapping:
            raise ValueError("Run convert() first to populate mappings!")
        
        rule_parts = ["-1"]  # Variable rule start
        
        for edge_tuple in metapath:
            if edge_tuple not in self.edge_type_mapping:
                available = list(self.edge_type_mapping.keys())[:5]
                raise ValueError(
                    f"Edge type {edge_tuple} not found. "
                    f"Available: {available}"
                )
            
            rule_parts.append("-2")  # Forward direction
            rule_parts.append(str(self.edge_type_mapping[edge_tuple]))
        
        # Instance rule part
        rule_parts.extend(["-5", "0", "-4", "-4"])
        
        return " ".join(rule_parts)
    
    def write_rule_file(self, rule_string: str, filename: str = "experiment.rule") -> str:
        """
        Writes rule string to file.
        
        Returns:
            Path to written rule file
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(rule_string)
        print(f"[Adapter] Rule written: {rule_string}")
        return path