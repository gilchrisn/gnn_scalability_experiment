"""
Python-to-C++ bridge for graph processing.
Handles data serialization for C++ ingestion and parsing of generated adjacency lists.
"""
import os
import subprocess
import time
import json
from typing import Dict, Optional, List
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data, HeteroData


class CppBridge:
    """
    Subprocess wrapper for the C++ graph processing executable.
    Adapts Python method calls to C++ CLI arguments.
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
                    k: Optional[int] = None,
                    l_val: Optional[int] = 1) -> float:
        """
        Invokes the C++ engine via subprocess.
        """
        cmd: List[str] = [
            self.executable,
            mode,
            self.data_dir,
            rule_file,
            output_file
        ]

        if mode == 'sketch':
            if k is None:
                raise ValueError("Argument 'k' is required for sketch mode.")
            cmd.append(str(k))
            cmd.append(str(l_val))

        print(f"   [C++] Exec: {' '.join(cmd)}")
        start = time.perf_counter()

        try:
            # Capture stdout/stderr to debug crashes
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            duration = time.perf_counter() - start
            return duration

        except subprocess.CalledProcessError as e:
            if "std::bad_alloc" in e.stderr:
                print("\n   [C++] CRITICAL: Memory allocation failed (OOM).")
                raise MemoryError("C++ backend exhausted available RAM.") from None
            
            print(f"\n   [C++] STDERR: {e.stderr}")
            print(f"   [C++] STDOUT: {e.stdout}")
            print(f"   [C++] Exit Code: {e.returncode} (Hex: {hex(e.returncode)})")
            
            raise RuntimeError(f"C++ binary failed with exit code {e.returncode}") from e

    def load_result_graph(self, 
                          filepath: str, 
                          num_nodes: int, 
                          node_offset: int) -> Data:
        """
        Parses the C++ adjacency list back into PyG format.
        """
        if not os.path.exists(filepath):
            print(f"   [C++] Warning: {filepath} not found. Returning empty graph.")
            return Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=num_nodes)

        srcs: List[int] = []
        dsts: List[int] = []

        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = list(map(int, line.strip().split()))
                    if not parts:
                        continue

                    # Format: <src_global_id> <dst1_global_id> <dst2_global_id> ...
                    u_global = parts[0]
                    u_local = u_global - node_offset

                    if u_local < 0 or u_local >= num_nodes:
                        continue

                    for v_global in parts[1:]:
                        v_local = v_global - node_offset
                        if 0 <= v_local < num_nodes:
                            srcs.append(u_local)
                            dsts.append(v_local)
        except ValueError as e:
            raise RuntimeError(f"Failed to parse C++ output file {filepath}: {e}")

        if not srcs:
            return Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=num_nodes)

        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        edge_index = pyg_utils.coalesce(edge_index, num_nodes=num_nodes)
        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=num_nodes)

        return Data(edge_index=edge_index, num_nodes=num_nodes)


class PyGToCppAdapter:
    """
    Format translator for converting HeteroData to C++ 'node.dat', 'link.dat', and 'meta.dat'.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.type_offsets: Dict[str, int] = {}
        self.node_type_mapping: Dict[str, int] = {}
        self.edge_type_mapping: Dict[tuple, int] = {}

    def convert(self, g_hetero: HeteroData) -> None:
        """
        Entry point for graph conversion. Generates node, link, and meta files.
        """
        node_file = os.path.join(self.output_dir, "node.dat")
        link_file = os.path.join(self.output_dir, "link.dat")
        meta_file = os.path.join(self.output_dir, "meta.dat")

        print(f"[Adapter] Serializing HeteroData to {self.output_dir}...")

        # Clean old files to prevent partial reads
        for f in [node_file, link_file, meta_file]:
            if os.path.exists(f): os.remove(f)

        total_nodes = self._write_nodes(g_hetero, node_file)
        self._write_edges(g_hetero, link_file)
        self._write_meta(meta_file, total_nodes)

        with open(os.path.join(self.output_dir, "offsets.json"), 'w') as f:
            json.dump({'offsets': self.type_offsets}, f)

        print(f"[Adapter] Success. Total graph size: {total_nodes} nodes.")

    def _write_nodes(self, g: HeteroData, filepath: str) -> int:
        """
        Writes node.dat using pandas for speed and correct formatting.
        Format: ID \t Name \t Type \t Info
        """
        node_types = sorted(g.node_types)
        self.node_type_mapping = {nt: i for i, nt in enumerate(node_types)}

        global_id = 0
        for ntype in tqdm(node_types, desc="Processing Node Types"):
            num_nodes = g[ntype].num_nodes
            self.type_offsets[ntype] = global_id
            type_id = self.node_type_mapping[ntype]

            # Use pandas to handle the writing efficiently and correctly
            df_nodes = pd.DataFrame({
                'name': [f"{ntype}_{i}" for i in range(num_nodes)],
                'type': type_id,
                'info': "" # Empty info column
            })
            
            # We explicitly construct the DataFrame to match the C++ expectation
            # The Global ID is implicit (line number), but some parsers might expect an ID column.
            # HNE standard is usually: Name \t Type \t Info (ID is line index)
            # However, previous successful runs suggested sticking to a simple format.
            # Let's write: Name \t Type \t Info
            
            df_nodes.to_csv(filepath, mode='a', sep='\t', header=False, index=False, quoting=3) # quoting=3 is csv.QUOTE_NONE
            global_id += num_nodes

        return global_id

    def _write_edges(self, g: HeteroData, filepath: str) -> None:
        """
        Writes link.dat using pandas.
        Format: SrcID \t DstID \t Type \t Weight
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

            srcs_global = edge_index[0] + src_offset
            dsts_global = edge_index[1] + dst_offset
            
            # Create DataFrame for bulk write
            edge_data = pd.DataFrame({
                'src': srcs_global,
                'dst': dsts_global,
                'type': etype_id,
                'weight': 1.0
            })
            
            # quoting=3 (QUOTE_NONE) ensures no quotes are added around numbers
            edge_data.to_csv(filepath, mode='a', sep='\t', header=False, index=False, quoting=3)

    def _write_meta(self, filepath: str, total_nodes: int) -> None:
        """
        Creates a valid meta.dat file to prevent parser crashes.
        """
        with open(filepath, 'w') as f:
            f.write(f"Total Nodes: {total_nodes}\n")
            f.write("Generated by PyGToCppAdapter\n")
            f.write("=" * 50 + "\n")

    def write_rule_file(self, rule_string: str, filename: str = "experiment.rule") -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(rule_string)
        return path