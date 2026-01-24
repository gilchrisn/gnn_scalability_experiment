"""
Python-to-C++ bridge adapter for graph processing.

Converts PyG HeteroData objects into the specific Tab-Separated Value (TSV) 
format required by the C++ engine (node.dat, link.dat, meta.dat).
"""
import os
import json
import pandas as pd
from typing import Dict, List, Any
from tqdm import tqdm
from torch_geometric.data import HeteroData

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
                    k: int = None,
                    l_val: int = 1) -> float:
        """
        Invokes the C++ engine via subprocess.
        """
        import time
        import subprocess

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

        print(f"    [C++] Exec: {' '.join(cmd)}")
        start = time.perf_counter()

        try:
            # Capture stdout/stderr to debug crashes
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            duration = time.perf_counter() - start
            return duration

        except subprocess.CalledProcessError as e:
            if "std::bad_alloc" in e.stderr:
                print("\n    [C++] CRITICAL: Memory allocation failed (OOM).")
                raise MemoryError("C++ backend exhausted available RAM.") from None
            
            print(f"\n    [C++] STDERR: {e.stderr}")
            print(f"    [C++] STDOUT: {e.stdout}")
            print(f"    [C++] Exit Code: {e.returncode}")
            
            raise RuntimeError(f"C++ binary failed with exit code {e.returncode}") from e

    def load_result_graph(self, 
                          filepath: str, 
                          num_nodes: int, 
                          node_offset: int) -> Any:
        """
        Parses the C++ adjacency list back into PyG format.
        """
        import torch
        import torch_geometric.utils as pyg_utils
        from torch_geometric.data import Data

        if not os.path.exists(filepath):
            print(f"    [C++] Warning: {filepath} not found. Returning empty graph.")
            return Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=num_nodes)

        srcs: List[int] = []
        dsts: List[int] = []

        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = list(map(int, line.strip().split()))
                    if not parts:
                        continue

                    u_global = parts[0]
                    u_local = u_global - node_offset

                    # Safety check for invalid IDs
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
    Adapts PyG HeteroData to C++ Engine format.
    
    Implements the Adapter Pattern to bridge Python graph structures 
    with C++ file-based input requirements. Handles data sanitization 
    and format conversion.
    """
    
    # Constant for the header string to ensure exact length match with C++ parser
    # C++ expects: "Node Number is : " (17 chars)
    _META_HEADER = "Node Number is : "

    def __init__(self, output_dir: str):
        """
        Initialize the adapter.

        Args:
            output_dir: Directory where C++ input files will be written.
        """
        self.output_dir = output_dir
        self._ensure_clean_directory()
        
        # State tracking for ID mapping
        self.type_offsets: Dict[str, int] = {}
        self.node_type_mapping: Dict[str, int] = {}
        self.edge_type_mapping: Dict[tuple, int] = {}

    def convert(self, g_hetero: HeteroData) -> None:
        """
        Orchestrates the conversion process.
        
        Pipeline:
        1. Validate & Fix Metadata (Handle dirty data)
        2. Serialize Nodes
        3. Serialize Edges
        4. Write Metadata
        5. Save Offsets
        
        Args:
            g_hetero: The input PyTorch Geometric HeteroData object.
        """
        print("[Adapter] Starting conversion pipeline...")
        
        # 1. Validate and fix node counts (Crucial for HGB datasets)
        real_counts = self._validate_and_fix_metadata(g_hetero)
        
        # 2. Write Nodes (node.dat)
        total_nodes = self._serialize_nodes(g_hetero, real_counts)
        
        # 3. Write Edges (link.dat)
        self._serialize_edges(g_hetero)
        
        # 4. Write Meta (meta.dat)
        self._write_meta(total_nodes)
        
        # 5. Save Offsets for backward mapping
        self._save_offsets()
        
        print(f"[Adapter] Conversion complete. Total Nodes: {total_nodes}")

    def write_rule_file(self, rule_string: str, filename: str = "experiment.rule") -> str:
        """
        Writes a rule string to a file for the C++ engine.

        Args:
            rule_string: The formatted rule string (e.g., "-1 -2 0 ...").
            filename: Name of the output file.

        Returns:
            Absolute path to the created rule file.
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(rule_string)
        return path

    # =========================================================================
    # Internal Helper Methods (Private)
    # =========================================================================

    def _ensure_clean_directory(self) -> None:
        """Creates directory and removes stale files."""
        os.makedirs(self.output_dir, exist_ok=True)
        for f in ["node.dat", "link.dat", "meta.dat"]:
            path = os.path.join(self.output_dir, f)
            if os.path.exists(path):
                os.remove(path)

    def _validate_and_fix_metadata(self, g: HeteroData) -> Dict[str, int]:
        """
        Scans edges to ensure node counts encompass all referenced IDs.
        Fixes 'dirty data' where edge indices exceed declared num_nodes.
        """
        print("  -> Validating graph integrity...")
        real_counts = {nt: g[nt].num_nodes for nt in g.node_types}
        
        for src, _, dst in g.edge_types:
            edge_index = g[src, _, dst].edge_index
            if edge_index.numel() == 0:
                continue
            
            # Find max ID actually used
            max_src = edge_index[0].max().item()
            max_dst = edge_index[1].max().item()
            
            # Auto-expand if necessary
            if max_src >= real_counts[src]:
                print(f"    [Fix] {src}: Found ID {max_src} >= num_nodes {real_counts[src]}. Expanding.")
                real_counts[src] = max_src + 1
                
            if max_dst >= real_counts[dst]:
                print(f"    [Fix] {dst}: Found ID {max_dst} >= num_nodes {real_counts[dst]}. Expanding.")
                real_counts[dst] = max_dst + 1
                
        return real_counts

    def _serialize_nodes(self, g: HeteroData, real_counts: Dict[str, int]) -> int:
        """
        Writes node.dat. Maps heterogeneous types to global homogeneous IDs.
        
        Returns:
            Total number of nodes written.
        """
        print("  -> Serializing nodes...")
        filepath = os.path.join(self.output_dir, "node.dat")
        total_nodes = 0
        node_types = sorted(g.node_types)
        
        self.node_type_mapping = {nt: i for i, nt in enumerate(node_types)}

        for ntype in node_types:
            count = real_counts[ntype]
            self.type_offsets[ntype] = total_nodes
            type_id = self.node_type_mapping[ntype]

            # Enforce column order for C++ parser safety
            df_nodes = pd.DataFrame({
                'col1': range(total_nodes, total_nodes + count),
                'col2': 'IGNORED',
                'type': type_id
            })
            
            df_nodes.to_csv(
                filepath, 
                mode='a', 
                sep='\t', 
                header=False, 
                index=False, 
                columns=['col1', 'col2', 'type'] # Strict ordering
            )
            
            total_nodes += count
            
        return total_nodes

    def _serialize_edges(self, g: HeteroData) -> None:
        """Writes link.dat. Converts local IDs to global IDs."""
        print("  -> Serializing edges...")
        filepath = os.path.join(self.output_dir, "link.dat")
        edge_types = sorted(g.edge_types)
        self.edge_type_mapping = {et: i for i, et in enumerate(edge_types)}

        for etype_tuple in tqdm(edge_types, desc="    Processing Edge Types"):
            src_type, _, dst_type = etype_tuple
            etype_id = self.edge_type_mapping[etype_tuple]

            edge_index = g[etype_tuple].edge_index.cpu().numpy()
            if edge_index.shape[1] == 0:
                continue

            # Map local IDs to global IDs
            src_offset = self.type_offsets[src_type]
            dst_offset = self.type_offsets[dst_type]

            srcs_global = edge_index[0] + src_offset
            dsts_global = edge_index[1] + dst_offset
            
            df_edges = pd.DataFrame({
                'src': srcs_global,
                'dst': dsts_global,
                'type': etype_id
            })
            
            df_edges.to_csv(
                filepath, 
                mode='a', 
                sep='\t', 
                header=False, 
                index=False, 
                columns=['src', 'dst', 'type'] # Strict ordering
            )

    def _write_meta(self, total_nodes: int) -> None:
        """Writes meta.dat with the required handshake header."""
        path = os.path.join(self.output_dir, "meta.dat")
        with open(path, 'w') as f:
            f.write(f"{self._META_HEADER}{total_nodes}\n")

    def _save_offsets(self) -> None:
        """Saves offsets to JSON for result reconstruction."""
        path = os.path.join(self.output_dir, "offsets.json")
        with open(path, 'w') as f:
            json.dump({'offsets': self.type_offsets}, f)