"""
Implements conversion from PyTorch Geometric to C++ Engine Format (TSV).
"""
import os
import json
import pandas as pd
import shutil
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from torch_geometric.data import HeteroData
from .base import GraphConverter

class PyGToCppAdapter(GraphConverter):
    """
    Adapts PyG HeteroData to the C++ Engine's node.dat/link.dat format.
    
    Implements the Adapter Pattern to bridge Python objects with C++ file I/O.
    """
    
    _META_HEADER = "Node Number is : "

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Directory where C++ input files will be written.
        """
        self.output_dir = output_dir
        self._ensure_clean_directory()
        
        # State tracking (ID Mappings)
        self.type_offsets: Dict[str, int] = {}
        self.node_type_mapping: Dict[str, int] = {}
        self.edge_type_mapping: Dict[tuple, int] = {}

    def convert(self, g_hetero: HeteroData) -> None:
        """
        Orchestrates the conversion pipeline.
        
        Pipeline:
        1. Validate & Fix Metadata (Handle dirty data)
        2. Serialize Nodes
        3. Serialize Edges
        4. Write Metadata & Offsets
        """
        real_counts = self._validate_and_fix_metadata(g_hetero)
        total_nodes = self._serialize_nodes(g_hetero, real_counts)
        self._serialize_edges(g_hetero)
        self._write_meta(total_nodes)
        self._save_offsets()

    def write_rule_file(self, rule_string: str, filename: str = "experiment.rule") -> str:
        """Writes the C++ compatible rule file."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(rule_string)
        return path

    # =========================================================================
    # Internal Helper Methods (SRP: Separate serialization logic)
    # =========================================================================

    def _ensure_clean_directory(self) -> None:
        """Creates directory and removes stale files."""
        os.makedirs(self.output_dir, exist_ok=True)
        for f in ["node.dat", "link.dat", "meta.dat", "offsets.json"]:
            path = os.path.join(self.output_dir, f)
            if os.path.exists(path):
                os.remove(path)

    def _validate_and_fix_metadata(self, g: HeteroData) -> Dict[str, int]:
        """
        Scans edges to ensure node counts encompass all referenced IDs.
        Fixes 'dirty data' where edge indices exceed declared num_nodes.
        """
        # Validate graph integrity
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
                print(f"      [Fix] {src}: ID {max_src} >= num_nodes {real_counts[src]}. Expanding.")
                real_counts[src] = max_src + 1
                
            if max_dst >= real_counts[dst]:
                print(f"      [Fix] {dst}: ID {max_dst} >= num_nodes {real_counts[dst]}. Expanding.")
                real_counts[dst] = max_dst + 1
                
        return real_counts

    def _serialize_nodes(self, g: HeteroData, counts: Dict[str, int]) -> int:
        """Writes node.dat mapping heterogeneous types to global IDs."""
        filepath = os.path.join(self.output_dir, "node.dat")
        total_nodes = 0
        node_types = sorted(g.node_types)
        
        self.node_type_mapping = {nt: i for i, nt in enumerate(node_types)}

        for ntype in node_types:
            count = counts[ntype]
            self.type_offsets[ntype] = total_nodes
            
            # Enforce column order for C++ parser safety
            df = pd.DataFrame({
                'col1': range(total_nodes, total_nodes + count),
                'col2': 'IGNORED',
                'type': self.node_type_mapping[ntype]
            })
            
            df.to_csv(
                filepath, 
                mode='a', 
                sep='\t', 
                header=False, 
                index=False,
                columns=['col1', 'col2', 'type']
            )
            
            total_nodes += count
            
        return total_nodes

    def _serialize_edges(self, g: HeteroData) -> None:
        """Writes link.dat converting local IDs to global IDs."""
        filepath = os.path.join(self.output_dir, "link.dat")
        edge_types = sorted(g.edge_types)
        self.edge_type_mapping = {et: i for i, et in enumerate(edge_types)}

        for etype_tuple in edge_types:
            src_type, _, dst_type = etype_tuple
            etype_id = self.edge_type_mapping[etype_tuple]

            edge_index = g[etype_tuple].edge_index.cpu().numpy()
            if edge_index.shape[1] == 0:
                continue

            # Map local IDs to global IDs
            src_offset = self.type_offsets[src_type]
            dst_offset = self.type_offsets[dst_type]

            df_edges = pd.DataFrame({
                'src': edge_index[0] + src_offset,
                'dst': edge_index[1] + dst_offset,
                'type': etype_id
            })
            
            df_edges.to_csv(
                filepath, 
                mode='a', 
                sep='\t', 
                header=False, 
                index=False,
                columns=['src', 'dst', 'type']
            )

    def _write_meta(self, total_nodes: int) -> None:
        """Writes meta.dat with handshake header."""
        path = os.path.join(self.output_dir, "meta.dat")
        with open(path, 'w') as f:
            f.write(f"{self._META_HEADER}{total_nodes}\n")

    def _save_offsets(self) -> None:
        """Saves offsets to JSON for result reconstruction."""
        path = os.path.join(self.output_dir, "offsets.json")
        with open(path, 'w') as f:
            json.dump({'offsets': self.type_offsets}, f)