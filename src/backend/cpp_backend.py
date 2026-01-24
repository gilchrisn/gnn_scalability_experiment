"""
C++ backend implementation.
Delegates computation to external C++ executable via subprocess.
Follows SOLID principles and handles file I/O determinism.
"""
import os
from typing import Dict, Any, List, Tuple, Optional
import torch
from torch_geometric.data import Data, HeteroData

from .base import GraphBackend
from ..bridge import CppEngine, PyGToCppAdapter


class CppBackend(GraphBackend):
    """
    C++ subprocess backend.
    Converts graph to C++ format, executes binary, and loads results.
    """

    def __init__(self, 
                 executable_path: str, 
                 temp_dir: str, 
                 num_sketches: int = 1,
                 **kwargs):
        """
        Args:
            executable_path: Path to C++ binary.
            temp_dir: Path for intermediate files.
            num_sketches: 'L' parameter. Number of ensemble graphs to generate.
        """
        self.executable_path = executable_path
        self.temp_dir = temp_dir
        self.num_sketches = num_sketches

        self._adapter: Optional[PyGToCppAdapter] = None
        self._engine: Optional[CppEngine] = None
        self._metapath: Optional[List[Tuple[str, str, str]]] = None
        self._info: Optional[Dict[str, Any]] = None
        self._target_ntype: Optional[str] = None
        self._target_offset: int = 0
        self._num_nodes: int = 0
        self._last_prep_time: float = 0.0
        self._rule_path: Optional[str] = None

        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"C++ executable not found: {executable_path}")

    @property
    def name(self) -> str:
        return "cpp"

    @property
    def supports_inference(self) -> bool:
        return True 

    def initialize(self, 
                   g_hetero: Optional[HeteroData], 
                   metapath: List[Tuple[str, str, str]],
                   info: Dict[str, Any]) -> None:
        """
        Prepares the C++ environment: converts data and writes rule files.
        """
        if g_hetero is None:
            raise ValueError("[CppBackend] g_hetero cannot be None. Full conversion required.")

        self._metapath = metapath
        self._info = info
        self._target_ntype = metapath[-1][2]

        # 1. Setup Adapter and Convert (PyG -> C++ Format)
        self._adapter = PyGToCppAdapter(self.temp_dir)
        self._adapter.convert(g_hetero)

        # 2. Calculate Offsets
        # We access the public state of the adapter to get offsets
        self._target_offset = self._adapter.type_offsets[self._target_ntype]

        # 3. Generate Rule File (Stack Machine Logic)
        rule_string = self._generate_rule_string(metapath, g_hetero.edge_types)
        self._rule_path = self._adapter.write_rule_file(rule_string)

        # 4. Initialize Engine
        self._engine = CppEngine(self.executable_path, self.temp_dir)
        self._num_nodes = info['features'].shape[0]

    def materialize_exact(self) -> Data:
        """Executes exact materialization."""
        if self._engine is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        output_file = os.path.join(self.temp_dir, "exact_result.txt")

        prep_time = self._engine.run_command(
            mode="materialize",
            rule_file=self._rule_path,
            output_file=output_file
        )

        g_result = self._engine.load_result(
            output_file, self._num_nodes, self._target_offset
        )

        self._attach_metadata(g_result)
        self._last_prep_time = prep_time
        return g_result

    def materialize_kmv(self, k: int) -> Data:
        """
        Execute C++ KMV materialization.
        """
        if self._engine is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        base_output_file = os.path.join(self.temp_dir, f"kmv_k{k}_L{self.num_sketches}.txt")

        # Run C++
        prep_time = self._engine.run_command(
            mode="sketch",
            rule_file=self._rule_path,
            output_file=base_output_file,
            k=k,
            l_val=self.num_sketches
        )

        # Logic Patch: Robust File Discovery
        actual_output_file = self._resolve_output_filename(base_output_file)

        g_result = self._engine.load_result(
            actual_output_file, self._num_nodes, self._target_offset
        )

        self._attach_metadata(g_result)
        self._last_prep_time = prep_time
        return g_result

    def materialize_kmv_ensemble(self, k: int) -> List[Data]:
        """
        Execute C++ KMV and load ALL L generated subgraphs.
        Returns a list of Data objects for dynamic cycling.
        """
        if self._engine is None:
            raise RuntimeError("Backend not initialized.")

        base_output_file = os.path.join(self.temp_dir, f"kmv_k{k}_L{self.num_sketches}.txt")

        # 1. Run C++ (Generates L files on disk)
        prep_time = self._engine.run_command(
            mode="sketch",
            rule_file=self._rule_path,
            output_file=base_output_file,
            k=k,
            l_val=self.num_sketches
        )
        self._last_prep_time = prep_time

        # 2. Load all L files
        graphs = []
        base_name, ext = os.path.splitext(base_output_file)
        
        # Handle L=1 case seamlessly
        if self.num_sketches == 1 and os.path.exists(base_output_file):
             g = self._engine.load_result(base_output_file, self._num_nodes, self._target_offset)
             self._attach_metadata(g)
             graphs.append(g)
        else:
            # Handle L > 1 or suffixed files
            for i in range(self.num_sketches):
                file_i = f"{base_name}_{i}{ext}"
                
                # Fallback: if i=0 and suffix file missing, check base file
                if not os.path.exists(file_i):
                    if i == 0 and os.path.exists(base_output_file):
                        file_i = base_output_file
                    else:
                        print(f"    [Warning] Expected output {file_i} missing. Skipping.")
                        continue

                g = self._engine.load_result(
                    file_i, self._num_nodes, self._target_offset
                )
                self._attach_metadata(g)
                graphs.append(g)
            
        print(f"    [CppBackend] Loaded {len(graphs)} ensemble graphs.")
        return graphs

    # --- Internal Helpers ---

    def _generate_rule_string(self, metapath: List[Tuple[str, str, str]], edge_types) -> str:
        """Generates the stack machine rule string."""
        sorted_edges = sorted(edge_types)
        edge_map = {et: i for i, et in enumerate(sorted_edges)}
        rule_parts: List[str] = []
        
        # Iterate all edges EXCEPT the last one
        for i in range(len(metapath) - 1):
            t_edge = tuple(metapath[i])
            if t_edge not in edge_map:
                raise ValueError(f"Edge {t_edge} not found in schema")
            
            eid = edge_map[t_edge]
            rule_parts.append("-2") # Opcode: Forward
            rule_parts.append(str(eid))
            
        # Handle the FINAL edge with the Trigger (-1)
        t_last_edge = tuple(metapath[-1])
        if t_last_edge not in edge_map:
             raise ValueError(f"Edge {t_last_edge} not found in schema")
             
        last_eid = edge_map[t_last_edge]
        
        rule_parts.append("-2") # Opcode: Forward
        rule_parts.append("-1") # Opcode: TRIGGER VARIABLE MODE
        rule_parts.append(str(last_eid))
        
        # Cleanup (Pop stack for every hop pushed)
        for _ in range(len(metapath)):
            rule_parts.append("-4") # Opcode: Pop

        return " ".join(rule_parts)

    def _resolve_output_filename(self, base_file: str) -> str:
        """Handles discovering _0 suffix if C++ added it."""
        if os.path.exists(base_file):
            return base_file
        
        base_name, ext = os.path.splitext(base_file)
        alt_file = f"{base_name}_0{ext}"
        
        if not os.path.exists(alt_file):
             raise FileNotFoundError(f"[CppBackend] Output missing. Checked {base_file} and {alt_file}")
        
        return alt_file

    def _attach_metadata(self, g_result: Data) -> None:
        """Helper to attach features, labels, and masks to the raw structure."""
        if self._info:
            g_result.x = self._info.get('features')
            g_result.y = self._info.get('labels')
            masks = self._info.get('masks', {})
            g_result.train_mask = masks.get('train')
            g_result.val_mask = masks.get('val')
            g_result.test_mask = masks.get('test')

    def get_prep_time(self) -> float:
        return self._last_prep_time

    def cleanup(self) -> None:
        self._adapter = None
        self._engine = None
        self._metapath = None
        self._info = None