"""
C++ backend implementation.
Delegates computation to external C++ executable via subprocess.
"""
import os
from typing import Dict, Any, List, Tuple
import torch
from torch_geometric.data import Data, HeteroData

from .base import GraphBackend
from ..bridge import CppBridge, PyGToCppAdapter


class CppBackend(GraphBackend):
    """
    C++ subprocess backend.
    Converts graph to C++ format, executes binary, and loads results.
    """
    
    def __init__(self, executable_path: str, temp_dir: str):
        """
        Args:
            executable_path: Path to compiled C++ binary
            temp_dir: Directory for intermediate files
        """
        self.executable_path = executable_path
        self.temp_dir = temp_dir
        
        self._adapter = None
        self._bridge = None
        self._g_hetero = None
        self._metapath = None
        self._info = None
        self._target_ntype = None
        self._target_offset = 0
        self._num_nodes = 0
        self._last_prep_time = 0.0
        self._rule_path = None
        
        # Validate executable exists
        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"C++ executable not found: {executable_path}")
    
    @property
    def name(self) -> str:
        return "cpp"
    
    @property
    def supports_inference(self) -> bool:
        return True  # Results can be used for inference
    
    def initialize(self,
                   g_hetero: HeteroData,
                   metapath: List[Tuple[str, str, str]],
                   info: Dict[str, Any]) -> None:
        """
        Convert graph to C++ format and prepare for execution.
        
        Args:
            g_hetero: Input graph
            metapath: Path definition
            info: Metadata
        """
        print(f"[CppBackend] Converting graph to C++ format...")
        
        self._g_hetero = g_hetero
        self._metapath = metapath
        self._info = info
        self._target_ntype = metapath[-1][2]
        
        # Convert to C++ format
        self._adapter = PyGToCppAdapter(self.temp_dir)
        self._adapter.convert(g_hetero)
        
        # Generate rule file
        rule_str = self._adapter.generate_cpp_rule(metapath)
        self._rule_path = self._adapter.write_rule_file(rule_str)
        
        # Setup bridge
        self._bridge = CppBridge(self.executable_path, self.temp_dir)
        
        # Calculate target offset for ID mapping
        self._target_offset = self._adapter.type_offsets[self._target_ntype]
        self._num_nodes = info['features'].shape[0]
        
        print(f"[CppBackend] Initialized (offset={self._target_offset}, nodes={self._num_nodes})")
    
    def materialize_exact(self) -> Data:
        """
        Execute C++ exact materialization.
        
        Returns:
            Materialized graph
        """
        if self._bridge is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        
        print(f"[CppBackend] Running exact materialization...")
        
        output_file = os.path.join(self.temp_dir, "exact_result.txt")
        
        # Execute C++ binary
        prep_time = self._bridge.run_command(
            "materialize",
            self._rule_path,
            output_file
        )
        
        # Load results
        g_result = self._bridge.load_result_graph(
            output_file,
            self._num_nodes,
            self._target_offset
        )
        
        # Attach metadata
        g_result.x = self._info['features']
        g_result.y = self._info['labels']
        g_result.train_mask = self._info['masks']['train']
        g_result.val_mask = self._info['masks']['val']
        g_result.test_mask = self._info['masks']['test']
        
        self._last_prep_time = prep_time
        return g_result
    
    def materialize_kmv(self, k: int) -> Data:
        """
        Execute C++ KMV materialization.
        
        Args:
            k: Sketch size
            
        Returns:
            Sampled graph
        """
        if self._bridge is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        
        print(f"[CppBackend] Running KMV (k={k})...")
        
        output_file = os.path.join(self.temp_dir, f"kmv_k{k}_result.txt")
        
        # Execute C++ binary
        prep_time = self._bridge.run_command(
            "sketch",
            self._rule_path,
            output_file,
            k=k
        )
        
        # Load results
        g_result = self._bridge.load_result_graph(
            output_file,
            self._num_nodes,
            self._target_offset
        )
        
        # Attach metadata
        g_result.x = self._info['features']
        g_result.y = self._info['labels']
        g_result.train_mask = self._info['masks']['train']
        g_result.val_mask = self._info['masks']['val']
        g_result.test_mask = self._info['masks']['test']
        
        self._last_prep_time = prep_time
        return g_result
    
    def get_prep_time(self) -> float:
        """Return last recorded preprocessing time."""
        return self._last_prep_time
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        # Optionally delete intermediate files
        self._adapter = None
        self._bridge = None
        self._g_hetero = None
        self._metapath = None
        self._info = None