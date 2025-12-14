import subprocess
import time
import os
import sys
import torch
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from . import config

class CppBridge:
    def __init__(self, output_dir):
        self.executable = config.CPP_EXECUTABLE
        self.output_dir = output_dir
        
    def run_command(self, mode, rule_file, output_file, k=None):
        """
        Executes the C++ binary.
        mode: 'materialize' or 'sketch'
        """
        cmd = [
            self.executable,
            mode,
            self.output_dir, # Input directory (where node.dat/link.dat are)
            rule_file,
            output_file
        ]
        if k is not None:
            cmd.append(str(k))
            
        print(f"   [Bridge] Running C++ {mode.upper()}...", end=" ", flush=True)
        start = time.perf_counter()
        try:
            # Run subprocess
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("\n   [Bridge] C++ CRASHED!")
            print("   Error Output:\n", e.stderr)
            print("   Standard Output:\n", e.stdout)
            # sys.exit(1)
            raise e
        
        duration = time.perf_counter() - start
        print(f"Done ({duration:.4f}s)")
        return duration

    def load_result_graph(self, filepath, num_nodes, node_offset):
        """
        Parses the C++ adjacency list.
        CRITICAL: Maps Global IDs back to Local IDs using node_offset.
        """
        srcs, dsts = [], []
        
        if not os.path.exists(filepath):
            print(f"   [Bridge] Error: Output file {filepath} not found.")
            return Data(edge_index=torch.empty((2,0), dtype=torch.long), num_nodes=num_nodes)

        with open(filepath, 'r') as f:
            for line in f:
                parts = list(map(int, line.strip().split()))
                if not parts: continue
                
                # C++ outputs: <GlobalSrc> <GlobalDst1> <GlobalDst2> ...
                u_global = parts[0]
                u_local = u_global - node_offset
                
                # Safety check: ensure this node belongs to our target type
                if u_local < 0 or u_local >= num_nodes: continue

                for v_global in parts[1:]:
                    v_local = v_global - node_offset
                    if 0 <= v_local < num_nodes:
                        srcs.append(u_local)
                        dsts.append(v_local)
        
        if not srcs:
            return Data(edge_index=torch.empty((2,0), dtype=torch.long), num_nodes=num_nodes)

        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        
        # Cleanup: Remove duplicates and add self-loops (standard GNN practice)
        edge_index = pyg_utils.coalesce(edge_index, num_nodes=num_nodes)
        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=num_nodes)
        
        return Data(edge_index=edge_index, num_nodes=num_nodes)