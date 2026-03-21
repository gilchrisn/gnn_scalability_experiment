"""
C++ Execution Engine implementation.
Handles subprocess calls and result parsing for the graph_prep binary.
"""
import cmd
import os
import time
import subprocess
from typing import List, Any
import torch
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from .base import ExecutionEngine

class CppEngine(ExecutionEngine):
    """
    Subprocess wrapper for the C++ graph processing executable.
    """

    def __init__(self, executable_path: str, data_dir: str):
        self.executable = executable_path
        self.data_dir = data_dir

        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"C++ binary missing: {executable_path}")

    def _sanitize_path(self, path: str) -> str:
        """
        Forces forward slashes for C++ argument compatibility, specifically on Windows.
        The C++ standard library or specific argument parsers can sometimes choke on backslashes
        in file paths when passed via subprocess.
        """
        return os.path.abspath(path).replace('\\', '/')
    
    def run_command(self, mode: str, rule_file: str, output_file: str,
                    k: int = None, l_val: int = 1, seed: int = None,
                    timeout: int = 600) -> float:
        """
        Invokes the C++ engine. Parses internal algorithmic timer if available,
        otherwise falls back to total end-to-end execution time.

        Args:
            timeout: Subprocess timeout in seconds. Raises RuntimeError on expiry.
        """
        bin_path = self._sanitize_path(self.executable)
        data_path = self._sanitize_path(self.data_dir)
        rule_path = self._sanitize_path(rule_file)
        out_path = self._sanitize_path(output_file)

        cmd: List[str] = [bin_path, mode, data_path, rule_path, out_path]

        if mode == 'sketch':
            if k is None:
                raise ValueError("Argument 'k' is required for sketch mode.")
            cmd.append(str(k))
            cmd.append(str(l_val))
            if seed is not None:
                cmd.append(str(seed))

        print(f"    [C++] Exec: {' '.join(cmd)}")
        start_fallback = time.perf_counter()

        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True,
                                 timeout=timeout)

            # 1. Try to intercept Pure C++ Algorithmic Time (For Benchmarking Mode)
            for line in res.stdout.split('\n'):
                line = line.strip().lower()
                if line.startswith("time:"):
                    try:
                        return float(line.split(":")[1].strip())
                    except ValueError:
                        pass

            # 2. Fallback to Total Pipeline Time (For Data Generation Mode like 'sketch')
            return time.perf_counter() - start_fallback

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"C++ binary timed out after {timeout}s (mode={mode}). "
                "Increase --timeout or skip this metapath."
            ) from None
        except subprocess.CalledProcessError as e:
            if "std::bad_alloc" in e.stderr:
                raise MemoryError("C++ backend exhausted available RAM.") from None

            print(f"\n    [C++] STDERR: {e.stderr}")
            print(f"    [C++] STDOUT: {e.stdout}")
            raise RuntimeError(f"C++ binary failed with exit code {e.returncode}") from e

    def load_result(self, filepath: str, num_nodes: int, node_offset: int) -> Data:
        """
        Parses the C++ adjacency list back into PyG format.
        """
        if not os.path.exists(filepath):
            print(f"    [C++] Warning: {filepath} not found. Returning empty graph.")
            return Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=num_nodes)

        srcs: List[int] = []
        dsts: List[int] = []

        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = list(map(int, line.strip().split()))
                    if not parts: continue

                    u_global = parts[0]
                    u_local = u_global - node_offset

                    if u_local < 0 or u_local >= num_nodes: continue

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