"""
Abstract base classes for the bridge layer.
Follows Interface Segregation Principle (ISP) and defines contracts for
adapters and execution engines.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from torch_geometric.data import HeteroData

class GraphConverter(ABC):
    """
    Strategy interface for converting PyG graphs to external formats.
    """
    
    @abstractmethod
    def convert(self, g_hetero: HeteroData) -> None:
        """
        Converts and serializes the graph to the target format.
        
        Args:
            g_hetero: The input heterogeneous graph.
        """
        pass

    @abstractmethod
    def write_rule_file(self, rule_string: str, filename: str) -> str:
        """
        Writes a rule string to a format the engine understands.
        
        Args:
            rule_string: The logical rule (e.g., "-2 0 -2 1").
            filename: The output filename.

        Returns:
            Path to the generated rule file.
        """
        pass

class RuleMiner(ABC):
    """
    Strategy interface for mining logical rules from graphs.
    """
    
    @abstractmethod
    def export_for_mining(self, g: HeteroData) -> None:
        """Exports graph to the format required by the miner."""
        pass

    @abstractmethod
    def run_mining(self, timeout: int, **kwargs) -> None:
        """Executes the mining process."""
        pass

    @abstractmethod
    def get_top_k_paths(self, k: int, min_conf: float) -> List[Tuple[float, str]]:
        """
        Parses mining results.

        Returns:
            List of (confidence, metapath_string) tuples.
        """
        pass

class ExecutionEngine(ABC):
    """
    Interface for running external binaries (Bridge Pattern).
    """
    
    @abstractmethod
    def run_command(self, mode: str, **kwargs) -> float:
        """
        Executes a command on the external engine.

        Returns:
            Execution time in seconds.
        """
        pass

    @abstractmethod
    def load_result(self, filepath: str, **kwargs) -> Any:
        """Loads results produced by the external engine."""
        pass