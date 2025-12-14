from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from torch_geometric.data import HeteroData

class GraphSampler(ABC):
    """
    Abstract base class for graph sampling strategies.
    Follows Strategy Pattern.
    """
    
    @abstractmethod
    def sample(self, g: HeteroData, config: Dict[str, Any]) -> HeteroData:
        """
        Samples a subgraph from the input graph.
        
        Args:
            g: Input heterogeneous graph
            config: Sampling configuration (seeds, hops, etc.)
            
        Returns:
            Sampled HeteroData object
        """
        pass