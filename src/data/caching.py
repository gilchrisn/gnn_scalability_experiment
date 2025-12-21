"""
Artifact caching logic to enable zero-copy execution.
Implements the Proxy Pattern for large dataset handling.
"""
import os
import torch
import json
from typing import Dict, Any, Optional

class ArtifactManager:
    """
    Manages persistence of graph metadata to decouple Python memory
    from C++ execution requirements.
    """
    
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.meta_path = os.path.join(root_dir, "metadata.pt")
        self.cpp_node_path = os.path.join(root_dir, "node.dat")
        self.cpp_link_path = os.path.join(root_dir, "link.dat")

    def exists(self) -> bool:
        """Checks if all necessary artifacts exist on disk."""
        return (os.path.exists(self.meta_path) and 
                os.path.exists(self.cpp_node_path) and 
                os.path.exists(self.cpp_link_path))

    def save_metadata(self, info: Dict[str, Any]) -> None:
        """
        Serializes lightweight metadata (labels, masks, feature dims).
        Avoids saving heavy edge indices.
        """
        os.makedirs(self.root_dir, exist_ok=True)
        torch.save(info, self.meta_path)

    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Loads metadata without instantiating the full graph."""
        if not os.path.exists(self.meta_path):
            return None
        return torch.load(self.meta_path)