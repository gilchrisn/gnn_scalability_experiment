"""
Utility functions for metapath generation, file management, and helpers.
"""
import os
import random
from typing import List, Tuple
from torch_geometric.data import HeteroData


def generate_random_metapath(g_hetero: HeteroData, 
                             start_ntype: str, 
                             length: int) -> List[Tuple[str, str, str]]:
    """
    Dynamically discovers the schema and generates a valid random cyclic meta-path.
    Ensures the path ends at the same node type it started with.
    
    Args:
        g_hetero: Heterogeneous graph
        start_ntype: Starting node type
        length: Desired metapath length
        
    Returns:
        List of edge type tuples forming a cyclic metapath
        
    Raises:
        ValueError: If no valid cyclic path can be found
    """
    # Build schema adjacency list
    schema_adj = {}
    for edge_type in g_hetero.edge_types:
        src, rel, dst = edge_type
        if src not in schema_adj:
            schema_adj[src] = []
        schema_adj[src].append(edge_type)
    
    if start_ntype not in schema_adj:
        raise ValueError(f"Start node type '{start_ntype}' has no outgoing edges")
    
    print(f"[Utils] Generating cyclic metapath (length={length}, start='{start_ntype}')...")
    
    max_attempts = 100
    for attempt in range(max_attempts):
        current_type = start_ntype
        metapath = []
        valid_path = True
        
        for i in range(length):
            if current_type not in schema_adj:
                valid_path = False
                break
            
            valid_edges = schema_adj[current_type]
            
            # For the last step, try to return to start
            if i == length - 1:
                candidates = [e for e in valid_edges if e[2] == start_ntype]
                if not candidates:
                    valid_path = False
                    break
                chosen_edge = random.choice(candidates)
            else:
                chosen_edge = random.choice(valid_edges)
            
            metapath.append(chosen_edge)
            current_type = chosen_edge[2]
        
        # Success: valid path that returns to start
        if valid_path and current_type == start_ntype:
            path_str = " -> ".join([rel for _, rel, _ in metapath])
            print(f"   ✓ Found: {path_str}")
            return metapath
    
    # Fallback: try to construct simple length-2 cycle
    print(f"   [Warning] Could not find random path of length {length}")
    
    if length >= 2 and start_ntype in schema_adj:
        first_edge = schema_adj[start_ntype][0]
        mid_type = first_edge[2]
        
        if mid_type in schema_adj:
            for back_edge in schema_adj[mid_type]:
                if back_edge[2] == start_ntype:
                    fallback = [first_edge, back_edge]
                    print(f"   ✓ Using fallback length-2 path")
                    return fallback
    
    raise ValueError(
        f"Could not generate cyclic metapath of length {length} "
        f"starting from '{start_ntype}'. The graph schema may not support it."
    )


def get_metapath_suffix(metapath: List[Tuple[str, str, str]]) -> str:
    """
    Generates a short, readable suffix from a metapath for file naming.
    
    Args:
        metapath: List of edge type tuples
        
    Returns:
        String suffix suitable for filenames
        
    Example:
        >>> metapath = [('author', 'to', 'paper'), ('paper', 'to', 'author')]
        >>> get_metapath_suffix(metapath)
        'a_p_a'
    """
    if not metapath:
        return "no_path"
    
    try:
        # Use first letter of each node type
        suffix_parts = [metapath[0][0][0]]  # First source type
        for _, _, dst_ntype in metapath:
            suffix_parts.append(dst_ntype[0])
        
        return "_".join(suffix_parts).lower()
    except (TypeError, IndexError):
        print(f"[Warning] Could not parse metapath {metapath}. Using 'custom_path'.")
        return "custom_path"


def get_model_checkpoint_path(model_dir: str, 
                              model_name: str, 
                              metapath: List[Tuple[str, str, str]]) -> str:
    """
    Generates standardized checkpoint path for a model.
    
    Args:
        model_dir: Directory for model checkpoints
        model_name: Name of the model (e.g., 'GCN', 'GAT')
        metapath: Metapath used for training
        
    Returns:
        Full path to checkpoint file
    """
    path_suffix = get_metapath_suffix(metapath)
    filename = f"{model_name.lower()}_{path_suffix}.ckpt"
    return os.path.join(model_dir, filename)


def ensure_dir(directory: str) -> None:
    """
    Ensures a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def get_edge_type_str(edge_type: Tuple[str, str, str]) -> str:
    """
    Converts edge type tuple to readable string.
    
    Args:
        edge_type: Tuple of (src_type, relation, dst_type)
        
    Returns:
        Formatted string
        
    Example:
        >>> get_edge_type_str(('author', 'writes', 'paper'))
        'author -[writes]-> paper'
    """
    src, rel, dst = edge_type
    return f"{src} -[{rel}]-> {dst}"


def format_metapath(metapath: List[Tuple[str, str, str]]) -> str:
    """
    Formats metapath for pretty printing.
    
    Args:
        metapath: List of edge type tuples
        
    Returns:
        Formatted string representation
    """
    if not metapath:
        return "Empty metapath"
    
    parts = [metapath[0][0]]  # Start node
    for _, rel, dst in metapath:
        parts.append(f"-[{rel}]->")
        parts.append(dst)
    
    return " ".join(parts)


class SchemaMatcher:
    """
    Robustly maps messy rule strings to actual graph edge types.
    Solves the mismatch between AnyBURL names (inverse_X) and Loader names (rev_X).
    """
    @staticmethod
    def match(relation_str: str, g_hetero) -> Tuple[str, str, str]:
        # 1. Clean the input string (remove known prefixes/separators)
        # e.g. "inverse_rev_>actorh" -> "actorh"
        # e.g. "paper_to_term" -> "papertoterm"
        clean_target = relation_str.replace("inverse_", "").replace("rev_", "").replace("_to_", "").replace(">", "").replace("<", "").lower()
        
        candidates = []
        
        # 2. Iterate over ALL valid edge types in the graph
        for src, rel, dst in g_hetero.edge_types:
            # Clean the graph's relation name similarly
            clean_rel = rel.replace("inverse_", "").replace("rev_", "").replace("_to_", "").replace(">", "").replace("<", "").lower()
            
            # Check 1: Exact normalized match
            if clean_rel == clean_target:
                candidates.append((src, rel, dst))
                continue
                
            # Check 2: Reverse match (if target is "papertoterm" but graph has "termtopaper")
            if f"{dst}{src}" in clean_target or f"{src}{dst}" in clean_target:
                 # This is a heuristic for when the relation name implies direction
                 if clean_rel in clean_target:
                     candidates.append((src, rel, dst))

        # 3. Selection Logic
        if not candidates:
            # Final Fallback: If absolutely nothing matches, return generic (will fail in C++, but we tried)
            print(f"      [Matcher] No schema match for '{relation_str}'. Defaulting to 'node'.")
            return ('node', relation_str, 'node')
        
        # If AnyBURL said "inverse", we prefer the edge type starting with "rev_" or "inverse_"
        if "inverse" in relation_str or "rev" in relation_str:
            for c in candidates:
                if "rev" in c[1] or "inverse" in c[1]:
                    return c
        
        # Otherwise return the first exact match (usually the forward edge)
        return candidates[0]