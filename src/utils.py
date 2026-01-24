"""
Utility module for graph operations and file management.

Contains pure functions for:
1. Metapath generation and formatting
2. File path management
3. Schema translation (Rule String -> Graph Edge Tuple)
"""
import os
import random
from typing import List, Tuple, Optional
from torch_geometric.data import HeteroData


def generate_random_metapath(g_hetero: HeteroData, 
                             start_ntype: str, 
                             length: int) -> List[Tuple[str, str, str]]:
    """
    Randomly walks the graph schema to generate a valid cyclic metapath.
    
    Args:
        g_hetero: The graph containing the schema.
        start_ntype: The starting node type (e.g., 'paper').
        length: The number of hops in the random walk.
        
    Returns:
        A list of edge tuples representing the path.
        
    Raises:
        ValueError: If a valid path cannot be found.
    """
    # Map node types to outgoing edge types for schema traversal
    schema_adj = {}
    for edge_type in g_hetero.edge_types:
        src, _, _ = edge_type
        if src not in schema_adj:
            schema_adj[src] = []
        schema_adj[src].append(edge_type)
    
    if start_ntype not in schema_adj:
        raise ValueError(f"Start node type '{start_ntype}' has no outgoing edges")
    
    print(f"[Utils] Sampling cyclic metapath (len={length}, root='{start_ntype}')...")
    
    max_attempts = 100
    for _ in range(max_attempts):
        current_type = start_ntype
        metapath = []
        valid_path = True
        
        for i in range(length):
            if current_type not in schema_adj:
                valid_path = False
                break
            
            valid_edges = schema_adj[current_type]
            
            # Constraints for the final step to enforce the cycle
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
        
        if valid_path and current_type == start_ntype:
            path_str = " -> ".join([rel for _, rel, _ in metapath])
            print(f"    Generated: {path_str}")
            return metapath
    
    # Fallback to a minimal length-2 cycle if the requested length search fails
    print(f"    [Warning] Length-{length} path search failed; attempting length-2 fallback.")
    
    if length >= 2 and start_ntype in schema_adj:
        first_edge = schema_adj[start_ntype][0]
        mid_type = first_edge[2]
        
        if mid_type in schema_adj:
            for back_edge in schema_adj[mid_type]:
                if back_edge[2] == start_ntype:
                    print(f"    Using minimal fallback path.")
                    return [first_edge, back_edge]
    
    raise ValueError(f"Schema constraints prevent cyclic metapath from '{start_ntype}'.")


def get_metapath_suffix(metapath: List[Tuple[str, str, str]]) -> str:
    """
    Creates a shorthand string (e.g., 'a_p_a') for file naming.
    """
    if not metapath:
        return "no_path"
    
    try:
        # Extract initial characters from the node sequence
        suffix_parts = [metapath[0][0][0]]
        for _, _, dst_ntype in metapath:
            suffix_parts.append(dst_ntype[0])
        
        return "_".join(suffix_parts).lower()
    except (TypeError, IndexError):
        return "custom_path"


def get_model_checkpoint_path(model_dir: str, 
                              model_name: str, 
                              metapath: List[Tuple[str, str, str]]) -> str:
    """Standardized naming convention for model persistence."""
    path_suffix = get_metapath_suffix(metapath)
    filename = f"{model_name.lower()}_{path_suffix}.ckpt"
    return os.path.join(model_dir, filename)


def ensure_dir(directory: str) -> None:
    """Wraps os.makedirs with exist_ok for safe directory initialization."""
    os.makedirs(directory, exist_ok=True)


def get_edge_type_str(edge_type: Tuple[str, str, str]) -> str:
    """ASCII representation of an edge type triple."""
    src, rel, dst = edge_type
    return f"{src} -[{rel}]-> {dst}"


def format_metapath(metapath: List[Tuple[str, str, str]]) -> str:
    """Readable string representation of the full metapath sequence."""
    if not metapath:
        return "Empty metapath"
    
    parts = [metapath[0][0]]
    for _, rel, dst in metapath:
        parts.append(f"-[{rel}]->")
        parts.append(dst)
    
    return " ".join(parts)


class SchemaMatcher:
    """
    Service for resolving string-based rule definitions to graph schema tuples.
    
    Handles the impedance mismatch between AnyBURL's rule output (e.g., "paper_to_author")
    and PyG's internal edge storage (e.g., ('paper', 'to', 'author')).
    """
    
    @staticmethod
    def match(relation_str: str, g_hetero: HeteroData) -> Tuple[str, str, str]:
        """
        Finds the exact edge type tuple corresponding to a relation string.
        
        Prioritizes structural matching (Source/Dest types) over name matching,
        as datasets like HGB normalize all relation names to 'to'.
        
        Args:
            relation_str: The raw string from a rule file (e.g., 'paper_to_author').
            g_hetero: The graph containing the schema definitions.
            
        Returns:
            The matching (Source, Relation, Destination) tuple.
            Defaults to ('node', relation_str, 'node') if no match found (fallback).
        """
        # 1. Clean Pre-processing
        # Remove AnyBURL prefixes but keep structural delimiters like '_to_'
        clean_input = relation_str.lower().replace("inverse_", "").replace("rev_", "")
        
        # Strategy A: Structure-based Matching (Primary)
        # HGB datasets use "src_to_dst" naming convention which is robust
        if "_to_" in clean_input:
            parts = clean_input.split("_to_")
            # We look for the last occurrence of splitting to handle potential type names with underscores
            # But a simple split usually suffices for standard benchmarks
            if len(parts) >= 2:
                # Assume format is roughly "typeA_to_typeB"
                req_src = parts[0]
                req_dst = parts[-1] 
                
                for src, rel, dst in g_hetero.edge_types:
                    if src.lower() == req_src and dst.lower() == req_dst:
                        return (src, rel, dst)

        # Strategy B: Relation Name Matching (Secondary)
        # Check for exact relation matches (e.g., "cites", "writes")
        candidates = []
        for src, rel, dst in g_hetero.edge_types:
            # Clean graph relation name similarly
            graph_rel_clean = rel.lower().replace("rev_", "").replace("inverse_", "")
            
            if graph_rel_clean == clean_input:
                candidates.append((src, rel, dst))
                
            # Also check if the cleaned input matches the explicit HGB style "src_to_dst"
            # created by our loaders
            if f"{src}_to_{dst}".lower() == clean_input:
                candidates.append((src, rel, dst))

        if candidates:
            # Tie-breaking: If input had "rev" or "inverse", prefer reverse edges
            if "inverse" in relation_str.lower() or "rev" in relation_str.lower():
                for c in candidates:
                    if "rev" in c[1].lower() or "inverse" in c[1].lower():
                        return c
            # Default to first match
            return candidates[0]

        # Strategy C: Fallback
        # If absolutely nothing matches, return placeholder to avoid crash, 
        # though this likely indicates a schema drift.
        print(f"      [Matcher] WARNING: No match found for '{relation_str}'.")
        return ('node', relation_str, 'node')