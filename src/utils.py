"""
Utility functions for metapath generation, file management, and schema mapping.
"""
import os
import random
from typing import List, Tuple
from torch_geometric.data import HeteroData


def generate_random_metapath(g_hetero: HeteroData, 
                             start_ntype: str, 
                             length: int) -> List[Tuple[str, str, str]]:
    """
    Randomly walks the graph schema to generate a valid cyclic metapath.
    Returns a sequence of edge types that starts and ends at start_ntype.
    """
    # Map node types to outgoing edge types for schema traversal
    schema_adj = {}
    for edge_type in g_hetero.edge_types:
        src, rel, dst = edge_type
        if src not in schema_adj:
            schema_adj[src] = []
        schema_adj[src].append(edge_type)
    
    if start_ntype not in schema_adj:
        raise ValueError(f"Start node type '{start_ntype}' has no outgoing edges")
    
    print(f"[Utils] Sampling cyclic metapath (len={length}, root='{start_ntype}')...")
    
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
            print(f"   Generated: {path_str}")
            return metapath
    
    # Fallback to a minimal length-2 cycle if the requested length search fails
    print(f"   [Warning] Length-{length} path search failed; attempting length-2 fallback.")
    
    if length >= 2 and start_ntype in schema_adj:
        first_edge = schema_adj[start_ntype][0]
        mid_type = first_edge[2]
        
        if mid_type in schema_adj:
            for back_edge in schema_adj[mid_type]:
                if back_edge[2] == start_ntype:
                    print(f"   Using minimal fallback path.")
                    return [first_edge, back_edge]
    
    raise ValueError(f"Schema constraints prevent cyclic metapath from '{start_ntype}'.")


def get_metapath_suffix(metapath: List[Tuple[str, str, str]]) -> str:
    """
    Creates a shorthand string (e.g., 'a_p_a') for file naming based on node type initials.
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
    Utility for resolving naming conflicts between rule mining outputs (AnyBURL) 
    and graph loader naming conventions (rev_X vs inverse_X).
    """
    @staticmethod
    def match(relation_str: str, g_hetero) -> Tuple[str, str, str]:
        # Normalize input to ignore syntactic differences like prefixes or direction arrows
        clean_target = relation_str.replace("inverse_", "").replace("rev_", "").replace("_to_", "").replace(">", "").replace("<", "").lower()
        
        candidates = []
        
        for src, rel, dst in g_hetero.edge_types:
            clean_rel = rel.replace("inverse_", "").replace("rev_", "").replace("_to_", "").replace(">", "").replace("<", "").lower()
            
            # Direct match on normalized strings
            if clean_rel == clean_target:
                candidates.append((src, rel, dst))
                continue
                
            # Directional heuristic for relation names containing node type identifiers
            if f"{dst}{src}" in clean_target or f"{src}{dst}" in clean_target:
                 if clean_rel in clean_target:
                     candidates.append((src, rel, dst))

        if not candidates:
            print(f"      [Matcher] No match for '{relation_str}'. Falling back to generic node type.")
            return ('node', relation_str, 'node')
        
        # Priority logic: align AnyBURL 'inverse' semantics with graph 'rev' attributes
        if "inverse" in relation_str or "rev" in relation_str:
            for c in candidates:
                if "rev" in c[1] or "inverse" in c[1]:
                    return c
        
        return candidates[0]