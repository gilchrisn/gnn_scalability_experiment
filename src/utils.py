import os
import random

def get_metapath_suffix(metapath):
    """
    Generates a short, readable suffix from a metapath.
    """
    if not metapath:
        return "no_path"
        
    try:
        suffix_parts = [metapath[0][0][0]]
        for _, _, dst_ntype in metapath:
            suffix_parts.append(dst_ntype[0])
            
        return "_".join(suffix_parts)
    except (TypeError, IndexError):
        print(f"Warning: Could not parse metapath {metapath}. Using 'custom_path'.")
        return "custom_path"

def get_model_checkpoint_path(model_dir: str, model_name: str, metapath: list) -> str:
    path_suffix = get_metapath_suffix(metapath)
    filename = f"{model_name.lower()}_{path_suffix}.ckpt"
    return os.path.join(model_dir, filename)

def generate_random_metapath(g_hetero, start_ntype, length):
    """
    Dynamically discovers the schema and generates a valid random meta-path.
    CRITICAL CHANGE: Ensures the path ends at the same node type it started with.
    """
    # 1. Build the Schema Graph
    schema_adj = {}
    for edge_type in g_hetero.edge_types:
        src, rel, dst = edge_type
        if src not in schema_adj:
            schema_adj[src] = []
        schema_adj[src].append(edge_type)

    print(f"[Utils] Generating valid metapath of len {length} starting at '{start_ntype}'...")
    
    max_attempts = 50
    for attempt in range(max_attempts):
        current_type = start_ntype
        metapath = []
        valid_path = True
        
        for i in range(length):
            if current_type not in schema_adj:
                valid_path = False
                break
                
            valid_edges = schema_adj[current_type]
            
            # Heuristic: If it's the last step, try to pick an edge that goes back to start
            if i == length - 1:
                candidates = [e for e in valid_edges if e[2] == start_ntype]
                if not candidates:
                    valid_path = False # Can't close the loop
                    break
                chosen_edge = random.choice(candidates)
            else:
                chosen_edge = random.choice(valid_edges)
            
            metapath.append(chosen_edge)
            current_type = chosen_edge[2]
            
        # Success Check: Did we finish validly and return to start?
        if valid_path and current_type == start_ntype:
             path_str = " -> ".join([t[1] for t in metapath])
             print(f"   Found valid cyclic path: {path_str}")
             return metapath

    # Fallback: If random walks fail (e.g. odd length on bipartite), force a simple length-2 path
    print(f"   [Warning] Could not find random cyclic path of length {length}. Falling back to length 2.")
    if start_ntype in schema_adj:
        first_edge = schema_adj[start_ntype][0]
        mid_type = first_edge[2]
        # Find return edge
        if mid_type in schema_adj:
            for back_edge in schema_adj[mid_type]:
                if back_edge[2] == start_ntype:
                    return [first_edge, back_edge]

    raise ValueError(f"Could not generate a valid cyclic metapath for {start_ntype}")