"""
Disk-based sampling strategies to prevent OOM on massive datasets.
"""
import os
import shutil
import random
from typing import Dict, Any, Set, List, Tuple

class HNESnowballDiskSampler:
    """
    Performs Snowball sampling and exports ONLY traversed edges.
    """
    
    def sample(self, source_dir: str, target_dir: str, config: Dict[str, Any]) -> None:
        seeds_count = config.get('seeds', 1000)
        hops = config.get('hops', 2)
        
        print(f"[DiskSampler] Sampling {source_dir} -> {target_dir}")
        
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir)
        
        # 1. Load Structure
        adj: Dict[int, List[int]] = {}
        all_nodes: List[int] = []
        link_file = os.path.join(source_dir, "link.dat")
        
        print("  1. Reading graph structure...")
        try:
            with open(link_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i % 5000000 == 0 and i > 0: print(f"     Line {i}...", end='\r')
                    parts = line.strip().split('\t')
                    if len(parts) < 2: continue
                    u, v = int(parts[0]), int(parts[1])
                    
                    if u not in adj: adj[u] = []
                    if v not in adj: adj[v] = []
                    adj[u].append(v)
                    adj[v].append(u)
                    
                    if len(all_nodes) < seeds_count * 10: all_nodes.append(u)
                    elif random.random() < 0.001: all_nodes[random.randint(0, len(all_nodes)-1)] = u
        except Exception as e:
            print(f"\n[Error] {e}")
            raise

        # 2. BFS with Edge Tracking
        current_frontier = set(random.sample(all_nodes, min(len(all_nodes), seeds_count)))
        sampled_nodes: Set[int] = set(current_frontier)
        
        # KEY CHANGE: Track valid edges to prevent density explosion
        # We store edges as sorted tuples to handle undirected matching
        valid_edges: Set[Tuple[int, int]] = set()
        
        print(f"\n  2. BFS (Hops={hops})...")
        for h in range(hops):
            print(f"     Hop {h+1} (Frontier: {len(current_frontier)})...")
            next_frontier = set()
            
            for u in current_frontier:
                if u in adj:
                    neighbors = adj[u]
                    
                    # 🛑 CAP NEIGHBORS (Strict Limit)
                    if len(neighbors) > 20: # Reduced from 50 to 20 for safety
                        neighbors = random.sample(neighbors, 20)
                        
                    for v in neighbors:
                        # Track this specific edge as valid
                        edge = tuple(sorted((u, v)))
                        valid_edges.add(edge)
                        
                        if v not in sampled_nodes:
                            sampled_nodes.add(v)
                            next_frontier.add(v)
            current_frontier = next_frontier
        
        print(f"  3. Final: {len(sampled_nodes)} nodes, {len(valid_edges)} edges.")
        print("     Exporting filtered dataset...")
        
        # 3. Strict Export
        with open(link_file, 'r', encoding='utf-8') as fin, \
             open(os.path.join(target_dir, "link.dat"), 'w', encoding='utf-8') as fout:
            
            for line in fin:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        # Check if this edge was explicitly traversed
                        edge = tuple(sorted((u, v)))
                        
                        # Only write if it's in our valid list
                        if edge in valid_edges:
                            fout.write(line)
                    except ValueError: pass

        # Filter node.dat
        node_file = os.path.join(source_dir, "node.dat")
        if os.path.exists(node_file):
            with open(node_file, 'r', encoding='utf-8') as fin, \
                 open(os.path.join(target_dir, "node.dat"), 'w', encoding='utf-8') as fout:
                for line in fin:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        try:
                            if int(parts[0]) in sampled_nodes:
                                fout.write(line)
                        except ValueError: pass

        meta_file = os.path.join(source_dir, "meta.dat")
        if os.path.exists(meta_file):
            shutil.copy(meta_file, os.path.join(target_dir, "meta.dat"))
            
        print(f"✅ Export complete.")