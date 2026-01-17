
import sys
import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import degree
from scipy.stats import entropy

# --- PATH SETUP ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.config import config
from src.data import DatasetFactory
from src.backend import BackendFactory
from src.models import get_model
from src.utils import SchemaMatcher

# --- CONFIGURATION ---
OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "paper_plots_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Full Dataset List
DATASETS = {
    "HGB_DBLP": [
        "author_to_paper,paper_to_author",
        "author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"
    ],
    "HGB_ACM": [
        "paper_to_author,author_to_paper",
        "paper_to_subject,subject_to_paper"
    ],
    "HGB_IMDB": [
        "movie_to_actor,actor_to_movie",
        "movie_to_director,director_to_movie"
    ],
    "PyG_IMDB": [
        "movie_to_actor,actor_to_movie",
        "movie_to_director,director_to_movie"
    ],
}

# Force CPU for analysis stability
config.DEVICE = torch.device('cpu')

def get_short_label(path_str):
    """
    Generates a unique, short acronym for the metapath (e.g., A-P-A).
    """
    try:
        parts = path_str.split(',')
        # Start node
        acronym = [parts[0].split('_to_')[0][0].upper()]
        
        for p in parts:
            # Target node of each hop
            dst = p.split('_to_')[-1]
            acronym.append(dst[0].upper())
            
        return "-".join(acronym)
    except:
        return path_str[:15]

def get_safe_filename(dataset, metapath_str, prefix):
    """Creates a collision-free filename."""
    # Replace separators to prevent directory confusion
    sanitized_path = metapath_str.replace(',', '_').replace(' ', '').replace('->', '_to_')
    # Truncate if insanely long, but keep suffix hash or unique end
    if len(sanitized_path) > 100:
        sanitized_path = sanitized_path[:50] + "..." + sanitized_path[-10:]
    return f"{prefix}_{dataset}_{sanitized_path}.png"

def get_model_wrapper(dataset, model_name, info):
    """Safe model loader helper"""
    model_path = config.get_model_path(dataset, model_name)
    if not os.path.exists(model_path): return None, None
    
    import json
    map_file = model_path.replace('.pt', '_mapper.json')
    if not os.path.exists(map_file): return None, None

    with open(map_file, 'r') as f:
        mapper_cfg = json.load(f)
    
    model = get_model(model_name, mapper_cfg['global_max_dim'], info['num_classes'], config.HIDDEN_DIM)
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except:
        return None, None
        
    model.eval()
    return model, mapper_cfg

# ==========================================
# 1. STRUCTURAL PROOF (Homophily & Sparsity)
# ==========================================
def analyze_structure():
    print(f"\n--- 1. Generating Structural Proof Plots (Homophily/Sparsity) ---")
    stats = []
    
    backend = BackendFactory.create('python', device=torch.device('cpu'))

    for ds_name, paths in DATASETS.items():
        for path_str in paths:
            try:
                cfg = config.get_dataset_config(ds_name)
                g_hetero, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
                
                path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in path_str.split(',')]
                backend.initialize(g_hetero, path_list, info)
                g_exact = backend.materialize_exact()
                
                # A. Degree Sparsity
                row, col = g_exact.edge_index
                degrees = torch.bincount(row, minlength=g_exact.num_nodes).float()
                pct_sparse = (degrees <= 32).float().mean().item() * 100
                
                # B. Homophily
                labels = info['labels']
                valid_mask = (labels[row] >= 0) & (labels[col] >= 0)
                if valid_mask.sum() > 0:
                    agreement = (labels[row[valid_mask]] == labels[col[valid_mask]]).float().mean().item()
                else:
                    agreement = 0.0

                meta_acronym = get_short_label(path_str)
                label = f"{ds_name}\n({meta_acronym})"
                
                stats.append({
                    "Label": label, 
                    "Dataset": ds_name,
                    "Homophily": agreement, 
                    "Sparsity": pct_sparse
                })
                print(f"   Processed {ds_name} [{meta_acronym}]: H={agreement:.2f}, S={pct_sparse:.1f}%")
                
                backend.cleanup()
            except Exception as e:
                print(f"   Skipping structure analysis for {ds_name}: {e}")

    if not stats: return

    # Generate ONE summary plot for the paper
    df = pd.DataFrame(stats)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Homophily
    sns.barplot(data=df, x="Label", y="Homophily", ax=ax1, palette="viridis", edgecolor="black")
    ax1.set_title("Structural Redundancy (Neighborhood Homophily)", fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    for i, v in enumerate(df["Homophily"]):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold', fontsize=9)

    # Plot 2: Sparsity
    sns.barplot(data=df, x="Label", y="Sparsity", ax=ax2, palette="magma", edgecolor="black")
    ax2.set_title("Graph Sparsity (% Nodes with Degree <= 32)", fontweight='bold')
    ax2.set_ylabel("% Nodes Fully Captured")
    ax2.set_ylim(0, 110)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    for i, v in enumerate(df["Sparsity"]):
        ax2.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "proof_of_mechanism_summary.png")
    plt.savefig(save_path, dpi=300)
    print(f"   -> Saved summary plot: {save_path}")
    plt.close()

# ==========================================
# 2. SEMANTIC DISTORTION (Exact vs KMV vs Random)
# ==========================================
def plot_semantic_distortion(dataset, metapath, k=4):
    print(f" [Distortion] Processing {dataset}...")
    try:
        cfg = config.get_dataset_config(dataset)
        g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        
        backend = BackendFactory.create('python', device='cpu')
        backend.initialize(g, [SchemaMatcher.match(s, g) for s in metapath.split(',')], info)
        
        # 1. Materialize All 3 Graphs
        g_exact = backend.materialize_exact()
        g_kmv = backend.materialize_kmv(k=k)
        g_rand = backend.materialize_random(k=k)
        
        # 2. Helper to get distribution
        num_classes = info['num_classes']
        def get_dist(graph):
            src, dst = graph.edge_index
            y = info['labels'][src]
            # One-hot aggregation of neighbors
            valid = y >= 0
            oh = F.one_hot(y[valid], num_classes).float()
            out = torch.zeros(graph.num_nodes, num_classes)
            out.index_add_(0, dst[valid], oh)
            # Normalize to probability distribution
            return out / (out.sum(1, keepdim=True) + 1e-6)

        P_exact = get_dist(g_exact)
        Q_kmv = get_dist(g_kmv)
        Q_rand = get_dist(g_rand)
        
        # 3. Calculate KL Divergence
        # KL(P || Q) = sum(P * log(P/Q))
        epsilon = 1e-8
        kl_kmv = torch.sum(P_exact * torch.log((P_exact+epsilon)/(Q_kmv+epsilon)), dim=1).numpy()
        kl_rand = torch.sum(P_exact * torch.log((P_exact+epsilon)/(Q_rand+epsilon)), dim=1).numpy()
        
        # 4. Stratify by Degree
        row, _ = g_exact.edge_index
        degs = degree(row, num_nodes=g_exact.num_nodes).numpy()
        
        buckets = [0, 5, 20, 100, 100000]
        labels = ["Low (0-5)", "Med (6-20)", "High (21-100)", "Super-Hub"]
        
        bar_data = []
        for i in range(len(buckets)-1):
            mask = (degs > buckets[i]) & (degs <= buckets[i+1])
            if mask.sum() > 0:
                # Add KMV score
                bar_data.append({
                    "Bucket": labels[i], 
                    "Method": f"KMV (k={k})", 
                    "KL Divergence": kl_kmv[mask].mean()
                })
                # Add Random score
                bar_data.append({
                    "Bucket": labels[i], 
                    "Method": f"Random (k={k})", 
                    "KL Divergence": kl_rand[mask].mean()
                })
        
        # 5. Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=pd.DataFrame(bar_data), 
            x="Bucket", 
            y="KL Divergence", 
            hue="Method",
            palette={'KMV (k=4)': '#2980b9', 'Random (k=4)': '#e74c3c'}
        )
        
        meta_acronym = get_short_label(metapath)
        plt.title(f"Semantic Distortion: KL Divergence (Lower is Better)\n{dataset} | {meta_acronym} | K={k}", fontsize=13)
        plt.ylabel("KL Divergence (Semantic Error)")
        plt.xlabel("Node Degree Bucket")
        
        save_path = os.path.join(OUTPUT_DIR, get_safe_filename(dataset, metapath, "distortion"))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"   -> Saved: {save_path}")

    except Exception as e: print(f"Skipping Distortion {dataset}: {e}")

# ==========================================
# 3. EFFECTIVE RANK (Dimensionality)
# ==========================================
def plot_effective_rank(dataset, metapath):
    print(f" [Rank] Processing {dataset}...")
    try:
        cfg = config.get_dataset_config(dataset)
        g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        
        backend = BackendFactory.create('python', device='cpu')
        backend.initialize(g, [SchemaMatcher.match(s, g) for s in metapath.split(',')], info)
        g_exact = backend.materialize_exact()
        
        row, col = g_exact.edge_index
        degs = degree(row, num_nodes=g_exact.num_nodes)
        _, hubs = torch.topk(degs, 100)
        
        ranks = []
        feats = info['features']
        
        for h in hubs:
            nb = col[row == h]
            if len(nb) < 2: continue
            X = feats[nb]
            try:
                # Stable Rank = ||X||_F^2 / ||X||_2^2
                # Use float64 for SVD stability
                X_centered = (X - X.mean(0)).double()
                _, S, _ = torch.linalg.svd(X_centered, full_matrices=False)
                if S[0] > 0:
                    rank = (S**2).sum() / (S[0]**2)
                    ranks.append(rank.item())
            except: pass
                
        return ranks
    except: return []

def run_rank_summary():
    print("\n--- 3. Generating Effective Rank Plot ---")
    all_ranks = []
    
    for name, paths in DATASETS.items():
        for path in paths:
            r = plot_effective_rank(name, path)
            # Use short label to match structure plot
            short_label = f"{name}\n({get_short_label(path)})"
            for val in r:
                all_ranks.append({"Scenario": short_label, "Rank": val})
            
    if not all_ranks: return
    
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=pd.DataFrame(all_ranks), x="Scenario", y="Rank", palette="viridis", hue="Scenario", legend=False)
    plt.axhline(y=4, color='black', linestyle='--', label="K=4 Sampling Limit")
    plt.title("Why K=4 Works: Neighborhood Feature Dimensionality (Top-100 Hubs)")
    plt.ylabel("Stable Rank (Effective Dimension)")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "effective_rank_summary.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved summary: {save_path}")

# ==========================================
# 4. ENTROPY SHIFT (Attention Entropy)
# ==========================================
def plot_entropy(dataset, metapath, k=4):
    print(f" [Entropy] Processing {dataset}...")
    
    cfg = config.get_dataset_config(dataset)
    g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    
    model, m_cfg = get_model_wrapper(dataset, "GAT", info)
    if not model:
        print("   -> No trained GAT model found. Skipping.")
        return

    backend = BackendFactory.create('python', device='cpu')
    path_list = [SchemaMatcher.match(s.strip(), g) for s in metapath.split(',')]
    backend.initialize(g, path_list, info)
    
    # Materialize Exact, KMV, AND Random
    g_exact = backend.materialize_exact()
    g_kmv = backend.materialize_kmv(k=k)
    g_rand = backend.materialize_random(k=k)
    
    results = []
    
    def compute(graph, label):
        target_dim = m_cfg['global_max_dim']
        x = graph.x
        if x.size(1) < target_dim:
            x = F.pad(x, (0, target_dim - x.size(1)))
            
        with torch.no_grad():
            # Hack: Directly call conv1 to get attention weights
            _, (att_edge, att_val) = model.conv1(x, graph.edge_index, return_attention_weights=True)
            
            # att_val shape: [num_edges, num_heads]
            att_mean = att_val.mean(dim=1)
            
            # Entropy: -sum(p * log p)
            eps = 1e-10
            ent = -att_mean * torch.log(att_mean + eps)
            
            # Aggregate per node
            from torch_geometric.utils import scatter
            node_ent = scatter(ent, att_edge[1], dim=0, dim_size=graph.num_nodes, reduce='sum')
            
            mask = info['masks']['test']
            valid_ents = node_ent[mask].numpy()
            
            for v in valid_ents:
                results.append({"Condition": label, "Entropy": v})

    try:
        compute(g_exact, "Exact")
        compute(g_kmv, f"KMV (K={k})")
        compute(g_rand, f"Random (K={k})")
        
        plt.figure(figsize=(9, 6))
        # Ensure palette covers all 3 conditions
        palette = {
            "Exact": "#34495e",          # Dark Blue/Grey
            f"KMV (K={k})": "#2980b9",   # Bright Blue
            f"Random (K={k})": "#e74c3c" # Red
        }
        
        sns.kdeplot(
            data=pd.DataFrame(results), 
            x="Entropy", 
            hue="Condition", 
            fill=True, 
            common_norm=False, 
            palette=palette,
            alpha=0.4,
            linewidth=2
        )
        
        meta_acronym = get_short_label(metapath)
        plt.title(f"Attention Entropy Shift (GAT Layer 1)\n{dataset} | {meta_acronym} | K={k}")
        plt.xlabel("Attention Entropy (Higher = More Uniform/Confused Attention)")
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(OUTPUT_DIR, get_safe_filename(dataset, metapath, "entropy"))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"   -> Saved: {save_path}")
        
    except Exception as e:
        print(f"   -> Entropy calculation failed: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"--- GENERATING ALL PAPER PLOTS ---")
    print(f"Output Directory: {OUTPUT_DIR}\n")
    
    # 1. Structural Proof (Homophily & Sparsity)
    analyze_structure()
    
    # 2. Per-Dataset Metrics (Iterate EVERY path)
    print("\n--- 2. Generating Per-Dataset Metrics (Distortion & Entropy) ---")
    for ds_name, paths in DATASETS.items():
        for path in paths:
            # We strictly loop through every single metapath defined in config
            plot_semantic_distortion(ds_name, path, k=4)
            plot_entropy(ds_name, path, k=4)
            
    # 3. Aggregate Rank
    run_rank_summary()
    
    print(f"\n[Done] All plots generated in: {OUTPUT_DIR}")