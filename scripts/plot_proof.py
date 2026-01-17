import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from src.config import config
from src.data import DatasetFactory
from src.backend import BackendFactory
from src.utils import SchemaMatcher
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# The 6 scenarios you just ran (based on your screenshots)
SCENARIOS = [
    ("HGB_DBLP", "author_to_paper,paper_to_author"),
    ("HGB_DBLP", "author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"),
    ("HGB_ACM",  "paper_to_author,author_to_paper"),
    ("HGB_ACM",  "paper_to_subject,subject_to_paper"),
    ("HGB_IMDB", "movie_to_actor,actor_to_movie"),
    ("HGB_IMDB", "movie_to_director,director_to_movie"),
    ("PyG_IMDB", "movie_to_actor,actor_to_movie"),
    ("PyG_IMDB", "movie_to_director,director_to_movie")
]

OUTPUT_DIR = "output/final_paper_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_structure():
    stats = []
    
    print(f"{'Dataset':<15} | {'Metapath':<20} | {'Avg Deg':<8} | {'Homophily':<8} | {'% Deg<=32':<10}")
    print("-" * 75)

    backend = BackendFactory.create('python', device=torch.device('cpu'))

    for ds_name, path_str in SCENARIOS:
        # 1. Load Data
        cfg = config.get_dataset_config(ds_name)
        g_hetero, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        
        # 2. Materialize EXACT Graph
        # We need the full adjacency matrix to measure the "True" redundancy
        path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in path_str.split(',')]
        backend.initialize(g_hetero, path_list, info)
        g_exact = backend.materialize_exact()
        
        # 3. Calculate Metrics
        # A. Degree Sparsity
        row, col = g_exact.edge_index
        degrees = torch.bincount(row, minlength=g_exact.num_nodes).float()
        avg_deg = degrees.mean().item()
        pct_sparse = (degrees <= 32).float().mean().item() * 100
        
        # B. Homophily (The "Echo Chamber" Effect)
        labels = info['labels']
        # Mask out missing labels (-1)
        valid_mask = (labels[row] >= 0) & (labels[col] >= 0)
        
        if valid_mask.sum() > 0:
            agreement = (labels[row[valid_mask]] == labels[col[valid_mask]]).float().mean().item()
        else:
            agreement = 0.0

        # Store for plotting
        short_path = path_str.split(',')[1] if ',' in path_str else path_str
        label = f"{ds_name}\n({short_path[:10]}..)"
        
        stats.append({
            "Label": label,
            "Dataset": ds_name,
            "Homophily": agreement,
            "Sparsity": pct_sparse, # % of nodes where K=32 is lossless
            "AvgDegree": avg_deg
        })
        
        print(f"{ds_name:<15} | {path_str[:20]:<20} | {avg_deg:<8.2f} | {agreement:<8.4f} | {pct_sparse:<10.1f}%")
        
        backend.cleanup()

    return pd.DataFrame(stats)

def plot_proof(df):
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Homophily (Why K=4 works)
    sns.barplot(data=df, x="Label", y="Homophily", ax=ax1, palette="viridis", edgecolor="black")
    ax1.set_title("Reason #1: Structural Redundancy\n(Neighborhood Homophily)", fontweight='bold')
    ax1.set_ylabel("Homophily (0-1)")
    ax1.set_ylim(0, 1.1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    
    # Add text labels
    for i, v in enumerate(df["Homophily"]):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

    # Plot 2: Sparsity (Why K=32 works)
    sns.barplot(data=df, x="Label", y="Sparsity", ax=ax2, palette="magma", edgecolor="black")
    ax2.set_title("Reason #2: Graph Sparsity\n(% Nodes with Degree <= 32)", fontweight='bold')
    ax2.set_ylabel("% Nodes Fully Captured")
    ax2.set_ylim(0, 110)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    
    # Add text labels
    for i, v in enumerate(df["Sparsity"]):
        ax2.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "proof_of_mechanism.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Proof plot saved to: {save_path}")

if __name__ == "__main__":
    df = analyze_structure()
    plot_proof(df)