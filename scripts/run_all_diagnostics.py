import sys
import os

# --- 1. PATH FIX (ROOT DIR) ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
# ------------------------------

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from src.config import config
from src.data import DatasetFactory
from src.backend import BackendFactory
from src.models import get_model
from src.utils import SchemaMatcher
from torch_geometric.utils import degree

# --- CONFIGURATION ---
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

config.DEVICE = torch.device('cpu')

def run_stratified_analysis(dataset, metapath, k_val):
    print(f"\n[{dataset}] Failure Diagnostics: {metapath[:50]}... (K={k_val})")
    
    try:
        # 1. Load Data
        try:
            cfg = config.get_dataset_config(dataset)
            g_hetero, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
        except Exception as e:
            print(f"Skipping {dataset}: {e}")
            return

        backend = BackendFactory.create('python', device=config.DEVICE)
        path_list = [SchemaMatcher.match(s.strip(), g_hetero) for s in metapath.split(',')]
        backend.initialize(g_hetero, path_list, info)
        
        # 2. Materialize
        print(" -> Materializing Exact...")
        g_exact = backend.materialize_exact()
        print(f" -> Materializing KMV (K={k_val})...")
        g_kmv = backend.materialize_kmv(k=k_val)
        print(f" -> Materializing Random (K={k_val})...")
        g_rand = backend.materialize_random(k=k_val)
        
        # 3. Calculate Exact Degrees
        row, _ = g_exact.edge_index
        degs = degree(row, num_nodes=g_exact.num_nodes).cpu().numpy()
        
        buckets = [0, 5, 20, 100, 100000]
        bucket_labels = ["Low (0-5)", "Med (6-20)", "High (21-100)", "Super-Hub (>100)"]
        
        results = []
        models = ["GCN", "SAGE", "GAT"]

        for model_name in models:
            # Load Model
            model_path = config.get_model_path(dataset, model_name)
            if not os.path.exists(model_path): continue
                
            import json
            mapper_path = model_path.replace('.pt', '_mapper.json')
            if not os.path.exists(mapper_path): continue
                
            with open(mapper_path, 'r') as f:
                mapper_cfg = json.load(f)
                
            model = get_model(model_name, mapper_cfg['global_max_dim'], info['num_classes'], config.HIDDEN_DIM).to(config.DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            model.eval()
            
            # Inference
            with torch.no_grad():
                def pad(x):
                    target = mapper_cfg['global_max_dim']
                    if x.size(1) == target: return x
                    return torch.nn.functional.pad(x, (0, target - x.size(1)))

                out_exact = model(pad(g_exact.x), g_exact.edge_index)
                out_kmv   = model(pad(g_kmv.x), g_kmv.edge_index)
                out_rand  = model(pad(g_rand.x), g_rand.edge_index)
                
                pred_exact = out_exact.argmax(dim=1)
                pred_kmv   = out_kmv.argmax(dim=1)
                pred_rand  = out_rand.argmax(dim=1)
                
                agree_kmv = (pred_exact == pred_kmv).numpy()
                agree_rand = (pred_exact == pred_rand).numpy()
                
                for i in range(len(buckets)-1):
                    low, high = buckets[i], buckets[i+1]
                    mask = (degs > low) & (degs <= high)
                    test_mask = info['masks']['test'].cpu().numpy()
                    final_mask = mask & test_mask
                    
                    if final_mask.sum() > 0:
                        results.append({
                            "Model": model_name,
                            "Method": "KMV",
                            "Degree Bucket": bucket_labels[i],
                            "Agreement": agree_kmv[final_mask].mean()
                        })
                        results.append({
                            "Model": model_name,
                            "Method": "Random",
                            "Degree Bucket": bucket_labels[i],
                            "Agreement": agree_rand[final_mask].mean()
                        })

        # Plotting
        if not results: return

        df = pd.DataFrame(results)
        g = sns.FacetGrid(df, col="Model", hue="Method", height=5, aspect=0.8, 
                          palette={'KMV': '#2980b9', 'Random': '#7f8c8d'})
        g.map(sns.lineplot, "Degree Bucket", "Agreement", marker="o", linewidth=2.5)
        
        safe_filename = metapath.replace(',', '_').replace(' ', '').replace('->', '_to_')[:100]
        
        g.fig.suptitle(f"Failure Analysis: {dataset} (K={k_val})\n{metapath[:40]}...", fontsize=14, y=1.05)
        g.set_axis_labels("Node Degree", "Agreement with Exact")
        g.add_legend(title="Method")
        
        for ax in g.axes.flat:
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.2)

        # --- FIX: DISTINCT DIRECTORY & PREFIX ---
        output_dir = os.path.join(ROOT_DIR, "output", "failure_diagnostics")
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"failure_case_{dataset}_{safe_filename}.png"
        save_path = os.path.join(output_dir, filename)
        
        plt.savefig(save_path, bbox_inches='tight')
        print(f" -> Saved plot: {save_path}")
        plt.close()

    except Exception as e:
        print(f"[CRITICAL SKIP] Error in {dataset}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--metapath', default=None)
    parser.add_argument('--k', type=int, default=4)
    args = parser.parse_args()
    
    if args.dataset and args.metapath:
        run_stratified_analysis(args.dataset, args.metapath, args.k)
    else:
        print("--- Batch Mode (Failure Analysis) ---")
        for ds_name, paths in DATASETS.items():
            for path in paths:
                run_stratified_analysis(ds_name, path, args.k)